# audio_detector.py
# ---------------------------------------------
# 실시간 사운드 감지 + PANNs 분류 + WebSocket 전송
# - ReSpeaker USB Mic Array + tuning
# - MIC_SAMPLE_RATE → 리샘플링(→ MODEL_SAMPLE_RATE=32kHz) 후 PANNs 추론
# - Node WebSocket 서버에 detection JSON 전송
# - Node 서버에서 오는 config_update 메시지로
#   CONF_THRESH / DETECT_DURATION / PRE_BUFFER_DURATION 실시간 변경
# ---------------------------------------------

import os
import time
import json
import queue
import threading

import numpy as np
import sounddevice as sd
import torch
import librosa
import usb.core
import usb.util
import websocket  # websocket-client
from websocket import WebSocketTimeoutException

from panns_inference.models import Cnn14
from tuning import Tuning

# {
#   "type": "config_update",
#   "config": {
#     "CONF_THRESH": 0.7,
#     "DETECT_DURATION": 0.8,
#     "PRE_BUFFER_DURATION": 0.3
#   }
# }

# -----------------------------
# 기본 설정 (환경변수로 override 가능)
# -----------------------------
MIC_SAMPLE_RATE = int(os.getenv("MIC_SAMPLE_RATE", "16000"))      # 입력(마이크) 레이트
MODEL_SAMPLE_RATE = int(os.getenv("MODEL_SAMPLE_RATE", "32000"))  # PANNs 기준 레이트 (32kHz)

PRE_BUFFER_DURATION = float(os.getenv("PRE_BUFFER_DURATION", "0.2"))  # 감지 이전 pre-buffer 길이(초)
DETECT_DURATION = float(os.getenv("DETECT_DURATION", "0.5"))          # 감지 구간 길이(초)

TARGET_CHANNEL = int(os.getenv("TARGET_CHANNEL", "0"))  # ReSpeaker 채널 인덱스
DEVICE_INDEX = int(os.getenv("DEVICE_INDEX", "1"))      # sounddevice 디바이스 인덱스
MIN_WAVEFORM_LENGTH = MODEL_SAMPLE_RATE                 # 최소 1초 @ 32kHz

CONF_THRESH = float(os.getenv("CONF_THRESH", "0.5"))    # softmax 임계값


LABEL_ORDER = [
    "/m/09x0r", "/m/05tny_", "/m/0bt9lr",
    "/m/0912c9", "/m/014zdl", "/m/07yv9", "/m/03kmc9"
]

BASE_CKPT_PATH = os.getenv(
    "BASE_CKPT_PATH",
    r"C:\Users\juuip\panns_data\Cnn14_mAP=0.431.pth"
)

FINETUNED_CKPT = os.getenv(
    "FINETUNED_CKPT",
    r"C:\OpenSourcePJ\server\best_panns6_acc0.828.pt"
)

# Node WS URL (환경변수로도 변경 가능)
NODE_WS_URL = os.getenv("NODE_WS_URL", "ws://localhost:8080")

# -----------------------------
# 설정값 공유용 (실시간 업데이트용)
# -----------------------------
config_lock = threading.Lock()
CONFIG = {
    "CONF_THRESH": CONF_THRESH,
    "DETECT_DURATION": DETECT_DURATION,
    "PRE_BUFFER_DURATION": PRE_BUFFER_DURATION,
}

# -----------------------------
#  오디오 큐 및 pre-buffer (입력 레이트 기준)
# -----------------------------
audio_queue = queue.Queue()
pre_buffer_size = int(MIC_SAMPLE_RATE * PRE_BUFFER_DURATION)
pre_buffer = np.zeros(pre_buffer_size, dtype=np.float32)


def apply_config(new_cfg: dict):
    """Node 서버에서 받은 설정을 런타임에 반영"""
    global CONF_THRESH, DETECT_DURATION, PRE_BUFFER_DURATION
    global CONFIG, pre_buffer_size, pre_buffer

    with config_lock:
        if "CONF_THRESH" in new_cfg:
            CONF_THRESH = float(new_cfg["CONF_THRESH"])
            CONFIG["CONF_THRESH"] = CONF_THRESH

        if "DETECT_DURATION" in new_cfg:
            DETECT_DURATION = float(new_cfg["DETECT_DURATION"])
            CONFIG["DETECT_DURATION"] = DETECT_DURATION

        if "PRE_BUFFER_DURATION" in new_cfg:
            PRE_BUFFER_DURATION = float(new_cfg["PRE_BUFFER_DURATION"])
            if PRE_BUFFER_DURATION < 0:
                PRE_BUFFER_DURATION = 0.0
            CONFIG["PRE_BUFFER_DURATION"] = PRE_BUFFER_DURATION

            # pre_buffer_size 재계산
            new_size = max(1, int(MIC_SAMPLE_RATE * PRE_BUFFER_DURATION))

            if new_size != pre_buffer_size:
                # 기존 버퍼의 꼬리 부분을 최대한 살려서 새 버퍼로 복사
                old_buf = pre_buffer
                if new_size <= len(old_buf):
                    new_buf = old_buf[-new_size:].copy()
                else:
                    new_buf = np.zeros(new_size, dtype=np.float32)
                    new_buf[-len(old_buf):] = old_buf

                pre_buffer_size = new_size
                pre_buffer = new_buf

        print("[CONFIG] updated:", CONFIG, flush=True)


# -----------------------------
# ReSpeaker 장치 초기화
# -----------------------------
dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
if not dev:
    raise RuntimeError("ReSpeaker 장치를 찾을 수 없습니다. (idVendor=0x2886, idProduct=0x0018)")
Mic_tuning = Tuning(dev)


# -----------------------------
# PANNs6Head 정의
# -----------------------------
class PANNs6Head(torch.nn.Module):
    def __init__(self, checkpoint_path: str, num_classes: int):
        super().__init__()
        # backbone: 원본 Cnn14
        self.backbone = Cnn14(
            sample_rate=MODEL_SAMPLE_RATE,   # 32kHz 기준
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527,
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        self.backbone.load_state_dict(state, strict=False)

        in_dim = self.backbone.fc_audioset.in_features
        self.head = torch.nn.Linear(in_dim, num_classes)

    def forward(self, wav):
        """
        wav: [B, T] (32kHz)
        return: {"logits": [B, num_classes]}
        """
        out = self.backbone(wav)
        emb = out["embedding"]
        logits = self.head(emb)
        return {"logits": logits}


# -----------------------------
#  파인튜닝 가중치 로드
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] device:", device)

audio_model = PANNs6Head(BASE_CKPT_PATH, num_classes=len(LABEL_ORDER)).to(device)

ckpt = torch.load(FINETUNED_CKPT, map_location=device)
LABEL_NAMES = ckpt.get(
    "label_names",
    ["Speech", "Bark", "Dog", "Vehicle horn", "Explosion", "Vehicle", "Siren"]
)
print("[INFO] LABEL_NAMES:", LABEL_NAMES)

audio_model.load_state_dict(ckpt["state_dict"], strict=False)
audio_model.eval()


# -----------------------------
#  WebSocket 클라이언트 (파이썬 → Node + Node → 파이썬)
# -----------------------------
ws_lock = threading.Lock()
ws_conn = None
ws_should_run = True


def _send_ws(obj: dict):
    """공용 WS send 함수 (예외/락 처리)"""
    global ws_conn
    msg = json.dumps(obj, ensure_ascii=False)
    with ws_lock:
        if ws_conn is None:
            return
        try:
            ws_conn.send(msg)
        except Exception as e:
            print("[WS] send error:", e)
            ws_conn = None


def handle_ws_message(msg: str):
    """서버 → 파이썬으로 오는 메시지 처리 (ping / config_update)"""
    try:
        payload = json.loads(msg)
    except json.JSONDecodeError:
        # 이상한 문자열이면 무시
        # print("[WS] invalid JSON:", msg)
        return

    mtype = payload.get("type")

    # ping 오면 바로 pong 응답 + 로그 X
    if mtype == "ping":
        t = payload.get("t")
        _send_ws({"type": "pong", "t": t})
        return

    if mtype == "config_update":
        cfg = payload.get("config", {})
        print("[WS] config_update received:", cfg, flush=True)
        apply_config(cfg)
        return

    # 필요하면 다른 타입도 여기서 처리
    # print("[WS] other message:", payload, flush=True)


def ws_connect_loop():
    """Node 서버와 WebSocket 연결 유지 (양방향: send + recv)"""
    global ws_conn
    while ws_should_run:
        try:
            print(f"[WS] trying connect to {NODE_WS_URL}", flush=True)
            ws = websocket.create_connection(NODE_WS_URL, timeout=3)
            ws.settimeout(1.0)

            with ws_lock:
                ws_conn = ws

            print("[WS] Connected OK", flush=True)

            while ws_should_run:
                try:
                    msg = ws.recv()
                    if not msg:
                        continue
                    # 내용은 handle_ws_message에서만 처리
                    handle_ws_message(msg)
                except WebSocketTimeoutException:
                    continue
                except Exception as e:
                    print("[WS] recv error:", repr(e), flush=True)
                    break

        except Exception as e:
            print("[WS] connection error in loop:", repr(e), flush=True)

        with ws_lock:
            ws_conn = None

        if ws_should_run:
            print("[WS] retry in 3s...", flush=True)
            time.sleep(3)


def audio_callback(indata, frames, time_info, status):
    """sounddevice InputStream 콜백: 오디오를 큐에 넣고, pre_buffer는 collector가 관리"""
    if status:
        print(status)
    # indata: [frames, channels], MIC_SAMPLE_RATE 기준
    audio_queue.put(indata[:, TARGET_CHANNEL].astype(np.float32))


def audio_collector():
    """실시간으로 pre_buffer(과거 PRE_BUFFER_DURATION 초 구간) 유지"""
    global pre_buffer
    while True:
        chunk = audio_queue.get()
        # pre_buffer_size는 apply_config에서 변경될 수 있음
        if len(chunk) >= pre_buffer_size:
            pre_buffer = chunk[-pre_buffer_size:]
        else:
            pre_buffer = np.roll(pre_buffer, -len(chunk))
            pre_buffer[-len(chunk):] = chunk


# -----------------------------
#  감지 + 분석 쓰레드
# -----------------------------
def audio_detector_analyzer():
    """
    Mic_tuning.is_voice() 가 true면
    - pre_buffer(PRE_BUFFER_DURATION 초) + 현재 detect_duration 초 오디오 묶어서
    - 32kHz로 리샘플링 → PANNs 추론 → top1 결과를 Node로 전송
    """
    while True:
        try:
            if Mic_tuning.is_voice():
                doa = Mic_tuning.direction

                # 설정값 읽기 (락으로 보호)
                with config_lock:
                    detect_duration = CONFIG["DETECT_DURATION"]
                    conf_thresh = CONFIG["CONF_THRESH"]

                # detect_size: 입력 레이트 기준 detect_duration 초
                detect_size = int(MIC_SAMPLE_RATE * detect_duration)
                collected = []

                while len(collected) < detect_size:
                    chunk = audio_queue.get()
                    collected.extend(chunk)

                collected = np.array(collected[:detect_size], dtype=np.float32)

                # pre_buffer + 현재 detect_duration 초 = 총 (pre_buffer_duration + detect_duration)초 @ MIC_SAMPLE_RATE
                waveform_in = np.concatenate([pre_buffer, collected])  # @ MIC_SAMPLE_RATE

                # 32kHz로 리샘플링
                waveform_32k = librosa.resample(
                    waveform_in,
                    orig_sr=MIC_SAMPLE_RATE,
                    target_sr=MODEL_SAMPLE_RATE,
                )

                # 최소 1초(32000샘플) 보장 (부족하면 뒤에 0 패딩)
                if len(waveform_32k) < MIN_WAVEFORM_LENGTH:
                    pad_len = MIN_WAVEFORM_LENGTH - len(waveform_32k)
                    waveform_32k = np.pad(waveform_32k, (0, pad_len))

                waveform_tensor = (
                    torch.from_numpy(waveform_32k)
                    .float()
                    .unsqueeze(0)  # [1, T]
                    .to(device)
                )

                with torch.no_grad():
                    out = audio_model(waveform_tensor)
                    logits = out["logits"][0]
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()

                scores = probs.tolist()
                pairs = [(LABEL_NAMES[i], scores[i]) for i in range(len(LABEL_NAMES))]

                # Dog / Bark 통합
                dog_score = next((s for l, s in pairs if l == "Dog"), 0.0)
                bark_score = next((s for l, s in pairs if l == "Bark"), 0.0)
                if (dog_score > 0.0) or (bark_score > 0.0):
                    combined_score = dog_score + bark_score
                    dog_label = "Dog"  # 필요하면 "Dog (Barking)" 등으로 변경 가능
                    pairs = [(l, s) for (l, s) in pairs if l not in ("Dog", "Bark")]
                    pairs.append((dog_label, combined_score))

                # Vehicle / Vehicle horn 통합
                vehicle_score = next((s for l, s in pairs if l == "Vehicle"), 0.0)
                horn_score = next((s for l, s in pairs if l == "Vehicle horn"), 0.0)
                if (vehicle_score > 0.0) or (horn_score > 0.0):
                    combined_score = vehicle_score + horn_score
                    vehicle_label = "Vehicle" if vehicle_score >= horn_score else "Vehicle (Horn)"
                    pairs = [(l, s) for (l, s) in pairs if l not in ("Vehicle", "Vehicle horn")]
                    pairs.append((vehicle_label, combined_score))

                # 정렬 후 top1만 사용
                pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
                if not pairs_sorted:
                    time.sleep(0.01)
                    continue

                top1_label, top1_score = pairs_sorted[0]

                if top1_score >= conf_thresh:
                    output = {
                        "type": "detection",
                        "timestamp": time.time_ns() // 1_000_000,  # ms 단위
                        "doa": doa,
                        "tags": [
                            {
                                "label": top1_label,
                                "score": round(float(top1_score), 2),
                            }
                        ],
                    }

                    # Node 서버로 전송
                    _send_ws(output)

                # 너무 자주 연속 추론되는 거 막고 싶으면 감지 구간만큼 sleep
                time.sleep(detect_duration)

            else:
                time.sleep(0.01)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print("[DETECTOR] error:", e)
            time.sleep(0.1)


# -----------------------------
#  시작 코드
# -----------------------------
if __name__ == "__main__":
    # WS 클라이언트 스레드 시작 (Node 서버 연결 + config_update 수신)
    ws_thread = threading.Thread(target=ws_connect_loop, daemon=True)
    ws_thread.start()

    # 오디오 스트리밍 + 감지/분석 스레드 시작
    stream = sd.InputStream(
        samplerate=MIC_SAMPLE_RATE,              # 입력 레이트 기준
        channels=6,
        device=DEVICE_INDEX,
        callback=audio_callback,
        blocksize=int(MIC_SAMPLE_RATE * 0.1),   # 0.1초 블록
    )

    collector_thread = threading.Thread(target=audio_collector, daemon=True)
    detector_thread = threading.Thread(target=audio_detector_analyzer, daemon=True)

    collector_thread.start()
    detector_thread.start()

    print("실시간 감지 + 리샘플링(→32k) + PANNs + WS 전송 시작...\n(CTRL+C로 종료)\n")
    print("--------------------------------------")
    print("NODE_WS_URL:", NODE_WS_URL)
    print("초기 CONFIG:", CONFIG)
    print("--------------------------------------")

    with stream:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n실시간 분석 종료")
