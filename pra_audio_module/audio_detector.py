import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import queue
import threading
import numpy as np
import sounddevice as sd
import torch
import librosa
import pyroomacoustics as pra
from pyroomacoustics import transform
from panns_inference.models import Cnn14

from web_socket_client import WebSocketClient

# =========================
# 설정
# =========================
MIC_SR = int(os.getenv("MIC_SAMPLE_RATE", "16000"))
MODEL_SR = int(os.getenv("MODEL_SAMPLE_RATE", "32000"))

IN_CHANNELS = 6
RAW_SLICE = slice(1, 5)  # ch1~ch4 (raw mic)

WIN_SEC = float(os.getenv("WIN_SEC", "1.0"))
HOP_SEC = float(os.getenv("HOP_SEC", "0.10"))

NFFT = int(os.getenv("NFFT", "512"))
STFT_HOP = int(os.getenv("STFT_HOP", str(NFFT // 2)))

FREQ_LOW = int(os.getenv("FREQ_LOW", "800"))
FREQ_HIGH = int(os.getenv("FREQ_HIGH", "3500"))
GRID_STEP_DEG = float(os.getenv("GRID_STEP_DEG", "2.0"))

C = float(os.getenv("C", "343.0"))
pra.parameters.constants.set("c", C)  # pyroomacoustics 내부 음속 맞춤

MIN_DB = float(os.getenv("MIN_DB", "-35.0"))
PRINT_MIN_INTERVAL = float(os.getenv("PRINT_MIN_INTERVAL", "0.20"))
INFER_MIN_INTERVAL = float(os.getenv("INFER_MIN_INTERVAL", "0.30"))

# 좌표계 보정
AZ_OFFSET_DEG = float(os.getenv("AZ_OFFSET_DEG", "40.0"))
AZ_FLIP = os.getenv("AZ_FLIP", "0") == "1"

# PANNs
CONF_THRESH = float(os.getenv("CONF_THRESH", "0.5"))
MIN_WAVEFORM_LENGTH = MODEL_SR  # 최소 1초 @ 32k

LABEL_ORDER = [
    "/m/09x0r", "/m/05tny_", "/m/0bt9lr",
    "/m/0912c9", "/m/014zdl", "/m/07yv9", "/m/03kmc9"
]
BASE_CKPT_PATH = os.getenv("BASE_CKPT_PATH", r"C:\Users\juuip\panns_data\Cnn14_mAP=0.431.pth")
FINETUNED_CKPT = os.getenv("FINETUNED_CKPT", r"C:\OpenSourcePJ\server\best_panns6_acc0.828.pt")

# WebSocket
NODE_WS_URL = os.getenv("NODE_WS_URL", "ws://localhost:8080")

# device
DEVICE_INDEX = os.getenv("DEVICE_INDEX")
DEVICE_INDEX = int(DEVICE_INDEX) if DEVICE_INDEX is not None else None

# AttributeError 에러 방지용)
if not hasattr(transform, "analysis"):
    from pyroomacoustics.transform import stft
    transform.analysis = stft.analysis
    transform.synthesis = stft.synthesis
    transform.compute_synthesis_window = stft.compute_synthesis_window

# 마이크 좌표 (2D, meters)
# ReSpeaker 4-mic array (usb_4_mic_array) 십자 형태 근사:
# (-0.032,0), (0,-0.032), (0.032,0), (0,0.032)
L2 = np.array([
    [-0.032,  0.000, +0.032,  0.000],
    [ 0.000, -0.032,  0.000, +0.032],
], dtype=np.float32)  # (2,4)

# =========================
# WS config_update 적용
# =========================
config_lock = threading.Lock()

def apply_config(new_cfg: dict):
    """Node 서버에서 받은 설정을 런타임에 반영"""
    global CONF_THRESH, MIN_DB, PRINT_MIN_INTERVAL, INFER_MIN_INTERVAL
    with config_lock:
        if "CONF_THRESH" in new_cfg:
            CONF_THRESH = float(new_cfg["CONF_THRESH"])
        if "MIN_DB" in new_cfg:
            MIN_DB = float(new_cfg["MIN_DB"])
        if "PRINT_MIN_INTERVAL" in new_cfg:
            PRINT_MIN_INTERVAL = float(new_cfg["PRINT_MIN_INTERVAL"])
        if "INFER_MIN_INTERVAL" in new_cfg:
            INFER_MIN_INTERVAL = float(new_cfg["INFER_MIN_INTERVAL"])

    print("[CONFIG] updated:", {
        "CONF_THRESH": CONF_THRESH,
        "MIN_DB": MIN_DB,
        "PRINT_MIN_INTERVAL": PRINT_MIN_INTERVAL,
        "INFER_MIN_INTERVAL": INFER_MIN_INTERVAL,
    }, flush=True)

def handle_ws_payload(payload: dict):
    """서버 -> 파이썬 메시지 처리 (config_update 등)"""
    if payload.get("type") == "config_update":
        cfg = payload.get("config", {})
        apply_config(cfg)

# # 마이크 각도 변환 유틸
def wrap_deg(d: float) -> float:
    d = float(d) % 360.0
    return d + 360.0 if d < 0 else d

def find_respeaker_device(min_channels: int = 6):
    try:
        devs = sd.query_devices()
    except Exception:
        return None
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) >= min_channels:
            name = d.get("name", "")
            if "ReSpeaker" in name or "respeaker" in name.lower():
                return i
    return None

# Shortcut Fourier transform
# 디지털 신호(DS)를 짧은 조각들(Shortcut)으로 잘라서, 각 조각들을 주파수로 변환하는 함수
# 즉, 디지털 신호를 주파수로 변환
# 입력 : x_mn(마이크 개수, 샘플 수), nfff(한 조각 길이), hop(다음 조각으로 이동하는 간격)
# 출력 : (마이크 개수, 주파수 bin 개수, 시간 프레임 개수)
# 예시) 1초 길이의 오디오 샘플(512 niff, 256 hop)
#   [--------- 1초 ---------]
#   [---512---]
#         [---512---]
#               [---512---]
def dbfs_from_multich(x_mn: np.ndarray) -> float:
    v = x_mn.mean(axis=0)
    rms = float(np.sqrt(np.mean(v * v)))
    return 20.0 * np.log10(rms + 1e-12)

def stft_mfs(x_mn: np.ndarray, nfft: int, hop: int) -> np.ndarray:
    M, N = x_mn.shape
    if N < nfft:
        x_mn = np.pad(x_mn, ((0, 0), (0, nfft - N)))
        N = x_mn.shape[1]

    win = np.hanning(nfft).astype(np.float32)
    S = 1 + (N - nfft) // hop   # 조각 개수
    F = nfft // 2 + 1           # 주파수 bin 개수로 nfft가 512라면, 실제 신호인 FT(푸리에 변환) 결과가 대칭(중복)이기 때문에 반만 사용

    X = np.empty((M, F, S), dtype=np.complex64)
    for s in range(S):
        st = s * hop
        frame = x_mn[:, st:st + nfft] * win[None, :]
        X[:, :, s] = np.fft.rfft(frame, n=nfft, axis=1) # fft(fast fourier transform, 고속 푸리에 변환), 실제 샘플을 주파수 성분(Hz)으로 바꿔줌
    return X

# SRP-PHAT (Steered Response Power with Phase Transform) 
# 소리 방향(음원 위치) 추정 및 추적
def srp_phat_scan(X_mfs: np.ndarray, L2: np.ndarray, fs: int, nfft: int,
                  f_low: int, f_high: int, grid_step_deg: float, c: float):
    M, F, S = X_mfs.shape

    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)     # 주파수 bin 선택
    band = (freqs >= f_low) & (freqs <= f_high) # 주파수 대역 게이트, 이후 특정 범위 주파수만 사용하여 엄청 낮은 저주파(바람/진동)이나 고주파(노이즈)를 제외
    f_sel = freqs[band].astype(np.float32)
    if f_sel.size < 4:
        raise ValueError("freq bins too small: increase NFFT or widen band")

    # 마이크 쌍 인덱스 생성 ex) (0,1)(0,2)(0,3)(1,2)(1,3)(2,3)
    pair_i, pair_j = np.triu_indices(M, k=1)

    # 특정 마이크 쌍의 STFT만 추출 (P,F,S)
    Xi = X_mfs[pair_i, :, :]
    Xj = X_mfs[pair_j, :, :]

    # GCC(Generalized Cross Corrlelation) 계산
    # Xi * np.conj(Xj) : 마이크 i와 j의 상관관계
    # 프레임마다 안정적인 값 사용을 위해 평균 사용
    G = np.mean(Xi * np.conj(Xj), axis=2)  

    G = G[:, band]  # 주파수 대역폭 자르기

    # PHAT 가중치 사용 (크기는 버리고 위상 중점으로)
    # G / |G| : 크기는 거의 1이 되고, 위상만 남는 형태로 변환
    Gphat = G / (np.abs(G) + 1e-12) # G/|G|

    # 각도 후보 그리드(0 ~ 360) 생성
    # grid_step_deg=1.0이면 0,1,2,3,~,349 : 360개 각도 후보 생성
    grid_deg = np.arange(0.0, 360.0, grid_step_deg, dtype=np.float32)
    th = np.deg2rad(grid_deg)

    # 생성된 각도 후보를 (평면파 가정) 2D 방향 벡터로 변환
    # ex) 300도 : (x, y) 벡터
    u = np.stack([np.cos(th), np.sin(th)], axis=0).astype(np.float32)  

    # 각도별 마이크 도달시간(기대 지연 시간) 즉, k 각도에서 m번 마이크 지연을 계산
    # p_m : 마이크 m의 위치, u : 각도 후보, c : 음속 
    # t_m(각도) = (u * p_m) / c
    # L2 = 마이크 위치 배열, 따라서 p_m = L2[:,m]
    tau_mk = (u.T @ L2).astype(np.float32) / float(c) 

    # GCC-PHAT와 통합하기 위해 
    # 위에 계산된 마이크 기대 지연 시간을 마이크 쌍의 지연 시간으로 변환
    # k 각도에서의 i번째 마이크 지연 시간 - j번째 마이크 지연 시간 
    tau_pk = tau_mk[:, pair_i] - tau_mk[:, pair_j]    

    P_grid = np.zeros((grid_deg.size,), dtype=np.float32)

    # Gphat : 앞서 계산된 마이크 i와 j 사이의 GCC-PHAT(상관 위상 계수 또는 위상차)
    # phase(이론적 위상차) : k각도에서 마이크 쌍의 지연 시간이 tau_pk일 때, 이론적인 두 마이크 사이 위상 차이
    # 따라서 실제 위상차(Gphat)와 이론적 위상차(phase)를 곱했을 때, 후보 각도 k가 소리 방향이 맞다면
    # 위상이 정렬되고, 아니라면 위상이 서로 엊갈려 상쇄됨
    # 이후 모든 마이크 쌍의 정보(값)와 대역 전체 정보를 합산해서 안정적인 DOA를 도출 
    for k in range(grid_deg.size):
        phase = np.exp(1j * 2.0 * np.pi * (f_sel[None, :] * tau_pk[k, :, None]))  # (P,Fb)
        P_grid[k] = float(np.real(Gphat * phase).sum())

    # 가장 큰 점수의 각도를 DOA로 선택
    best_idx = int(np.argmax(P_grid))
    return float(grid_deg[best_idx]), grid_deg, P_grid

# PANNs
class PANNs6Head(torch.nn.Module):
    def __init__(self, checkpoint_path: str, num_classes: int):
        super().__init__()
        self.backbone = Cnn14(
            sample_rate=MODEL_SR,
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
        out = self.backbone(wav)
        emb = out["embedding"]
        return {"logits": self.head(emb)}

def postprocess_pairs(label_names, probs):
    pairs = [(label_names[i], float(probs[i])) for i in range(len(label_names))]

    # Dog/Bark 통합
    dog = next((s for l, s in pairs if l == "Dog"), 0.0)
    bark = next((s for l, s in pairs if l == "Bark"), 0.0)
    if dog > 0.0 or bark > 0.0:
        pairs = [(l, s) for (l, s) in pairs if l not in ("Dog", "Bark")]
        pairs.append(("Dog", dog + bark))

    # Vehicle/Horn 통합
    veh = next((s for l, s in pairs if l == "Vehicle"), 0.0)
    horn = next((s for l, s in pairs if l == "Vehicle horn"), 0.0)
    if veh > 0.0 or horn > 0.0:
        pairs = [(l, s) for (l, s) in pairs if l not in ("Vehicle", "Vehicle horn")]
        pairs.append(("Vehicle", veh + horn))

    return sorted(pairs, key=lambda x: x[1], reverse=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] torch device:", device)

audio_model = PANNs6Head(BASE_CKPT_PATH, num_classes=len(LABEL_ORDER)).to(device)
ckpt = torch.load(FINETUNED_CKPT, map_location=device)
LABEL_NAMES = ckpt.get("label_names", ["Speech", "Bark", "Dog", "Vehicle horn", "Explosion", "Vehicle", "Siren"])
audio_model.load_state_dict(ckpt["state_dict"], strict=False)
audio_model.eval()
print("[INFO] LABEL_NAMES:", LABEL_NAMES)

def run_panns(y_16k: np.ndarray):
    y_32k = librosa.resample(y_16k, orig_sr=MIC_SR, target_sr=MODEL_SR)
    if len(y_32k) < MIN_WAVEFORM_LENGTH:
        y_32k = np.pad(y_32k, (0, MIN_WAVEFORM_LENGTH - len(y_32k)))
    wav = torch.from_numpy(y_32k).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = audio_model(wav)["logits"][0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    pairs = postprocess_pairs(LABEL_NAMES, probs)
    return pairs[0][0], float(pairs[0][1])

# 메인
def main():
    # WS 시작
    ws = WebSocketClient(NODE_WS_URL, on_payload=handle_ws_payload, reconnect_sec=3.0)
    ws.start()

    dev_idx = DEVICE_INDEX if DEVICE_INDEX is not None else find_respeaker_device(IN_CHANNELS)
    win_samples = int(MIC_SR * WIN_SEC)
    blocksize = int(MIC_SR * HOP_SEC)

    ring = np.zeros((win_samples, IN_CHANNELS), dtype=np.float32)
    q = queue.Queue(maxsize=200)

    # pyroomacoustics Beamformer 
    # L2 : 마이크 좌표, fs : 오디오 샘플레이트, N : FT 프레임 길이, hop : 프레임 간격, Lg : 필터 길이
    bf = pra.Beamformer(L2, fs=MIC_SR, N=NFFT, hop=STFT_HOP, Lg=NFFT)

    def cb(indata, frames, time_info, status):
        if status:
            pass
        try:
            q.put_nowait(indata.copy())
        except queue.Full:
            try:
                _ = q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(indata.copy())
            except queue.Full:
                pass

    print("\nStart... Ctrl+C to stop")
    print(f"[AUDIO] device={dev_idx}, SR={MIC_SR}, in_ch={IN_CHANNELS}, raw=ch1~ch4")
    print(f"[DOA ] SRP-PHAT manual, band={FREQ_LOW}-{FREQ_HIGH}Hz, step={GRID_STEP_DEG}deg")
    print(f"[BF  ] pyroomacoustics Beamformer.far_field_weights + process(FD=True)")
    print(f"[GATE] MIN_DB={MIN_DB} dBFS (db만 사용)")
    print(f"[PANN] CONF_THRESH={CONF_THRESH}")
    print(f"[WS  ] NODE_WS_URL={NODE_WS_URL}\n")

    last_print = 0.0
    last_infer = 0.0

    try:
        with sd.InputStream(
            device=dev_idx,
            samplerate=MIC_SR,
            channels=IN_CHANNELS,
            dtype="float32",
            blocksize=blocksize,
            callback=cb,
        ):
            while True:
                chunk = q.get()  # (frames,6)

                # ring update
                n = chunk.shape[0]
                if n >= win_samples:
                    ring[:] = chunk[-win_samples:, :]
                else:
                    ring[:-n, :] = ring[n:, :]
                    ring[-n:, :] = chunk

                raw4 = ring[:, RAW_SLICE]  # (win,4)
                x_mn = raw4.T              # (4,win)

                with config_lock:
                    min_db = MIN_DB
                    conf_thresh = CONF_THRESH
                    infer_min_interval = INFER_MIN_INTERVAL
                    print_min_interval = PRINT_MIN_INTERVAL

                db = dbfs_from_multich(x_mn)
                if db < min_db:
                    continue

                # DC bias 제거
                # 마이크 내부 회로나 드라이버 처리 과정에서 전체적인 파형이 상하로 밀릴 수 있음
                # Respeaker Mic Raw 채널에서는 전처리 과정을 거치지 않는다고 하지만
                # SRP-PHAT 같은 위상의 상관을 계산하는 경우, 편향이 생길 수 있어 안정성 측면에서 사용하는 방법
                x_mn = x_mn - x_mn.mean(axis=1, keepdims=True)

                # DOA용 STFT
                X = stft_mfs(x_mn, nfft=NFFT, hop=STFT_HOP)

                # DOA (SRP-PHAT manual)
                best_deg, _, _ = srp_phat_scan(
                    X, L2, MIC_SR, NFFT, FREQ_LOW, FREQ_HIGH, GRID_STEP_DEG, C
                )

                # 마이크 각도 보정
                if AZ_FLIP:
                    best_deg = (-best_deg) % 360.0
                best_deg = wrap_deg(best_deg + AZ_OFFSET_DEG)

                now = time.time()
                if now - last_infer < infer_min_interval:
                    continue

                # ---------- pyroomacoustics beamforming ----------
                bf.signals = x_mn.astype(np.float32)  # (M,N)
                phi = float(np.deg2rad(best_deg))     # rad
                bf.far_field_weights(phi)             # steer
                y = bf.process(FD=True).astype(np.float32)
                y = y[:x_mn.shape[1]]

                # PANNs
                label, score = run_panns(y)

                # (Node로 전송) - PASS일 때만 전송
                if score >= conf_thresh:
                    ws.send_json({
                        "type": "detection",
                        "timestamp": time.time_ns() // 1_000_000,  # ms
                        "doa": float(best_deg),
                        "db": float(db),
                        "tags": [{
                            "label": label,
                            "score": round(float(score), 2),
                        }],
                    })

                if now - last_print >= print_min_interval:
                    tag = "PASS" if score >= conf_thresh else "low"
                    # Windows cp949 콘솔에서 '≈', '°' 같은 문자가 터질 수 있어서 ASCII로 출력
                    print(f"DOA~{best_deg:6.1f} deg | dB={db:6.1f} | {label:>10s} {score:.2f} | {tag}")
                    last_print = now

                last_infer = now

    finally:
        ws.stop()
