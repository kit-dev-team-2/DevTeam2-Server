# audio_detector.py
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

from panns_inference.models import Cnn14
from tuning import Tuning


class PANNs6Head(torch.nn.Module):
    def __init__(self, checkpoint_path: str, num_classes: int, model_sr: int):
        super().__init__()
        self.backbone = Cnn14(
            sample_rate=model_sr,
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
        logits = self.head(emb)
        return {"logits": logits}


class AudioDetector:
    def __init__(self):
        # -----------------------------
        # 기본 설정 (환경변수 override 가능)
        # -----------------------------
        self.MIC_SAMPLE_RATE = int(os.getenv("MIC_SAMPLE_RATE", "16000"))
        self.MODEL_SAMPLE_RATE = int(os.getenv("MODEL_SAMPLE_RATE", "32000"))

        self.PRE_BUFFER_DURATION = float(os.getenv("PRE_BUFFER_DURATION", "0.2"))
        self.DETECT_DURATION = float(os.getenv("DETECT_DURATION", "0.5"))

        self.TARGET_CHANNEL = int(os.getenv("TARGET_CHANNEL", "0"))
        self.DEVICE_INDEX = int(os.getenv("DEVICE_INDEX", "1"))
        self.MIN_WAVEFORM_LENGTH = self.MODEL_SAMPLE_RATE

        self.CONF_THRESH = float(os.getenv("CONF_THRESH", "0.5"))

        self.BASE_CKPT_PATH = os.getenv(
            "BASE_CKPT_PATH", r"C:\Users\juuip\panns_data\Cnn14_mAP=0.431.pth"
        )
        self.FINETUNED_CKPT = os.getenv(
            "FINETUNED_CKPT", r"C:\OpenSourcePJ\server\best_panns6_acc0.828.pt"
        )

        # 공유 config
        self.config_lock = threading.Lock()
        self.CONFIG = {
            "CONF_THRESH": self.CONF_THRESH,
            "DETECT_DURATION": self.DETECT_DURATION,
            "PRE_BUFFER_DURATION": self.PRE_BUFFER_DURATION,
        }

        # 오디오 큐를 2개로 분리(collector용 / detector용)
        self.q_collect = queue.Queue()
        self.q_detect = queue.Queue()

        self.prebuf_lock = threading.Lock()
        self.pre_buffer_size = int(self.MIC_SAMPLE_RATE * self.PRE_BUFFER_DURATION)
        self.pre_buffer = np.zeros(self.pre_buffer_size, dtype=np.float32)

        # ReSpeaker
        dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        if not dev:
            raise RuntimeError("ReSpeaker 장치를 찾을 수 없습니다. (0x2886, 0x0018)")
        self.Mic_tuning = Tuning(dev)

        # Model load
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[INFO] device:", self.device)

        self.audio_model = PANNs6Head(
            self.BASE_CKPT_PATH,
            num_classes=7,  # label_names 기반(아래에서 로드)
            model_sr=self.MODEL_SAMPLE_RATE,
        ).to(self.device)

        ckpt = torch.load(self.FINETUNED_CKPT, map_location=self.device)
        self.LABEL_NAMES = ckpt.get(
            "label_names",
            ["Speech", "Bark", "Dog", "Vehicle horn", "Explosion", "Vehicle", "Siren"],
        )
        print("[INFO] LABEL_NAMES:", self.LABEL_NAMES)

        self.audio_model.load_state_dict(ckpt["state_dict"], strict=False)
        self.audio_model.eval()

        # WSClient (main에서 주입)
        self.ws_client = None

        # threads / stream
        self._threads = []
        self._stream = None
        self._run = True

    def set_ws_client(self, ws_client):
        self.ws_client = ws_client

    # ---------- WS 수신 처리 ----------
    def on_ws_message(self, msg: str):
        try:
            payload = json.loads(msg)
        except json.JSONDecodeError:
            return

        t = payload.get("type")
        if t == "ping":
            if self.ws_client:
                self.ws_client.send_json({"type": "pong", "t": payload.get("t")})
            return

        if t == "config_update":
            cfg = payload.get("config", {})
            print("[WS] config_update:", cfg, flush=True)
            self.apply_config(cfg)
            return

    # ---------- config 적용 ----------
    def apply_config(self, new_cfg: dict):
        with self.config_lock:
            if "CONF_THRESH" in new_cfg:
                self.CONF_THRESH = float(new_cfg["CONF_THRESH"])
                self.CONFIG["CONF_THRESH"] = self.CONF_THRESH

            if "DETECT_DURATION" in new_cfg:
                self.DETECT_DURATION = float(new_cfg["DETECT_DURATION"])
                self.CONFIG["DETECT_DURATION"] = self.DETECT_DURATION

            if "PRE_BUFFER_DURATION" in new_cfg:
                self.PRE_BUFFER_DURATION = max(0.0, float(new_cfg["PRE_BUFFER_DURATION"]))
                self.CONFIG["PRE_BUFFER_DURATION"] = self.PRE_BUFFER_DURATION

                new_size = max(1, int(self.MIC_SAMPLE_RATE * self.PRE_BUFFER_DURATION))
                with self.prebuf_lock:
                    if new_size != self.pre_buffer_size:
                        old = self.pre_buffer
                        if new_size <= len(old):
                            self.pre_buffer = old[-new_size:].copy()
                        else:
                            nb = np.zeros(new_size, dtype=np.float32)
                            nb[-len(old):] = old
                            self.pre_buffer = nb
                        self.pre_buffer_size = new_size

        print("[CONFIG] updated:", self.CONFIG, flush=True)

    # ---------- audio ----------
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        chunk = indata[:, self.TARGET_CHANNEL].astype(np.float32)
        self.q_collect.put(chunk)
        self.q_detect.put(chunk)

    def audio_collector(self):
        while self._run:
            chunk = self.q_collect.get()
            with self.prebuf_lock:
                n = len(chunk)
                if n >= self.pre_buffer_size:
                    self.pre_buffer = chunk[-self.pre_buffer_size:].copy()
                else:
                    self.pre_buffer = np.roll(self.pre_buffer, -n)
                    self.pre_buffer[-n:] = chunk

    def _send_detection(self, output: dict):
        if self.ws_client:
            self.ws_client.send_json(output)

    def audio_detector_analyzer(self):
        # “연속 추론 방지”는 sleep 대신 rising-edge로(놓침 최소화)
        last_voice = False

        while self._run:
            try:
                voice = bool(self.Mic_tuning.is_voice())

                if voice and not last_voice:
                    doa = self.Mic_tuning.direction

                    with self.config_lock:
                        detect_duration = self.CONFIG["DETECT_DURATION"]
                        conf_thresh = self.CONFIG["CONF_THRESH"]

                    detect_size = int(self.MIC_SAMPLE_RATE * detect_duration)
                    collected = []
                    while len(collected) < detect_size:
                        collected.extend(self.q_detect.get())

                    collected = np.array(collected[:detect_size], dtype=np.float32)

                    with self.prebuf_lock:
                        pre = self.pre_buffer.copy()

                    waveform_in = np.concatenate([pre, collected])

                    waveform_32k = librosa.resample(
                        waveform_in, orig_sr=self.MIC_SAMPLE_RATE, target_sr=self.MODEL_SAMPLE_RATE
                    )

                    if len(waveform_32k) < self.MIN_WAVEFORM_LENGTH:
                        waveform_32k = np.pad(
                            waveform_32k, (0, self.MIN_WAVEFORM_LENGTH - len(waveform_32k))
                        )

                    wav = torch.from_numpy(waveform_32k).float().unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        logits = self.audio_model(wav)["logits"][0]
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()

                    pairs = [(self.LABEL_NAMES[i], float(probs[i])) for i in range(len(self.LABEL_NAMES))]
                    t_detect_ms = time.time_ns() // 1_000_000

                    # Dog/Bark 통합 (버그 수정)
                    dog_score = next((s for l, s in pairs if l == "Dog"), 0.0)
                    bark_score = next((s for l, s in pairs if l == "Bark"), 0.0)
                    if dog_score > 0.0 or bark_score > 0.0:
                        combined = dog_score + bark_score
                        dog_label = "Bark" if bark_score >= dog_score else "Dog"
                        pairs = [(l, s) for (l, s) in pairs if l not in ("Dog", "Bark")]
                        pairs.append((dog_label, combined))

                    # Vehicle/Vehicle horn 통합
                    vehicle_score = next((s for l, s in pairs if l == "Vehicle"), 0.0)
                    horn_score = next((s for l, s in pairs if l == "Vehicle horn"), 0.0)
                    if vehicle_score > 0.0 or horn_score > 0.0:
                        combined = vehicle_score + horn_score
                        v_label = "Vehicle" if vehicle_score >= horn_score else "Vehicle horn"
                        pairs = [(l, s) for (l, s) in pairs if l not in ("Vehicle", "Vehicle horn")]
                        pairs.append((v_label, combined))

                    pairs.sort(key=lambda x: x[1], reverse=True)
                    if pairs:
                        top1_label, top1_score = pairs[0]
                        if top1_score >= conf_thresh:
                            output = {
                                "type": "detection",
                                # 전송시각/감지시각 둘 다 넣어줌
                                # "t_detect_ms": t_detect_ms,
                                "timestamp": time.time_ns() // 1_000_000,
                                "doa": doa,
                                "tags": [{"label": top1_label, "score": round(top1_score, 2)}],
                            }
                            self._send_detection(output)

                last_voice = voice
                time.sleep(0.01)

            except Exception as e:
                print("[DETECTOR] error:", repr(e), flush=True)
                time.sleep(0.1)

    # ---------- start ----------
    def start(self):
        self._stream = sd.InputStream(
            samplerate=self.MIC_SAMPLE_RATE,
            channels=6,
            device=self.DEVICE_INDEX,
            callback=self.audio_callback,
            blocksize=int(self.MIC_SAMPLE_RATE * 0.1),
        )

        t1 = threading.Thread(target=self.audio_collector, daemon=True)
        t2 = threading.Thread(target=self.audio_detector_analyzer, daemon=True)
        self._threads.extend([t1, t2])

        t1.start()
        t2.start()

        print("--------------------------------------")
        print("실시간 감지 + 리샘플링(→32k) + PANNs + WS 전송 시작")
        print("초기 CONFIG:", self.CONFIG)
        print("--------------------------------------")

        self._stream.start()

    def stop(self):
        self._run = False
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass


