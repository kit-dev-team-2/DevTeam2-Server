# Audio Detection System (ReSpeaker + PANNs + WebSocket + GUI)

ReSpeaker USB Mic Array, PANNs 기반 분류 모델, Node WebSocket 서버, Tkinter GUI를 묶어서  
**실시간 소리 감지 → 분류 → WebSocket 전송 → 클라이언트 표시** 까지 동작하는 시스템입니다.

구성 요소:

1. **Node WebSocket 서버** (예: `server3.js`)
2. **Python GUI 런처** (`gui.py` / `ScriptManagerGUI`)
3. **오디오 감지 + 분류 스크립트** (`audio_detector.py`)
4. **외부 WebSocket 클라이언트** (브라우저, Meta Quest 등)

---

## 1. 전체 아키텍처

데이터 흐름:

1. **ReSpeaker USB Mic Array**
   - 6채널 오디오를 MIC_SAMPLE_RATE(기본 16kHz) 로 출력
2. **`audio_detector.py`**
   - ReSpeaker `Mic_tuning.is_voice()` 로 소리 유무(SAD) 확인
   - 소리 감지 시:
     - `PRE_BUFFER_DURATION` 초 pre-buffer + `DETECT_DURATION` 초 구간을 합쳐서
     - 32kHz 로 리샘플링 → PANNs 모델로 분류
     - Top-1 라벨이 `CONF_THRESH` 이상이면 `detection` JSON 을 WebSocket으로 전송
3. **Node WebSocket 서버**
   - Python → Node: `detection` 메시지 수신 후, 모든 클라이언트에 브로드캐스트
   - 클라이언트 → Node: `config_update` / `pong` 수신
   - Node → Python & GUI: `config_update` 전달, `ping` 전송
4. **GUI (`gui.py`)**
   - Node 서버 & Python 스크립트 실행/종료
   - Node/Python 로그 표시
   - 연결된 WebSocket 클라이언트 목록 + ping RTT 표시
   - ReSpeaker SAD(소리 감지 기준 dB) 슬라이더 제공
5. **클라이언트 (브라우저 / Meta Quest 등)**
   - WebSocket으로 접속
   - `detection` 메시지를 받아서 UI에 반영
   - `config_update` 로 서버/모델 파라미터 실시간 조정

---

## 2. 파일 구조 예시

```text
project-root/
├─ server3.js             # Node WebSocket 서버
├─ gui.py                 # Tkinter GUI 런처 (ScriptManagerGUI)
├─ audio_detector.py      # 오디오 감지 + PANNs 분류 + WS 클라이언트
├─ tuning.py              # ReSpeaker 제어용 (Seeed 제공 예제)
├─ Cnn14_mAP=0.431.pth    # BASE_CKPT_PATH (PANNs 원본)
└─ best_panns6_acc0.828.pt# FINETUNED_CKPT (파인튜닝된 7클래스 모델)
