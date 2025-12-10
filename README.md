# Audio Detection System (ReSpeaker + PANNs + WebSocket + GUI)

ReSpeaker USB Mic Array, PANNs 기반 분류 모델, Node WebSocket 서버, Tkinter GUI를 묶어서  
**실시간 소리 감지 → 분류 → WebSocket 전송 → 클라이언트** 까지 동작하는 시스템입니다.

## 1-1. 하드웨어
- ReSpeaker USB Mic Array (6 Mic / 6채널 버전) 
- USB 포트가 있는 PC 또는 노트북

## 1-2. 소프트웨어
- Node.js 22.18.0
- npm 10.9.3  
- Python 3.12.4

## 1-3 Node.js 패키지
```bash
npm install ws
npm install express
```

## 1-4 Python 패키지
```bash
pip install sounddevice numpy torch librosa usb websocket-client panns-inference
```

