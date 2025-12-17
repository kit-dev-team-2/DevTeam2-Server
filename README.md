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

## 1-3. Node.js 패키지
```bash
npm install ws
npm install express
```

## 1-4. Python 패키지
```bash
pip install sounddevice numpy torch librosa usb websocket-client panns-inference
```

## 1-5. PANNs CheckPoints
https://drive.google.com/drive/folders/1ozmmY240myc-u5-6DVxisbggEtaHsAYg?usp=sharing
- Cnn14_mAP=0.431 (Base CKP)
- best_panns6_acc0.828 (fine-tuned CKP)
- 구글 드라이브 링크에서 두 ckp 모두 다운로드

## 2. 실행
- GUI 프로그램 실행
```bash
python gui.py
```
<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/09906230-c33d-4ac8-8f75-acb287b56a2b" />

- 각 파일 경로 설정 및 Node WS 서버 실행 버튼 클릭
- WS 서버가 정상 실행된 후, PANNs 감지 시작 버튼 클릭

## 3. 파일
- audio_detector.py : 오디오 감지 및 분류
- sound.py : main 실행 파일
- web_socket_client : 웹소켓 클라이언트


