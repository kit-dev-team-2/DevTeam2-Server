# main.py
import os
import time

from audio_detector import AudioDetector
from web_socket_client import WSClient


def main():
    node_ws_url = os.getenv("NODE_WS_URL", "ws://localhost:8080")

    detector = AudioDetector()

    ws = WSClient(
        url=node_ws_url,
        on_message=detector.on_ws_message,  # 수신 → detector가 처리(config_update/ping)
    )

    detector.set_ws_client(ws)  # detector가 send할 때 ws 사용

    ws.start()
    detector.start()

    print("[MAIN] NODE_WS_URL:", node_ws_url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[MAIN] stopping...")
        detector.stop()
        ws.stop()


if __name__ == "__main__":
    main()
