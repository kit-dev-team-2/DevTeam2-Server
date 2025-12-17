# web_socket_client.py
# ---------------------------------------------
# WebSocket 클라이언트 (Python -> Node, Node -> Python)
# - 자동 재연결
# - ping 수신 시 pong 자동 응답
# - config_update 수신 시 콜백으로 전달
# ---------------------------------------------

import json
import threading
import time

import websocket  # websocket-client
from websocket import WebSocketTimeoutException


class WebSocketClient:
    def __init__(self, url: str, on_payload=None, reconnect_sec: float = 3.0):
        self.url = url
        self.on_payload = on_payload  # callable(dict) | None
        self.reconnect_sec = reconnect_sec

        self._lock = threading.Lock()
        self._ws = None
        self._run = False
        self._th = None

    def start(self):
        if self._th is not None and self._th.is_alive():
            return
        self._run = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._run = False
        with self._lock:
            try:
                if self._ws is not None:
                    self._ws.close()
            except Exception:
                pass
            self._ws = None

    def send_json(self, obj: dict):
        msg = json.dumps(obj, ensure_ascii=False)
        with self._lock:
            ws = self._ws
        if ws is None:
            return
        try:
            ws.send(msg)
        except Exception:
            with self._lock:
                self._ws = None

    def _handle_text(self, text: str):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return

        mtype = payload.get("type")

        # ping 오면 pong 자동 응답
        if mtype == "ping":
            t = payload.get("t")
            self.send_json({"type": "pong", "t": t})
            return

        if self.on_payload is not None:
            try:
                self.on_payload(payload)
            except Exception:
                pass

    def _loop(self):
        while self._run:
            try:
                print(f"[WS] trying connect to {self.url}", flush=True)
                ws = websocket.create_connection(self.url, timeout=3)
                ws.settimeout(1.0)

                with self._lock:
                    self._ws = ws

                print("[WS] Connected OK", flush=True)

                while self._run:
                    try:
                        msg = ws.recv()
                        if msg:
                            self._handle_text(msg)
                    except WebSocketTimeoutException:
                        continue
                    except Exception as e:
                        print("[WS] recv error:", repr(e), flush=True)
                        break

            except Exception as e:
                print("[WS] connection error in loop:", repr(e), flush=True)

            with self._lock:
                self._ws = None

            if self._run:
                time.sleep(self.reconnect_sec)
