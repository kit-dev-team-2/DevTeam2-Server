# web_socket_client.py
import json
import threading
import time

import websocket
from websocket import WebSocketTimeoutException


class WSClient:
    def __init__(self, url: str, on_message=None):
        self.url = url
        self.on_message = on_message

        self._lock = threading.Lock()
        self._conn = None
        self._should_run = True
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self._thread

    def stop(self):
        self._should_run = False
        with self._lock:
            try:
                if self._conn:
                    self._conn.close()
            except Exception:
                pass
            self._conn = None

    def send_json(self, obj: dict) -> bool:
        msg = json.dumps(obj, ensure_ascii=False)
        with self._lock:
            if self._conn is None:
                return False
            try:
                self._conn.send(msg)
                return True
            except Exception as e:
                print("[WS] send error:", repr(e), flush=True)
                self._conn = None
                return False

    def _loop(self):
        while self._should_run:
            try:
                print(f"[WS] trying connect to {self.url}", flush=True)
                ws = websocket.create_connection(self.url, timeout=3)
                ws.settimeout(1.0)

                with self._lock:
                    self._conn = ws

                print("[WS] Connected OK", flush=True)

                while self._should_run:
                    try:
                        msg = ws.recv()
                        if msg and self.on_message:
                            self.on_message(msg)
                    except WebSocketTimeoutException:
                        continue
                    except Exception as e:
                        print("[WS] recv error:", repr(e), flush=True)
                        break

            except Exception as e:
                print("[WS] connection error:", repr(e), flush=True)

            with self._lock:
                self._conn = None

            if self._should_run:
                time.sleep(3)
