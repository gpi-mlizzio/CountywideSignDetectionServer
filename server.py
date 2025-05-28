# threaded_server.py
# Python 3.10+  

import json
import socketserver
import struct
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np


from pathlib import Path

# script_dir = folder that contains server.py
script_dir    = Path(__file__).parent.resolve()
INCOMING_DIR  = script_dir / "debug_tools" / "data" / "incoming"
ANNOTATED_DIR = script_dir / "debug_tools" / "data" / "annotated"
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
INCOMING_DIR.mkdir(parents=True, exist_ok=True)

from yoloe.thumbnail_generator import predict_boxes, annotate_and_save


# ── CONFIG ─────────────────────────────────────────────────────────────
HOST, PORT       = "0.0.0.0", 5001

# message types
TYPE_DETECT = 1
TYPE_TRACK  = 2

# ── HELPER: reliable recv ──────────────────────────────────────────────
def recv_all(sock, n):
    data = bytearray()
    while len(data) < n:
        pkt = sock.recv(n - len(data))
        if not pkt:
            raise ConnectionError("client disconnected early")
        data += pkt
    return data

# ── REQUEST HANDLER ────────────────────────────────────────────────────
class ClientHandler(socketserver.BaseRequestHandler):
    def handle(self):
        sock = self.request
        try:
            # 1) read header
            hdr = recv_all(sock, 1 + 4)
            msg_type = hdr[0]
            meta_len  = struct.unpack("!i", hdr[1:])[0]

            # 2) optional JSON meta
            meta = {}
            if meta_len:
                meta = json.loads(recv_all(sock, meta_len).decode())

            # 3) read image
            img_len = struct.unpack("!i", recv_all(sock, 4))[0]
            img_data = recv_all(sock, img_len)
            img = cv2.imdecode(
                np.frombuffer(img_data, np.uint8),
                cv2.IMREAD_COLOR
            )

            # archive raw
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            raw_path = INCOMING_DIR / f"{stamp}.jpg"
            raw_path.write_bytes(img_data)

            # dispatch
            if msg_type == TYPE_DETECT:
                boxes = predict_boxes(img)

                # annotate for debug
                annotate_and_save(img.copy(), boxes, str(ANNOTATED_DIR / f"{stamp}.jpg"))

                # reply: count + each box as 4 floats
                sock.sendall(struct.pack("!i", len(boxes)))
                for x1, y1, x2, y2 in boxes:
                    sock.sendall(struct.pack("ffff", x1, y1, x2, y2))

            elif msg_type == TYPE_TRACK:
                # your existing tracker logic goes here...
                response = {"capture_id": meta.get("capture_id"), "ok": True}
                data = json.dumps(response).encode()
                sock.sendall(struct.pack("!i", len(data)))
                sock.sendall(data)

            else:
                sock.sendall(b"\x00")

        except Exception as e:
            print(f"[{self.client_address}] error: {e}")

# ── THREAD POOL SERVER ─────────────────────────────────────────────────
class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads      = True
    allow_reuse_address = True

# ── MAIN ───────────────────────────────────────────────────────────────
# if __name__ == "__main__":
print(f"Listening on {HOST}:{PORT}")
with ThreadedTCPServer((HOST, PORT), ClientHandler) as srv:
    srv.serve_forever()



# pyinstaller --onefile --name sign_server --add-data "yolo/yolo-11s-seg.pt;." server.py
