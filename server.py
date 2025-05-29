#!/usr/bin/env python3
# simple_server.py  – threaded accept loop with remote kill

import json
import socket
import struct
import threading
import sys
from pathlib import Path
from datetime import datetime
from PIL import Image
import cv2
import numpy as np

from yoloe.thumbnail_generator import predict_boxes, annotate_and_save
from yoloe.track_frames       import perform_segment_tracking, match_reference_to_track

HOST, PORT              = "0.0.0.0", 5001
TYPE_DETECT, TYPE_TRACK, TYPE_KILL, TYPE_SIGNMAP = 1, 2, 3, 4

BASE_DIR      = Path(__file__).parent.resolve()
INCOMING_DIR  = BASE_DIR / "debug_tools" / "data" / "incoming"
ANNOTATED_DIR = BASE_DIR / "debug_tools" / "data" / "annotated"
CAPTURE_DIR = BASE_DIR / "data" / "captures"
INCOMING_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

def recv_all(conn: socket.socket, n: int) -> bytes:
    buf = b''
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Client closed early")
        buf += chunk
    return buf

def handle_client(conn: socket.socket, addr):
    print(f"[{addr}] Connected")
    try:
        # 1) read header: 1 byte type + 4 byte meta length
        hdr    = recv_all(conn, 1 + 4)
        msg_t  = hdr[0]
        meta_l = struct.unpack("!i", hdr[1:])[0]

        # --------------- HANDLE KILL ----------------
        if msg_t == TYPE_KILL:
            print(f"[{addr}] Received KILL command - shutting down.")
            # ack back a zero-length OK
            conn.sendall(struct.pack("!i", 0))
            conn.close()
            # exit process immediately
            import os
            os._exit(0)

        # 2) optional JSON meta
        meta = {}
        if meta_l > 0:
            meta = json.loads(recv_all(conn, meta_l).decode())

        # 3) read image size + image bytes
        img_l     = struct.unpack("!i", recv_all(conn, 4))[0]
        img_bytes = recv_all(conn, img_l)
        frame     = cv2.imdecode(np.frombuffer(img_bytes, np.uint8),
                                 cv2.IMREAD_COLOR)

        # archive raw
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        (INCOMING_DIR/f"{stamp}.jpg").write_bytes(img_bytes)

        # ─── dispatch ─────────────────────────────────
        if msg_t == TYPE_DETECT:
            print(f"[{addr}] DETECT")
            boxes = predict_boxes(frame)
            annotate_and_save(frame.copy(), boxes,
                              str(ANNOTATED_DIR/f"{stamp}.jpg"))
            # send
            conn.sendall(struct.pack("!i", len(boxes)))
            for x1,y1,x2,y2 in boxes:
                conn.sendall(struct.pack("ffff", x1,y1,x2,y2))
            print(f"[{addr}] Sent {len(boxes)} boxes")

        elif msg_t == TYPE_TRACK:
            # unpack metadata
            capture_id  = meta["capture_id"]
            frames_path = meta["frames_path"]
            sign_id = meta.get("sign_hash_code") 

            print(f"[{addr}] TRACK {capture_id} / {frames_path}", flush=True)

            # 1) run or load the full segment tracking
            results = perform_segment_tracking(capture_id, frames_path)
            print(f"[{addr}] Full track results keys: {list(results.keys())[:5]} ...", flush=True)

            ref_image = None
            if sign_id:
                cap_file = CAPTURE_DIR / f"capture_{capture_id}.json"
                if cap_file.exists():
                    sign_map = json.loads(cap_file.read_text())
                    bbox = sign_map.get(str(sign_id))
                    if bbox:
                        xa, ya, xb, yb = map(int, bbox)
                        crop_bgr = frame[ya:yb, xa:xb]
                        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                        ref_image = Image.fromarray(crop_rgb)
                        print(f"[{addr}] Using stored bbox for sign {sign_id}", flush=True)
                    else:
                        print(f"[{addr}] sign {sign_id} not in map — falling back", flush=True)

            # if we still don’t have ref_image, fall back to largest box / full frame (old code)
            if 'ref_image' not in locals():
                boxes = predict_boxes(frame)
                if boxes:
                    x1,y1,x2,y2 = max(boxes, key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
                    xa,ya,xb,yb = map(int,(x1,y1,x2,y2))
                    crop_bgr = frame[ya: yb, xa: xb]
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    ref_image = Image.fromarray(crop_rgb)
                else:
                    ref_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # 3) run the single‐frame matcher on this cropped ref
            match = match_reference_to_track(ref_image, capture_id)
            print(f"[{addr}] Single‐frame match: {match}", flush=True)

            # 4) send the full JSON results back to the client
            payload = json.dumps(results).encode()
            conn.sendall(struct.pack("!i", len(payload)))
            conn.sendall(payload)
            print(f"[{addr}] Sent TRACK payload ({len(payload)} bytes)", flush=True)

        elif msg_t == TYPE_SIGNMAP:
            # meta must contain the capture we are mapping into
            cap_id = meta["capture_id"]
            print(f"[{addr}] SIGNMAP for capture {cap_id}", flush=True)

            # body length
            body_len = struct.unpack("!i", recv_all(conn, 4))[0]

            if body_len:                                # ↰ only parse if > 0
                mapping = json.loads(recv_all(conn, body_len).decode())
            else:
                mapping = {}                            # nothing to map

            # write / overwrite  data/captures/capture_<id>.json
            out_file = CAPTURE_DIR / f"capture_{cap_id}.json"
            out_file.write_text(json.dumps(mapping, indent=2))
            print(f"[{addr}] wrote mapping → {out_file}", flush=True)

            # send 0‑length ACK so client can unblock
            conn.sendall(struct.pack("!i", 0))



        else:
            print(f"[{addr}] UNKNOWN msg_type {msg_t}")
            conn.sendall(b"\x00")

    except SystemExit:
        # re-raise so thread/process exits
        raise

    except Exception as ex:
        print(f"[{addr}] ERROR: {ex}")

    finally:
        try: conn.close()
        except: pass
        print(f"[{addr}] Connection closed")

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"Server listening on {HOST}:{PORT}")

    try:
        while True:
            conn, addr = server.accept()
            print(f"Got a connection from {addr}")
            t = threading.Thread(target=handle_client,
                                 args=(conn, addr),
                                 daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("\nShutting down server (got Ctrl+C)…")
    finally:
        server.close()
        print("Server socket closed.")

if __name__ == "__main__":
    main()
