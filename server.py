#!/usr/bin/env python3
# simple_server.py  – threaded accept loop with remote kill and sign-map

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
from utils.sign_triangulator import Locator
from utils.sign_dimensions import DetectionSummary

HOST, PORT = "0.0.0.0", 5001
TYPE_DETECT, TYPE_TRACK, TYPE_KILL, TYPE_SIGNMAP, TYPE_PROCESS_DETECTIONS = 1, 2, 3, 4, 5

BASE_DIR      = Path(__file__).parent.resolve()
INCOMING_DIR  = BASE_DIR / "debug_tools" / "data" / "incoming"
ANNOTATED_DIR = BASE_DIR / "debug_tools" / "data" / "annotated"
CAPTURE_DIR   = BASE_DIR / "data" / "captures"

INCOMING_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

# ─── global per-capture lock registry ───────────────────────────────────
_lock_registry: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()

def _get_capture_lock(capture_id: str) -> threading.Lock:
    with _registry_lock:
        if capture_id not in _lock_registry:
            _lock_registry[capture_id] = threading.Lock()
        return _lock_registry[capture_id]

def recv_all(conn: socket.socket, n: int) -> bytes:
    buf = b''
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Client closed early")
        buf += chunk
    return buf

def handle_client(conn: socket.socket, addr):
    print(f"[{addr}] Connected", flush=True)
    try:
        # 1) header: type + meta length
        hdr    = recv_all(conn, 1 + 4)
        msg_t  = hdr[0]
        meta_l = struct.unpack("!i", hdr[1:])[0]

        # 2) meta JSON
        meta = {}
        if meta_l > 0:
            meta = json.loads(recv_all(conn, meta_l).decode())

        # ── KILL ──
        if msg_t == TYPE_KILL:
            print(f"[{addr}] Received KILL, exiting.", flush=True)
            conn.sendall(struct.pack("!i", 0))
            conn.close()
            import os; os._exit(0)

        # ── SIGNMAP ── (no image follows)
        if msg_t == TYPE_SIGNMAP:
            cap_id   = meta["capture_id"]
            print(f"[{addr}] SIGNMAP for capture {cap_id}", flush=True)
            body_len = struct.unpack("!i", recv_all(conn, 4))[0]
            if body_len:
                mapping = json.loads(recv_all(conn, body_len).decode())
            else:
                mapping = {}
            out_file = CAPTURE_DIR / f"capture_{cap_id}.json"
            out_file.write_text(json.dumps(mapping, indent=2))
            print(f"[{addr}] wrote mapping → {out_file}", flush=True)
            conn.sendall(struct.pack("!i", 0))
            return
        
        elif msg_t == TYPE_PROCESS_DETECTIONS:
            capture_id = meta["capture_id"]
            sign_id    = meta["sign_id"]
            print(f"[{addr}] PROCESS_DETECTIONS for capture {capture_id}, sign {sign_id}", flush=True)

            # locate
            track_dir = BASE_DIR / "data" / "tracks" / f"track_{capture_id}"
            det_file  = track_dir / "detections.json"
            # print(f"[{addr}]    Looking for {det_file}", flush=True)
            if not det_file.exists():
                print(f"[{addr}] detections.json missing → abort", flush=True)
                conn.sendall(struct.pack("!i", 0))
                return

            # load
            with det_file.open() as f:
                detections = json.load(f)
            # triangulate
            # print(f"[{addr}]    Running sign_triangulator.Locator.main()", flush=True)
            locator = Locator()
            augmented = locator.main(detections_path=str(det_file))
            # print(f"[{addr}]    Triangulation complete; augmented keys: {augmented}", flush=True)

            # dimension summary
            # print(f"[{addr}]    Running sign_dimensions.DetectionSummary.process()", flush=True)
            summarizer = DetectionSummary()
            summary = summarizer.process(augmented)
            # print(f"[{addr}]    Dimension summary computed; sign IDs: {summary}", flush=True)

            # inject into memory
            sign_key = str(sign_id)
            #print(f"[{addr}]    Injecting dims for sign {sign_key} into in-memory detections...", flush=True)
            if sign_key in summary:
                dims = summary[sign_key]
                for frame_name, frame_rec in detections.items():
                    for det in frame_rec.get("detections", []):
                        if det.get("id") == sign_id:
                            # print(f"[{addr}]      Frame {frame_name}: tagging det {det} with dims {dims}", flush=True)
                            det.update({
                                "distance": dims.get("distance", det.get("distance")),
                                "enu_east_m": dims.get("enu_east_m", det.get("enu_east_m")),
                                "enu_north_m": dims.get("enu_north_m", det.get("enu_north_m")),
                                "sign_location": {
                                    "latitude":  dims["latitude"],
                                    "longitude": dims["longitude"]
                                },
                                "estimated_height_inches": dims["estimated_height_inches"],
                                "estimated_width_inches":  dims["estimated_width_inches"]
                            })
                # write out per-sign file
                sign_json = track_dir / "signs.json"
                with sign_json.open("w") as sf:
                    json.dump({sign_key: dims}, sf, indent=2)
            else:
                print(f"[{addr}] No summary found for sign {sign_key}", flush=True)

            # reply
            reply_dict = {sign_key: summary.get(sign_key, {})}
            payload = json.dumps(reply_dict).encode()
            # print(f"[{addr}]    Sending back payload: {reply_dict}", flush=True)
            conn.sendall(struct.pack("!i", len(payload)))
            conn.sendall(payload)
            print(f"[{addr}] PROCESS_DETECTIONS done", flush=True)
            return


        # ── DETECT or TRACK ── both expect an image !!!!! IMPORTANT do not put code that doesn't expect a image below this section !!!!!
        img_l     = struct.unpack("!i", recv_all(conn, 4))[0]
        img_bytes = recv_all(conn, img_l)
        frame     = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        # archive raw
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        (INCOMING_DIR / f"{stamp}.jpg").write_bytes(img_bytes)

        if msg_t == TYPE_DETECT:
            print(f"[{addr}] DETECT", flush=True)
            boxes = predict_boxes(frame)
            annotate_and_save(frame.copy(), boxes, str(ANNOTATED_DIR/f"{stamp}.jpg"))
            conn.sendall(struct.pack("!i", len(boxes)))
            for x1,y1,x2,y2 in boxes:
                conn.sendall(struct.pack("ffff", x1,y1,x2,y2))
            print(f"[{addr}] Sent {len(boxes)} boxes", flush=True)

        elif msg_t == TYPE_TRACK:
            # ── TRACK ── with per-capture lock
            capture_id  = meta["capture_id"]
            frames_path = meta["frames_path"]
            sign_id     = meta.get("sign_hash_code")
            print(f"[{addr}] TRACK {capture_id} / {frames_path}", flush=True)

            lock = _get_capture_lock(capture_id)
            with lock:
                # full segment tracking
                results = perform_segment_tracking(capture_id, frames_path)
                print(f"[{addr}] Full track keys: {list(results.keys())[:5]} …", flush=True)
                
                # choose ref_image via stored bbox if provided
                ref_image = None # 
                if sign_id is not None:
                    cap_file = CAPTURE_DIR / f"capture_{capture_id}.json"
                    if cap_file.exists():
                        sign_map = json.loads(cap_file.read_text())
                        bbox = sign_map.get(str(sign_id))
                        if bbox:
                            xa,ya,xb,yb = map(int, bbox)
                            crop_bgr    = frame[ya:yb, xa:xb]
                            crop_rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                            ref_image   = Image.fromarray(crop_rgb)
                            print(f"[{addr}] Using stored bbox for sign {sign_id}", flush=True)
                        else:
                            print(f"[{addr}] sign {sign_id} not in map; fallback", flush=True)

                # fallback: largest detection thumbnail
                if ref_image is None:
                    boxes = predict_boxes(frame)
                    if boxes:
                        x1,y1,x2,y2 = max(boxes, key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
                        xa,ya,xb,yb= map(int,(x1,y1,x2,y2))
                        crop_bgr   = frame[ya:yb, xa:xb]
                        crop_rgb   = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                        ref_image  = Image.fromarray(crop_rgb)
                    else:
                        ref_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # single-frame match
                match = match_reference_to_track(ref_image, capture_id)
                print(f"[{addr}] Single-frame match: {match}", flush=True)

                if match is not None and sign_id is not None:
                    # get the numeric subtrack id, e.g. "track_0003" → 3
                    sub_num = int("".join(match["subtrack"].split("_")[-1]))
                    # inject sign_id into every matching detection in your in-memory results
                    for frame_rec in results.values():
                        for det in frame_rec["detections"]:
                            if det["id"] == sub_num:
                                det["sign_id"] = sign_id
                    # write the updated results back to disk
                    track_folder = BASE_DIR / "data" / "tracks" / f"track_{capture_id}"
                    det_path     = track_folder / "detections.json"
                    det_path.write_text(json.dumps(results, indent=2))

                    print(f"[{addr}] injected sign_id={sign_id} into {det_path}", flush=True)


                # send full results as JSON
                payload = json.dumps(results).encode()
                conn.sendall(struct.pack("!i", len(payload)))
                conn.sendall(payload)
                print(f"[{addr}] Sent TRACK payload ({len(payload)} bytes)", flush=True)

        else:
            print(f"[{addr}] UNKNOWN msg_type {msg_t}", flush=True)
            conn.sendall(b"\x00")

    except SystemExit:
        raise

    except Exception as ex:
        print(f"[{addr}] ERROR: {ex}", flush=True)

    finally:
        try: conn.close()
        except: pass
        print(f"[{addr}] Connection closed", flush=True)

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"Server listening on {HOST}:{PORT}", flush=True)

    try:
        while True:
            conn, addr = server.accept()
            print(f"Got connection from {addr}", flush=True)
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        print("\nShutting down server…", flush=True)
    finally:
        server.close()
        print("Server socket closed.", flush=True)

if __name__ == "__main__":
    main()
