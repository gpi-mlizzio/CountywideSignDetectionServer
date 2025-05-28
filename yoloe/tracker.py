#!/usr/bin/env python3
"""
track_frames.py – detect + appearance+spatial-track YOLO-E sign detections,
dump crops into ./detections/track_XXXX/, and save per-frame visualizations
into ./vision/, merging in GPS + heading data from a CSV.

Michael Lizzio · 2025-05-20
"""

import os
import glob
import json
import argparse
import itertools
import math
import shutil
import csv

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLOE

# ───────── user-tweakable constants ─────────
# FRAMES_DIR    = "../frames/fe43aa02-8209-4479-a791-3e7ba840344a"
FRAMES_DIR    = "./frames"
GPS_CSV       = "coordinates.csv"
OUTPUT_JSON   = "detections_light_2.json"
WEIGHTS_PATH  = "yoloe-11s-seg.pt"
CONF_THR      = 0.35
SIM_THR       = 0.80    # appearance threshold
DET_ROOT      = "./detections"
VISION_ROOT   = "./vision"
# ────────────────────────────────────────────

PROMPT_NAMES = [
    'crt screen','pole','traffic sign','neon light','milestone','payphone',
    'tv genre','exit','flare','shape','warning sign','scoreboard',
    'projection screen','speed limit sign','tv sitcom','score','flyer',
    'basketball backboard','darkness','screen','hamburg','rectangle',
    'triangle','torch','movie poster','portrait','parking sign','fixture',
    'moth','sundial','road sign','tv drama','plaque','sign','billboard',
    'stop at','hail','solar battery'
]

# prepare feature extractor
_feat_model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.DEFAULT
).features
_feat_model.eval().requires_grad_(False)
_preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
def feat_vec(crop: Image.Image) -> torch.Tensor:
    if crop.mode != "RGB":
        crop = crop.convert("RGB")
    x = _preprocess(crop).unsqueeze(0)
    with torch.no_grad():
        f = _feat_model(x).mean([2,3])
    return F.normalize(f[:, :128], p=2, dim=1)

def load_gps(frames_dir):
    csv_path = os.path.join(frames_dir, GPS_CSV)
    gps_map, order = {}, []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row['image']
            order.append(img)
            gps_map[img] = {
                'latitude':  float(row['latitude']),
                'longitude': float(row['longitude']),
                'timestamp': row['timestamp']
            }
    return gps_map, order

def detect_and_track(frames_dir, out_json):
    # clear outputs
    for d in (DET_ROOT, VISION_ROOT):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    gps_map, img_order = load_gps(frames_dir)

    model = YOLOE(WEIGHTS_PATH)
    model.set_classes(PROMPT_NAMES, model.get_text_pe(PROMPT_NAMES))

    tracks  = {}
    id_iter = itertools.count(1)
    results = {}

    # process in CSV order
    for img_name in img_order:
        img_path = os.path.join(frames_dir, img_name)
        if not os.path.isfile(img_path):
            continue

        pil = Image.open(img_path)
        frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        H, W  = frame.shape[:2]

        # detect
        pred = model.predict(source=frame, imgsz=(H,W),
                             conf=CONF_THR, device=0, verbose=False)[0]

        # visualize
        vis = frame.copy()
        for (x1,y1,x2,y2), cf, cl in zip(
            pred.boxes.xyxy.cpu().numpy(),
            pred.boxes.conf.cpu().numpy(),
            pred.boxes.cls.cpu().numpy().astype(int)
        ):
            cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)),
                          (0,0,255), 2)
            cv2.putText(vis, f"{model.names[cl]} {cf:.2f}",
                        (int(x1),int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.imwrite(os.path.join(VISION_ROOT, img_name), vis)

        # ROI limits
        x_min, x_max = W*0.25, W*0.75
        y_min, y_max = H*0.25, H*0.75
        MAX_DIST     = 0.5 * max(W,H)

        frame_dets = []
        for (x1,y1,x2,y2), cf, cl in zip(
            pred.boxes.xyxy.cpu().numpy(),
            pred.boxes.conf.cpu().numpy(),
            pred.boxes.cls.cpu().numpy().astype(int)
        ):
            cx, cy = (x1+x2)/2, (y1+y2)/2
            if not (x_min<cx<x_max and y_min<cy<y_max):
                continue

            # appearance vector
            crop = pil.crop((int(x1),int(y1),int(x2),int(y2)))
            vec  = feat_vec(crop)

            # match to existing track
            best_id, best_sim = None, SIM_THR
            for tid,data in tracks.items():
                sim = float(torch.matmul(vec, data['feat'].T))
                if sim <= best_sim:
                    continue
                px, py = data['center']
                if math.hypot(cx-px, cy-py) <= MAX_DIST:
                    best_id, best_sim = tid, sim
            if best_id is None:
                best_id = next(id_iter)

            # update track
            tracks[best_id] = {'feat': vec, 'center': (cx,cy)}

            # save crop
            tdir = os.path.join(DET_ROOT, f"track_{best_id:04d}")
            os.makedirs(tdir, exist_ok=True)
            base = os.path.splitext(img_name)[0] + ".png"
            crop.save(os.path.join(tdir, base))

            # prepare JSON entry
            det = {
                "id":    best_id,
                "bbox":  {
                    "left":   int(x1),
                    "top":    int(y1),
                    "width":  int(x2-x1),
                    "height": int(y2-y1)
                },
                "label": model.names[cl],
                "conf":  round(float(cf),4)
            }
            frame_dets.append(det)

        # assemble frame record, include GPS+heading
        record = {
            "latitude":  gps_map[img_name]['latitude'],
            "longitude": gps_map[img_name]['longitude'],
            "timestamp": gps_map[img_name]['timestamp'],
            "detections": frame_dets
        }
        results[img_name] = record
        print(f"[+] {img_name:>15} → {len(frame_dets)} detections")

    # write JSON
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON → {out_json}")

def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--dir", default=FRAMES_DIR, help="Frame folder")
    ap.add_argument("--out", default=OUTPUT_JSON, help="JSON output")
    args = ap.parse_args()
    detect_and_track(args.dir, args.out)

if __name__=="__main__":
    main()
