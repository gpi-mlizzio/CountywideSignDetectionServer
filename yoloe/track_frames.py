#!/usr/bin/env python3
"""
track_frames.py – detect + appearance+spatial-track YOLO-E sign detections
and store each track under ../data/tracks/track_<track_id>.
"""

import os, json, itertools, math
from pathlib import Path
from PIL import Image
import shutil, csv

import torch
import torch.nn.functional as F
from torchvision import models, transforms
import cv2, numpy as np
from ultralytics import YOLOE
import logging

# ── BASE PATH FOR ASSETS ──────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.resolve()
LOCAL_WEIGHTS  = BASE_DIR / "yoloe-11s-seg.pt"
LOCAL_TEXT_TS  = BASE_DIR / "mobileclip_blt.ts"

# ─── user-tweakable constants ────────────────────────────────────────
GPS_CSV  = "coordinates.csv"
CONF_THR = 0.35
SIM_THR  = 0.80    # appearance threshold

PROMPT_NAMES = [  # ... as before ...
    'crt screen','pole','traffic sign','neon light','milestone','payphone',
    'tv genre','exit','flare','shape','warning sign','scoreboard',
    'projection screen','speed limit sign','tv sitcom','score','flyer',
    'basketball backboard','darkness','screen','hamburg','rectangle',
    'triangle','torch','movie poster','portrait','parking sign','fixture',
    'moth','sundial','road sign','tv drama','plaque','sign','billboard',
    'stop at','hail','solar battery'
]

# ─── feature extractor ────────────────────────────────────────────────
_feat_mdl = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.DEFAULT
).features
_feat_mdl.eval().requires_grad_(False)
_preproc = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
def feat_vec(crop: Image.Image):
    if crop.mode!="RGB": crop = crop.convert("RGB")
    x = _preproc(crop).unsqueeze(0)
    with torch.no_grad():
        f = _feat_mdl(x).mean((2,3))
    return F.normalize(f[:,:128], p=2, dim=1)

def load_gps(frames_dir):
    gps_map, order = {}, []
    with open(Path(frames_dir)/GPS_CSV, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            img = row["image"]
            order.append(img)
            gps_map[img] = {
                "latitude":  float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "timestamp": row["timestamp"]
            }
    return gps_map, order

# ─── core detect & track, writes to arbitrary DET_ROOT / VIZ_ROOT ─────
def __detect_and_track(frames_dir, det_root: Path, viz_root: Path):
    gps_map, img_order = load_gps(frames_dir)

    # instantiate YOLOE with local weight + text-encoder paths
    model = YOLOE(str(LOCAL_WEIGHTS), str(LOCAL_TEXT_TS))
    model.set_classes(PROMPT_NAMES, model.get_text_pe(PROMPT_NAMES))

    tracks  = {}
    id_iter = itertools.count(1)
    results = {}

    # clear + mkdir
    for d in (det_root, viz_root):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    for img_name in img_order:
        img_path = Path(frames_dir)/img_name
        if not img_path.exists(): continue

        pil   = Image.open(img_path)
        frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        H, W  = frame.shape[:2]

        pred = model.predict(
            source=frame,
            imgsz=(H,W),
            conf=CONF_THR,
            device=0,
            verbose=False
        )[0]

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
        cv2.imwrite(str(viz_root/img_name), vis)

        # ROI + track logic
        frame_dets = []
        for (x1,y1,x2,y2), cf, cl in zip(
            pred.boxes.xyxy.cpu().numpy(),
            pred.boxes.conf.cpu().numpy(),
            pred.boxes.cls.cpu().numpy().astype(int)
        ):
            cx, cy = (x1+x2)/2, (y1+y2)/2
            if not (W*0.25<cx<W*0.75 and H*0.25<cy<H*0.75):
                continue

            vec = feat_vec(pil.crop((int(x1),int(y1),int(x2),int(y2))))

            # spatial+appearance match
            best_id, best_sim = None, SIM_THR
            for tid,data in tracks.items():
                sim = float(torch.matmul(vec, data["feat"].T))
                if sim>best_sim:
                    px, py = data["center"]
                    if math.hypot(cx-px, cy-py) <= 0.5*max(W,H):
                        best_id, best_sim = tid, sim

            if best_id is None:
                best_id = next(id_iter)
            tracks[best_id] = {"feat":vec, "center":(cx,cy)}

            # save crop under track_<tid>
            tdir = det_root/f"track_{best_id:04d}"
            tdir.mkdir(exist_ok=True)
            crop_name = Path(img_name).stem + ".png"
            pil.crop((int(x1),int(y1),int(x2),int(y2))).save(tdir/crop_name)

            frame_dets.append({
                "id":    best_id,
                "bbox":  {"left":int(x1),"top":int(y1),
                          "width":int(x2-x1),"height":int(y2-y1)},
                "label": model.names[cl],
                "conf":  round(float(cf),4)
            })

        results[img_name] = {
            **gps_map[img_name],
            "detections": frame_dets
        }

    return results

# ─── public API ────────────────────────────────────────────────────────
def perform_segment_tracking(track_id: str, frames_dir: str):
    """
    Run or re-use detections for the given track_id and frame folder.
    Results and crops/vision are stored under:
      ../data/tracks/track_<track_id>/
         ├─ detections.json
         ├─ vision/…
         └─ detections/…
    Returns the detections dict loaded from JSON or freshly computed.
    """
    root     = Path(f"{BASE_DIR}/../data/tracks")/f"track_{track_id}"
    out_json = root/"detections.json"
    viz_root = root/"vision"
    det_root = root/"detections"

    logging.info(str(root) + " " + str(out_json) + " " + str(viz_root) + " " + str(det_root))

    if out_json.exists():
        return json.loads(out_json.read_text())

    root.mkdir(parents=True, exist_ok=True)
    results = __detect_and_track(frames_dir, det_root, viz_root)

    logging.info(f"Results: {results}")

    out_json.write_text(json.dumps(results, indent=2))
    return results

# ─── match reference image to an existing track ────────────────────────
def match_reference_to_track(
        ref, 
        track_id: str, 
        sim_threshold: float = 0.15, 
        delta: float = 0.1, 
        max_images: int = 4
    ):
    """
    As before, but when a sub-track has no i-th image, subtract 0.015
    from its last sim (or from 0.0 if none) instead of dropping it.
    """
    # prepare debug folder
    debug_dir = Path(f"{BASE_DIR}/../data/tracks/track_{track_id}") / "match_debug"
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True)

    from PIL import Image
    # save ref image
    if isinstance(ref, str):
        ref_img = Image.open(ref)
    else:
        ref_img = ref
    ref_img.save(debug_dir/"ref_image.png")
    ref_vec = feat_vec(ref_img)

    base    = Path(f"{BASE_DIR}/../data/tracks")/f"track_{track_id}"/"detections"
    print("DEBUG: matching in", base, flush=True)
    if not base.exists():
        print("DEBUG: no detections folder", flush=True)
        return None

    subdirs     = sorted(d for d in base.iterdir() if d.is_dir())
    sims        = {d.name: [] for d in subdirs}
    best_overall= {"subtrack":None, "image":None, "similarity":0.0}

    PENALTY = 0.015

    for i in range(max_images):
        print(f"\n--- ROUND {i+1} ---", flush=True)
        round_sims = {}

        for sub in sims:
            imgs = sorted((base/sub).glob("*.png"))
            if i < len(imgs):
                img_path = imgs[i]
                print(f"  [{sub}] testing {img_path.name}", flush=True)
                crop = Image.open(img_path)
                crop.save(debug_dir/f"{sub}_r{i+1}.png")

                vec = feat_vec(crop)
                sim = float(torch.matmul(ref_vec, vec.T))
                print(f"    sim = {sim:.4f}", flush=True)
            else:
                # no image: apply a small penalty
                last = sims[sub][-1] if sims[sub] else 0.0
                sim = last - PENALTY
                print(f"  [{sub}] no image #{i+1}, penalize {last:.4f}→{sim:.4f}", flush=True)
                # save placeholder
                Image.new("RGB", (100,100)).save(debug_dir/f"{sub}_r{i+1}_P.png")

            sims[sub].append(sim)
            round_sims[sub] = sim

        # prune by threshold only (no hard-dropping for missing)
        if i < 2:
            for sub in list(sims):
                if sims[sub][-1] < sim_threshold:
                    print(f"  [{sub}] sim {sims[sub][-1]:.4f} < threshold; pruning", flush=True)
                    sims.pop(sub)
            if not sims:
                print("  all pruned early", flush=True)
                break

        if not sims:
            print("  no candidates left", flush=True)
            break

        # identify leader & runner-up
        sorted_round = sorted(round_sims.items(), key=lambda kv:kv[1], reverse=True)
        leader, lead_sim = sorted_round[0]
        runner_sim = sorted_round[1][1] if len(sorted_round)>1 else 0.0
        print(f"  Leader: {leader} ({lead_sim:.4f}), Runner-up: {runner_sim:.4f}", flush=True)

        # update best overall
        if lead_sim > best_overall["similarity"]:
            first_img = sorted((base/leader).glob("*.png"))[0].name
            best_overall.update({
                "subtrack":   leader,
                "image":      str(base/leader/first_img),
                "similarity": lead_sim
            })
            print(f"  New best_overall: {best_overall}", flush=True)

        # early exit if clearly ahead
        if lead_sim >= sim_threshold and (lead_sim - runner_sim) >= delta:
            print(f"  Exiting early: Δ={lead_sim-runner_sim:.4f} ≥ {delta}", flush=True)
            return best_overall

    print(f"\nFINAL best_overall: {best_overall}", flush=True)
    return best_overall if best_overall["similarity"] >= sim_threshold else None




# ─── command-line debug ────────────────────────────────────────────────
if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("track_id")
    p.add_argument("frames_dir")
    args = p.parse_args()
    dets = perform_segment_tracking(args.track_id, args.frames_dir)
    print(f"Finished track {args.track_id}: {len(dets)} frames with detections")
