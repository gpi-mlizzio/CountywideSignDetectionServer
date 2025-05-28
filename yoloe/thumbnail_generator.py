# yoloe/thumbnail_generator.py

import os
import json
import random
import cv2
import numpy as np
from ultralytics import YOLOE

# ── CONFIG ────────────────────────────────────────────────────────────
_WEIGHTS    = os.environ.get("YOLOE_WEIGHTS", "yoloe-11s-seg.pt")
_PROMPTS    = "crt screen, pole, traffic sign, neon light, milestone, payphone, tv genre, exit, flare, shape, warning sign, scoreboard, projection screen, speed limit sign, tv sitcom, score, flyer, basketball backboard, darkness, screen, hamburg, rectangle, triangle, torch, movie poster, ticket booth, portrait, parking sign, fixture, moth, sundial, road sign, tv drama, plaque, sign, billboard, stop at, hail, solar battery"
_CONF_THRESH = float(os.environ.get("YOLOE_CONF", 0.35))
_DEVICE      = 0  # or "cuda:0"/"cpu"

# ── SETUP ─────────────────────────────────────────────────────────────
# load model ONCE
_model = YOLOE(_WEIGHTS)

# extract and register our prompt classes
_names = [s.strip() for s in _PROMPTS.split(",") if s.strip()]
_tokens = _model.get_text_pe(_names)
_model.set_classes(_names, _tokens)


def predict_boxes(img: np.ndarray) -> list[tuple[float,float,float,float]]:
    """
    Run YOLOE on a single OpenCV image (BGR ndarray) 
    and return a list of (x1, y1, x2, y2) float tuples.
    """
    # ultralytics can accept numpy arrays directly
    results = _model.predict(
        source=img,            # pass the ndarray
        imgsz=640,
        conf=_CONF_THRESH,
        device=_DEVICE,
        verbose=False,
    )
    # assume first (and only) batch
    xyxy = results[0].boxes.xyxy.cpu().tolist()
    return [(float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2, *_ in xyxy]


def annotate_and_save(img: np.ndarray, boxes: list[tuple[float,float,float,float]], out_path: str):
    """
    Draws boxes on the image and writes out to disk.
    """
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      color=(0,255,0), thickness=2)
    cv2.imwrite(out_path, img)
