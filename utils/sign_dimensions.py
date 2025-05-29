#!/usr/bin/env python3
import json, math
import statistics
from pathlib import Path
from collections import defaultdict
import pprint

# ───── INPUT / OUTPUT ─────────────────────────────────────
INPUT_JSON  = Path("interpolated_detections_with_distances.json")
OUTPUT_SUM  = Path("detections_summary_e45.json")

# ───── YOUR REFERENCE DATA & FORMULAS ─────────────────────
measurements = {
    60: {'width': 29, 'height': 35},
    20: {'width': 76, 'height': 80},
}

def calc_focal_length_from_known_height(p1, p2, d, known_height):
    return (d * p1 * p2) / (known_height * (p2 - p1))

def sign_height(p1, p2, d, f_px):
    return (d * p1 * p2) / (f_px * (p2 - p1))

# calibrate focal length using 60 ft vs 20 ft and a 2 ft sign
REF_DIST_FT = 40.0
F2_HT = measurements[20]['height']
F1_HT = measurements[60]['height']
F2_WD = measurements[20]['width']
F1_WD = measurements[60]['width']
FPX = calc_focal_length_from_known_height(F1_HT, F2_HT, REF_DIST_FT, 2.0)

# ───── HELPER TO PROJECT ENU BACK TO LAT/LON ─────────────
class Geo:
    R = 6371000
    @staticmethod
    def enu_to_latlon(east, north, ref_lat, ref_lon):
        lat = ref_lat + math.degrees(north/Geo.R)
        lon = ref_lon + math.degrees(east/(Geo.R * math.cos(math.radians(ref_lat))))
        return lat, lon

# ───── AGGREGATE OBSERVATIONS ─────────────────────────────
agg = defaultdict(lambda: {
    "lats": [], "lons": [], "heights": [], "widths": []
})

data = json.loads(INPUT_JSON.read_text())

for frame, info in data.items():
    cam_lat, cam_lon = info["latitude"], info["longitude"]
    for det in info.get("detections", []):
        if "distance" not in det or "enu_east_m" not in det:
            continue
        sid = str(det["id"])
        # back-project GPS
        lat_s, lon_s = Geo.enu_to_latlon(
            det["enu_east_m"], det["enu_north_m"], cam_lat, cam_lon
        )
        dist_ft = det["distance"] / 0.3048

        # estimate sign size via your two-photo formula
        h_ft = abs(sign_height(
            p1 = det["bbox"]["height"],
            p2 = F2_HT,
            d  = abs(dist_ft - 20),
            f_px = FPX
        ))
        w_ft = abs(sign_height(
            p1 = det["bbox"]["width"],
            p2 = F2_WD,
            d  = abs(dist_ft - 20),
            f_px = FPX
        ))

        agg[sid]["lats"].append(lat_s)
        agg[sid]["lons"].append(lon_s)
        agg[sid]["heights"].append(h_ft)
        agg[sid]["widths"].append(w_ft)

pprint.pprint(agg)

# ───── COMPUTE MEDIAN PER SIGN ID ─────────────────────────
summary = {}
for sid, vals in agg.items():
    # median latitude/longitude
    med_lat = statistics.median(vals["lats"])
    med_lon = statistics.median(vals["lons"])
    # median height/width
    med_h  = statistics.median(vals["heights"])
    med_w  = statistics.median(vals["widths"])

    med_h  = min(vals["heights"])*12
    med_w  = min(vals["widths"])*12

    summary[sid] = {
        "latitude":             round(med_lat,  8),
        "longitude":            round(med_lon,  8),
        "estimated_height_inches":  round(med_h, 2),
        "estimated_width_inches":   round(med_w, 2)
    }

OUTPUT_SUM.write_text(json.dumps(summary, indent=2))
print(f"✅ Wrote aggregated summary with medians → {OUTPUT_SUM}")
