# utils/sign_triangulator.py 
import json
import math
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from pprint import pprint
from typing import Dict, List, Any

MERGE_THRESHOLD = 5

# Ray lengths in meters
HEADING_RAY_LEN = 20.0
ANGLE_RAY_LEN   = 20.0

# Fusion weights (tweak as desired)
WEIGHTS = {
    "first_two": 1.0,
    "last_two":  5.0,
    "closest":   5.0,
    "furthest":  1.0,
}

# FOV and screen width for angle calculation
FOV = 140
SCREEN_WIDTH = 1920


class Locator:
    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371000
        φ1, φ2 = map(math.radians, (lat1, lat2))
        dφ = math.radians(lat2 - lat1)
        dλ = math.radians(lon2 - lon1)
        a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
        return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

    def filter_detections(self, raw):
        grouped = defaultdict(dict)
        for frame, info in raw.items():
            for det in info.get("detections", []):
                prev = grouped[det["id"]].get(frame)
                if prev is None or det["conf"] > prev["conf"]:
                    grouped[det["id"]][frame] = {
                        **det,
                        "frame": frame,
                        "latitude": info["latitude"],
                        "longitude": info["longitude"],
                        "timestamp": info["timestamp"],
                        "heading": info.get("heading", None)
                    }
        return {i: list(d.values()) for i,d in grouped.items() if len(d)>=2}

    def retrieve_first_two(self, dets):
        out = {}
        for i,l in dets.items():
            uniq = list({d["frame"]:d for d in l}.values())
            sorted_l = sorted(uniq, key=lambda d:d["timestamp"])
            out[i] = tuple(sorted_l[:2])
        return out

    def retrieve_last_two(self, dets):
        out = {}
        for i,l in dets.items():
            uniq = list({d["frame"]:d for d in l}.values())
            sorted_l = sorted(uniq, key=lambda d:d["timestamp"])
            out[i] = tuple(sorted_l[-2:])
        return out

    def retrieve_closest(self, dets):
        out = {}
        for i,l in dets.items():
            uniq = list({d["frame"]:d for d in l}.values())
            best = None; md = float('inf')
            for a in range(len(uniq)):
                for b in range(a+1,len(uniq)):
                    d1,d2 = uniq[a],uniq[b]
                    if (d1["latitude"],d1["longitude"]) == (d2["latitude"],d2["longitude"]):
                        continue
                    dist = self.haversine(d1["latitude"],d1["longitude"],
                                          d2["latitude"],d2["longitude"])
                    if dist < md:
                        md, best = dist, (d1,d2)
            out[i] = best
        return out

    def retrieve_furthest(self, dets):
        out = {}
        for i,l in dets.items():
            uniq = list({d["frame"]:d for d in l}.values())
            best = None; Md = -1
            for a in range(len(uniq)):
                for b in range(a+1,len(uniq)):
                    d1,d2 = uniq[a],uniq[b]
                    dist = self.haversine(d1["latitude"],d1["longitude"],
                                          d2["latitude"],d2["longitude"])
                    if dist > Md:
                        Md, best = dist, (d1,d2)
            out[i] = best
        return out

    def latlon_to_enu(self, lat, lon, ref_lat, ref_lon):
        R = 6371000
        dφ = math.radians(lat - ref_lat)
        dλ = math.radians(lon - ref_lon)
        north = R * dφ
        east  = R * math.cos(math.radians(ref_lat)) * dλ
        return east, north

    def enu_to_latlon(self, e, n, ref_lat, ref_lon):
        R = 6371000
        lat = ref_lat + math.degrees(n / R)
        lon = ref_lon + math.degrees(e / (R * math.cos(math.radians(ref_lat))))
        return lat, lon

    def bearing(self, p1, p2):
        φ1,φ2,λ1,λ2 = map(math.radians,
            [p1["latitude"],p2["latitude"],p1["longitude"],p2["longitude"]])
        dλ = λ2 - λ1
        x = math.sin(dλ) * math.cos(φ2)
        y = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(dλ)
        return (math.degrees(math.atan2(x, y)) + 360) % 360

    def angle_offset(self, det):
        return (det["bbox"]["left"] / SCREEN_WIDTH) * FOV - (FOV / 2)

    def intersect(self, p1, p2):
        ref_lat, ref_lon = p1["latitude"], p1["longitude"]
        e2,n2 = self.latlon_to_enu(p2["latitude"], p2["longitude"], ref_lat, ref_lon)
        b = self.bearing(p1, p2)
        a1, a2 = self.angle_offset(p1), self.angle_offset(p2)
        θ1, θ2 = math.radians(a1 + b), math.radians(a2 + b)
        dx1, dy1 = math.sin(θ1), math.cos(θ1)
        dx2, dy2 = math.sin(θ2), math.cos(θ2)
        denom = dx1 * dy2 - dy1 * dx2
        if abs(denom) < 1e-6:
            return None
        t1 = (e2 * dy2 - n2 * dx2) / denom
        xi, yi = t1 * dx1, t1 * dy1
        lat_i, lon_i = self.enu_to_latlon(xi, yi, ref_lat, ref_lon)

        # build rays
        heading = p1.get("heading", None)
        hray = None
        if heading is not None:
            θh = math.radians(heading)
            ex = HEADING_RAY_LEN * math.sin(θh)
            ny = HEADING_RAY_LEN * math.cos(θh)
            hray = [
                [p1["latitude"], p1["longitude"]],
                self.enu_to_latlon(ex, ny, ref_lat, ref_lon)
            ]
        ar1 = [
            [p1["latitude"], p1["longitude"]],
            self.enu_to_latlon(dx1 * ANGLE_RAY_LEN, dy1 * ANGLE_RAY_LEN, ref_lat, ref_lon)
        ]
        ar2 = [
            [p2["latitude"], p2["longitude"]],
            self.enu_to_latlon(dx2 * ANGLE_RAY_LEN, dy2 * ANGLE_RAY_LEN, ref_lat, ref_lon)
        ]
        return {"pt":[lat_i,lon_i], "heading_ray":hray, "angle_rays":(ar1,ar2)}


    # Extra json formatter and merger
    def group_and_merge(self, raw) -> Dict[int, List[dict]]:
        # 1) per-frame merge
        cleaned_per_frame: Dict[str, List[dict]] = {}
        for frame, info in raw.items():
            merged: List[dict] = []
            for det in info.get("detections", []):
                # bbox center
                left, top, w, h = det["bbox"].values()
                cx, cy = left + w/2, top + h/2

                for m in merged:
                    if m["id"] != det["id"]:
                        continue
                    mx, my = m["_center"]
                    if abs(cx - mx) <= MERGE_THRESHOLD and abs(cy - my) <= MERGE_THRESHOLD:
                        # pick higher conf & fuse labels
                        if det["conf"] > m["conf"]:
                            m["conf"], m["bbox"] = det["conf"], det["bbox"]
                        if det["label"] not in m["labels"]:
                            m["labels"].append(det["label"])
                        break
                else:
                    entry = det.copy()
                    entry["labels"] = [det["label"]]
                    entry["_center"] = (cx, cy)
                    merged.append(entry)

            # strip helper key
            for m in merged:
                del m["_center"]
            cleaned_per_frame[frame] = merged

        # 2) flatten by id
        by_id: Dict[int, List[dict]] = defaultdict(list)
        for frame, info in raw.items():
            lat, lon, ts = info["latitude"], info["longitude"], info["timestamp"]
            for det in cleaned_per_frame[frame]:
                record = {
                    **det,
                    "frame":     frame,
                    "latitude":  lat,
                    "longitude": lon,
                    "timestamp": ts,
                }
                record.pop("label", None)
                by_id[record["id"]].append(record)

        return by_id

    def main(self, detections=None, detections_path=None):
        raw = None

        # Load & filter
        if detections_path:
            path = Path(detections_path)
            with path.open() as f:
                raw = json.load(f)
            detections = self.filter_detections(raw)

        elif detections:
            raw = {"frame": {"latitude": d["latitude"],
                        "longitude": d["longitude"],
                        "detections": []}
                for lst in detections.values() for d in lst}
            detections = self.filter_detections(detections)

        else:
            return None

        # pprint(detections)

        # Prepare intersection methods
        methods = {
            "first_two": self.retrieve_first_two(detections),
            "last_two":  self.retrieve_last_two(detections),
            "closest":   self.retrieve_closest(detections),
            "furthest":  self.retrieve_furthest(detections),
        }

        # Compute fused intersections
        fused = {}
        for det_id in detections:
            pts = {}
            for name, pairs in methods.items():
                p1, p2 = pairs.get(det_id, (None, None))
                if not p1 or not p2:
                    continue
                res = self.intersect(p1, p2)
                if res:
                    pts[name] = res["pt"]
            if not pts:
                continue
            W = sum(WEIGHTS[n] for n in pts)
            fused[det_id] = (
                sum(pts[n][0] * WEIGHTS[n] for n in pts) / W,
                sum(pts[n][1] * WEIGHTS[n] for n in pts) / W
            )

        # Now augment the **raw** JSON with distances
        for frame_key, frame_info in raw.items():
            cam_lat = frame_info["latitude"]
            cam_lon = frame_info["longitude"]
            for det in frame_info.get("detections", []):
                det_id = det["id"]
                if det_id not in fused:
                    continue
                obj_lat, obj_lon = fused[det_id]
                e, n = self.latlon_to_enu(obj_lat, obj_lon, cam_lat, cam_lon)
                hyp = math.hypot(e, n)
                det["distance"]      = float(hyp)
                det["enu_east_m"]    = float(e)
                det["enu_north_m"]   = float(n)
                det["sign_location"] = {"latitude": obj_lat, "longitude": obj_lon}


        # (Optional) write back to file or return the augmented JSON
        return raw


if __name__ == "__main__":
    loc = Locator()
    computed_detections = loc.main(detections_path=Path("../data/tracks/track_5AC9A5EC/detections.json"))
    # for example, save it:
    with open("augmented_detections.json", "w") as out:
        json.dump(computed_detections, out, indent=2)

    grouped = loc.group_and_merge(computed_detections)
    with open("grouped_by_id.json", "w") as f:
        json.dump(grouped, f, indent=2)

