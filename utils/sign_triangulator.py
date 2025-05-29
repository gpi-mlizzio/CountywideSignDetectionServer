#!/usr/bin/env python3
import json
import math
from pathlib import Path
from collections import defaultdict
from datetime import datetime

INPUT_JSON  = Path("./detections_e45.json")
OUTPUT_VIEW  = Path("get_position/view_data.json")
OUTPUT_JSON  = Path("interpolated_detections_with_distances.json")

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


if __name__=="__main__":
    raw = json.loads(INPUT_JSON.read_text())
    loc = Locator()
    dets = loc.filter_detections(raw)

    methods = {
        "first_two": loc.retrieve_first_two(dets),
        "last_two":  loc.retrieve_last_two(dets),
        "closest":   loc.retrieve_closest(dets),
        "furthest":  loc.retrieve_furthest(dets),
    }

    # compute fused intersections
    fused = {}
    for det_id in dets:
        pts = {}
        for name,pair in methods.items():
            p1,p2 = pair.get(det_id,(None,None))
            if not p1 or not p2: continue
            res = loc.intersect(p1,p2)
            if res: pts[name] = res["pt"]
        if not pts: continue
        W = sum(WEIGHTS[n] for n in pts)
        fused[det_id] = (
            sum(pts[n][0]*WEIGHTS[n] for n in pts)/W,
            sum(pts[n][1]*WEIGHTS[n] for n in pts)/W
        )

    # build view_data.json as before
    colors = {
        "first_two":"#FF8800","last_two":"#8800FF",
        "closest":"#00CCCC","furthest":"#CC00CC"
    }
    view_pts, view_lines = [], []
    for det_id in dets:
        # plot each method + rays, then fused
        for name,pair in methods.items():
            p1,p2 = pair.get(det_id,(None,None))
            if not p1 or not p2: continue
            res = loc.intersect(p1,p2)
            if not res: continue
            col = colors[name]
            view_pts.append({"coords":res["pt"],"label":f"{det_id}:{name}","color":col})
            if res["heading_ray"]:
                view_lines.append({"coords":res["heading_ray"],"color":"#800080"})
            ar1,ar2 = res["angle_rays"]
            view_lines.append({"coords":ar1,"color":col})
            view_lines.append({"coords":ar2,"color":col})
        lat,lon = fused[det_id]
        view_pts.append({"coords":[lat,lon],"label":f"{det_id}:fused","color":"#FFFF00"})
        # connect method pts
        # instead of the broken line, do:
        pts = []
        for name, method_dict in methods.items():
            pair = method_dict.get(det_id)
            if not pair:
                continue
            p1, p2 = pair
            res = loc.intersect(p1, p2)
            if res:
                pts.append(res["pt"])

                view_lines.append({"coords":pts,"color":"#444444"})

    # write view_data
    center = view_pts[0]["coords"] if view_pts else [0,0]
    OUTPUT_VIEW.write_text(json.dumps({
        "points":view_pts,"lines":view_lines,"center":center,"zoom":18
    }, indent=2))

    # now augment the raw JSON with distances
    for frame, info in raw.items():
        cam_lat, cam_lon = info["latitude"], info["longitude"]
        for det in info.get("detections", []):
            det_id = det["id"]
            if det_id not in fused:
                continue
            obj_lat, obj_lon = fused[det_id]
            e,n = loc.latlon_to_enu(obj_lat, obj_lon, cam_lat, cam_lon)
            hyp = math.hypot(e,n)
            det["distance"]    = float(hyp)
            det["enu_east_m"]  = float(e)
            det["enu_north_m"] = float(n)

    # write augmented input
    OUTPUT_JSON.write_text(json.dumps(raw, indent=2))
    print(f"✅ Wrote view data → {OUTPUT_VIEW}")
    print(f"✅ Augmented detections → {OUTPUT_JSON}")
