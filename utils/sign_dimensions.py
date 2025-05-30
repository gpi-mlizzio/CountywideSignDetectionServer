# utils/sign_dimensions.py
import json
import math
import statistics
from pathlib import Path
from collections import defaultdict


class DetectionSummary:
    def __init__(self):
        # calibration parameters
        measurements = {
            60: {'width': 29, 'height': 35},
            20: {'width': 76, 'height': 80},
        }
        REF_DIST_FT = 40.0
        F2_HT = measurements[20]['height']
        F1_HT = measurements[60]['height']
        # focal length in px
        self.FPX = (REF_DIST_FT * F1_HT * F2_HT) / (2.0 * (F2_HT - F1_HT))
        self.reference_width = measurements[20]['width']
        self.reference_height = measurements[20]['height']

    @staticmethod
    def enu_to_latlon(east, north, ref_lat, ref_lon):
        R = 6371000
        lat = ref_lat + math.degrees(north / R)
        lon = ref_lon + math.degrees(east / (R * math.cos(math.radians(ref_lat))))
        return lat, lon

    def sign_height(self, p1, p2, d):
        # returns size in feet
        return (d * p1 * p2) / (self.FPX * (p2 - p1))

    def process(self, data) -> dict:
        """
        Takes a frame-sorted detection dict (loaded JSON), 
        returns a summary dict keyed by sign ID.
        """
        agg = defaultdict(lambda: {
            "lats": [], "lons": [], "heights": [], "widths": []
        })

        for frame, info in data.items():
            cam_lat = info["latitude"]
            cam_lon = info["longitude"]

            for det in info.get("detections", []):
                if "distance" not in det or "enu_east_m" not in det:
                    continue

                sid = str(det.get("sign_id", det["id"]))

                # use precomputed location if present
                if "sign_location" in det:
                    lat_s = det["sign_location"]["latitude"]
                    lon_s = det["sign_location"]["longitude"]
                else:
                    lat_s, lon_s = self.enu_to_latlon(
                        det["enu_east_m"], det["enu_north_m"], cam_lat, cam_lon
                    )

                dist_ft = det["distance"] / 0.3048

                h_ft = abs(self.sign_height(
                    p1=det["bbox"]["height"],
                    p2=self.reference_height,
                    d=abs(dist_ft - 20)
                ))
                w_ft = abs(self.sign_height(
                    p1=det["bbox"]["width"],
                    p2=self.reference_width,
                    d=abs(dist_ft - 20)
                ))

                agg[sid]["lats"].append(lat_s)
                agg[sid]["lons"].append(lon_s)
                agg[sid]["heights"].append(h_ft)
                agg[sid]["widths"].append(w_ft)

        # build summary
        summary = {}
        for sid, vals in agg.items():
            med_lat = statistics.median(vals["lats"])
            med_lon = statistics.median(vals["lons"])
            med_h   = min(vals["heights"]) * 12  # inches
            med_w   = min(vals["widths"])  * 12  # inches

            summary[sid] = {
                "latitude": round(med_lat, 8),
                "longitude": round(med_lon, 8),
                "estimated_height_inches": round(med_h, 2),
                "estimated_width_inches":  round(med_w, 2),
            }

        return summary


def main():
    # load frame-sorted detections
    data = json.loads(Path("./augmented_detections.json").read_text())

    # compute summary
    summarizer = DetectionSummary()
    summary = summarizer.process(data)

    # save to file
    Path("./final_info.json").write_text(json.dumps(summary, indent=2))
    print(f"✅ Wrote aggregated summary → {Path("./final_info.json")}")

    # return summary dict if needed elsewhere
    return summary


if __name__ == "__main__":
    main()
