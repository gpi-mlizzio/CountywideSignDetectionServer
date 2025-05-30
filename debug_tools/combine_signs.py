#!/usr/bin/env python3
"""
Walk every data/tracks/track_*/signs.json,
merge into debug_tools/map/combined_signs.json,
and refresh it every 30 s.
"""

import json, time, sys
from pathlib import Path

SLEEP = 30  # seconds between scans
ROOT   = Path(__file__).resolve().parent
TRACKS = ROOT / ".." / "data" / "tracks"
OUTPUT = ROOT / "map" / "combined_signs.json"

def combine():
    merged = {}
    for path in TRACKS.glob("track_*"):
        sj = path / "signs.json"
        if not sj.exists():
            continue
        try:
            data = json.loads(sj.read_text())
        except Exception as e:
            print(f"[WARN] bad JSON {sj}: {e}")
            continue
        cap_id = path.name.split("_",1)[1]
        for sid, info in data.items():
            info["capture_id"] = cap_id
            merged[sid] = info
    return merged

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"[combiner] Watching {TRACKS} → {OUTPUT} every {SLEEP}s")
    while True:
        combined = combine()
        try:
            with OUTPUT.open("w") as f:
                json.dump(combined, f, indent=2)
            print(f"[{time.strftime('%H:%M:%S')}] wrote {len(combined)} signs")
        except Exception as e:
            print(f"[ERROR] writing output: {e}", file=sys.stderr)
        time.sleep(SLEEP)

if __name__ == "__main__":
    main()
