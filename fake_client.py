#!/usr/bin/env python3
"""
kill_server.py – tell simple_server.py to shut itself down.

Usage:
    python kill_server.py [host] [port]

Defaults are localhost:5001 (the same as simple_server.py).
"""

import socket
import struct
import sys

# ── read cmd‑line or fallback ─────────────────────────────────────────
HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 5001

TYPE_KILL = 3               # must match the server constant

def main():
    print(f"→ Connecting to {HOST}:{PORT} …")
    with socket.create_connection((HOST, PORT), timeout=5) as s:
        # 1‑byte type 3  +  4‑byte zero json‑length
        pkt = bytes([TYPE_KILL]) + struct.pack("!I", 0)
        s.sendall(pkt)
        # read 4‑byte ACK length (should be 0)
        ack = s.recv(4)
        if len(ack) != 4:
            print("✖ No ACK received – server may already be down.")
            return
        length = struct.unpack("!I", ack)[0]
        if length == 0:
            print("✓ Kill ACK received – server will exit now.")
        else:
            print(f"⚠ Unexpected ACK length: {length}")

if __name__ == "__main__":
    try:
        main()
    except ConnectionRefusedError:
        print("✖ Could not connect – is the server running?")
    except Exception as e:
        print("✖ Error:", e)


# #!/usr/bin/env python3
# import socket
# import struct
# import sys

# HOST, PORT = "localhost", 5001
# TYPE_DETECT = 1

# def send_image(path: str):
#     # read image bytes
#     with open(path, "rb") as f:
#         img = f.read()

#     with socket.create_connection((HOST, PORT)) as sock:
#         # 1-byte msg type + 4-byte meta length (0) + 4-byte img length + img
#         sock.sendall(struct.pack("!B I", TYPE_DETECT, 0))
#         sock.sendall(struct.pack("!I", len(img)))
#         sock.sendall(img)

#         # read back number of boxes
#         count = struct.unpack("!I", sock.recv(4))[0]
#         print(f"Detected {count} boxes:")
#         for _ in range(count):
#             x1, y1, x2, y2 = struct.unpack("!ffff", sock.recv(16))
#             print(f"  box: ({x1:.1f}, {y1:.1f}) → ({x2:.1f}, {y2:.1f})")

# if __name__ == "__main__":
#     send_image(f"./debug_tools/data/incoming/20250529_082854_316682.jpg")
