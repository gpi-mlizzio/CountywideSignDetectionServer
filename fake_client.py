#!/usr/bin/env python3
import socket
import struct
import sys

HOST, PORT = "localhost", 5001
TYPE_DETECT = 1

def send_image(path: str):
    # read image bytes
    with open(path, "rb") as f:
        img = f.read()

    with socket.create_connection((HOST, PORT)) as sock:
        # 1-byte msg type + 4-byte meta length (0) + 4-byte img length + img
        sock.sendall(struct.pack("!B I", TYPE_DETECT, 0))
        sock.sendall(struct.pack("!I", len(img)))
        sock.sendall(img)

        # read back number of boxes
        count = struct.unpack("!I", sock.recv(4))[0]
        print(f"Detected {count} boxes:")
        for _ in range(count):
            x1, y1, x2, y2 = struct.unpack("!ffff", sock.recv(16))
            print(f"  box: ({x1:.1f}, {y1:.1f}) → ({x2:.1f}, {y2:.1f})")

if __name__ == "__main__":
    send_image(f"./debug_tools/data/incoming/20250529_082854_316682.jpg")
