#!/usr/bin/env python3
"""End-to-end proof: submit a real image to the serverless endpoint, poll the
job, and validate that the returned artifact is a genuine binary GLB.

Usage:
  RUNPOD_API_KEY=... python3 deploy/runpod/e2e_generate.py --endpoint <id>
  ... --image path/to/character.png      # defaults to a generated test image
"""
import argparse
import base64
import json
import os
import struct
import sys
import time
import urllib.request
import zlib

RUNTIME_API = "https://api.runpod.ai/v2"


def make_test_png(size=256):
    """Dependency-free PNG: white background with a dark rounded blob, which is
    enough for the shape pipeline to produce a mesh."""
    rows = []
    c, r = size // 2, size // 3
    for y in range(size):
        row = bytearray([0])  # filter: none
        for x in range(size):
            body = (x - c) ** 2 + (y - c) ** 2 < r * r
            head = (x - c) ** 2 + (y - c + r) ** 2 < (r // 2) ** 2
            row += bytes([40, 90, 200] if body or head else [255, 255, 255])
        rows.append(bytes(row))
    raw = zlib.compress(b"".join(rows), 9)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", raw) + chunk(b"IEND", b"")


def call(key, url, body=None, timeout=90):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), method="POST" if body else "GET")
    req.add_header("Authorization", "Bearer " + key)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True)
    p.add_argument("--image", default="", help="PNG/JPG to send; generated test image when omitted")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--timeout-minutes", type=int, default=25)
    p.add_argument("--output", default="/tmp/spriteengine_e2e.glb")
    args = p.parse_args()
    key = os.getenv("RUNPOD_API_KEY", "").strip()
    if not key:
        sys.exit("RUNPOD_API_KEY is required")

    image_bytes = open(args.image, "rb").read() if args.image else make_test_png()
    print(f"input image: {args.image or 'generated test blob'} ({len(image_bytes)} bytes)")

    base = f"{RUNTIME_API}/{args.endpoint}"
    health = call(key, base + "/health")
    print("health:", json.dumps(health))

    job = call(key, base + "/run", {"input": {
        "image": base64.b64encode(image_bytes).decode("ascii"),
        "seed": 1234,
        "steps": args.steps,
        "guidance_scale": 5.0,
    }})
    job_id = job["id"]
    print(f"job submitted: {job_id}")

    deadline = time.time() + args.timeout_minutes * 60
    status = {}
    while time.time() < deadline:
        try:
            status = call(key, f"{base}/status/{job_id}")
        except Exception as exc:  # transient gateway blips must not abort the poll
            print(f"[{time.strftime('%H:%M:%S')}] status poll error, retrying: {exc}", flush=True)
            time.sleep(10)
            continue
        state = status.get("status")
        print(f"[{time.strftime('%H:%M:%S')}] {state} delay={status.get('delayTime')} exec={status.get('executionTime')}", flush=True)
        if state == "COMPLETED":
            break
        if state in ("FAILED", "CANCELLED", "TIMED_OUT"):
            sys.exit(f"job {state}: {json.dumps(status, indent=2)[:4000]}")
        time.sleep(15)
    else:
        sys.exit(f"job {job_id} did not complete in {args.timeout_minutes} minutes")

    output = status.get("output") or {}
    glb = base64.b64decode(output.get("glb_base64", ""))
    if glb[:4] != b"glTF":
        sys.exit(f"artifact is not a binary GLB (magic={glb[:4]!r}, keys={list(output)})")
    with open(args.output, "wb") as fh:
        fh.write(glb)
    print(json.dumps({
        "ok": True,
        "model": output.get("model"),
        "bytes": len(glb),
        "glbMagic": "glTF",
        "savedTo": args.output,
        "delayTimeMs": status.get("delayTime"),
        "executionTimeMs": status.get("executionTime"),
    }, indent=2))


if __name__ == "__main__":
    main()
