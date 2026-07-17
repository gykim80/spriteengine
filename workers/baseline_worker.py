#!/usr/bin/env python3
"""SpriteEngine JSON Lines worker baseline.

The worker intentionally uses only Python's standard library so the desktop
orchestrator can validate its process/progress/artifact contract before large
GPU environments are installed.
"""
import hashlib
import json
import os
import shutil
import struct
import sys
import time


def emit(event_type, **payload):
    print(json.dumps({"type": event_type, **payload}, separators=(",", ":")), flush=True)


def dimensions(path):
    with open(path, "rb") as f:
        head = f.read(32)
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return struct.unpack(">II", head[16:24])
        if head[:2] == b"\xff\xd8":
            f.seek(2)
            while True:
                marker = f.read(2)
                if len(marker) < 2:
                    break
                if marker[0] != 0xFF:
                    continue
                length_raw = f.read(2)
                if len(length_raw) < 2:
                    break
                length = struct.unpack(">H", length_raw)[0]
                if marker[1] in range(0xC0, 0xC4):
                    data = f.read(5)
                    return struct.unpack(">HH", data[1:5])[::-1]
                f.seek(length - 2, 1)
        if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
            # Dimensions are optional for this baseline; format validation passed.
            return 0, 0
    raise ValueError("unsupported or corrupt image")


def run(req):
    job = req["jobId"]
    stage = req["stage"]
    workspace = os.path.abspath(req["workspace"])
    os.makedirs(workspace, exist_ok=True)
    emit("progress", jobId=job, progress=.08, message="Worker environment ready")
    if stage == "prepare":
        source = os.path.abspath(req["input"])
        width, height = dimensions(source)
        with open(source, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        out_dir = os.path.join(workspace, "prepare")
        os.makedirs(out_dir, exist_ok=True)
        output = os.path.join(out_dir, "reference" + os.path.splitext(source)[1].lower())
        shutil.copy2(source, output)
        emit("progress", jobId=job, progress=.62, message="Validated image and provenance")
        metrics = {"width": width, "height": height, "sha256": digest, "alphaRequired": True}
        emit("artifact", jobId=job, kind="reference", path=output, metrics=metrics)
    else:
        # Adapter handshake artifact. GPU model adapters replace this branch.
        out_dir = os.path.join(workspace, stage)
        os.makedirs(out_dir, exist_ok=True)
        output = os.path.join(out_dir, "adapter-request.json")
        with open(output, "w", encoding="utf-8") as f:
            json.dump(req, f, indent=2)
        emit("progress", jobId=job, progress=.7, message=f"Prepared {stage} adapter request")
        metrics = {"adapter": req.get("adapter", "baseline"), "requiresModel": True}
        emit("artifact", jobId=job, kind="adapter-request", path=output, metrics=metrics)
    time.sleep(.03)
    emit("done", jobId=job, stage=stage, progress=1, metrics=metrics)


for line in sys.stdin:
    try:
        request = json.loads(line)
        if request.get("type") != "run":
            raise ValueError("expected run message")
        run(request)
    except Exception as exc:
        emit("error", message=str(exc))
        sys.exit(1)
