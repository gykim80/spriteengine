#!/usr/bin/env python3
"""Pre-download Hunyuan3D weights into the RunPod network volume.

Nothing is downloaded to the local machine. This script:
  1. ensures the `spriteengine-model-cache` network volume exists,
  2. launches a small CPU pod attached to that volume,
  3. runs `huggingface_hub.snapshot_download` inside the pod so the weights
     land directly in /runpod-volume/huggingface (the same HF_HOME the
     serverless worker uses),
  4. waits for the pod to exit and terminates it.

Usage:
  RUNPOD_API_KEY=... python3 deploy/runpod/warm_volume.py
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://rest.runpod.io/v1"
VOLUME_NAME = "spriteengine-model-cache"
POD_NAME = "spriteengine-volume-warmup"

WARMUP_SCRIPT = r"""
set -euo pipefail
export HF_HOME=/runpod-volume/huggingface
export HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface/hub
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install --no-cache-dir 'huggingface_hub[hf_transfer]==0.27.1'
python - <<'PY'
from huggingface_hub import snapshot_download
import os
model = os.environ["WARMUP_MODEL"]
path = snapshot_download(model, max_workers=8)
print("snapshot complete:", path, flush=True)
PY
touch /runpod-volume/huggingface/.spriteengine-warmup-complete
du -sh /runpod-volume/huggingface
"""


def request(key, method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(API + path, data=data, method=method)
    req.add_header("Authorization", "Bearer " + key)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"RunPod {method} {path}: HTTP {exc.code}: {detail}") from exc


def items(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("items", "data", "networkVolumes", "pods"):
            if isinstance(value.get(key), list):
                return value[key]
    return []


def ensure_volume(key, region, size_gb):
    existing = next(
        (v for v in items(request(key, "GET", "/networkvolumes")) if v.get("name") == VOLUME_NAME),
        None,
    )
    if existing:
        print(f"network volume reuse: {existing['id']} ({existing.get('dataCenterId')})")
        return existing
    created = request(key, "POST", "/networkvolumes", {
        "name": VOLUME_NAME, "size": size_gb, "dataCenterId": region,
    })
    print(f"network volume created: {created['id']} ({region}, {size_gb}GB)")
    return created


def launch_pod(key, volume_id, model):
    stale = next(
        (p for p in items(request(key, "GET", "/pods")) if p.get("name") == POD_NAME),
        None,
    )
    if stale:
        print(f"terminating stale warmup pod {stale['id']}")
        request(key, "DELETE", "/pods/" + stale["id"])
    pod = request(key, "POST", "/pods", {
        "name": POD_NAME,
        "imageName": "python:3.11-slim",
        "computeType": "CPU",
        "cloudType": "SECURE",
        "vcpuCount": 4,
        "containerDiskInGb": 10,
        "networkVolumeId": volume_id,
        "volumeMountPath": "/runpod-volume",
        "ports": [],
        "env": {"WARMUP_MODEL": model},
        "dockerEntrypoint": [],
        "dockerStartCmd": ["bash", "-c", WARMUP_SCRIPT],
    })
    print(f"warmup pod started: {pod['id']}")
    return pod


def wait_for_exit(key, pod_id, timeout_minutes):
    deadline = time.time() + timeout_minutes * 60
    last = ""
    while time.time() < deadline:
        pod = request(key, "GET", "/pods/" + pod_id)
        status = pod.get("desiredStatus") or ""
        change = pod.get("lastStatusChange") or ""
        line = f"{status} | {change}"
        if line != last:
            print(f"[{time.strftime('%H:%M:%S')}] {line}", flush=True)
            last = line
        if status in ("EXITED", "TERMINATED"):
            return status
        time.sleep(30)
    raise TimeoutError(f"warmup pod {pod_id} did not exit within {timeout_minutes} minutes")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="tencent/Hunyuan3D-2.1")
    p.add_argument("--region", default="US-CA-2")
    p.add_argument("--volume-gb", type=int, default=100)
    p.add_argument("--timeout-minutes", type=int, default=120)
    p.add_argument("--keep-pod", action="store_true", help="Skip terminating the pod after exit")
    args = p.parse_args()
    key = os.getenv("RUNPOD_API_KEY", "").strip()
    if not key:
        sys.exit("RUNPOD_API_KEY is required")

    volume = ensure_volume(key, args.region, args.volume_gb)
    pod = launch_pod(key, volume["id"], args.model)
    try:
        status = wait_for_exit(key, pod["id"], args.timeout_minutes)
    finally:
        if not args.keep_pod:
            request(key, "DELETE", "/pods/" + pod["id"])
            print(f"warmup pod terminated: {pod['id']}")
    print(json.dumps({
        "volumeId": volume["id"],
        "model": args.model,
        "podFinalStatus": status,
    }, indent=2))


if __name__ == "__main__":
    main()
