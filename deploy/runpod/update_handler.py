#!/usr/bin/env python3
"""Push the local handler.py to the RunPod network volume (no re-bootstrap).

The serverless worker execs /runpod-volume/spriteengine/handler.py, so a
handler-only change just needs a cheap CPU pod to (1) overwrite that file and
(2) pre-download the rembg u2net model into $V/u2net so the first GPU request
doesn't spend its timeout on a cold model download.

Usage:
  RUNPOD_API_KEY=... python3 deploy/runpod/update_handler.py
"""
import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

from warm_volume import VOLUME_NAME, create_pod, fetch_progress, items, request

POD_NAME = "spriteengine-handler-update"

UPDATE_SCRIPT = r"""
mkdir -p /srv/progress
cd /srv/progress
: > progress.log
python3 -m http.server 8888 --bind 0.0.0.0 >/dev/null 2>&1 &
{
    set -euo pipefail
    apt-get update -qq && apt-get install -y -qq libgomp1 libgl1 libglib2.0-0
    V=/runpod-volume
    mkdir -p "$V/spriteengine" "$V/u2net"
    echo "$HANDLER_B64" | base64 -d > "$V/spriteengine/handler.py"
    python3 -c "import ast; ast.parse(open('/runpod-volume/spriteengine/handler.py').read()); print('handler syntax ok')"
    # u2net 프리워밍 + rembg 스모크 테스트 (pydeps311는 cp311 대상 — 이 이미지와 일치)
    export U2NET_HOME="$V/u2net"
    PYTHONPATH="$V/pydeps311" python3 - <<'PY'
import os
from PIL import Image
from rembg import new_session, remove
session = new_session()  # u2net.onnx를 U2NET_HOME(볼륨)에 다운로드
img = Image.new("RGB", (64, 64), (200, 200, 200))
out = remove(img, session=session, bgcolor=[255, 255, 255, 0])
assert out.mode == "RGBA", out.mode
print("rembg smoke ok; u2net cache:", sorted(os.listdir(os.environ["U2NET_HOME"])))
PY
} >> progress.log 2>&1 && echo UPDATE_OK >> progress.log || echo UPDATE_FAIL >> progress.log
sleep 3600
"""


def wait_for_update(pod_id, timeout_minutes):
    deadline = time.time() + timeout_minutes * 60
    printed = 0
    while time.time() < deadline:
        progress = fetch_progress(pod_id)
        if progress is not None:
            if len(progress) > printed:
                sys.stdout.write(progress[printed:])
                sys.stdout.flush()
                printed = len(progress)
            if "UPDATE_OK" in progress:
                return
            if "UPDATE_FAIL" in progress:
                raise RuntimeError("handler update failed inside the pod; see log above")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] progress endpoint not reachable yet", flush=True)
        time.sleep(15)
    raise TimeoutError(f"update pod {pod_id} did not finish within {timeout_minutes} minutes")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timeout-minutes", type=int, default=20)
    args = p.parse_args()
    key = os.getenv("RUNPOD_API_KEY", "").strip()
    if not key:
        sys.exit("RUNPOD_API_KEY is required")

    volume = next((v for v in items(request(key, "GET", "/networkvolumes"))
                   if v.get("name") == VOLUME_NAME), None)
    if not volume:
        sys.exit(f"network volume {VOLUME_NAME!r} not found")
    print(f"volume: {volume['id']} ({volume.get('dataCenterId')})")

    handler = Path(__file__).with_name("handler.py").read_bytes()
    stale = next((x for x in items(request(key, "GET", "/pods")) if x.get("name") == POD_NAME), None)
    if stale:
        print(f"terminating stale update pod {stale['id']}")
        request(key, "DELETE", "/pods/" + stale["id"])
    pod = create_pod(key, {
        "name": POD_NAME,
        "imageName": "python:3.11-slim",
        "computeType": "CPU",
        "cloudType": "SECURE",
        "vcpuCount": 2,
        "containerDiskInGb": 10,
        "networkVolumeId": volume["id"],
        "volumeMountPath": "/runpod-volume",
        "ports": ["8888/http"],
        "env": {"HANDLER_B64": base64.b64encode(handler).decode("ascii")},
        "dockerEntrypoint": [],
        "dockerStartCmd": ["bash", "-c", UPDATE_SCRIPT],
    })
    pod_id = pod["id"]
    print(f"update pod started: {pod_id}")
    try:
        wait_for_update(pod_id, args.timeout_minutes)
    finally:
        try:
            request(key, "DELETE", "/pods/" + pod_id)
            print(f"\nupdate pod terminated: {pod_id}")
        except RuntimeError as exc:
            print(f"\npod cleanup failed (terminate manually): {exc}")
    print(json.dumps({"volumeId": volume["id"], "handlerBytes": len(handler)}, indent=2))


if __name__ == "__main__":
    main()
