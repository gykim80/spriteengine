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
    V=/runpod-volume
    mkdir -p "$V/spriteengine" "$V/u2net"
    echo "$HANDLER_B64" | base64 -d > "$V/spriteengine/handler.py"
    python3 -c "import ast; ast.parse(open('/runpod-volume/spriteengine/handler.py').read()); print('handler syntax ok')"
    # u2net 프리워밍: rembg/onnxruntime 임포트 스모크 테스트는 CPU SECURE pod의
    # executable-stack 제약으로 .so 로딩이 거부되어 항상 실패한다(GPU 워커는 정상).
    # 실동작 검증은 프로덕션 recon의 background_removed 메트릭이 담당하므로,
    # 여기서는 모델 파일을 직접 다운로드·md5 검증하는 방식으로 프리워밍만 한다.
    U2NET_HOME="$V/u2net" python3 - <<'PY'
import hashlib
import os
import urllib.request

URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
MD5 = "60024c5c889badc19c04ad937298a77b"  # rembg가 세션 생성 시 검증하는 known_hash
path = os.path.join(os.environ["U2NET_HOME"], "u2net.onnx")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if os.path.exists(path) and md5(path) == MD5:
    print(f"u2net already cached: {path} {os.path.getsize(path)} bytes")
else:
    urllib.request.urlretrieve(URL, path + ".tmp")
    got = md5(path + ".tmp")
    assert got == MD5, f"u2net md5 mismatch: {got}"
    os.replace(path + ".tmp", path)
    print(f"u2net downloaded: {path} {os.path.getsize(path)} bytes")
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
