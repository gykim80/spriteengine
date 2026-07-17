#!/usr/bin/env python3
"""Create a working Hunyuan3D serverless endpoint without building any image.

RunPod's GitHub-integration build is console-only and GHCR builds are blocked,
so this script deploys with a public base image instead:

  Phase A (bootstrap pod, observable over the RunPod HTTP proxy):
    - a cheap CPU pod attached to the `spriteengine-model-cache` volume
      installs Hunyuan3D-2.1 code + python deps into /runpod-volume
      (pip --target, cp311 wheels) and writes handler.py there.
  Phase B (REST): create a serverless template + endpoint that runs the
    public `runpod/pytorch` py3.11 image with PYTHONPATH pointing at the
    volume — no registry push, no console step.

Usage:
  RUNPOD_API_KEY=... python3 deploy/runpod/bootstrap_endpoint.py
  ... --skip-bootstrap        # volume already bootstrapped, only make endpoint
  ... --pod-id <id>           # re-attach to a running bootstrap pod
"""
import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

from warm_volume import VOLUME_NAME, create_pod, fetch_progress, items, request

WORKER_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
BOOTSTRAP_POD_NAME = "spriteengine-endpoint-bootstrap"
TEMPLATE_NAME = "spriteengine-hy3d21-volboot"
ENDPOINT_NAME = "spriteengine-hunyuan3d21"
# Shape generation needs ~10GB VRAM, so 24GB-class cards are acceptable
# fallbacks when the big cards are unavailable in the volume's datacenter.
GPU_TYPES = [
    "NVIDIA A100 80GB PCIe",
    "NVIDIA L40S",
    "NVIDIA RTX A6000",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A5000",
    "NVIDIA L4",
    "NVIDIA GeForce RTX 3090",
]

# Packages that the shape-only worker never imports (texture/demo/training
# extras) or that cannot install outside their original environment.
REQ_FILTER = "^--extra-index-url|^tb_nightly|^torchmetrics|^deepspeed|^bpy|^gradio|^fastapi|^uvicorn|^basicsr|^realesrgan|^cupy"

BOOTSTRAP_SCRIPT = r"""
mkdir -p /srv/progress
cd /srv/progress
: > progress.log
python3 -m http.server 8888 --bind 0.0.0.0 >/dev/null 2>&1 &
{
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    V=/runpod-volume
    apt-get update -qq
    apt-get install -y -qq git build-essential cmake libgl1 libglib2.0-0
    if [ ! -e "$V/hunyuan3d21/requirements.txt" ]; then
        rm -rf "$V/hunyuan3d21"
        git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git "$V/hunyuan3d21"
    fi
    grep -v -E "__REQ_FILTER__" "$V/hunyuan3d21/requirements.txt" > /tmp/req.txt
    printf 'torch==2.5.1\ntorchvision==0.20.1\nrunpod==1.7.13\n' >> /tmp/req.txt
    echo "=== filtered requirements ==="
    cat /tmp/req.txt
    rm -rf "$V/pydeps311"
    pip install --no-cache-dir --target "$V/pydeps311" -r /tmp/req.txt
    mkdir -p "$V/spriteengine"
    echo "$HANDLER_B64" | base64 -d > "$V/spriteengine/handler.py"
    PYTHONPATH="$V/pydeps311:$V/hunyuan3d21/hy3dshape" python3 - <<'PY'
import runpod
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
print("imports ok:", Hunyuan3DDiTFlowMatchingPipeline.__name__)
PY
    touch "$V/.spriteengine-bootstrap-complete"
    du -sh "$V/pydeps311" "$V/hunyuan3d21"
} >> progress.log 2>&1 && echo BOOTSTRAP_OK >> progress.log || echo BOOTSTRAP_FAIL >> progress.log
sleep 3600
""".replace("__REQ_FILTER__", REQ_FILTER)

# Worker start: deps/code/weights all live on the volume; the base image only
# provides python 3.11 + CUDA userland. libgl is needed by opencv at import.
WORKER_CMD = (
    "apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null 2>&1 || true; "
    "exec python -u /runpod-volume/spriteengine/handler.py"
)
WORKER_ENV = {
    "PYTHONPATH": "/runpod-volume/pydeps311:/runpod-volume/hunyuan3d21/hy3dshape",
    "HF_HOME": "/runpod-volume/huggingface",
    "HUGGINGFACE_HUB_CACHE": "/runpod-volume/huggingface/hub",
    "HUNYUAN_MODEL": "tencent/Hunyuan3D-2.1",
}


def find_named(key, path, name):
    return next((x for x in items(request(key, "GET", path)) if x.get("name") == name), None)


def launch_bootstrap_pod(key, volume_id):
    handler = Path(__file__).with_name("handler.py").read_bytes()
    stale = find_named(key, "/pods", BOOTSTRAP_POD_NAME)
    if stale:
        print(f"terminating stale bootstrap pod {stale['id']}")
        request(key, "DELETE", "/pods/" + stale["id"])
    pod = create_pod(key, {
        "name": BOOTSTRAP_POD_NAME,
        "imageName": "python:3.11-slim",
        "computeType": "CPU",
        "cloudType": "SECURE",
        "vcpuCount": 8,
        "containerDiskInGb": 15,
        "networkVolumeId": volume_id,
        "volumeMountPath": "/runpod-volume",
        "ports": ["8888/http"],
        "env": {"HANDLER_B64": base64.b64encode(handler).decode("ascii")},
        "dockerEntrypoint": [],
        "dockerStartCmd": ["bash", "-c", BOOTSTRAP_SCRIPT],
    })
    print(f"bootstrap pod started: {pod['id']}")
    return pod["id"]


def wait_for_bootstrap(key, pod_id, timeout_minutes):
    deadline = time.time() + timeout_minutes * 60
    printed = 0
    while time.time() < deadline:
        progress = fetch_progress(pod_id)
        if progress is not None:
            if len(progress) > printed:
                sys.stdout.write(progress[printed:])
                sys.stdout.flush()
                printed = len(progress)
            if "BOOTSTRAP_OK" in progress:
                return
            if "BOOTSTRAP_FAIL" in progress:
                raise RuntimeError("bootstrap failed inside the pod; see log above")
        else:
            try:
                pod = request(key, "GET", "/pods/" + pod_id)
                print(f"[{time.strftime('%H:%M:%S')}] pod {pod.get('desiredStatus')}, progress endpoint not reachable yet", flush=True)
            except RuntimeError as exc:
                print(f"[{time.strftime('%H:%M:%S')}] poll error, retrying: {exc}", flush=True)
        time.sleep(20)
    raise TimeoutError(f"bootstrap pod {pod_id} did not finish within {timeout_minutes} minutes")


def ensure_endpoint(key, volume):
    template = find_named(key, "/templates", TEMPLATE_NAME)
    if not template:
        template = request(key, "POST", "/templates", {
            "name": TEMPLATE_NAME,
            "imageName": WORKER_IMAGE,
            "isServerless": True,
            "containerDiskInGb": 30,
            "volumeInGb": 0,
            "env": WORKER_ENV,
            "dockerEntrypoint": [],
            "dockerStartCmd": ["bash", "-c", WORKER_CMD],
            "ports": [],
        })
        print(f"template created: {template['id']}")
    else:
        print(f"template reuse: {template['id']}")

    endpoint = find_named(key, "/endpoints", ENDPOINT_NAME)
    if not endpoint:
        endpoint = request(key, "POST", "/endpoints", {
            "name": ENDPOINT_NAME,
            "templateId": template["id"],
            "networkVolumeId": volume["id"],
            "dataCenterIds": [volume.get("dataCenterId") or "US-CA-2"],
            "computeType": "GPU",
            "gpuTypeIds": GPU_TYPES,
            "gpuCount": 1,
            "workersMin": 0,
            "workersMax": 1,
            "idleTimeout": 60,
            "scalerType": "QUEUE_DELAY",
            "scalerValue": 4,
            "executionTimeoutMs": 900000,
        })
        print(f"endpoint created: {endpoint['id']}")
    else:
        print(f"endpoint reuse: {endpoint['id']}")
    return template, endpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timeout-minutes", type=int, default=60)
    p.add_argument("--skip-bootstrap", action="store_true")
    p.add_argument("--pod-id", default="", help="Attach to an already-running bootstrap pod")
    p.add_argument("--keep-pod", action="store_true")
    args = p.parse_args()
    key = os.getenv("RUNPOD_API_KEY", "").strip()
    if not key:
        sys.exit("RUNPOD_API_KEY is required")

    volume = find_named(key, "/networkvolumes", VOLUME_NAME)
    if not volume:
        sys.exit(f"network volume {VOLUME_NAME!r} not found — run warm_volume.py first")
    print(f"volume: {volume['id']} ({volume.get('dataCenterId')})")

    if not args.skip_bootstrap:
        pod_id = args.pod_id or launch_bootstrap_pod(key, volume["id"])
        try:
            wait_for_bootstrap(key, pod_id, args.timeout_minutes)
        finally:
            if not args.keep_pod:
                try:
                    request(key, "DELETE", "/pods/" + pod_id)
                    print(f"\nbootstrap pod terminated: {pod_id}")
                except RuntimeError as exc:
                    print(f"\npod cleanup failed (terminate manually): {exc}")

    template, endpoint = ensure_endpoint(key, volume)
    print(json.dumps({
        "endpointId": endpoint["id"],
        "templateId": template["id"],
        "volumeId": volume["id"],
        "image": WORKER_IMAGE,
    }, indent=2))


if __name__ == "__main__":
    main()
