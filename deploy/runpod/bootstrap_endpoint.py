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
# Shape generation needs ~10GB VRAM; texture(hy3dpaint)까지 켜면 ~29GB가 필요해
# 24GB급 카드에서는 paint 단계가 OOM으로 실패할 수 있다 — handler가 shape GLB로
# graceful fallback 하므로 가용성을 위해 24GB 카드도 후순위로 유지한다.
GPU_TYPES = [
    "NVIDIA A100 80GB PCIe",
    "NVIDIA L40S",
    "NVIDIA RTX A6000",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A5000",
    "NVIDIA L4",
    "NVIDIA GeForce RTX 3090",
]

# Packages that the worker never imports (demo/training extras) or that cannot
# install outside their original environment. basicsr/realesrgan은 paint 단계에
# 필요하지만 gfpgan/facexlib/numba 의존성 연쇄를 피하려고 아래에서 --no-deps로
# 별도 설치한다 (cupy는 requirements.txt에만 있고 코드에서 import되지 않음).
REQ_FILTER = "^--extra-index-url|^tb_nightly|^torchmetrics|^deepspeed|^bpy|^gradio|^fastapi|^uvicorn|^basicsr|^realesrgan|^cupy"

# 주의: 작업 본문을 `{ ... } && echo OK || echo FAIL`로 감싸면 bash가 종료코드를
# 테스트하는 컴파운드 내부에서 set -e를 무력화해 중간 실패(pip 오류 등)가 조용히
# 무시된다. 본문을 파일로 저장해 자식 bash로 실행해야 errexit이 온전히 동작한다.
BOOTSTRAP_SCRIPT = r"""
mkdir -p /srv/progress
cd /srv/progress
: > progress.log
python3 -m http.server 8888 --bind 0.0.0.0 >/dev/null 2>&1 &
cat > /srv/bootstrap_work.sh <<'WORK'
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    V=/runpod-volume
    apt-get update -qq
    apt-get install -y -qq git build-essential cmake libgl1 libglib2.0-0 python3-dev wget curl \
        libegl1 libxrender1 libxxf86vm1 libxfixes3 libxi6 libxkbcommon0 libsm6
    if [ ! -e "$V/hunyuan3d21/requirements.txt" ]; then
        rm -rf "$V/hunyuan3d21"
        git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git "$V/hunyuan3d21"
    fi
    grep -v -E "__REQ_FILTER__" "$V/hunyuan3d21/requirements.txt" > /tmp/req.txt
    printf 'torch==2.5.1\ntorchvision==0.20.1\nrunpod==1.7.13\n' >> /tmp/req.txt
    echo "=== filtered requirements ==="
    cat /tmp/req.txt
    # A previous run may have finished the (very long) pip install before being
    # interrupted; only reinstall when the installed tree is not importable.
    if PYTHONPATH="$V/pydeps311:$V/hunyuan3d21/hy3dshape" python3 -c "import runpod, hy3dshape.pipelines" 2>/dev/null; then
        echo "pydeps311 already importable; skipping pip install"
    else
        rm -rf "$V/pydeps311"
        pip install --no-cache-dir --target "$V/pydeps311" -r /tmp/req.txt
    fi
    # ---- hy3dpaint (texture) 단계 ----
    # basicsr/realesrgan은 REQ_FILTER로 걸러졌으므로 --no-deps 증분 설치
    # (gfpgan/facexlib/numba 연쇄 방지). 순수 파이썬 소의존성만 별도 추가.
    if ! PYTHONPATH="$V/pydeps311" python3 -c "import basicsr, realesrgan" 2>/dev/null; then
        pip install --no-cache-dir --upgrade --target "$V/pydeps311" --no-deps basicsr==1.4.2 realesrgan==0.3.0
        pip install --no-cache-dir --upgrade --target "$V/pydeps311" addict future lmdb yapf
    fi
    # hy3dpaint의 DifferentiableRenderer/mesh_utils.py는 최상단에서 bpy(Blender)를
    # import하고 최종 OBJ→GLB 변환도 Blender로 수행한다 — cp311 휠을 볼륨에 설치.
    # (4.2는 Python 3.11 대상의 LTS 라인; 4.1.x는 PyPI에서 제공되지 않는다)
    if ! PYTHONPATH="$V/pydeps311" python3 -c "import bpy" 2>/dev/null; then
        pip install --no-cache-dir --upgrade --target "$V/pydeps311" --no-deps bpy==4.2.22 zstandard
    fi
    # basicsr 1.4.2는 torchvision>=0.17에서 제거된 functional_tensor를 import한다.
    sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
        "$V/pydeps311/basicsr/data/degradations.py" || true
    # custom_rasterizer CUDA 확장: 볼륨의 torch 2.5.1 기준으로 빌드해 pydeps311에 설치.
    # (부트스트랩 pod은 GPU가 없어도 nvcc만 있으면 됨 — ARCH_LIST로 타깃 카드 지정)
    if ! PYTHONPATH="$V/pydeps311" python3 -c "import custom_rasterizer" 2>/dev/null; then
        ( cd "$V/hunyuan3d21/hy3dpaint/custom_rasterizer" && \
          TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" PYTHONPATH="$V/pydeps311" PATH="$V/pydeps311/bin:$PATH" \
          pip wheel --no-cache-dir --no-build-isolation --no-deps -w /tmp/wheels . )
        pip install --no-deps --target "$V/pydeps311" /tmp/wheels/custom_rasterizer*.whl
    fi
    # DifferentiableRenderer의 pybind11 인페인트 모듈(.so)을 볼륨 안에 컴파일.
    # 업스트림 compile_mesh_painter.sh는 python3-config(apt의 3.10)를 쓰므로
    # cpython-310 접미사가 붙어 3.11 워커에서 import되지 않는다 — 실행 중인
    # 인터프리터의 EXT_SUFFIX로 직접 컴파일한다.
    if ! PYTHONPATH="$V/pydeps311:$V/hunyuan3d21/hy3dpaint" python3 -c "from DifferentiableRenderer.mesh_inpaint_processor import meshVerticeInpaint" 2>/dev/null; then
        ( cd "$V/hunyuan3d21/hy3dpaint/DifferentiableRenderer" && \
          SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))") && \
          c++ -O3 -Wall -shared -std=c++11 -fPIC \
              $(PYTHONPATH="$V/pydeps311" python3 -m pybind11 --includes) \
              mesh_inpaint_processor.cpp -o "mesh_inpaint_processor$SUFFIX" )
    fi
    # RealESRGAN 체크포인트 (imageSuperNet 업스케일러)
    mkdir -p "$V/hunyuan3d21/hy3dpaint/ckpt"
    if [ ! -s "$V/hunyuan3d21/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" ]; then
        wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
            -O "$V/hunyuan3d21/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" || \
        curl -fsSL https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
            -o "$V/hunyuan3d21/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    fi
    # paint 가중치 프리워밍: 첫 텍스처 요청의 콜드 스타트에서 수 GB 다운로드로
    # 20분 타임아웃에 걸리지 않도록 볼륨 HF 캐시에 미리 받아 둔다.
    HF_HOME="$V/huggingface" HUGGINGFACE_HUB_CACHE="$V/huggingface/hub" PYTHONPATH="$V/pydeps311" python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("tencent/Hunyuan3D-2.1", allow_patterns=["hunyuan3d-paintpbr-v2-1/*"])
snapshot_download("facebook/dinov2-giant", allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("paint weights warmed")
PY
    # diffusers custom_pipeline 동적 모듈 캐시 프리시드: diffusers 0.30 로더는
    # pipeline.py의 `from .unet.modules import ...` 같은 서브패키지 상대 import를
    # 캐시(HF_HOME/modules/diffusers_modules/local)로 복사하지 못해 첫 로드가
    # ModuleNotFoundError(diffusers_modules.local.*)로 실패한다. 패키지/평탄 두
    # 레이아웃을 모두 심어 어느 import 경로로도 해결되게 한다.
    M="$V/huggingface/modules/diffusers_modules/local"
    mkdir -p "$M"
    touch "$V/huggingface/modules/diffusers_modules/__init__.py" "$M/__init__.py"
    cp "$V/hunyuan3d21/hy3dpaint/hunyuanpaintpbr/pipeline.py" "$M/"
    rm -rf "$M/unet"
    cp -r "$V/hunyuan3d21/hy3dpaint/hunyuanpaintpbr/unet" "$M/unet"
    touch "$M/unet/__init__.py"
    cp "$V/hunyuan3d21/hy3dpaint/hunyuanpaintpbr/unet/"*.py "$M/"
    mkdir -p "$V/spriteengine"
    echo "$HANDLER_B64" | base64 -d > "$V/spriteengine/handler.py"
    PYTHONPATH="$V/pydeps311:$V/hunyuan3d21/hy3dshape" python3 - <<'PY'
import sys
import runpod
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
sys.path.insert(0, "/runpod-volume/hunyuan3d21/hy3dpaint")
import basicsr
import custom_rasterizer
import realesrgan
import bpy
# MeshRender는 인페인트 모듈 부재를 try/except로 삼키므로 직접 import해 검증한다.
from DifferentiableRenderer.mesh_inpaint_processor import meshVerticeInpaint
# 전체 paint 파이프라인 모듈을 import해 누락 의존성(bpy 등)을 여기서 잡는다.
import textureGenPipeline
# 프리시드한 diffusers 동적 모듈 캐시가 실제로 import 가능한지 검증 (평탄+패키지)
sys.path.insert(0, "/runpod-volume/huggingface/modules")
import diffusers_modules.local.modules
import diffusers_modules.local.pipeline
print("imports ok:", Hunyuan3DDiTFlowMatchingPipeline.__name__, "+ paint deps + bpy", bpy.app.version_string)
PY
    touch "$V/.spriteengine-bootstrap-complete"
    du -sh "$V/pydeps311" "$V/hunyuan3d21"
WORK
if bash /srv/bootstrap_work.sh >> progress.log 2>&1; then
    echo BOOTSTRAP_OK >> progress.log
else
    echo BOOTSTRAP_FAIL >> progress.log
fi
sleep 3600
""".replace("__REQ_FILTER__", REQ_FILTER)

# Worker start: deps/code/weights all live on the volume; the base image only
# provides python 3.11 + CUDA userland. libgl은 opencv, X/EGL 계열은 bpy(Blender)
# import에 필요하다.
WORKER_CMD = (
    "apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq libgl1 libegl1 libglib2.0-0 "
    "libxrender1 libxxf86vm1 libxfixes3 libxi6 libxkbcommon0 libsm6 >/dev/null 2>&1 || true; "
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
        # custom_rasterizer CUDA 확장 컴파일에 nvcc가 필요해 worker와 같은
        # devel 이미지를 쓴다 (GPU는 불필요 — CPU pod에서 컴파일만 수행).
        "imageName": WORKER_IMAGE,
        "computeType": "CPU",
        "cloudType": "SECURE",
        "vcpuCount": 8,
        "containerDiskInGb": 40,
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
        # WORKER_CMD/WORKER_ENV가 바뀌어도 기존 템플릿에 반영되도록 동기화한다.
        request(key, "PATCH", "/templates/" + template["id"], {
            "name": TEMPLATE_NAME,
            "imageName": WORKER_IMAGE,
            "env": WORKER_ENV,
            "dockerEntrypoint": [],
            "dockerStartCmd": ["bash", "-c", WORKER_CMD],
        })
        print(f"template synced: {template['id']}")

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
