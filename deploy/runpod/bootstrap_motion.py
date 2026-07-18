#!/usr/bin/env python3
"""Provision HY-Motion-1.0 (text→motion) onto the existing network volume.

기존 Hunyuan3D 엔드포인트(handler.py)가 task=="motion" 요청을 처리할 수 있도록
볼륨에 다음을 준비한다 (bootstrap_endpoint.py와 같은 CPU pod + progress.log 패턴):

  1. HY-Motion-1.0 저장소 → $V/hymotion10 (git-lfs로 stats/*.npy 실파일 확보)
  2. pydeps311_motion 오버레이 — transformers 4.53(Qwen3 지원)은 Hunyuan 스택과
     충돌하므로 별도 디렉터리에 설치하고 motion 서브프로세스의 PYTHONPATH에서만
     pydeps311보다 앞에 둔다. torch 2.5.1 등 공용 스택은 pydeps311을 그대로 공유.
  3. 가중치 프리워밍 — HY-Motion-1.0-Lite(repo 상대 ckpts 경로), Qwen3-8B·CLIP
     텍스트 인코더(HF 캐시, USE_HF_MODELS=1 로드 경로)
  4. 갱신된 handler.py + motion_worker.py를 $V/spriteengine/에 배치
  5. CPU에서 가능한 임포트 검증 (모델 로드는 GPU 워커의 첫 job에서 수행)

서버리스 템플릿/엔드포인트는 변경이 불필요하다 — 워커는 볼륨의 handler.py를
실행하므로 파일 교체만으로 task=motion 라우팅이 활성화된다.

Usage:
  RUNPOD_API_KEY=... python3 deploy/runpod/bootstrap_motion.py
  ... --pod-id <id>   # re-attach to a running bootstrap pod
"""
import argparse
import base64
import os
import sys
import time
from pathlib import Path

from warm_volume import VOLUME_NAME, create_pod, fetch_progress, items, request

WORKER_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
BOOTSTRAP_POD_NAME = "spriteengine-motion-bootstrap"

# 주의: 본문을 파일로 저장해 자식 bash로 실행해야 set -e(errexit)가 온전히
# 동작한다 (bootstrap_endpoint.py와 동일한 이유).
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
    apt-get install -y -qq git git-lfs curl
    git lfs install --skip-repo
    # ---- 1. HY-Motion 저장소 (stats/*.npy는 LFS — 포인터가 아닌 실파일 필요) ----
    if [ ! -e "$V/hymotion10/hymotion/utils/t2m_runtime.py" ]; then
        rm -rf "$V/hymotion10"
        git clone --depth 1 https://github.com/Tencent-Hunyuan/HY-Motion-1.0.git "$V/hymotion10"
    fi
    ( cd "$V/hymotion10" && git lfs pull || true )
    # ---- 2. pydeps311_motion 오버레이 ----
    # transformers 4.53.3(Qwen3)은 Hunyuan 스택(pydeps311)과 버전 충돌 →
    # 별도 target에 설치. 무거운 공통 의존성(torch/numpy/scipy/HF hub 등)은
    # pydeps311에 이미 있으므로 --no-deps로 연쇄 설치를 차단한다.
    if PYTHONPATH="$V/pydeps311_motion:$V/pydeps311" python3 -c "import transformers; assert transformers.__version__.startswith('4.53'); import torchdiffeq, transforms3d, openai" 2>/dev/null; then
        echo "pydeps311_motion already importable; skipping pip install"
    else
        rm -rf "$V/pydeps311_motion"
        pip install --no-cache-dir --target "$V/pydeps311_motion" --no-deps \
            transformers==4.53.3 "tokenizers>=0.21,<0.22" torchdiffeq==0.2.5 transforms3d==0.4.2
        # prompt_rewrite.py가 모듈 최상단에서 openai를 import한다(리라이터를 꺼도
        # import 자체는 필요) — httpx/jiter 등 소의존성 포함 설치.
        pip install --no-cache-dir --target "$V/pydeps311_motion" openai==1.78.1
    fi
    # ---- 3. 가중치 프리워밍 (콜드 스타트에서 수십 GB 다운로드 방지) ----
    HF_HOME="$V/huggingface" HUGGINGFACE_HUB_CACHE="$V/huggingface/hub" PYTHONPATH="$V/pydeps311" python3 - <<'PY'
from huggingface_hub import snapshot_download
# Lite 체크포인트는 T2MRuntime이 repo 상대경로 ckpts/tencent/HY-Motion-1.0-Lite로 연다
snapshot_download("tencent/HY-Motion-1.0", allow_patterns=["HY-Motion-1.0-Lite/*"],
                  local_dir="/runpod-volume/hymotion10/ckpts/tencent")
print("HY-Motion-1.0-Lite checkpoint ready")
# 텍스트 인코더 2종은 USE_HF_MODELS=1 경로로 HF 캐시에서 로드된다
snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("Qwen3-8B ready")
snapshot_download("openai/clip-vit-large-patch14",
                  allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin"])
print("CLIP-L ready")
PY
    # ---- 4. 워커 코드 배치 ----
    mkdir -p "$V/spriteengine"
    echo "$HANDLER_B64" | base64 -d > "$V/spriteengine/handler.py"
    echo "$MOTION_WORKER_B64" | base64 -d > "$V/spriteengine/motion_worker.py"
    # ---- 5. CPU에서 가능한 검증 (모델 로드 제외) ----
    PYTHONPATH="$V/pydeps311_motion:$V/pydeps311:$V/hymotion10" python3 - <<'PY'
from pathlib import Path
import transformers
assert transformers.__version__.startswith("4.53"), transformers.__version__
import torchdiffeq, transforms3d, openai  # noqa: F401
from hymotion.utils.t2m_runtime import T2MRuntime  # noqa: F401
from hymotion.pipeline.body_model import construct_smpl_data_dict  # noqa: F401
cfg = Path("/runpod-volume/hymotion10/ckpts/tencent/HY-Motion-1.0-Lite/config.yml")
assert cfg.is_file(), f"missing {cfg}"
# latest.ckpt가 없으면 T2MRuntime이 경고만 찍고 랜덤 가중치로 생성한다(치명적) —
# 실파일(수 GB)인지까지 확인한다.
ckpt = cfg.with_name("latest.ckpt")
assert ckpt.is_file() and ckpt.stat().st_size > 10**9, f"missing/short {ckpt}"
# stats/*.npy 등 LFS 파일이 포인터 텍스트로 남아 있지 않은지 확인
for p in Path("/runpod-volume/hymotion10").rglob("*.npy"):
    with open(p, "rb") as f:
        assert f.read(7) != b"version", f"LFS pointer not fetched: {p}"
print("motion imports ok; transformers", transformers.__version__)
PY
    touch "$V/.spriteengine-motion-bootstrap-complete"
    du -sh "$V/pydeps311_motion" "$V/hymotion10" "$V/huggingface"
WORK
if bash /srv/bootstrap_work.sh >> progress.log 2>&1; then
    echo BOOTSTRAP_OK >> progress.log
else
    echo BOOTSTRAP_FAIL >> progress.log
fi
sleep 3600
"""


def find_named(key, path, name):
    return next((x for x in items(request(key, "GET", path)) if x.get("name") == name), None)


def launch_bootstrap_pod(key, volume_id):
    here = Path(__file__).parent
    stale = find_named(key, "/pods", BOOTSTRAP_POD_NAME)
    if stale:
        print(f"terminating stale bootstrap pod {stale['id']}")
        request(key, "DELETE", "/pods/" + stale["id"])
    pod = create_pod(key, {
        "name": BOOTSTRAP_POD_NAME,
        "imageName": WORKER_IMAGE,
        "computeType": "CPU",
        "cloudType": "SECURE",
        "vcpuCount": 8,
        "containerDiskInGb": 40,
        "networkVolumeId": volume_id,
        "volumeMountPath": "/runpod-volume",
        "ports": ["8888/http"],
        "env": {
            "HANDLER_B64": base64.b64encode((here / "handler.py").read_bytes()).decode("ascii"),
            "MOTION_WORKER_B64": base64.b64encode((here / "motion_worker.py").read_bytes()).decode("ascii"),
        },
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timeout-minutes", type=int, default=90)
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

    pod_id = args.pod_id or launch_bootstrap_pod(key, volume["id"])
    try:
        wait_for_bootstrap(key, pod_id, args.timeout_minutes)
        print("\nmotion bootstrap complete — handler.py/motion_worker.py deployed to the volume")
    finally:
        if not args.keep_pod:
            try:
                request(key, "DELETE", "/pods/" + pod_id)
                print(f"bootstrap pod terminated: {pod_id}")
            except RuntimeError as exc:
                print(f"pod cleanup failed (terminate manually): {exc}")


if __name__ == "__main__":
    main()
