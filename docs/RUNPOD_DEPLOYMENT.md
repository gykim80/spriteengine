# RunPod deployment

Production worker는 **Hunyuan3D-2.1 shape pipeline**으로 입력 이미지를 실제 GLB mesh로 생성한다. RunPod network volume은 `/runpod-volume`에 mount되며 Hugging Face model cache를 worker 재시작 간 유지한다.

## Architecture

- Worker repo: `https://github.com/gykim80/spriteengine-hunyuan3d-worker` (RunPod GitHub 연동 빌드)
- Image: worker repo 루트 `Dockerfile` (원본: `deploy/runpod/Dockerfile`)
- Handler: `handler.py` (원본: `deploy/runpod/handler.py`)
- Volume warm-up: `deploy/runpod/warm_volume.py` — weights를 로컬이 아닌 RunPod network volume에 직접 다운로드
- GPU preference: A100 80GB → H100 80GB → L40S
- Scale: min 0 / max 1, 5초 idle timeout
- Persistent volume: `spriteengine-model-cache` 100GB, **US-CA-2** (network volume 지원 region), model weights 전용
- Request timeout: 15분

## Deployment flow (GitHub integration)

RunPod이 GitHub repo를 직접 clone하여 RunPod 인프라에서 이미지를 빌드하므로 GHCR push가 필요 없다. (기존 GHCR workflow는 제거됨 — GitHub Actions는 계정 billing lock으로 실행 불가하며, 어차피 불필요하다.)

1. Network volume 생성 + weights pre-download (로컬 다운로드 없음):
   ```sh
   RUNPOD_API_KEY=... python3 deploy/runpod/warm_volume.py
   ```
   임시 CPU pod가 `tencent/Hunyuan3D-2.1`을 `/runpod-volume/huggingface`에 받고 자동 종료된다.
2. RunPod Console → Serverless → New Endpoint → **GitHub Repo** → `spriteengine-hunyuan3d-worker` 선택
   - Branch `main`, Dockerfile path `Dockerfile`
   - GPU: A100 80GB / H100 80GB / L40S, workers 0~1, idle timeout 5s, execution timeout 900s
   - Network volume `spriteengine-model-cache` attach (mount `/runpod-volume`)
   - Datacenter는 volume region(US-CA-2)과 일치해야 한다
3. 생성된 endpoint ID와 API key를 Studio Settings에 저장한다.

`provision.py`는 registry image 기반 대체 경로(idempotent)로 유지한다. Network volume은 endpoint보다 먼저 만들어져야 하며, network volume을 지원하는 region(US-CA-2 등)만 사용할 수 있다. API key와 deployment manifest는 commit하지 않는다.

## 401 authentication 복구

`401 Invalid authentication credentials`는 worker image 문제가 아니라 RunPod API gateway가 request를 거부한 상태다.

- RunPod Console의 **Settings → API Keys**에서 새 key를 생성한다. Endpoint ID나 endpoint URL은 API key가 아니다.
- UI에 표시되는 가려진 값(`****`) 대신 생성 직후의 실제 key 전체를 Studio Settings에 붙여넣는다.
- `Bearer ` 또는 `RUNPOD_API_KEY=`까지 붙여넣어도 Studio가 정규화하지만, key 원문만 넣는 것을 권장한다.
- **Verify & save**가 성공하기 전에는 key가 저장되지 않으며, 이전 정상 credential도 덮어쓰지 않는다.
- GitHub `GITHUB_TOKEN`, GHCR token, Hugging Face token은 RunPod API key를 대신할 수 없다.

## Handler contract

Input:

```json
{"input":{"image":"<base64 또는 data URL>","seed":1234,"steps":30,"guidance_scale":5.0}}
```

Output은 `format`, `model`, `bytes`, `glb_base64`를 포함한다. 현재 desktop orchestration의 기존 stage protocol과 실제 remote generation artifact 연결은 별도 adapter에서 수행한다.

## Cost and operational notes

Hunyuan3D-2.1 README 기준 shape generation은 약 10GB VRAM, texture는 21GB, 동시 shape+texture는 29GB를 요구한다. 첫 cold start에는 image pull과 model download가 발생한다. Model cache가 채워진 뒤 network volume을 통해 이후 cold start를 줄인다. 현재 handler는 안정성을 위해 shape GLB만 생성하고 texture/rig/motion은 후속 worker stage로 분리한다.
