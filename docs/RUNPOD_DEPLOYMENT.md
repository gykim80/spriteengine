# RunPod deployment

Production worker는 **Hunyuan3D-2.1 shape pipeline**으로 입력 이미지를 실제 GLB mesh로 생성한다. RunPod network volume은 `/runpod-volume`에 mount되며 Hugging Face model cache를 worker 재시작 간 유지한다.

## Architecture

- Image: `deploy/runpod/Dockerfile`
- Handler: `deploy/runpod/handler.py`
- Provisioner: `deploy/runpod/provision.py`
- GPU preference: A100 80GB → H100 80GB → L40S
- Scale: min 0 / max 1, 5초 idle timeout
- Persistent volume: 100GB, model weights 전용
- Request timeout: 15분

## Deployment prerequisites

1. Repository를 GitHub에 push하면 workflow가 GHCR image를 build/push한다.
2. RunPod API key를 `RUNPOD_API_KEY` environment variable로 제공한다.
3. `provision.py --image ghcr.io/<owner>/spriteengine-hunyuan3d21:latest`를 실행한다.
4. 출력된 endpoint ID와 API key를 Studio Settings에 저장한다.

Provisioning은 동일 이름 resource를 재사용하는 idempotent 방식이다. Network volume은 지정 region에 endpoint보다 먼저 만들어져야 한다. API key와 generated deployment manifest는 commit하지 않는다.

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
