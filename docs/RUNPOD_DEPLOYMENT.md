# RunPod deployment

Production worker는 **Hunyuan3D-2.1 shape pipeline**으로 입력 이미지를 실제 GLB mesh로 생성하고, `texture: true` 요청 시 **hy3dpaint paint pipeline**으로 멀티뷰 PBR 텍스처(뒷면 포함)까지 입힌다. RunPod network volume은 `/runpod-volume`에 mount되며 Hugging Face model cache를 worker 재시작 간 유지한다.

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
{"input":{"image":"<base64 또는 data URL>","seed":1234,"steps":30,"guidance_scale":5.0,
  "texture":true,"max_num_view":6,"texture_resolution":512}}
```

- `texture` (기본 `false`): hy3dpaint 멀티뷰 PBR 텍스처 생성을 opt-in 한다. 구버전 handler는 이 필드를 무시하므로 클라이언트는 하위 호환이다.
- `max_num_view` 4–9, `texture_resolution` 256–1024. 값이 클수록 품질↑, VRAM/응답 크기↑.

Output은 `format`, `model`, `bytes`, `textured`, `glb_base64`를 포함한다. Paint 단계가 실패하면 job을 실패시키지 않고 **shape-only GLB로 폴백**하며 사유를 `texture_error`에 담는다 (Studio는 artifact metrics의 `textured`/`textureError`로 노출). 이 경우 로컬 retopo 단계의 front-projection bake가 임시 텍스처를 입힌다. 텍스처가 이미 있는 GLB는 bake가 건너뛴다.

## Texture (hy3dpaint) 볼륨 요구사항

`bootstrap_endpoint.py`가 볼륨에 추가로 준비하는 것:

- `basicsr==1.4.2`/`realesrgan==0.3.0` (`--no-deps` 증분 설치 + torchvision `functional_tensor` sed 패치)
- `custom_rasterizer` CUDA 확장 — nvcc가 필요해 부트스트랩 pod이 worker와 같은 `runpod/pytorch` devel 이미지를 사용한다 (GPU 불필요, `TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9`)
- `DifferentiableRenderer/mesh_inpaint_processor*.so` (pybind11 컴파일)
- `hy3dpaint/ckpt/RealESRGAN_x4plus.pth` 체크포인트
- HF 캐시 프리워밍: `tencent/Hunyuan3D-2.1`의 `hunyuan3d-paintpbr-v2-1/*` + `facebook/dinov2-giant` — 첫 텍스처 요청이 20분 타임아웃에 걸리지 않게 한다

기존 볼륨에는 `RUNPOD_API_KEY=... python3 deploy/runpod/bootstrap_endpoint.py`를 다시 실행하면 된다 — shape 의존성은 import 검사로 건너뛰고 paint 단계만 증분으로 채워진다. 템플릿/엔드포인트 env 변경은 필요 없다 (handler가 `HUNYUAN_PAINT_ROOT` 기본값으로 sys.path를 스스로 구성).

## Cost and operational notes

Hunyuan3D-2.1 README 기준 shape generation은 약 10GB VRAM, texture는 21GB, 동시 shape+texture는 29GB를 요구한다. 24GB급 카드(4090/A5000/3090/L4)에 배정되면 paint 단계가 OOM으로 실패할 수 있으나 shape GLB 폴백으로 job 자체는 성공한다. 안정적인 텍스처 생성을 원하면 endpoint GPU를 A100/L40S/A6000(48GB+)로 제한한다. 첫 cold start에는 image pull과 model download가 발생한다. Model cache가 채워진 뒤 network volume을 통해 이후 cold start를 줄인다.
