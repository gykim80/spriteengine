#!/usr/bin/env python3
"""RunPod Serverless handler for Hunyuan3D-2.1 image-to-GLB generation.

The model cache lives on the attached network volume via HF_HOME. Responses use
RunPod's object storage when available; base64 is retained as a portable fallback.
"""
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import runpod

_pipeline = None
_paint = None


def pipeline():
    global _pipeline
    if _pipeline is None:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            os.getenv("HUNYUAN_MODEL", "tencent/Hunyuan3D-2.1")
        )
    return _pipeline


def paint_pipeline(max_num_view: int, resolution: int):
    """hy3dpaint 멀티뷰 PBR 텍스처 파이프라인 (지연 로딩 싱글턴).

    설정(max_num_view/resolution)이 달라져도 첫 로딩 값이 유지된다 —
    serverless 워커는 요청마다 재기동될 수 있고 모델 로딩이 지배적이므로
    단일 캐시로 충분하다.
    """
    global _paint
    if _paint is None:
        # hy3dpaint 모듈은 저장소의 hy3dpaint/ 디렉터리를 sys.path 루트로 가정한다.
        paint_dir = Path(os.getenv("HUNYUAN_PAINT_ROOT", "/runpod-volume/hunyuan3d21")) / "hy3dpaint"
        if str(paint_dir) not in sys.path:
            sys.path.insert(0, str(paint_dir))
        from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
        config = Hunyuan3DPaintConfig(max_num_view, resolution)
        # 기본 설정은 repo 루트 cwd 기준 상대경로라 serverless에서 깨진다 — 절대경로로 교정.
        config.realesrgan_ckpt_path = str(paint_dir / "ckpt" / "RealESRGAN_x4plus.pth")
        config.multiview_cfg_path = str(paint_dir / "cfgs" / "hunyuan-paint-pbr.yaml")
        config.custom_pipeline = str(paint_dir / "hunyuanpaintpbr")
        _paint = Hunyuan3DPaintPipeline(config)
    return _paint


def decode_image(value: str, target: Path) -> None:
    if not value:
        raise ValueError("input.image (base64 or data URL) is required")
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    target.write_bytes(base64.b64decode(value, validate=True))
    # Fail fast with a clear message: hy3dshape otherwise crashes deep inside
    # its preprocessor (cv2.imread returns None) on undecodable image bytes.
    from PIL import Image
    try:
        with Image.open(target) as probe:
            probe.verify()
    except Exception as exc:
        raise ValueError(f"input.image is not a decodable image: {exc}") from exc


# RunPod drops the whole `output` of a COMPLETED job when it exceeds the
# serverless response payload limit (~20MB), so the mesh must be reduced to a
# game-ready face count before base64 encoding.
MAX_ENCODED_BYTES = 18 * 1024 * 1024


def motion_handler(payload):
    """HY-Motion-1.0 텍스트→모션 생성. motion_worker.py 서브프로세스로 실행해
    의존성(transformers 4.53/Qwen3)과 VRAM(~18GB)을 이 프로세스로부터 격리한다."""
    prompts = payload.get("prompts") or []
    if not prompts:
        return {"error": "input.prompts is required for task=motion"}
    if len(prompts) > 20:
        return {"error": "too many prompts (max 20 per job)"}
    volume = Path(os.getenv("RUNPOD_VOLUME", "/runpod-volume"))
    env = dict(os.environ)
    # 오버레이 우선 순서: 모션 전용 deps → 공용 torch 스택 → HY-Motion 저장소
    env["PYTHONPATH"] = os.pathsep.join([
        str(volume / "pydeps311_motion"),
        str(volume / "pydeps311"),
        str(volume / "hymotion10"),
    ])
    env["HYMOTION_ROOT"] = str(volume / "hymotion10")
    env["USE_HF_MODELS"] = "1"  # Qwen3-8B/CLIP을 볼륨 HF 캐시에서 로드
    request = json.dumps({
        "prompts": prompts,
        "seed": int(payload.get("seed", 42)),
        "cfg_scale": float(payload.get("cfg_scale", 5.0)),
    })
    proc = subprocess.run(
        [sys.executable, str(volume / "spriteengine" / "motion_worker.py")],
        input=request, capture_output=True, text=True, env=env, timeout=1500,
    )
    if proc.returncode != 0:
        return {"error": "motion worker failed", "stderr": proc.stderr[-4000:]}
    # 워커 stdout에는 모델 로그가 섞인다 — 마지막 JSON 줄만 파싱한다.
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            result = json.loads(line)
            result["model"] = "HY-Motion-1.0-Lite"
            return result
    return {"error": "motion worker produced no JSON output", "stderr": proc.stderr[-4000:]}


def handler(job):
    payload = job.get("input") or {}
    if payload.get("task") == "motion":
        return motion_handler(payload)
    seed = int(payload.get("seed", 1234))
    steps = max(1, min(int(payload.get("steps", 30)), 100))
    guidance = float(payload.get("guidance_scale", 5.0))
    octree = max(64, min(int(payload.get("octree_resolution", 256)), 512))
    faces = max(1000, min(int(payload.get("face_count", 40000)), 500000))
    # opt-in 텍스처 단계: 뒷면 포함 멀티뷰 PBR 텍스처를 hy3dpaint로 생성한다.
    # 텍스처 GLB는 크기가 커지므로 기본 face_count/해상도는 응답 한도 안쪽으로 잡는다.
    texture = bool(payload.get("texture", False))
    max_num_view = max(4, min(int(payload.get("max_num_view", 6)), 9))
    tex_resolution = max(256, min(int(payload.get("texture_resolution", 512)), 1024))
    with tempfile.TemporaryDirectory(prefix="spriteengine-") as tmp:
        root = Path(tmp)
        image = root / "input.png"
        output = root / "character.glb"
        decode_image(payload.get("image", ""), image)
        meshes = pipeline()(
            image=str(image),
            seed=seed,
            num_inference_steps=steps,
            guidance_scale=guidance,
            octree_resolution=octree,
        )
        from hy3dshape.postprocessors import (
            DegenerateFaceRemover,
            FaceReducer,
            FloaterRemover,
        )
        mesh = meshes[0]
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh, max_facenum=faces)
        mesh.export(str(output))
        textured = False
        texture_error = ""
        if texture:
            # 실패 시 shape-only GLB로 graceful fallback — 텍스처는 부가 단계이므로
            # 전체 job을 실패시키지 않는다.
            try:
                # hy3dpaint는 output_mesh_path(.obj)에 OBJ를 쓰고, save_glb=True면
                # 같은 이름의 .glb를 옆에 생성한다 (반환값은 OBJ 경로).
                painted_obj = root / "character_textured.obj"
                painted = root / "character_textured.glb"
                paint_pipeline(max_num_view, tex_resolution)(
                    mesh_path=str(output),
                    image_path=str(image),
                    output_mesh_path=str(painted_obj),
                    save_glb=True,
                )
                if painted.exists() and painted.stat().st_size > 0:
                    output = painted
                    textured = True
                else:
                    texture_error = "paint pipeline produced no GLB output"
            except Exception as exc:  # noqa: BLE001 — GPU/모델 오류 종류가 다양함
                texture_error = f"{type(exc).__name__}: {exc}"
        data = output.read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        if len(encoded) > MAX_ENCODED_BYTES:
            return {
                "error": (
                    f"GLB is {len(data)} bytes; exceeds the serverless response"
                    " limit. Lower face_count/octree_resolution"
                    + ("/texture_resolution." if texture else ".")
                ),
                "bytes": len(data),
            }
        response = {
            "model": "Hunyuan3D-2.1",
            "format": "glb",
            "seed": seed,
            "faces": faces,
            "octree_resolution": octree,
            "textured": textured,
            "glb_base64": encoded,
            "bytes": len(data),
        }
        if texture_error:
            response["texture_error"] = texture_error
        return response


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
