#!/usr/bin/env python3
"""RunPod Serverless handler for Hunyuan3D-2.1 image-to-GLB generation.

The model cache lives on the attached network volume via HF_HOME. Responses use
RunPod's object storage when available; base64 is retained as a portable fallback.
"""
import base64
import os
import tempfile
from pathlib import Path

import runpod

_pipeline = None


def pipeline():
    global _pipeline
    if _pipeline is None:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            os.getenv("HUNYUAN_MODEL", "tencent/Hunyuan3D-2.1")
        )
    return _pipeline


def decode_image(value: str, target: Path) -> None:
    if not value:
        raise ValueError("input.image (base64 or data URL) is required")
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    target.write_bytes(base64.b64decode(value, validate=True))


# RunPod drops the whole `output` of a COMPLETED job when it exceeds the
# serverless response payload limit (~20MB), so the mesh must be reduced to a
# game-ready face count before base64 encoding.
MAX_ENCODED_BYTES = 18 * 1024 * 1024


def handler(job):
    payload = job.get("input") or {}
    seed = int(payload.get("seed", 1234))
    steps = max(1, min(int(payload.get("steps", 30)), 100))
    guidance = float(payload.get("guidance_scale", 5.0))
    octree = max(64, min(int(payload.get("octree_resolution", 256)), 512))
    faces = max(1000, min(int(payload.get("face_count", 40000)), 500000))
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
        data = output.read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        if len(encoded) > MAX_ENCODED_BYTES:
            return {
                "error": (
                    f"GLB is {len(data)} bytes; exceeds the serverless response"
                    " limit. Lower face_count/octree_resolution."
                ),
                "bytes": len(data),
            }
        return {
            "model": "Hunyuan3D-2.1",
            "format": "glb",
            "seed": seed,
            "faces": faces,
            "octree_resolution": octree,
            "glb_base64": encoded,
            "bytes": len(data),
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
