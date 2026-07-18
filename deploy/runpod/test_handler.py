#!/usr/bin/env python3
"""Unit tests for the RunPod handler that do not require CUDA/model downloads."""
import base64
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

# The production module imports runpod at module load time. Supply a tiny test double.
runpod = types.ModuleType("runpod")
runpod.serverless = types.SimpleNamespace(start=lambda _: None)
sys.modules.setdefault("runpod", runpod)


class _PassthroughPostprocessor:
    """Test double for hy3dshape postprocessors: callable that returns the mesh."""

    def __call__(self, mesh, **kwargs):
        return mesh


# handler() imports hy3dshape.postprocessors lazily; stub it so tests run without CUDA deps.
hy3dshape = types.ModuleType("hy3dshape")
postprocessors = types.ModuleType("hy3dshape.postprocessors")
postprocessors.FloaterRemover = lambda: _PassthroughPostprocessor()
postprocessors.DegenerateFaceRemover = lambda: _PassthroughPostprocessor()
postprocessors.FaceReducer = lambda: _PassthroughPostprocessor()
hy3dshape.postprocessors = postprocessors
sys.modules.setdefault("hy3dshape", hy3dshape)
sys.modules.setdefault("hy3dshape.postprocessors", postprocessors)
spec = importlib.util.spec_from_file_location("spriteengine_handler", Path(__file__).with_name("handler.py"))
handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handler)


class FakeMesh:
    def export(self, path):
        # GLB magic plus enough payload to exercise encoding.
        Path(path).write_bytes(b"glTF" + b"\x00" * 20)


def tiny_png_bytes() -> bytes:
    """A real 1x1 PNG so the handler's decodability check passes."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGBA", (1, 1), (255, 0, 0, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


class HandlerTests(unittest.TestCase):
    def test_requires_image(self):
        with self.assertRaisesRegex(ValueError, "input.image"):
            handler.handler({"input": {}})

    def test_rejects_undecodable_image(self):
        image = base64.b64encode(b"not-a-real-png-but-valid-input-bytes").decode()
        with self.assertRaisesRegex(ValueError, "not a decodable image"):
            handler.handler({"input": {"image": image}})

    def test_generation_contract(self):
        image = base64.b64encode(tiny_png_bytes()).decode()
        fake_pipeline = mock.Mock(return_value=[FakeMesh()])
        with mock.patch.object(handler, "pipeline", return_value=fake_pipeline):
            output = handler.handler({"input": {"image": image, "seed": 7, "steps": 999}})
        self.assertEqual(output["format"], "glb")
        self.assertEqual(output["seed"], 7)
        self.assertEqual(base64.b64decode(output["glb_base64"])[:4], b"glTF")
        self.assertEqual(fake_pipeline.call_args.kwargs["num_inference_steps"], 100)

    def test_texture_stage_success(self):
        """texture=True면 paint 파이프라인이 만든 GLB가 응답에 실린다."""
        image = base64.b64encode(tiny_png_bytes()).decode()
        fake_pipeline = mock.Mock(return_value=[FakeMesh()])

        def fake_painter(mesh_path, image_path, output_mesh_path, save_glb):
            # 실제 hy3dpaint 계약: .obj 경로를 받아 .glb를 옆에 생성하고 obj 경로 반환
            self.assertTrue(output_mesh_path.endswith(".obj"))
            self.assertTrue(save_glb)
            glb = Path(output_mesh_path.replace(".obj", ".glb"))
            glb.write_bytes(b"glTF" + b"\x01" * 32)
            return output_mesh_path

        with mock.patch.object(handler, "pipeline", return_value=fake_pipeline), \
             mock.patch.object(handler, "paint_pipeline", return_value=fake_painter) as paint:
            output = handler.handler({"input": {
                "image": image, "texture": True, "max_num_view": 8, "texture_resolution": 768,
            }})
        paint.assert_called_once_with(8, 768)
        self.assertTrue(output["textured"])
        self.assertNotIn("texture_error", output)
        self.assertEqual(base64.b64decode(output["glb_base64"]), b"glTF" + b"\x01" * 32)

    def test_texture_stage_falls_back_to_shape_glb(self):
        """paint 실패 시 job 전체를 실패시키지 않고 shape GLB로 폴백한다."""
        image = base64.b64encode(tiny_png_bytes()).decode()
        fake_pipeline = mock.Mock(return_value=[FakeMesh()])

        def broken_painter(**kwargs):
            raise RuntimeError("CUDA out of memory")

        with mock.patch.object(handler, "pipeline", return_value=fake_pipeline), \
             mock.patch.object(handler, "paint_pipeline", return_value=broken_painter):
            output = handler.handler({"input": {"image": image, "texture": True}})
        self.assertFalse(output["textured"])
        self.assertIn("CUDA out of memory", output["texture_error"])
        self.assertEqual(base64.b64decode(output["glb_base64"])[:4], b"glTF")

    def test_texture_disabled_by_default(self):
        image = base64.b64encode(tiny_png_bytes()).decode()
        fake_pipeline = mock.Mock(return_value=[FakeMesh()])
        with mock.patch.object(handler, "pipeline", return_value=fake_pipeline), \
             mock.patch.object(handler, "paint_pipeline") as paint:
            output = handler.handler({"input": {"image": image}})
        paint.assert_not_called()
        self.assertFalse(output["textured"])
        self.assertNotIn("texture_error", output)

    def test_data_url(self):
        target = Path(tempfile.mkdtemp()) / "input.png"
        png = tiny_png_bytes()
        handler.decode_image("data:image/png;base64," + base64.b64encode(png).decode(), target)
        self.assertEqual(target.read_bytes(), png)


if __name__ == "__main__":
    unittest.main()
