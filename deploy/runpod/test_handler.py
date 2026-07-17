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
spec = importlib.util.spec_from_file_location("spriteengine_handler", Path(__file__).with_name("handler.py"))
handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handler)


class FakeMesh:
    def export(self, path):
        # GLB magic plus enough payload to exercise encoding.
        Path(path).write_bytes(b"glTF" + b"\x00" * 20)


class HandlerTests(unittest.TestCase):
    def test_requires_image(self):
        with self.assertRaisesRegex(ValueError, "input.image"):
            handler.handler({"input": {}})

    def test_generation_contract(self):
        image = base64.b64encode(b"not-a-real-png-but-valid-input-bytes").decode()
        fake_pipeline = mock.Mock(return_value=[FakeMesh()])
        with mock.patch.object(handler, "pipeline", return_value=fake_pipeline):
            output = handler.handler({"input": {"image": image, "seed": 7, "steps": 999}})
        self.assertEqual(output["format"], "glb")
        self.assertEqual(output["seed"], 7)
        self.assertEqual(base64.b64decode(output["glb_base64"])[:4], b"glTF")
        self.assertEqual(fake_pipeline.call_args.kwargs["num_inference_steps"], 100)

    def test_data_url(self):
        target = Path(tempfile.mkdtemp()) / "input.png"
        handler.decode_image("data:image/png;base64," + base64.b64encode(b"png").decode(), target)
        self.assertEqual(target.read_bytes(), b"png")


if __name__ == "__main__":
    unittest.main()
