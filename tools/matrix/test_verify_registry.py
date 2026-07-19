#!/usr/bin/env python3
"""verify_registry 감사 로직 회귀 테스트 (합성 GLB로 네트워크·앱 데이터 불필요)."""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_registry as vr  # noqa: E402


def make_glb(gltf: dict) -> bytes:
    """최소 유효 GLB 바이너리 생성 (JSON 청크만)."""
    payload = json.dumps(gltf).encode()
    payload += b" " * (-len(payload) % 4)  # 4바이트 정렬
    header = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(payload))
    chunk = struct.pack("<I4s", len(payload), b"JSON") + payload
    return header + chunk


def neutral_gltf(clips=10):
    return {
        "materials": [{"pbrMetallicRoughness": {"metallicFactor": 0.0,
                                                "roughnessFactor": 1.0}}],
        "animations": [{"name": f"clip{i}"} for i in range(clips)],
    }


class GlbJsonTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".glb") as f:
            f.write(make_glb(neutral_gltf()))
            f.flush()
            g = vr.glb_json(f.name)
        self.assertEqual(len(g["animations"]), 10)


class CheckExportTests(unittest.TestCase):
    def _run(self, gltf):
        ws = Path(tempfile.mkdtemp())
        (ws / "export").mkdir()
        (ws / "export" / "character-final.glb").write_bytes(make_glb(gltf))
        return vr.check_export(ws)

    def test_neutralized_material_passes(self):
        self.assertEqual(self._run(neutral_gltf()), [])

    def test_metallic_material_fails(self):
        """실측 2026-07-20: 플라스틱 광택의 원인이던 metallic 기본값 1.0 검출."""
        g = neutral_gltf()
        g["materials"][0]["pbrMetallicRoughness"]["metallicFactor"] = 1.0
        self.assertIn("metallicFactor!=0", self._run(g))

    def test_specular_extension_fails(self):
        g = neutral_gltf()
        g["materials"][0]["extensions"] = {"KHR_materials_specular": {}}
        self.assertIn("KHR_materials_specular present", self._run(g))

    def test_mr_texture_fails(self):
        g = neutral_gltf()
        g["materials"][0]["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": 0}
        self.assertIn("metallicRoughnessTexture present", self._run(g))

    def test_too_few_clips_fails(self):
        issues = self._run(neutral_gltf(clips=3))
        self.assertTrue(any(i.startswith("clips=3") for i in issues))

    def test_missing_glb_fails(self):
        ws = Path(tempfile.mkdtemp())
        (ws / "export").mkdir()
        self.assertEqual(vr.check_export(ws), ["no character-final.glb"])


class CheckStructureTests(unittest.TestCase):
    def _ws(self):
        ws = Path(tempfile.mkdtemp())
        (ws / "source.png").write_bytes(b"\x89PNG")
        for d in vr.STAGE_DIRS:
            (ws / d).mkdir()
        return ws

    def _job(self):
        return {"status": "complete", "progress": 100,
                "stages": [{"id": "rig", "status": "done"}]}

    def test_sound_job_passes(self):
        self.assertEqual(vr.check_structure(self._job(), self._ws()), [])

    def test_incomplete_status_fails(self):
        job = self._job()
        job["status"] = "processing"
        issues = vr.check_structure(job, self._ws())
        self.assertTrue(any("status=processing" in i for i in issues))

    def test_pending_stage_fails(self):
        job = self._job()
        job["stages"][0]["status"] = "pending"
        issues = vr.check_structure(job, self._ws())
        self.assertIn("stage rig=pending", issues)

    def test_missing_stage_dir_fails(self):
        ws = self._ws()
        (ws / "export").rmdir()
        self.assertIn("missing export", vr.check_structure(self._job(), ws))


if __name__ == "__main__":
    unittest.main()
