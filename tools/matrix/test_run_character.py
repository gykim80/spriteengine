#!/usr/bin/env python3
"""run_character 드라이버 로직 회귀 테스트 (네트워크 없이 실행)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matrix_pipeline as mp  # noqa: E402
import run_character as rc  # noqa: E402
import validate_character as vc  # noqa: E402


class MeshResolutionTests(unittest.TestCase):
    """저해상도 진흙 기각 게이트 (실측 2026-07-20 gladiator 11,483 verts)."""

    def _glb(self, count):
        import json as _json
        import struct as _struct
        import tempfile
        gltf = {"meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                "accessors": [{"count": count}]}
        payload = _json.dumps(gltf).encode()
        payload += b" " * (-len(payload) % 4)
        data = (_struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(payload))
                + _struct.pack("<I4s", len(payload), b"JSON") + payload)
        f = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        f.write(data)
        f.close()
        return f.name

    def test_mud_resolution_below_threshold(self):
        self.assertLess(vc.mesh_vertex_count(self._glb(11483)),
                        vc.RECON_MIN_VERTICES)

    def test_normal_resolution_above_threshold(self):
        self.assertGreaterEqual(vc.mesh_vertex_count(self._glb(23876)),
                                vc.RECON_MIN_VERTICES)


class ChunkTests(unittest.TestCase):
    def test_30_prompts_split_evenly(self):
        """Go chunkMotionPrompts와 동일하게 30개 → 15+15."""
        batches = rc.chunk(list(range(30)), 20)
        self.assertEqual([len(b) for b in batches], [15, 15])


class InfraStallTests(unittest.TestCase):
    """적응형 재시도 전 인프라 정지 감지 (실측 2026-07-20 explorer-v2 낭비)."""

    def setUp(self):
        self._rp = mp.rp

    def tearDown(self):
        mp.rp = self._rp

    def test_zero_workers_is_stalled(self):
        mp.rp = lambda *a, **k: {
            "workers": {"idle": 0, "ready": 0, "running": 0},
            "jobs": {"inQueue": 2},
        }
        stalled, queue = rc._infra_stalled()
        self.assertTrue(stalled)
        self.assertEqual(queue, {"inQueue": 2})

    def test_running_worker_is_not_stalled(self):
        mp.rp = lambda *a, **k: {"workers": {"idle": 0, "running": 1}, "jobs": {}}
        stalled, _ = rc._infra_stalled()
        self.assertFalse(stalled)

    def test_health_error_does_not_block_retry(self):
        def boom(*a, **k):
            raise RuntimeError("health unavailable")
        mp.rp = boom
        stalled, queue = rc._infra_stalled()
        self.assertFalse(stalled)
        self.assertIsNone(queue)


if __name__ == "__main__":
    unittest.main()
