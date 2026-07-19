#!/usr/bin/env python3
"""run_character 드라이버 로직 회귀 테스트 (네트워크 없이 실행)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matrix_pipeline as mp  # noqa: E402
import run_character as rc  # noqa: E402


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
