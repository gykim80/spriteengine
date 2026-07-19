#!/usr/bin/env python3
"""폴링 데드라인 초과로 드라이버가 포기했지만 GPU에서 계속 실행 중인
HY-Motion 배치를 수동 회수해 hy_motion.json을 완성한다.

실측 2026-07-20: 엔드포인트 혼잡으로 gladiator/skater 모션 배치가 드라이버
타임아웃 뒤에도 큐/GPU에 살아 있었다. 이 도구로 job id를 넘겨 결과를
회수하거나, id가 없으면 해당 배치를 새로 제출한다.

사용: python3 recover_motion.py <name> [batch0_job_id] [batch1_job_id]
"""
import json
import sys

import matrix_pipeline as mp
import run_character as rc


def main() -> None:
    name = sys.argv[1]
    job_ids = sys.argv[2:]

    prompts = [{"id": i, "text": t, "duration": d} for i, t, d in rc.MOTIONS_30]
    batches = rc.chunk(prompts, 20)

    merged = {"model": "", "errors": {}, "motions": []}
    for bi, batch in enumerate(batches):
        if bi < len(job_ids):
            jid = job_ids[bi]
            print(f"[recover] {name}: batch {bi+1} reuse existing job {jid}", flush=True)
        else:
            payload = {"input": {"task": "motion", "prompts": batch, "seed": 42, "cfg_scale": 5.0}}
            job = mp.rp("POST", "run", payload, timeout=120)
            jid = job["id"]
            print(f"[recover] {name}: batch {bi+1} submitted {jid}", flush=True)
        out = mp.poll({f"{name}-b{bi}": jid}, minutes=120)[f"{name}-b{bi}"]
        if not out or out.get("error"):
            sys.exit(f"[recover] {name}: batch {bi+1} failed: {(out or {}).get('error')}")
        merged["model"] = out.get("model") or merged["model"]
        merged["motions"].extend(out.get("motions", []))
        merged["errors"].update(out.get("errors", {}))
        print(f"[recover] {name}: batch {bi+1} -> {len(out.get('motions', []))} clips", flush=True)

    expected = len(rc.MOTIONS_30)
    got = len(merged["motions"])
    if got != expected:
        sys.exit(f"[recover] {name}: clips {got} != expected {expected} errors={merged['errors']}")

    target = mp.WS / name / "motion" / "hy_motion.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(merged))
    print(f"[recover] {name}: OK model={merged['model']} clips={got} -> {target}", flush=True)


if __name__ == "__main__":
    main()
