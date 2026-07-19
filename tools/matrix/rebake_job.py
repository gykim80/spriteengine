#!/usr/bin/env python3
"""기존 앱 job(수정 전 파이프라인으로 처리돼 애니메이션이 없는 export)을
수정된 워커 + HY-Motion으로 재처리해 제자리 갱신한다.

reconstruct GLB는 이미 앱 프로젝트에 있으므로 그 이후 단계만 재실행:
  retopo → rig(1차 검증) → HY-Motion 30클립 → 베이킹(워커 게이트) → export
통과 시 앱 프로젝트의 retopo/rig/motion/export 파일을 백업(.bak-rebake2) 후
교체하고 jobs.json의 artifacts/stages를 갱신한다.

사용: python3 rebake_job.py <app-job-id> <ws-name>
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matrix_pipeline as mp  # noqa: E402
import run_character as rc  # noqa: E402

APP_ROOT = Path.home() / "Library/Application Support/SpriteEngine"


def main():
    job_id, name = sys.argv[1], sys.argv[2]
    jobs_path = APP_ROOT / "jobs.json"
    jobs = json.loads(jobs_path.read_text())
    job = next(j for j in jobs if j["id"] == job_id)
    proj = Path(job["workspace"])
    recon_src = proj / "reconstruct" / "hunyuan3d21.glb"
    assert recon_src.exists(), recon_src

    ws = mp.WS / name
    (ws / "reconstruct").mkdir(parents=True, exist_ok=True)
    recon = ws / "reconstruct" / "hunyuan3d21.glb"
    if not recon.exists():
        recon.write_bytes(recon_src.read_bytes())
    print(f"[seed] {name}: reconstruct {recon.stat().st_size} bytes", flush=True)

    retopo_path, rig_path = rc.retopo_and_rig(name, recon)
    rc.generate_motion(name)
    motion_path, animations, model = rc.bake_and_validate(name, rig_path)
    exported = mp.run_worker("export", ws, motion_path)
    assert exported.get("metrics", {}).get("validated"), exported.get("metrics")
    export_path = exported["path"]

    # 앱 프로젝트에 반영 (기존 파일 백업 후 교체)
    new_paths = {}
    for stage, src in (("retopo", retopo_path), ("rig", rig_path),
                       ("motion", motion_path), ("export", export_path)):
        dst_dir = proj / stage
        dst_dir.mkdir(parents=True, exist_ok=True)
        for old in dst_dir.glob("*.glb"):
            bak = old.with_suffix(old.suffix + ".bak-rebake2")
            if not bak.exists():
                shutil.copy2(old, bak)
            old.unlink()
        dst = dst_dir / Path(src).name
        dst.write_bytes(Path(src).read_bytes())
        new_paths[stage] = str(dst)
        print(f"[apply] {name}: {stage} -> {dst.name}", flush=True)

    details = {
        "retopo": "Completed · local-baseline (rebaked)",
        "rig": "Completed · auto-rig-bbox (skinned, upright-normalized)",
        "motion": f"Completed · RunPod HY-Motion ({animations} clips)",
        "export": "Completed · passthrough-offline (validated)",
    }
    metrics = {
        "retopo": {"adapter": "front-projection", "validated": True},
        "rig": {"adapter": "auto-rig-bbox", "skinned": True, "renderValid": True},
        "motion": {"adapter": "hy-motion-retarget", "animations": animations,
                   "model": model, "renderValid": True},
        "export": {"adapter": "passthrough-offline", "validated": True},
    }
    for s in job["stages"]:
        if s["id"] in details:
            s["detail"] = details[s["id"]]
    for a in job["artifacts"]:
        st = a.get("stage")
        if st in new_paths:
            a["path"] = new_paths[st]
            a["metrics"] = {**a.get("metrics", {}), **metrics[st]}
    job.setdefault("logs", []).append({
        "time": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"), "stage": "system",
        "level": "info",
        "message": f"수정된 파이프라인으로 재베이킹 (auto-rig 직립 정규화 + HY-Motion {animations}클립, 렌더링 게이트 통과)",
    })
    jobs_path.write_text(json.dumps(jobs, ensure_ascii=False, indent=1))
    print(f"REBAKE_OK job={job_id} name={name} clips={animations}", flush=True)


if __name__ == "__main__":
    main()
