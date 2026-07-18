#!/usr/bin/env python3
"""10 캐릭터 × 10 모션 검증 파이프라인 드라이버.

앱(Wails) 없이 동일 경로를 재현한다:
  1. gpt-image-2 원본 → RunPod Hunyuan3D-2.1 reconstruct (shape+texture, 동시 제출)
  2. baseline_worker retopo → rig (로컬)
  3. RunPod HY-Motion task=motion 프롬프트 10개 → hy_motion.json (1 job)
  4. 캐릭터별 motion 단계: SMPL→auto-rig 리타겟·GLB 베이킹 (10 clips each)
  5. GLB 구조 검증: animations/channels/duration

사용: python3 matrix_pipeline.py [--stage recon|motion|bake|all]
경로: $MATRIX_ROOT (기본 /tmp/spriteengine_matrix), 워커는 repo 상대 경로로 자동 해석.
RunPod 자격 증명: 앱 설정 파일(~/Library/Application Support/SpriteEngine/runpod.json).
"""
import argparse
import base64
import json
import os
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_character  # noqa: E402  렌더링 정상성 게이트(직립/관절계층/스킨변형)

ROOT = Path(os.environ.get("MATRIX_ROOT", "/tmp/spriteengine_matrix"))
CHARS = ROOT / "characters"
WS = ROOT / "ws"
WORKER = str(Path(__file__).resolve().parents[2] / "workers" / "baseline_worker.py")

cfg = json.loads((Path.home() / "Library/Application Support/SpriteEngine/runpod.json").read_text())
KEY, EP = cfg["apiKey"], cfg["endpointId"]
BASE = cfg.get("baseUrl") or "https://api.runpod.ai/v2"

MOTIONS = [
    ("run",     "a person runs forward quickly", 5.0),
    ("jump",    "a person jumps high in place", 4.0),
    ("wave",    "a person waves hello with the right hand", 4.0),
    ("dance",   "a person dances energetically", 6.0),
    ("punch",   "a person throws two strong punches", 4.0),
    ("kick",    "a person performs a high kick", 4.0),
    ("walk",    "a person walks forward casually", 5.0),
    ("bow",     "a person bows politely", 4.0),
    ("spin",    "a person spins around in place", 4.0),
    ("flykick", "a person leaps forward and performs a flying kick", 5.0),
]


def rp(method, path, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{BASE}/{EP}/{path}", data=data, method=method,
                                 headers={"Authorization": "Bearer " + KEY,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def poll(job_ids, minutes=40):
    """여러 job을 함께 폴링해 {id: output} 반환. 실패는 예외 대신 None."""
    pending = dict(job_ids)  # name -> job id
    results = {}
    deadline = time.time() + minutes * 60
    while pending and time.time() < deadline:
        time.sleep(6)
        for name, jid in list(pending.items()):
            st = rp("GET", "status/" + jid)
            s = st.get("status")
            if s in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                del pending[name]
                if s == "COMPLETED":
                    results[name] = st.get("output") or {}
                    print(f"[recon] {name}: COMPLETED", flush=True)
                else:
                    results[name] = None
                    print(f"[recon] {name}: {s} {str(st.get('error'))[:300]}", flush=True)
        if pending:
            print(f"[poll] waiting: {sorted(pending)}", flush=True)
    for name in pending:
        results[name] = None
        print(f"[poll] {name}: timeout", flush=True)
    return results


def run_worker(stage, workspace, input_path, adapter="local-baseline"):
    """baseline_worker 한 stage 실행, artifact 경로/메트릭 반환."""
    req = {"type": "run", "jobId": f"matrix-{workspace.name}", "stage": stage,
           "workspace": str(workspace), "input": str(input_path), "adapter": adapter}
    proc = subprocess.run(["python3", WORKER], input=json.dumps(req) + "\n",
                          capture_output=True, text=True, timeout=600)
    artifact, error = None, None
    for line in proc.stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "artifact":
            artifact = ev
        if ev.get("type") == "error":
            error = ev.get("message")
    if proc.returncode != 0 or error:
        raise RuntimeError(f"{stage} failed: {error or proc.stderr[-500:]}")
    if not artifact:
        raise RuntimeError(f"{stage} produced no artifact")
    return artifact


def stage_recon():
    jobs = {}
    for png in sorted(CHARS.glob("*.png")):
        name = png.stem
        out = WS / name / "reconstruct" / "hunyuan3d21.glb"
        if out.exists():
            print(f"[recon] {name}: exists, skip", flush=True)
            continue
        payload = {"input": {
            "image": base64.b64encode(png.read_bytes()).decode(), "seed": 1234,
            "steps": 30, "guidance_scale": 5.0, "octree_resolution": 256,
            "face_count": 40000, "texture": True, "max_num_view": 6,
            "texture_resolution": 512,
        }}
        job = rp("POST", "run", payload, timeout=120)
        jobs[name] = job["id"]
        print(f"[recon] {name}: submitted {job['id']}", flush=True)
    if not jobs:
        return
    for name, out in poll(jobs).items():
        if not out or not out.get("glb_base64"):
            print(f"[recon] {name}: FAILED {str((out or {}).get('error'))[:300]}", flush=True)
            continue
        glb = base64.b64decode(out["glb_base64"])
        assert glb[:4] == b"glTF", name
        target = WS / name / "reconstruct" / "hunyuan3d21.glb"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(glb)
        print(f"[recon] {name}: {len(glb)} bytes textured={out.get('textured')}", flush=True)


def stage_motion():
    target = ROOT / "hy_motion_10.json"
    if target.exists():
        print("[motion] hy_motion_10.json exists, skip", flush=True)
        return
    prompts = [{"id": i, "text": t, "duration": d} for i, t, d in MOTIONS]
    payload = {"input": {"task": "motion", "prompts": prompts, "seed": 42, "cfg_scale": 5.0}}
    job = rp("POST", "run", payload, timeout=120)
    print(f"[motion] submitted {job['id']}", flush=True)
    out = poll({"motion10": job["id"]})["motion10"]
    if not out or out.get("error"):
        sys.exit(f"[motion] failed: {(out or {}).get('error')}")
    print(f"[motion] model={out.get('model')} clips={len(out.get('motions', []))} "
          f"errors={out.get('errors')}", flush=True)
    target.write_text(json.dumps(out))


def parse_glb(path):
    data = path.read_bytes()
    assert data[:4] == b"glTF"
    length = struct.unpack_from("<I", data, 12)[0]
    assert struct.unpack_from("<I", data, 16)[0] == 0x4E4F534A  # JSON
    return json.loads(data[20:20 + length])


def stage_bake():
    motion_json = (ROOT / "hy_motion_10.json").read_text()
    report = {}
    for char_ws in sorted(WS.iterdir()):
        name = char_ws.name
        recon = char_ws / "reconstruct" / "hunyuan3d21.glb"
        if not recon.exists():
            report[name] = "no reconstruct GLB"
            continue
        try:
            retopo = run_worker("retopo", char_ws, recon)
            rig = run_worker("rig", char_ws, retopo["path"])
            if not rig.get("metrics", {}).get("skinned"):
                report[name] = f"rig not skinned: {rig.get('metrics')}"
                continue
            (char_ws / "motion").mkdir(exist_ok=True)
            (char_ws / "motion" / "hy_motion.json").write_text(motion_json)
            baked = run_worker("motion", char_ws, rig["path"], adapter="hy-motion-retarget")
            m = baked.get("metrics", {})
            if m.get("adapter") != "hy-motion-retarget":
                report[name] = f"bake fell back to {m.get('adapter')}"
                continue
            gltf = parse_glb(Path(baked["path"]))
            anims = gltf.get("animations", [])
            chans = sorted({len(a.get("channels", [])) for a in anims})
            names = [a.get("name") for a in anims]
            # 렌더링 정상성 게이트: 직립/관절계층/스킨변형이 전부 통과해야
            # "정상적으로 렌더링되는 캐릭터"로 인정한다.
            check = validate_character.validate(baked["path"])
            report[name] = {"animations": len(anims), "channels": chans,
                            "names": names, "path": baked["path"],
                            "render_valid": check["ok"],
                            "render_issues": (check["upright"]["issues"]
                                              + check["hierarchy"]["issues"]
                                              + check["deformation"]["issues"])}
            status = "OK" if check["ok"] else "RENDER-INVALID"
            print(f"[bake] {name}: {len(anims)} clips, channels={chans}, render={status}", flush=True)
            if not check["ok"]:
                for issue in report[name]["render_issues"]:
                    print(f"[bake]   {name}: {issue}", flush=True)
        except Exception as exc:  # noqa: BLE001
            report[name] = f"ERROR: {exc}"
            print(f"[bake] {name}: ERROR {exc}", flush=True)
    (ROOT / "matrix_report.json").write_text(json.dumps(report, indent=2))
    ok = sum(1 for v in report.values()
             if isinstance(v, dict) and v["animations"] == len(MOTIONS) and v["render_valid"])
    print(f"[bake] {ok}/{len(report)} characters fully baked, {len(MOTIONS)} clips, and render-valid", flush=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["recon", "motion", "bake", "all"])
    args = p.parse_args()
    WS.mkdir(parents=True, exist_ok=True)
    if args.stage in ("recon", "all"):
        stage_recon()
    if args.stage in ("motion", "all"):
        stage_motion()
    if args.stage in ("bake", "all"):
        stage_bake()
    print("MATRIX_DONE")
