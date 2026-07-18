#!/usr/bin/env python3
"""캐릭터 1종을 gpt-image-2 생성부터 30모션 검증까지 전 구간 실행하고,
통과하면 실제 앱(jobs.json)에 프로젝트로 등록한다.

단계마다 실패하면 즉시 중단해 다음 캐릭터로 넘어가지 않는다 (하나씩 만들고
평가하면서 완성하는 워크플로우):
  1. gpt-image-2 원본 이미지 생성 (없으면)
  2. RunPod Hunyuan3D-2.1 reconstruct
  3. baseline_worker retopo → rig (로컬) → 1차 검증(직립/관절계층)
  4. RunPod HY-Motion 30개 프롬프트 (15+15 배치) → hy_motion.json
  5. baseline_worker motion 베이킹 → 2차 검증(직립/관절계층/스킨변형, 30클립)
  6. 통과 시 실제 앱 jobs.json + projects/<id>/에 등록

사용: python3 run_character.py <name> [--set 2]
"""
import argparse
import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_characters  # noqa: E402
import matrix_pipeline as mp  # noqa: E402  rp/poll/run_worker/WORKER 재사용
import validate_character  # noqa: E402

APP_ROOT = Path.home() / "Library/Application Support/SpriteEngine"

MOTIONS_30 = [
    ("idle",       "a person stands still, breathing calmly and shifting weight slightly", 4.0),
    ("walk",       "a person walks forward casually", 5.0),
    ("run",        "a person runs forward quickly", 5.0),
    ("jump",       "a person jumps high in place", 4.0),
    ("wave",       "a person waves hello with the right hand", 4.0),
    ("dance",      "a person dances energetically", 6.0),
    ("punch",      "a person throws two strong punches", 4.0),
    ("kick",       "a person performs a high kick", 4.0),
    ("bow",        "a person bows politely", 4.0),
    ("spin",       "a person spins around in place", 4.0),
    ("flykick",    "a person leaps forward and performs a flying kick", 5.0),
    ("crouch",     "a person crouches down low and holds the position", 4.0),
    ("roll",       "a person rolls forward on the ground and stands back up", 5.0),
    ("climb",      "a person climbs up an invisible ladder", 5.0),
    ("crawl",      "a person crawls forward on hands and knees", 5.0),
    ("sit",        "a person sits down on the ground cross-legged", 4.0),
    ("standup",    "a person stands up from a seated position", 4.0),
    ("lookaround", "a person looks around curiously, turning their head left and right", 4.0),
    ("pickup",     "a person bends down and picks up an object from the ground", 4.0),
    ("push",       "a person pushes hard against something heavy in front of them", 4.0),
    ("pull",       "a person pulls hard on a rope behind them", 4.0),
    ("throw",      "a person winds up and throws an object overhand", 4.0),
    ("block",      "a person raises both arms to block an incoming attack", 3.0),
    ("dodge",      "a person quickly dodges to the side", 3.0),
    ("sprint",     "a person sprints forward at full speed", 4.0),
    ("stumble",    "a person stumbles and loses balance briefly", 4.0),
    ("cheer",      "a person raises both arms and cheers with excitement", 4.0),
    ("salute",     "a person stands at attention and salutes", 3.0),
    ("faint",      "a person faints and collapses to the ground", 4.0),
    ("shrug",      "a person shrugs their shoulders in confusion", 3.0),
]


def chunk(prompts, max_size=20):
    """Go의 chunkMotionPrompts와 동일한 균등 분할 (30 → 15+15)."""
    n = len(prompts)
    count = (n + max_size - 1) // max_size
    size = (n + count - 1) // count
    return [prompts[i:i + size] for i in range(0, n, size)]


def ensure_image(name, desc, char_set):
    png = mp.CHARS / f"{name}.png"
    if png.exists():
        print(f"[image] {name}: exists, skip", flush=True)
        return png
    print(f"[image] {name}: generating via gpt-image-2", flush=True)
    gen_characters.generate(name, desc)
    return png


def reconstruct(name, png):
    ws = mp.WS / name
    out = ws / "reconstruct" / "hunyuan3d21.glb"
    if out.exists():
        print(f"[recon] {name}: exists, skip", flush=True)
        return out
    payload = {"input": {
        "image": base64.b64encode(png.read_bytes()).decode(), "seed": 1234,
        "steps": 30, "guidance_scale": 5.0, "octree_resolution": 256,
        "face_count": 40000, "texture": True, "max_num_view": 6,
        "texture_resolution": 512,
    }}
    job = mp.rp("POST", "run", payload, timeout=120)
    print(f"[recon] {name}: submitted {job['id']}", flush=True)
    result = mp.poll({name: job["id"]})[name]
    if not result or not result.get("glb_base64"):
        sys.exit(f"[recon] {name}: FAILED {str((result or {}).get('error'))[:300]}")
    glb = base64.b64decode(result["glb_base64"])
    assert glb[:4] == b"glTF"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(glb)
    print(f"[recon] {name}: {len(glb)} bytes textured={result.get('textured')}", flush=True)
    return out


def retopo_and_rig(name, recon_glb):
    ws = mp.WS / name
    retopo = mp.run_worker("retopo", ws, recon_glb)
    rig = mp.run_worker("rig", ws, retopo["path"])
    if not rig.get("metrics", {}).get("skinned"):
        sys.exit(f"[rig] {name}: not skinned — {rig.get('metrics')}")
    check = validate_character.validate(rig["path"])
    print(f"[rig] {name}: upright={check['upright']['ok']} hierarchy={check['hierarchy']['ok']}", flush=True)
    if not check["upright"]["ok"] or not check["hierarchy"]["ok"]:
        for issue in check["upright"]["issues"] + check["hierarchy"]["issues"]:
            print(f"[rig]   {name}: {issue}", flush=True)
        sys.exit(f"[rig] {name}: FAILED render-sanity check before motion generation (stopping to save GPU time)")
    return retopo["path"], rig["path"]


def generate_motion(name):
    ws = mp.WS / name
    target = ws / "motion" / "hy_motion.json"
    if target.exists():
        print(f"[motion] {name}: hy_motion.json exists, skip", flush=True)
        return json.loads(target.read_text())
    prompts = [{"id": i, "text": t, "duration": d} for i, t, d in MOTIONS_30]
    batches = chunk(prompts, 20)
    merged = {"model": "", "errors": {}, "motions": []}
    for bi, batch in enumerate(batches):
        payload = {"input": {"task": "motion", "prompts": batch, "seed": 42, "cfg_scale": 5.0}}
        job = mp.rp("POST", "run", payload, timeout=120)
        print(f"[motion] {name}: batch {bi+1}/{len(batches)} ({len(batch)} prompts) submitted {job['id']}", flush=True)
        out = mp.poll({f"{name}-b{bi}": job["id"]})[f"{name}-b{bi}"]
        if not out or out.get("error"):
            sys.exit(f"[motion] {name}: batch {bi+1} failed: {(out or {}).get('error')}")
        merged["model"] = out.get("model") or merged["model"]
        merged["motions"].extend(out.get("motions", []))
        merged["errors"].update(out.get("errors", {}))
    print(f"[motion] {name}: model={merged['model']} clips={len(merged['motions'])} errors={merged['errors']}", flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(merged))
    return merged


def bake_and_validate(name, rig_glb):
    ws = mp.WS / name
    baked = mp.run_worker("motion", ws, rig_glb, adapter="hy-motion-retarget")
    m = baked.get("metrics", {})
    if m.get("adapter") != "hy-motion-retarget":
        sys.exit(f"[bake] {name}: fell back to {m.get('adapter')}")
    check = validate_character.validate(baked["path"])
    print(f"[bake] {name}: {m.get('animations')} clips, render_valid={check['ok']}", flush=True)
    if not check["ok"]:
        for section in ("upright", "hierarchy", "deformation"):
            for issue in check[section]["issues"]:
                print(f"[bake]   {name}: {issue}", flush=True)
        sys.exit(f"[bake] {name}: FAILED final render-sanity check")
    return baked["path"], m.get("animations", 0), m.get("model", "")


def register_in_app(name, retopo_path, rig_path, motion_path, animations, model):
    jobs_path = APP_ROOT / "jobs.json"
    jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else []
    job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().int % 1000000:06d}"
    proj = APP_ROOT / "projects" / job_id
    for stage, src in (("retopo", retopo_path), ("rig", rig_path), ("motion", motion_path)):
        dst_dir = proj / stage
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / Path(src).name
        dst.write_bytes(Path(src).read_bytes())
    recon_src = mp.WS / name / "reconstruct" / "hunyuan3d21.glb"
    (proj / "reconstruct").mkdir(parents=True, exist_ok=True)
    (proj / "reconstruct" / "hunyuan3d21.glb").write_bytes(recon_src.read_bytes())
    png_src = mp.CHARS / f"{name}.png"
    (proj / "source.png").write_bytes(png_src.read_bytes())

    job = {
        "id": job_id, "name": f"matrix2-{name}",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "status": "ready", "progress": 83,
        "image": str(proj / "source.png"), "imageHash": "",
        "workspace": str(proj),
        "stages": [
            {"id": "prepare", "name": "Image cleanup", "status": "done", "detail": "Reference secured from matrix2 validation"},
            {"id": "reconstruct", "name": "3D reconstruction", "status": "done", "detail": "Completed · RunPod Hunyuan3D-2.1"},
            {"id": "retopo", "name": "Mesh cleanup", "status": "done", "detail": "Completed · local-baseline"},
            {"id": "rig", "name": "Auto rig", "status": "done", "detail": "Completed · auto-rig-bbox (skinned, upright-normalized)"},
            {"id": "motion", "name": "Animation", "status": "done", "detail": f"Completed · RunPod HY-Motion ({animations} clips)"},
            {"id": "export", "name": "Export", "status": "ready", "detail": "GLB / FBX / USDZ package"},
        ],
        "artifacts": [
            {"stage": "prepare", "kind": "reference", "path": str(proj / "source.png")},
            {"stage": "reconstruct", "kind": "mesh", "path": str(proj / "reconstruct" / "hunyuan3d21.glb"),
             "metrics": {"adapter": "runpod-hunyuan3d21", "textured": True}},
            {"stage": "retopo", "kind": "clean-mesh", "path": str(proj / "retopo" / Path(retopo_path).name)},
            {"stage": "rig", "kind": "rigged-model", "path": str(proj / "rig" / Path(rig_path).name),
             "metrics": {"adapter": "auto-rig-bbox", "skinned": True}},
            {"stage": "motion", "kind": "animated-model", "path": str(proj / "motion" / Path(motion_path).name),
             "metrics": {"adapter": "hy-motion-retarget", "animations": animations, "model": model}},
        ],
        "logs": [{"time": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"), "stage": "system", "level": "info",
                  "message": f"matrix2 검증 산출물에서 등록 (gpt-image-2 → Hunyuan3D → auto-rig(직립 정규화) → HY-Motion {animations}클립)"}],
    }
    jobs.insert(0, job)
    jobs_path.write_text(json.dumps(jobs, indent=2))
    print(f"[register] {name}: registered as job {job_id} ({animations} clips)", flush=True)
    return job_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--set", choices=["1", "2"], default="2")
    args = ap.parse_args()
    pool = gen_characters.CHARACTERS if args.set == "1" else gen_characters.CHARACTERS_SET2
    if args.name not in pool:
        sys.exit(f"unknown character {args.name!r} in set {args.set}: {sorted(pool)}")
    desc = pool[args.name]

    mp.WS.mkdir(parents=True, exist_ok=True)
    png = ensure_image(args.name, desc, args.set)
    recon_glb = reconstruct(args.name, png)
    retopo_path, rig_path = retopo_and_rig(args.name, recon_glb)
    generate_motion(args.name)
    motion_path, animations, model = bake_and_validate(args.name, rig_path)
    job_id = register_in_app(args.name, retopo_path, rig_path, motion_path, animations, model)
    print(f"CHARACTER_OK name={args.name} job={job_id} clips={animations}")


if __name__ == "__main__":
    main()
