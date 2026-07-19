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

사용: python3 run_character.py <name> [--set 1|2|q]  (q = 4족 quadruped 세트)
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


# 4족은 단일 이미지 복원이 가려진 쪽 다리를 웹/융합으로 환각하기 쉽다
# (실측: 3/4 시점 시바견 → 앞다리 웹). 실측 결과 RunPod Hunyuan 엔드포인트는
# seed가 셰이프에 반영되지 않아(시드 3개 POSITION/텍스처 해시 완전 동일)
# 시드 변주는 무의미했다 — 환각의 실제 변동원은 원본 이미지이므로, 4족은
# 이미지를 여러 장 생성해 각각 복원한 뒤 leg_quality 최고 후보를 선택한다.
QUAD_IMAGE_CANDIDATES = 3

# 휴머노이드도 간혹 누운/대각선 자세로 복원된다 (실측: vampire — raw XY 평면
# 대각 32° 누움, 리깅의 노드 회전 베이크로는 교정 불가). 4족과 달리 발생
# 빈도가 낮으므로 이미지를 선제 3장 만들지 않고, 복원 직후 직립 품질 게이트
# (humanoid_upright_quality)에 실패한 경우에만 이미지를 추가 생성해 재복원하는
# 적응형 재시도를 쓴다 (최대 3장).
HUMANOID_MAX_IMAGE_ATTEMPTS = 3


def image_variants(name, count=QUAD_IMAGE_CANDIDATES):
    return [name] + [f"{name}-v{i}" for i in range(2, count + 1)]


def ensure_images(name, desc, char_set):
    names = image_variants(name) if char_set == "q" else [name]
    pngs = []
    for n in names:
        png = mp.CHARS / f"{n}.png"
        if png.exists():
            print(f"[image] {n}: exists, skip", flush=True)
        else:
            print(f"[image] {n}: generating via gpt-image-2", flush=True)
            gen_characters.generate(n, desc)
        pngs.append(png)
    return pngs


def _recon_candidates(name, pngs, out_dir, quadruped):
    """이미지들을 복원해 (score, ok, stem, glb경로, png경로) 후보 리스트 반환.

    이미 복원된 candidate-*.glb가 있으면 재제출 없이 재평가만 한다
    (재시도 시 GPU 시간 절약).
    """
    jobs = {}
    for png in pngs:
        if (out_dir / f"candidate-{png.stem}.glb").exists():
            print(f"[recon] {name}: image={png.stem} candidate exists, re-scoring", flush=True)
            continue
        payload = {"input": {
            "image": base64.b64encode(png.read_bytes()).decode(), "seed": 1234,
            "steps": 30, "guidance_scale": 5.0, "octree_resolution": 256,
            "face_count": 40000, "texture": True, "max_num_view": 6,
            "texture_resolution": 512,
        }}
        job = mp.rp("POST", "run", payload, timeout=120)
        jobs[png.stem] = job["id"]
        print(f"[recon] {name}: submitted image={png.stem} {job['id']}", flush=True)
    results = mp.poll(jobs) if jobs else {}
    candidates = []
    for png in pngs:
        cand = out_dir / f"candidate-{png.stem}.glb"
        if not cand.exists():
            result = results.get(png.stem)
            if not result or not result.get("glb_base64"):
                print(f"[recon] {name}: image={png.stem} FAILED {str((result or {}).get('error'))[:200]}", flush=True)
                continue
            glb = base64.b64decode(result["glb_base64"])
            assert glb[:4] == b"glTF"
            cand.write_bytes(glb)
        # 바닥 판(base slab) 위에 서 있는 복원(실측: gladiator 3장, vampire-v1)은
        # 슬래브만 제거하면 정상 후보다 — 스코어링 전에 잘라낸다 (없으면 no-op).
        # 단, 제거 후 캐릭터가 미니어처(재정규화 배율 초과)면 저해상도 진흙
        # 품질이므로 후보에서 제외한다 → 적응형 이미지 재생성으로 폴백.
        try:
            stripped = validate_character.worker.strip_base_plane(str(cand), str(cand))
        except validate_character.worker.BasePlaneRescueError as exc:
            print(f"[recon] {name}: image={png.stem} REJECTED — {exc}", flush=True)
            continue
        if stripped:
            print(f"[recon] {name}: image={png.stem} stripped base plane ({stripped} verts)", flush=True)
        # 저해상도 진흙 기각 — 실측(gladiator, 사용자 신고 "완전 심각"): 11,483
        # verts 진흙 복원이 직립/파편/슬래브 게이트를 전부 통과해 등록까지 갔다.
        # 정상은 23.9k~29.5k verts(등록 22종 실측)라 하한으로 안전하게 분리된다.
        verts = validate_character.mesh_vertex_count(str(cand))
        if verts < validate_character.RECON_MIN_VERTICES:
            print(f"[recon] {name}: image={png.stem} REJECTED — low-res mud "
                  f"({verts} verts < {validate_character.RECON_MIN_VERTICES})", flush=True)
            continue
        if quadruped:
            q = validate_character.leg_quality(str(cand))
            score, ok = q["score"], q["score"] >= 0
            print(f"[recon] {name}: image={png.stem} {cand.stat().st_size} bytes leg_quality={q}", flush=True)
        else:
            q = validate_character.humanoid_upright_quality(str(cand))
            score, ok = q["score"], q["ok"]
            print(f"[recon] {name}: image={png.stem} {cand.stat().st_size} bytes upright_quality={q}", flush=True)
        candidates.append((score, ok, png.stem, cand, png))
    return candidates


def _infra_stalled():
    """엔드포인트 워커가 0이면 True — 크레딧 소진/GPU 재고 고갈 신호.

    실측 2026-07-20: 크레딧 소진으로 recon job이 실행조차 못 됐는데 이를
    이미지 품질 실패로 오인, 변주 이미지 생성 + 중복 recon job 제출로
    고아 job(explorer-v2)과 이미지 비용이 낭비됐다. 워커 0이면 이미지
    문제가 아니므로 적응형 재시도를 중단해야 한다.
    """
    try:
        h = mp.rp("GET", "health")
        return sum(h.get("workers", {}).values()) == 0, h.get("jobs")
    except Exception:  # noqa: BLE001 — health 조회 실패는 판단 보류
        return False, None


def reconstruct(name, desc, pngs, quadruped=False):
    ws = mp.WS / name
    out = ws / "reconstruct" / "hunyuan3d21.glb"
    if out.exists():
        print(f"[recon] {name}: exists, skip", flush=True)
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    pngs = list(pngs)
    candidates = _recon_candidates(name, pngs, out.parent, quadruped)
    if not quadruped:
        # 적응형 재시도: 직립 게이트 통과 후보가 없으면 이미지를 새로 만들어
        # 재복원한다 (환각의 변동원은 원본 이미지 — 시드 변주는 무의미).
        while not any(c[1] for c in candidates) and len(pngs) < HUMANOID_MAX_IMAGE_ATTEMPTS:
            stalled, queue = _infra_stalled()
            if stalled:
                sys.exit(f"[recon] {name}: endpoint has ZERO workers — infrastructure "
                         f"stall (credit balance / GPU capacity), not an image quality "
                         f"problem; retry when workers recover ({queue})")
            variant = f"{name}-v{len(pngs) + 1}"
            print(f"[recon] {name}: no upright candidate yet — generating variant image {variant}", flush=True)
            gen_characters.generate(variant, desc)
            png = mp.CHARS / f"{variant}.png"
            pngs.append(png)
            candidates += _recon_candidates(name, [png], out.parent, quadruped)
    if not candidates:
        sys.exit(f"[recon] {name}: all reconstructions FAILED")
    best_score, best_ok, best_stem, best, best_png = max(candidates)
    if not best_ok:
        kind = "4 sound leg columns" if quadruped else "an upright body"
        sys.exit(f"[recon] {name}: no candidate reconstructed {kind} "
                 f"— adjust the character prompt and retry")
    out.write_bytes(best.read_bytes())
    # 앱 등록 시 실제 채택된 원본 이미지를 쓰도록 함께 보존한다.
    (out.parent / "source.png").write_bytes(best_png.read_bytes())
    print(f"[recon] {name}: selected image={best_stem} (score={best_score})", flush=True)
    return out


def retopo_and_rig(name, recon_glb):
    ws = mp.WS / name
    retopo = mp.run_worker("retopo", ws, recon_glb)
    rig = mp.run_worker("rig", ws, retopo["path"])
    if not rig.get("metrics", {}).get("skinned"):
        sys.exit(f"[rig] {name}: not skinned — {rig.get('metrics')}")
    body_type = rig.get("metrics", {}).get("bodyType", "humanoid")
    check = validate_character.validate(rig["path"])
    print(f"[rig] {name}: bodyType={body_type} upright={check['upright']['ok']} "
          f"hierarchy={check['hierarchy']['ok']} legs={check['legs']['ok']} "
          f"leg_columns={check['legs'].get('legs')}", flush=True)
    if not (check["upright"]["ok"] and check["hierarchy"]["ok"] and check["legs"]["ok"]):
        for issue in check["upright"]["issues"] + check["hierarchy"]["issues"] + check["legs"]["issues"]:
            print(f"[rig]   {name}: {issue}", flush=True)
        sys.exit(f"[rig] {name}: FAILED render-sanity check before motion generation (stopping to save GPU time)")
    return retopo["path"], rig["path"], body_type


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
        # 실측 2026-07-20: gladiator 배치1이 GPU에서 정상 실행 중인데 기본 40분
        # 폴링 데드라인에 걸려 timeout 처리 → sys.exit. 엔드포인트 혼잡 시
        # 15프롬프트 배치가 40분을 넘길 수 있어 90분으로 여유를 둔다.
        out = mp.poll({f"{name}-b{bi}": job["id"]}, minutes=90)[f"{name}-b{bi}"]
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
        for section in ("upright", "hierarchy", "legs", "deformation"):
            for issue in check[section]["issues"]:
                print(f"[bake]   {name}: {issue}", flush=True)
        sys.exit(f"[bake] {name}: FAILED final render-sanity check")
    return baked["path"], m.get("animations", 0), m.get("model", "")


def register_in_app(name, retopo_path, rig_path, motion_path, export_path, animations, model,
                    body_type="humanoid"):
    jobs_path = APP_ROOT / "jobs.json"
    jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else []
    job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().int % 1000000:06d}"
    proj = APP_ROOT / "projects" / job_id
    for stage, src in (("retopo", retopo_path), ("rig", rig_path), ("motion", motion_path),
                       ("export", export_path)):
        dst_dir = proj / stage
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / Path(src).name
        dst.write_bytes(Path(src).read_bytes())
    recon_src = mp.WS / name / "reconstruct" / "hunyuan3d21.glb"
    (proj / "reconstruct").mkdir(parents=True, exist_ok=True)
    (proj / "reconstruct" / "hunyuan3d21.glb").write_bytes(recon_src.read_bytes())
    # 멀티이미지 복원이 채택한 원본이 있으면 그것을 등록한다.
    png_src = mp.WS / name / "reconstruct" / "source.png"
    if not png_src.exists():
        png_src = mp.CHARS / f"{name}.png"
    (proj / "source.png").write_bytes(png_src.read_bytes())

    job = {
        "id": job_id, "name": f"matrix2-{name}",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "status": "complete", "progress": 100,
        "image": str(proj / "source.png"), "imageHash": "",
        "workspace": str(proj),
        "stages": [
            {"id": "prepare", "name": "Image cleanup", "status": "done", "detail": "Reference secured from matrix2 validation"},
            {"id": "reconstruct", "name": "3D reconstruction", "status": "done", "detail": "Completed · RunPod Hunyuan3D-2.1"},
            {"id": "retopo", "name": "Mesh cleanup", "status": "done", "detail": "Completed · local-baseline"},
            {"id": "rig", "name": "Auto rig", "status": "done", "detail": "Completed · auto-rig-bbox (skinned, upright-normalized)"},
            {"id": "motion", "name": "Animation", "status": "done", "detail": f"Completed · RunPod HY-Motion ({animations} clips)"},
            {"id": "export", "name": "Export", "status": "done", "detail": "Completed · passthrough-offline (validated)"},
        ],
        "artifacts": [
            {"stage": "prepare", "kind": "reference", "path": str(proj / "source.png")},
            {"stage": "reconstruct", "kind": "mesh", "path": str(proj / "reconstruct" / "hunyuan3d21.glb"),
             "metrics": {"adapter": "runpod-hunyuan3d21", "textured": True}},
            {"stage": "retopo", "kind": "clean-mesh", "path": str(proj / "retopo" / Path(retopo_path).name)},
            {"stage": "rig", "kind": "rigged-model", "path": str(proj / "rig" / Path(rig_path).name),
             "metrics": {"adapter": "auto-rig-bbox", "skinned": True, "bodyType": body_type}},
            {"stage": "motion", "kind": "animated-model", "path": str(proj / "motion" / Path(motion_path).name),
             "metrics": {"adapter": "hy-motion-retarget", "animations": animations, "model": model}},
            {"stage": "export", "kind": "package", "path": str(proj / "export" / Path(export_path).name),
             "metrics": {"adapter": "passthrough-offline", "validated": True, "previewOnly": True}},
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
    ap.add_argument("--set", choices=["1", "2", "q"], default="2")
    args = ap.parse_args()
    pool = {"1": gen_characters.CHARACTERS, "2": gen_characters.CHARACTERS_SET2,
            "q": gen_characters.CHARACTERS_QUADRUPED}[args.set]
    if args.name not in pool:
        sys.exit(f"unknown character {args.name!r} in set {args.set}: {sorted(pool)}")
    desc = pool[args.name]

    mp.WS.mkdir(parents=True, exist_ok=True)
    pngs = ensure_images(args.name, desc, args.set)
    recon_glb = reconstruct(args.name, desc, pngs, quadruped=(args.set == "q"))
    retopo_path, rig_path, body_type = retopo_and_rig(args.name, recon_glb)
    generate_motion(args.name)
    motion_path, animations, model = bake_and_validate(args.name, rig_path)
    exported = mp.run_worker("export", mp.WS / args.name, motion_path)
    assert exported.get("metrics", {}).get("validated"), exported.get("metrics")
    job_id = register_in_app(args.name, retopo_path, rig_path, motion_path,
                             exported["path"], animations, model, body_type)
    print(f"CHARACTER_OK name={args.name} job={job_id} clips={animations}")


if __name__ == "__main__":
    main()
