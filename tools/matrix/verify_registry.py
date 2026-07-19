#!/usr/bin/env python3
"""앱에 등록된 전체 job을 전수 검증한다 (jobs.json 레지스트리 감사).

이번 세션(2026-07-20)에서 임시 스크립트로 반복 수행한 전수 검증을 영구화:
구형 파이프라인 잔재(플라스틱 광택 재질, 손 결손 rig)가 정상 자산으로
남아 있는 것을 이 감사로 발견했다. 신규 등록·재베이킹 후에는 반드시
이 스크립트로 레지스트리 전체를 재확인한다.

검사 항목 (job당):
  1. 구조 무결성 — status/progress/stages, 스테이지 디렉토리·source.png 존재
  2. 최종 export GLB — 재질 중화(metallic=0, MR 텍스처·specular 확장 없음),
     애니메이션 클립 수(>= 10)
  3. rig GLB — validate_character 전체 게이트
     (upright/hierarchy/legs/deformation/arm_pose/skinning/hands)

사용: python3 verify_registry.py [--rig]   (--rig: rig 게이트까지, 기본은 1+2만)
종료 코드: 0 = 전수 통과, 1 = 실패 존재
"""
import argparse
import json
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_character  # noqa: E402

APP_ROOT = Path.home() / "Library/Application Support/SpriteEngine"
MIN_CLIPS = 10
STAGE_DIRS = ("reconstruct", "retopo", "rig", "motion", "export")


def load_jobs():
    jobs = json.loads((APP_ROOT / "jobs.json").read_text())
    return jobs if isinstance(jobs, list) else jobs.get("jobs", [])


def glb_json(path):
    data = Path(path).read_bytes()
    length, = struct.unpack("<I", data[12:16])
    return json.loads(data[20:20 + length])


def check_structure(job, ws):
    issues = []
    if job.get("status") != "complete" or job.get("progress") != 100:
        issues.append(f"status={job.get('status')} progress={job.get('progress')}")
    for s in job.get("stages", []):
        if s.get("status") != "done":
            issues.append(f"stage {s.get('id')}={s.get('status')}")
    for rel in ("source.png",) + STAGE_DIRS:
        if not (ws / rel).exists():
            issues.append(f"missing {rel}")
    img = job.get("image")
    if img and not os.path.exists(img):
        issues.append("image path broken")
    return issues


def check_export(ws):
    final = ws / "export" / "character-final.glb"
    if not final.exists():
        return ["no character-final.glb"]
    g = glb_json(final)
    issues = []
    for m in g.get("materials", []):
        pbr = m.get("pbrMetallicRoughness", {})
        if pbr.get("metallicFactor", 1.0) != 0.0:
            issues.append("metallicFactor!=0")
        if "metallicRoughnessTexture" in pbr:
            issues.append("metallicRoughnessTexture present")
        if "KHR_materials_specular" in m.get("extensions", {}):
            issues.append("KHR_materials_specular present")
    clips = len(g.get("animations", []))
    if clips < MIN_CLIPS:
        issues.append(f"clips={clips} < {MIN_CLIPS}")
    return sorted(set(issues))


def _validate_glb(path):
    if not path.exists():
        return [f"no {path.name}"]
    rep = validate_character.validate(str(path))
    return [k for k, v in rep.items() if isinstance(v, dict) and not v.get("ok", True)]


def check_rig(ws):
    return _validate_glb(ws / "rig" / "character-rigged.glb")


def check_animated(ws):
    """모션 베이킹 산출물 검증 — rig 통과 후 bake 단계에서 생긴 변형 붕괴를 잡는다."""
    return _validate_glb(ws / "motion" / "character-animated.glb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", action="store_true", help="rig 전체 게이트까지 검사")
    ap.add_argument("--animated", action="store_true",
                    help="모션 베이킹 GLB 게이트까지 검사")
    args = ap.parse_args()

    jobs = load_jobs()
    failures = {}
    for job in jobs:
        name = job.get("name", job.get("id", "?"))
        ws = Path(job.get("workspace") or APP_ROOT / "projects" / job["id"])
        issues = check_structure(job, ws)
        if not any(i.startswith("missing export") for i in issues):
            issues += check_export(ws)
        if args.rig:
            issues += [f"rig:{k}" for k in check_rig(ws)]
        if args.animated:
            issues += [f"animated:{k}" for k in check_animated(ws)]
        if issues:
            failures[name] = issues
        print(f"{name:26s} {'PASS' if not issues else 'FAIL ' + '; '.join(issues)}",
              flush=True)

    total = len(jobs)
    if failures:
        print(f"\nREGISTRY_FAIL {len(failures)}/{total} job(s) with issues")
        sys.exit(1)
    print(f"\nREGISTRY_PASS {total}/{total}")


if __name__ == "__main__":
    main()
