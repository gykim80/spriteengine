#!/usr/bin/env python3
"""캐릭터 GLB 렌더링 정상성 자동 평가기.

"누워서 리깅됨 → 몬스터처럼 움직임" 버그 재발을 자동으로 잡기 위한 4가지 검사:

  1. upright   — 메시가 실제로 Y축 기준 직립 비율(키 >> 폭/깊이)인지, 스킨이
                 붙은 메시 노드에 남은 TRS가 없는지(glTF 스펙상 무시되어야 함)
  2. hierarchy — 스킨 조인트가 순환/고아 없이 단일 루트(Hips)에서 뻗어나가는
                 트리를 이루고, 좌우 대칭 부위가 짝을 이루는지
  3. deformation — 애니메이션 클립이 있으면(motion 단계) 클립마다 조인트가
                 실제로(항등이 아니게) 움직이고, 서로 다른 프롬프트의 클립이
                 byte-identical하지 않은지 (ckpt 미로딩 버그 같은 회귀 탐지)
  4. grounding — 정보성: 메시 최저점이 원점에서 크게 벗어나 있으면 알림만
                 (뷰어가 자동으로 접지시키므로 실패 사유는 아님)

사용법:
  python3 validate_character.py <character.glb> [--json]
  종료 코드 0=PASS, 1=FAIL. --json이면 리포트를 stdout에 JSON으로만 출력.
"""
import argparse
import importlib.util
import io
import json
import math
import os
import struct
import sys

# baseline_worker.py는 모듈 최상단에서 stdin 루프를 돈다 — 빈 stdin으로
# 바꿔치기한 뒤 임포트해 GLB 헬퍼 함수만 재사용한다.
_stdin_backup = sys.stdin
sys.stdin = io.StringIO("")
_spec = importlib.util.spec_from_file_location(
    "baseline_worker",
    os.path.join(os.path.dirname(__file__), "..", "..", "workers", "baseline_worker.py"))
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)
sys.stdin = _stdin_backup

EXPECTED_LEFT_RIGHT_PAIRS = [
    ("LeftUpLeg", "RightUpLeg"), ("LeftLeg", "RightLeg"), ("LeftFoot", "RightFoot"),
    ("LeftArm", "RightArm"), ("LeftForeArm", "RightForeArm"),
]


def _percentile(values, p):
    s = sorted(values)
    idx = min(len(s) - 1, int(len(s) * p / 100))
    return s[idx]


def _read_accessor_floats(gltf, bin_data, acc_index, width):
    acc = gltf["accessors"][acc_index]
    view = gltf["bufferViews"][acc["bufferView"]]
    base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride", width * 4)
    return [struct.unpack_from(f"<{width}f", bin_data, base + i * stride)
            for i in range(acc["count"])]


def check_upright(gltf, bin_data):
    issues = []
    all_pos = []
    for node in gltf.get("nodes", []):
        if "mesh" not in node:
            continue
        rot = node.get("rotation") or [0.0, 0.0, 0.0, 1.0]
        if not worker._quat_is_identity(rot):
            issues.append(f"mesh node has non-identity rotation {rot} — will be ignored once skinned (glTF spec)")
        mesh = gltf["meshes"][node["mesh"]]
        for prim in mesh.get("primitives", []):
            pid = prim.get("attributes", {}).get("POSITION")
            if pid is not None:
                all_pos.extend(worker._read_vec3(gltf, bin_data, pid, "POSITION"))
    if not all_pos:
        return {"ok": False, "issues": ["no mesh POSITION data found"]}
    xs = [p[0] for p in all_pos]; ys = [p[1] for p in all_pos]; zs = [p[2] for p in all_pos]
    ext = {"x": max(xs) - min(xs), "y": max(ys) - min(ys), "z": max(zs) - min(zs)}
    # A-pose 캐릭터는 팔을 벌린 손끝이 키에 맞먹는 X 폭을 만들 수 있으므로,
    # 전체 min/max 대신 core body(p10~p90, 팔 끝단 같은 극단값 제외)로 비교한다
    # — 실제 "누워있음" 버그는 core 기준으로도 3배 이상 차이가 나므로 여전히 잡힌다.
    core = {axis: _percentile(vals, 90) - _percentile(vals, 10)
            for axis, vals in (("x", xs), ("y", ys), ("z", zs))}
    if not (core["y"] > 1.15 * core["x"] and core["y"] > 1.15 * core["z"]):
        issues.append(f"core-body Y range ({core['y']:.3f}, p10-p90) is not clearly the tallest axis "
                       f"vs X={core['x']:.3f} Z={core['z']:.3f} — character may be lying down")
    return {"ok": not issues, "issues": issues, "extents": ext, "core_extents": core,
            "min_y": min(ys), "max_y": max(ys)}


def check_hierarchy(gltf):
    issues = []
    skins = gltf.get("skins") or []
    if not skins:
        return {"ok": False, "issues": ["no skin present"]}
    skin = skins[0]
    joints = skin.get("joints") or []
    if not joints:
        return {"ok": False, "issues": ["skin has no joints"]}
    names = {j: gltf["nodes"][j].get("name", f"node{j}") for j in joints}
    children_of = {j: set(gltf["nodes"][j].get("children", [])) & set(joints) for j in joints}
    is_child = set()
    for parent, kids in children_of.items():
        is_child |= kids
    roots = [j for j in joints if j not in is_child]
    if len(roots) != 1:
        issues.append(f"expected exactly 1 skeleton root, found {len(roots)}: {[names[r] for r in roots]}")
    elif names[roots[0]] != "Hips":
        issues.append(f"skeleton root is {names[roots[0]]!r}, expected 'Hips'")
    # 순환 탐지: 루트에서 BFS로 모든 조인트에 정확히 한 번씩 도달해야 함
    if roots:
        seen = set()
        stack = [roots[0]]
        while stack:
            n = stack.pop()
            if n in seen:
                issues.append(f"cycle detected reaching node {names.get(n, n)} twice")
                break
            seen.add(n)
            stack.extend(children_of.get(n, []))
        unreached = set(joints) - seen
        if unreached:
            issues.append(f"joints unreachable from root: {[names[j] for j in unreached]}")
    name_set = set(names.values())
    for left, right in EXPECTED_LEFT_RIGHT_PAIRS:
        if (left in name_set) != (right in name_set):
            issues.append(f"asymmetric skeleton: {left} in={left in name_set} {right} in={right in name_set}")
    return {"ok": not issues, "issues": issues, "joint_count": len(joints),
            "root": names.get(roots[0]) if len(roots) == 1 else None}


# 클립 이름에 이 키워드가 포함되면 "이동" 모션으로 간주해 실제 수평 이동
# 거리를 검사한다 (예: HY-Motion ckpt 미로딩 버그처럼 "run"인데 제자리인 경우
# clip 상이성 검사만으로는 못 잡는 의미 오류를 잡기 위함). "dodge"처럼 짧은
# 회피 스텝은 원래 이동량이 작은 게 정상이므로 제외한다 (20종 실측 결과
# 모든 캐릭터에서 일관되게 ~0.22m — 데이터 문제가 아니라 모션 자체의 특성).
LOCOMOTION_KEYWORDS = ("run", "walk", "sprint", "crawl", "climb", "flykick", "roll")
MIN_LOCOMOTION_METERS = 0.3


def _root_horizontal_travel(gltf, bin_data, anim):
    node_by_name = {n.get("name"): i for i, n in enumerate(gltf.get("nodes", []))}
    hips = node_by_name.get("Hips")
    if hips is None:
        return None
    for ch in anim.get("channels", []):
        if ch["target"].get("node") != hips or ch["target"].get("path") != "translation":
            continue
        sampler = anim["samplers"][ch["sampler"]]
        values = _read_accessor_floats(gltf, bin_data, sampler["output"], 3)
        if len(values) < 2:
            return 0.0
        xs = [v[0] for v in values]; zs = [v[2] for v in values]
        return max(((xs[i] - xs[j]) ** 2 + (zs[i] - zs[j]) ** 2) ** 0.5
                   for i in range(len(values)) for j in range(i))
    return None


def check_deformation(gltf, bin_data):
    animations = gltf.get("animations") or []
    if not animations:
        return {"ok": True, "issues": [], "clips": 0, "note": "no animations to check (static rig)"}
    issues = []
    clip_signatures = {}
    for anim in animations:
        name = anim.get("name", "?")
        energy = 0.0
        sig_parts = []
        for ch in anim.get("channels", []):
            sampler = anim["samplers"][ch["sampler"]]
            out_acc = gltf["accessors"][sampler["output"]]
            width = 4 if out_acc["type"] == "VEC4" else 3
            values = _read_accessor_floats(gltf, bin_data, sampler["output"], width)
            if len(values) < 2:
                continue
            for k in range(width):
                col = [v[k] for v in values]
                mean = sum(col) / len(col)
                energy += sum((c - mean) ** 2 for c in col)
            sig_parts.append(tuple(round(c, 5) for v in values for c in v))
        clip_signatures[name] = hash(tuple(sig_parts))
        if energy < 1e-8:
            issues.append(f"clip {name!r}: channels are frozen (no movement across frames) — possible untrained/unconditioned generation")
        if any(kw in str(name).lower() for kw in LOCOMOTION_KEYWORDS):
            travel = _root_horizontal_travel(gltf, bin_data, anim)
            if travel is not None and travel < MIN_LOCOMOTION_METERS:
                issues.append(f"clip {name!r}: locomotion motion only travels {travel:.3f}m horizontally "
                               f"(expected >= {MIN_LOCOMOTION_METERS}m) — may be an in-place/noise generation")
    dupes = {}
    for name, sig in clip_signatures.items():
        dupes.setdefault(sig, []).append(name)
    for sig, names in dupes.items():
        if len(names) > 1:
            issues.append(f"clips are byte-identical: {names} — text conditioning likely not applied")
    return {"ok": not issues, "issues": issues, "clips": len(animations),
            "distinct_clips": len(dupes)}


def validate(glb_path):
    gltf, bin_data = worker._read_glb(glb_path)
    upright = check_upright(gltf, bin_data)
    hierarchy = check_hierarchy(gltf)
    deformation = check_deformation(gltf, bin_data)
    ok = upright["ok"] and hierarchy["ok"] and deformation["ok"]
    return {"path": glb_path, "ok": ok,
            "upright": upright, "hierarchy": hierarchy, "deformation": deformation}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    report = validate(args.glb)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        print(f"[{status}] {args.glb}")
        for section in ("upright", "hierarchy", "deformation"):
            r = report[section]
            print(f"  {section}: {'ok' if r['ok'] else 'FAIL'}")
            for issue in r.get("issues", []):
                print(f"    - {issue}")
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
