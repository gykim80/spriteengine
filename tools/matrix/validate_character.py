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
  5. arm_pose  — 좌우 명명이 SMPL/glTF 규약(+X=Left)인지, 그리고 클립들을
                 FK로 실측한 팔(Arm→ForeArm) elevation 중앙값이 아래를 향하는지
                 ("팔이 위로 꺾여 머리 옆에 붙음" 미러 리깅 버그 재발 탐지)
  6. skinning  — 클립 프레임들을 실제 LBS(linear blend skinning)로 변형해
                 메시 엣지 신장률을 실측 — 겨드랑이/팔-몸통 사이가 막(웨빙)처럼
                 늘어나는 웨이트 배정 오류를 정량 탐지

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
# 바꿔치기한 뒤 임포트해 GLB 헬퍼 함수만 재사용한다. repo 체크아웃에서는
# ../../workers/에, 앱 임베드 추출(runtime/) 환경에서는 형제 경로에 있다.
_here = os.path.dirname(os.path.abspath(__file__))
_worker_path = next(p for p in (os.path.join(_here, "..", "..", "workers", "baseline_worker.py"),
                                os.path.join(_here, "baseline_worker.py"))
                    if os.path.exists(p))
_stdin_backup = sys.stdin
sys.stdin = io.StringIO("")
_spec = importlib.util.spec_from_file_location("baseline_worker", _worker_path)
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


def _rig_type(gltf):
    """auto_rig가 skin 이름에 남긴 체형 마커 — 4족이면 upright 기준이 다르다."""
    skins = gltf.get("skins") or []
    name = skins[0].get("name", "") if skins else ""
    return "quadruped" if name == "AutoQuadrupedRig" else "humanoid"


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
    rig_type = _rig_type(gltf)
    if rig_type == "quadruped":
        # 4족 정상 자세: 체장(Z)이 가장 긴 축, 좌우 폭(X)보다 명확히 길어야 함.
        if not (core["z"] >= core["y"] and core["z"] > 1.15 * core["x"]):
            issues.append(f"quadruped core-body Z length ({core['z']:.3f}, p10-p90) is not the longest axis "
                           f"vs X={core['x']:.3f} Y={core['y']:.3f} — animal may be tipped over")
        # 평면 카드 검출은 체장 대비 나머지 두 축으로 판정
        depth_ratio = min(core["x"], core["y"]) / core["z"] if core["z"] > 0 else 0.0
    else:
        if not (core["y"] > 1.15 * core["x"] and core["y"] > 1.15 * core["z"]):
            issues.append(f"core-body Y range ({core['y']:.3f}, p10-p90) is not clearly the tallest axis "
                           f"vs X={core['x']:.3f} Z={core['z']:.3f} — character may be lying down")
        # 평면 카드/부조(relief) 복원 실패 검출: 실측(20종)에서 정상 캐릭터의
        # 최소 수평축/키 비율은 >= 0.125, 카드로 복원된 실패 사례는 0.007이었다
        # — 0.06이면 2배 마진으로 안전하게 분리된다.
        depth_ratio = min(core["x"], core["z"]) / core["y"] if core["y"] > 0 else 0.0
    if depth_ratio < 0.06:
        issues.append(f"core depth ratio {depth_ratio:.3f} (min secondary axis / main axis) < 0.06 "
                       f"— mesh is a flat card/relief, reconstruction likely failed")
    return {"ok": not issues, "issues": issues, "extents": ext, "core_extents": core,
            "rig_type": rig_type, "depth_ratio": depth_ratio, "min_y": min(ys), "max_y": max(ys)}


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
    # 휴머노이드 비례 게이트: 팔을 벌린 전체 mesh bbox를 어깨 폭으로 오인하면
    # Arm 조인트 간격이 신장의 절반 이상이 되는 회귀를 수치로 차단한다.
    # 성인/스타일라이즈드 캐릭터를 넉넉히 포괄하는 범위는 신장 대비 0.18~0.36.
    if skin.get("name") != "AutoQuadrupedRig":
        by_name = {gltf["nodes"][j].get("name"): j for j in joints}
        if all(n in by_name for n in ("Hips", "Head", "LeftArm", "RightArm")):
            parent = {}
            for i, node in enumerate(gltf.get("nodes", [])):
                for child in node.get("children", []):
                    parent[child] = i

            def rest_pos(idx):
                p = [0.0, 0.0, 0.0]
                seen_nodes = set()
                while idx is not None and idx not in seen_nodes:
                    seen_nodes.add(idx)
                    t = gltf["nodes"][idx].get("translation", (0.0, 0.0, 0.0))
                    p = [p[k] + t[k] for k in range(3)]
                    idx = parent.get(idx)
                return p

            left = rest_pos(by_name["LeftArm"]); right = rest_pos(by_name["RightArm"])
            hips = rest_pos(by_name["Hips"]); head = rest_pos(by_name["Head"])
            shoulder_width = math.sqrt(sum((left[k] - right[k]) ** 2 for k in range(3)))
            torso_height = abs(head[1] - hips[1])
            ratio = shoulder_width / torso_height if torso_height > 1e-8 else float("inf")
            if not 0.35 <= ratio <= 0.90:
                issues.append(f"implausible shoulder rig width: shoulder/hips-to-head ratio {ratio:.3f} "
                              f"(expected 0.35..0.90) — full arm-span may have been mistaken for shoulder width")
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


# 4족 다리 개수 게이트: 옆모습 원본 이미지는 반대편 다리가 가려져 3D 복원에서
# 다리가 소실/융합될 수 있다 (실측: 측면 시바견 → 다리 결손 리포트). 접지
# 슬라이스(최저점 위 15% 높이)를 XZ 평면에서 반경 클러스터링해 유의미한 다리
# 기둥이 4개 이상이고 전후×좌우 4사분면을 모두 덮는지, 그리고 특정 다리가
# 퇴화(버텍스 질량이 중앙값의 30% 미만)하지 않았는지 실측한다.
LEG_GROUND_BAND = 0.15
LEG_CLUSTER_RADIUS = 0.08   # × max(체장, 폭)
LEG_SAMPLE_MAX = 500
LEG_MIN_MASS_RATIO = 0.3


def check_legs(gltf, bin_data):
    if _rig_type(gltf) != "quadruped":
        return {"ok": True, "issues": [], "note": "humanoid rig — quadruped leg check skipped"}
    all_pos = []
    for node in gltf.get("nodes", []):
        if "mesh" not in node:
            continue
        for prim in gltf["meshes"][node["mesh"]].get("primitives", []):
            pid = prim.get("attributes", {}).get("POSITION")
            if pid is not None:
                all_pos.extend(worker._read_vec3(gltf, bin_data, pid, "POSITION"))
    if not all_pos:
        return {"ok": False, "issues": ["no mesh POSITION data found"]}
    xs = [p[0] for p in all_pos]; ys = [p[1] for p in all_pos]; zs = [p[2] for p in all_pos]
    min_y = min(ys); h = (max(ys) - min_y) or 1e-9
    xc = (max(xs) + min(xs)) / 2.0; zc = (max(zs) + min(zs)) / 2.0
    pts = [(p[0], p[2]) for p in all_pos if p[1] < min_y + LEG_GROUND_BAND * h]
    step = max(1, len(pts) // LEG_SAMPLE_MAX)
    pts = pts[::step]
    if len(pts) < 4:
        return {"ok": False, "issues": [f"only {len(pts)} ground-contact vertices — mesh does not reach the ground"]}
    radius = LEG_CLUSTER_RADIUS * max(max(xs) - min(xs), max(zs) - min(zs))
    sig_min = max(3, len(pts) // 50)
    legs = []
    for comp in _cluster_xz(pts, radius):
        if len(comp) < sig_min:
            continue
        cx = sum(p[0] for p in comp) / len(comp)
        cz = sum(p[1] for p in comp) / len(comp)
        legs.append({"verts": len(comp), "x": round(cx, 3), "z": round(cz, 3)})
    issues = []
    quadrants = {(leg["x"] > xc, leg["z"] > zc) for leg in legs}
    if len(legs) < 4:
        issues.append(f"only {len(legs)} ground-contact leg column(s) found (expected 4) "
                      f"— reconstruction likely lost/merged occluded legs")
    elif len(quadrants) < 4:
        issues.append(f"leg columns cover only {len(quadrants)}/4 front-rear x left-right quadrants "
                      f"— legs merged or missing on one side")
    else:
        masses = sorted(leg["verts"] for leg in legs)
        median = masses[len(masses) // 2]
        thin = [leg for leg in legs if leg["verts"] < LEG_MIN_MASS_RATIO * median]
        if thin:
            issues.append(f"degenerate thin leg(s) {thin} — vertex mass under {LEG_MIN_MASS_RATIO:.0%} "
                          f"of median leg ({median}), occluded-side leg poorly reconstructed")
    return {"ok": not issues, "issues": issues, "legs": legs, "ground_verts": len(pts)}


def _cluster_xz(pts, radius):
    """XZ 평면 반경 클러스터링 — [(x,z), ...] → 클러스터별 점 리스트."""
    r2 = radius * radius
    unvisited = set(range(len(pts)))
    comps = []
    while unvisited:
        seed = unvisited.pop()
        comp = [seed]; frontier = [seed]
        while frontier:
            i = frontier.pop()
            near = [j for j in unvisited
                    if (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2 <= r2]
            unvisited -= set(near)
            comp.extend(near); frontier.extend(near)
        comps.append([pts[i] for i in comp])
    return comps


# 단일 이미지 복원은 가려진 쪽 다리를 웹(막)/융합 형태로 환각하기 쉽다
# (실측: 3/4 시점 시바견 → 앞다리쌍이 30% 높이에서 웹으로 붙음). 절대
# 임계값으로는 양불 판정이 어려우므로, 시드별 복원 후보를 상대 비교해
# 최적을 고르는 점수를 제공한다.
LEG_SEP_BAND = 0.05      # pair separation 스캔 밴드 높이 (h 비율)
LEG_SEP_SCAN_TOP = 0.5   # 지면에서 50% 높이까지만 스캔


def leg_quality(glb_path):
    """4족 복원 후보의 다리 품질 점수 (상대 비교용, 높을수록 좋음).

    리깅 전 원시 메시에도 동작한다(방향 정규화 전이므로 몸 길이 축을
    수평 범위가 더 긴 축으로 자동 판정).
      - 접지 다리 기둥 4개 미만 또는 4사분면 미커버 → score=-1.0 (실격)
      - score = front_sep + rear_sep + 0.5 * balance
        · pair separation: 전/후 다리쌍의 좌/우가 별도 기둥으로 유지되는
          최대 높이 비율 — 웹/융합으로 뭉개진 복원은 낮은 높이에서 붙는다
        · balance: 접지 기둥 질량 min/median — 퇴화 다리가 있으면 낮다
    """
    gltf, bin_data = worker._read_glb(glb_path)
    all_pos = []
    for node in gltf.get("nodes", []):
        if "mesh" not in node:
            continue
        # 원시 Hunyuan 복원은 Z-up 정점 + 노드 회전(+90° X)으로 Y-up을 만든다.
        # 스킨 메시는 glTF 스펙상 노드 TRS가 무시되므로 비스킨 노드에만 적용.
        rot = node.get("rotation") if "skin" not in node else None
        scale = node.get("scale") if "skin" not in node else None
        trans = node.get("translation") if "skin" not in node else None
        for prim in gltf["meshes"][node["mesh"]].get("primitives", []):
            pid = prim.get("attributes", {}).get("POSITION")
            if pid is None:
                continue
            for p in worker._read_vec3(gltf, bin_data, pid, "POSITION"):
                if scale:
                    p = (p[0] * scale[0], p[1] * scale[1], p[2] * scale[2])
                if rot and not worker._quat_is_identity(rot):
                    p = worker._quat_rotate_vec3(rot, p)
                if trans:
                    p = (p[0] + trans[0], p[1] + trans[1], p[2] + trans[2])
                all_pos.append(p)
    if not all_pos:
        return {"score": -1.0, "reason": "no mesh POSITION data"}
    xs = [p[0] for p in all_pos]; ys = [p[1] for p in all_pos]; zs = [p[2] for p in all_pos]
    min_y = min(ys); h = (max(ys) - min_y) or 1e-9
    xc = (max(xs) + min(xs)) / 2.0; zc = (max(zs) + min(zs)) / 2.0
    radius = LEG_CLUSTER_RADIUS * max(max(xs) - min(xs), max(zs) - min(zs))

    # 1) 접지 기둥 — check_legs와 동일 기준 (pts는 (x, z)로 저장)
    ground = [(p[0], p[2]) for p in all_pos if p[1] < min_y + LEG_GROUND_BAND * h]
    step = max(1, len(ground) // LEG_SAMPLE_MAX)
    ground = ground[::step]
    if len(ground) < 4:
        return {"score": -1.0, "reason": "mesh does not reach the ground"}
    cols = [c for c in _cluster_xz(ground, radius)
            if len(c) >= max(3, len(ground) // 50)]
    quadrants = {(sum(p[0] for p in c) / len(c) > xc, sum(p[1] for p in c) / len(c) > zc)
                 for c in cols}
    if len(cols) < 4 or len(quadrants) < 4:
        return {"score": -1.0, "columns": len(cols), "quadrants": len(quadrants),
                "reason": "missing/merged ground leg columns"}
    masses = sorted(len(c) for c in cols)
    balance = masses[0] / masses[len(masses) // 2]

    # 2) pair separation — 몸 길이 축(수평 범위가 긴 축)으로 전/후를 나눈다
    li, lc = ((1, zc) if (max(zs) - min(zs)) >= (max(xs) - min(xs)) else (0, xc))
    sep = {}
    for front, key in ((True, "front"), (False, "rear")):
        top = 0.0
        for band in range(int(LEG_SEP_SCAN_TOP / LEG_SEP_BAND)):
            y0 = min_y + h * LEG_SEP_BAND * band
            y1 = min_y + h * LEG_SEP_BAND * (band + 1)
            pts = [(p[0], p[2]) for p in all_pos
                   if y0 <= p[1] < y1 and ((p[2] if li == 1 else p[0]) > lc) == front]
            stp = max(1, len(pts) // 300)
            pts = pts[::stp]
            if len(pts) < 6:
                continue  # 데이터가 희소한 밴드는 판단 보류 (합성 픽스처 대응)
            sig = max(2, len(pts) // 30)
            n = sum(1 for c in _cluster_xz(pts, radius) if len(c) >= sig)
            if n >= 2:
                top = LEG_SEP_BAND * (band + 1)
            elif band >= 1:
                break  # 다리가 붙기 시작한 높이 위쪽은 스캔 불필요
        sep[key] = round(top, 3)
    score = sep["front"] + sep["rear"] + 0.5 * balance
    return {"score": round(score, 3), "columns": len(cols), "quadrants": len(quadrants),
            "balance": round(balance, 3), "front_sep": sep["front"], "rear_sep": sep["rear"]}


# 팔 elevation 중앙값 상한(°): 정상 모션 셋은 팔이 대부분 매달림(−60~−85°)이고
# 일시적으로만 올라간다. 미러 리깅 버그는 거의 전 프레임 +70°대로 나타난다.
MAX_MEDIAN_ARM_ELEVATION_DEG = 30.0


def _node_parents(gltf):
    parent = {}
    for i, n in enumerate(gltf.get("nodes", [])):
        for c in n.get("children", []):
            parent[c] = i
    return parent


def _anim_tracks(gltf, bin_data, anim):
    """애니메이션 채널 → {node: {"rotation"/"translation": frames}}와 프레임 수."""
    tracks = {}
    for ch in anim.get("channels", []):
        sampler = anim["samplers"][ch["sampler"]]
        path = ch["target"].get("path")
        if path not in ("rotation", "translation"):
            continue
        width = 4 if path == "rotation" else 3
        tracks.setdefault(ch["target"].get("node"), {})[path] = \
            _read_accessor_floats(gltf, bin_data, sampler["output"], width)
    if not tracks:
        return {}, 0
    return tracks, min(len(v) for t in tracks.values() for v in t.values())


def _fk_world(gltf, tracks, parent, idx, frame, memo):
    """해당 프레임의 노드 월드 (쿼터니언, 위치) — 채널이 없으면 rest TRS 사용."""
    if idx in memo:
        return memo[idx]
    node = gltf["nodes"][idx]
    t = tracks.get(idx, {})
    q = tuple(t["rotation"][frame]) if "rotation" in t else tuple(node.get("rotation", (0, 0, 0, 1)))
    tr = tuple(t["translation"][frame]) if "translation" in t else tuple(node.get("translation", (0, 0, 0)))
    p = parent.get(idx)
    if p is None:
        memo[idx] = (q, tr)
    else:
        pq, pp = _fk_world(gltf, tracks, parent, p, frame, memo)
        rt = worker._quat_rotate_vec3(pq, tr)
        memo[idx] = (worker._quat_mul(pq, q), tuple(pp[k] + rt[k] for k in range(3)))
    return memo[idx]


def check_arm_pose(gltf, bin_data):
    node_by_name = {n.get("name"): i for i, n in enumerate(gltf.get("nodes", []))}
    needed = ("Hips", "LeftArm", "LeftForeArm", "RightArm", "RightForeArm")
    if any(n not in node_by_name for n in needed):
        return {"ok": True, "issues": [], "note": "no humanoid arm joints to check"}
    issues = []
    rest = worker._rig_rest_world(gltf, node_by_name)
    # 좌우 명명 규약: 캐릭터가 +Z를 향할 때 Left*는 +X 쪽 (SMPL/glTF 휴머노이드).
    # 반대면 SMPL 좌팔 회전이 미러로 적용돼 팔이 위로 접힌다.
    if not (rest["LeftArm"][0] > 0.0 > rest["RightArm"][0]):
        issues.append(f"rig left/right mirrored vs SMPL convention: LeftArm x={rest['LeftArm'][0]:.3f} "
                      f"RightArm x={rest['RightArm'][0]:.3f} (expected Left on +X)")

    parent = _node_parents(gltf)
    elevations = []
    for anim in gltf.get("animations") or []:
        tracks, frames = _anim_tracks(gltf, bin_data, anim)
        if not frames:
            continue
        for frame in sorted({0, frames // 4, frames // 2, 3 * frames // 4, frames - 1}):
            memo = {}
            for side in ("Left", "Right"):
                a = _fk_world(gltf, tracks, parent, node_by_name[side + "Arm"], frame, memo)[1]
                f = _fk_world(gltf, tracks, parent, node_by_name[side + "ForeArm"], frame, memo)[1]
                v = [f[k] - a[k] for k in range(3)]
                n = math.sqrt(sum(c * c for c in v))
                if n > 1e-9:
                    elevations.append(math.degrees(math.asin(max(-1.0, min(1.0, v[1] / n)))))
    median = None
    if elevations:
        median = sorted(elevations)[len(elevations) // 2]
        if median > MAX_MEDIAN_ARM_ELEVATION_DEG:
            issues.append(f"arms point upward: median elevation {median:+.1f}deg across "
                          f"{len(elevations)} samples (limit {MAX_MEDIAN_ARM_ELEVATION_DEG:+.0f}deg) "
                          f"— mirrored rig or retarget regression")
    return {"ok": not issues, "issues": issues,
            "median_elevation_deg": None if median is None else round(median, 1),
            "samples": len(elevations)}


# 엣지 신장률 p99 상한: 웨빙(팔-몸통 사이 막) 버그는 반대편 본에 끌려간
# 버텍스가 엣지를 수십 배로 늘인다. 25종 실측 캘리브레이션 — 정상(팔 방향
# 수정 후) p99 분포 2.8~11.3, 미러 리깅 버그본은 22.8 — 15.0이면 정상 전부
# 통과하면서 심각한 웨이트 붕괴를 잡는다 (경계 사례는 arm_pose 검사가 보완).
MAX_EDGE_STRETCH_P99 = 15.0
SKIN_EDGE_SAMPLES = 1500

# 팔↔다리 교차 웨빙: 순수 거리 기반 웨이트 시절 손 버텍스가 UpLeg과
# 블렌드되어, 팔을 들면 손-허벅지 사이가 막처럼 늘어났다(실측 mechanic
# ForeArm↔UpLeg 과신장 엣지 333건). 체인 인접 제한 수정 후에는 팔 체인과
# 다리 체인 지배 버텍스를 잇는 엣지가 5배 이상 늘어나는 일이 없어야 한다.
ARM_CHAIN = ("LeftArm", "LeftForeArm", "RightArm", "RightForeArm")
LEG_CHAIN = ("LeftUpLeg", "LeftLeg", "LeftFoot",
             "RightUpLeg", "RightLeg", "RightFoot")
CROSS_WEB_RATIO = 5.0
MAX_CROSS_CHAIN_WEB = 10  # 표본 엣지×프레임 관측 건수 상한


def _read_accessor_uints(gltf, bin_data, acc_index, comps):
    acc = gltf["accessors"][acc_index]
    view = gltf["bufferViews"][acc["bufferView"]]
    base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    fmt = {5121: "B", 5123: "H", 5125: "I"}[acc["componentType"]]
    size = struct.calcsize(fmt)
    stride = view.get("byteStride", size * comps)
    return [struct.unpack_from(f"<{comps}{fmt}", bin_data, base + i * stride)
            for i in range(acc["count"])]


def _quat_to_mat3(q):
    x, y, z, w = q
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))


def check_skinning(gltf, bin_data):
    skins = gltf.get("skins") or []
    animations = gltf.get("animations") or []
    if not skins or not animations:
        return {"ok": True, "issues": [], "note": "no skin/animations to check"}
    skin = skins[0]
    joints = skin["joints"]
    ibms = _read_accessor_floats(gltf, bin_data, skin["inverseBindMatrices"], 16)

    prim = next((p for m in gltf.get("meshes", []) for p in m.get("primitives", [])
                 if "JOINTS_0" in p.get("attributes", {})), None)
    if prim is None or prim.get("indices") is None:
        return {"ok": True, "issues": [], "note": "no skinned indexed primitive"}
    attrs = prim["attributes"]
    positions = worker._read_vec3(gltf, bin_data, attrs["POSITION"], "POSITION")
    vjoints = _read_accessor_uints(gltf, bin_data, attrs["JOINTS_0"], 4)
    vweights = _read_accessor_floats(gltf, bin_data, attrs["WEIGHTS_0"], 4)
    indices = [i[0] for i in _read_accessor_uints(gltf, bin_data, prim["indices"], 1)]

    # 삼각형 엣지 표본: 중복 제거 후 결정적 스트라이드로 최대 N개
    edges = set()
    for t in range(0, len(indices) - 2, 3):
        a, b, c = indices[t:t + 3]
        for e in ((a, b), (b, c), (c, a)):
            edges.add((min(e), max(e)))
    edges = sorted(edges)
    step = max(1, len(edges) // SKIN_EDGE_SAMPLES)
    edges = edges[::step][:SKIN_EDGE_SAMPLES]
    rest_len = {}
    for a, b in edges:
        pa, pb = positions[a], positions[b]
        d = math.sqrt(sum((pa[k] - pb[k]) ** 2 for k in range(3)))
        if d > 1e-6:
            rest_len[(a, b)] = d
    verts = sorted({v for e in rest_len for v in e})
    parent = _node_parents(gltf)
    jname = [gltf["nodes"][j].get("name") for j in joints]
    dom = {}
    for vi in verts:
        slot = max(range(4), key=lambda s: vweights[vi][s])
        dom[vi] = jname[vjoints[vi][slot]] if vjoints[vi][slot] < len(jname) else None

    def skin_vertex(mats, vi):
        out = [0.0, 0.0, 0.0]
        for slot in range(4):
            wgt = vweights[vi][slot]
            if wgt <= 0.0:
                continue
            rot, trn = mats[vjoints[vi][slot]]
            v = positions[vi]
            for r in range(3):
                out[r] += wgt * (rot[r][0] * v[0] + rot[r][1] * v[1] + rot[r][2] * v[2] + trn[r])
        return out

    stretches = []
    worst = (0.0, None, None)  # (ratio, clip, edge)
    cross_web = 0
    cross_web_clip = None
    for anim in animations:
        tracks, frames = _anim_tracks(gltf, bin_data, anim)
        if not frames:
            continue
        for frame in sorted({0, frames // 2, frames - 1}):
            memo = {}
            mats = {}
            for ji, node in enumerate(joints):
                q, p = _fk_world(gltf, tracks, parent, node, frame, memo)
                rq = _quat_to_mat3(q)
                m = ibms[ji]  # column-major: 회전 3x3 + 이동(12..14)
                ia = ((m[0], m[4], m[8]), (m[1], m[5], m[9]), (m[2], m[6], m[10]))
                ib = (m[12], m[13], m[14])
                rot = tuple(tuple(sum(rq[r][k] * ia[k][c] for k in range(3)) for c in range(3))
                            for r in range(3))
                trn = tuple(sum(rq[r][k] * ib[k] for k in range(3)) + p[r] for r in range(3))
                mats[ji] = (rot, trn)
            deformed = {vi: skin_vertex(mats, vi) for vi in verts}
            for (a, b), rl in rest_len.items():
                da, db = deformed[a], deformed[b]
                d = math.sqrt(sum((da[k] - db[k]) ** 2 for k in range(3)))
                ratio = d / rl
                stretches.append(ratio)
                if ratio > worst[0]:
                    worst = (ratio, anim.get("name"), (a, b))
                if ratio > CROSS_WEB_RATIO:
                    ra, rb = dom.get(a), dom.get(b)
                    if (ra in ARM_CHAIN and rb in LEG_CHAIN) or \
                       (rb in ARM_CHAIN and ra in LEG_CHAIN):
                        cross_web += 1
                        cross_web_clip = anim.get("name")
    issues = []
    p99 = None
    if stretches:
        p99 = sorted(stretches)[min(len(stretches) - 1, int(len(stretches) * 0.99))]
        if p99 > MAX_EDGE_STRETCH_P99:
            issues.append(f"skinned edges over-stretched: p99 ratio {p99:.2f} (limit {MAX_EDGE_STRETCH_P99}) "
                          f"worst {worst[0]:.2f} in clip {worst[1]!r} — webbing/weight assignment problem")
        if cross_web > MAX_CROSS_CHAIN_WEB:
            issues.append(f"arm-leg cross-chain webbing: {cross_web} stretched(>{CROSS_WEB_RATIO}x) edges "
                          f"between arm/leg dominated vertices (limit {MAX_CROSS_CHAIN_WEB}, "
                          f"e.g. clip {cross_web_clip!r}) — hand vertices blended with leg bones")
    return {"ok": not issues, "issues": issues,
            "edge_stretch_p99": None if p99 is None else round(p99, 3),
            "edge_stretch_max": round(worst[0], 3) if stretches else None,
            "worst_clip": worst[1], "cross_chain_web": cross_web,
            "edges_sampled": len(rest_len)}


# 손 품질/리깅 게이트. 현재 14-joint rig에는 Hand joint가 없으므로 손 형상은
# 전완 지배 버텍스의 말단 클러스터로 평가한다. 좌우 손이 모두 존재하고,
# hand cluster가 충분한 vertex mass/3D thickness를 가지며 ForeArm에 지배되어야 한다.
HAND_MIN_VERTICES = 6
HAND_MIN_THICKNESS_HEIGHT = 0.006


def check_hands(gltf, bin_data):
    if _rig_type(gltf) != "humanoid":
        return {"ok": True, "issues": [], "note": "quadruped rig — hand check skipped"}
    skins = gltf.get("skins") or []
    if not skins:
        return {"ok": False, "issues": ["no skin present"]}
    skin = skins[0]; joints = skin.get("joints") or []
    names = [gltf["nodes"][j].get("name") for j in joints]
    prim = next((p for m in gltf.get("meshes", []) for p in m.get("primitives", [])
                 if "JOINTS_0" in p.get("attributes", {}) and "WEIGHTS_0" in p.get("attributes", {})), None)
    if prim is None:
        return {"ok": False, "issues": ["no skinned primitive for hand validation"]}
    attrs = prim["attributes"]
    pos = worker._read_vec3(gltf, bin_data, attrs["POSITION"], "POSITION")
    # 극저해상도 placeholder/단위 테스트 mesh는 손 자체를 표현할 topology가 없다.
    # 실제 reconstruction 품질 게이트(수천~수만 vertices)에만 형상 판정을 적용한다.
    if len(pos) < 100:
        return {"ok": True, "issues": [], "note": f"only {len(pos)} vertices — hand geometry check skipped"}
    vj = _read_accessor_uints(gltf, bin_data, attrs["JOINTS_0"], 4)
    vw = _read_accessor_floats(gltf, bin_data, attrs["WEIGHTS_0"], 4)
    ys = [p[1] for p in pos]; height = (max(ys) - min(ys)) or 1.0
    node_by_name = {n.get("name"): i for i, n in enumerate(gltf.get("nodes", []))}
    world = worker._rig_rest_world(gltf, node_by_name)
    issues, details = [], {}
    for side in ("Left", "Right"):
        name = side + "ForeArm"
        if name not in names or name not in world:
            issues.append(f"{name} missing — hand cannot be rigged")
            continue
        ji = names.index(name); origin = world[name]
        dominated = []
        for i, p in enumerate(pos):
            slot = max(range(4), key=lambda s: vw[i][s])
            if vj[i][slot] == ji:
                dominated.append(p)
        # ForeArm origin에서 가장 먼 20%를 손 말단으로 간주한다.
        dominated.sort(key=lambda p: -sum((p[k] - origin[k]) ** 2 for k in range(3)))
        hand = dominated[:max(HAND_MIN_VERTICES, len(dominated) // 5)] if dominated else []
        if len(hand) < HAND_MIN_VERTICES:
            issues.append(f"{side} hand has only {len(hand)} ForeArm-dominated vertices "
                          f"(expected >= {HAND_MIN_VERTICES}) — hand may be missing/fused")
            details[side.lower()] = {"vertices": len(hand)}
            continue
        ext = [max(p[k] for p in hand) - min(p[k] for p in hand) for k in range(3)]
        thickness = sorted(ext)[0] / height
        details[side.lower()] = {"vertices": len(hand), "extents": [round(x, 4) for x in ext],
                                 "min_extent_height_ratio": round(thickness, 4)}
        if thickness < HAND_MIN_THICKNESS_HEIGHT:
            issues.append(f"{side} hand is nearly flat: min extent/height {thickness:.4f} "
                          f"(expected >= {HAND_MIN_THICKNESS_HEIGHT}) — malformed hand geometry")
    return {"ok": not issues, "issues": issues, "hands": details}


def validate(glb_path):
    gltf, bin_data = worker._read_glb(glb_path)
    upright = check_upright(gltf, bin_data)
    hierarchy = check_hierarchy(gltf)
    legs = check_legs(gltf, bin_data)
    deformation = check_deformation(gltf, bin_data)
    arm_pose = check_arm_pose(gltf, bin_data)
    skinning = check_skinning(gltf, bin_data)
    hands = check_hands(gltf, bin_data)
    ok = (upright["ok"] and hierarchy["ok"] and legs["ok"] and deformation["ok"]
          and arm_pose["ok"] and skinning["ok"] and hands["ok"])
    return {"path": glb_path, "ok": ok, "upright": upright,
            "hierarchy": hierarchy, "legs": legs, "deformation": deformation,
            "arm_pose": arm_pose, "skinning": skinning, "hands": hands}


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
        for section in ("upright", "hierarchy", "legs", "deformation", "arm_pose", "skinning", "hands"):
            r = report[section]
            print(f"  {section}: {'ok' if r['ok'] else 'FAIL'}")
            for issue in r.get("issues", []):
                print(f"    - {issue}")
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
