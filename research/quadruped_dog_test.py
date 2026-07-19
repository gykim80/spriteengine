#!/usr/bin/env python3
"""4족 보행(quadruped) 리깅 실증 테스트 — 절차적 개 메시로 auto_rig() 한계 실측.

목적: 현재 파이프라인(workers/baseline_worker.py auto_rig)이 휴머노이드
bbox 비율 가정을 4족 동물에 적용하면 무엇이 깨지는지 실측한다.

절차:
  1. 절차적 개 메시 GLB 생성 (Y-up, 머리 +Z, 어깨높이 0.62m, 체장 1.4m)
  2. auto_rig() 실행 → 조인트 월드 배치를 개 해부학과 대조
  3. 부위별(앞다리/뒷다리/머리/꼬리/몸통) 지배 조인트 분포 실측
  4. 합성 걷기 클립 리타겟(bake_animation) 후 validate_character 게이트 실행

사용법: python3 research/quadruped_dog_test.py [--keep]
출력물: /tmp/spriteengine_quadruped/dog.glb, dog-rigged.glb, dog-animated.glb
"""
import importlib.util
import io
import json
import math
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "/tmp/spriteengine_quadruped"
os.makedirs(OUT, exist_ok=True)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# baseline_worker는 모듈 최상단에서 stdin 루프를 돌므로 빈 stdin으로 임포트
_stdin = sys.stdin
sys.stdin = io.StringIO("")
worker = _load_module("baseline_worker", os.path.join(HERE, "..", "workers", "baseline_worker.py"))
sys.stdin = _stdin
validator = _load_module("validate_character", os.path.join(HERE, "..", "tools", "matrix", "validate_character.py"))


# --- 1. 절차적 개 메시 -------------------------------------------------------
# 실제 중형견 비례(리트리버급): 어깨높이 ~0.62m, 체장(가슴~엉덩이) ~0.8m,
# 머리 포함 전장 ~1.4m. Y-up, 머리 +Z (휴머노이드 규약과 동일 시점).
positions = []   # (x, y, z)
part_of = []     # 버텍스 → 해부학 부위 라벨
indices = []


def add_box(part, mn, mx, splits=1, axis=2):
    """축 방향으로 splits등분한 박스를 추가한다 (부위 라벨 기록)."""
    global indices
    lo, hi = mn[axis], mx[axis]
    for s in range(splits):
        a, b = list(mn), list(mx)
        a[axis] = lo + (hi - lo) * s / splits
        b[axis] = lo + (hi - lo) * (s + 1) / splits
        base = len(positions)
        for dx in (a[0], b[0]):
            for dy in (a[1], b[1]):
                for dz in (a[2], b[2]):
                    positions.append((dx, dy, dz))
                    part_of.append(part)
        # 8버텍스 박스의 12삼각형 (인덱스: x*4 + y*2 + z)
        quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
                 (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        for q in quads:
            indices += [base + q[0], base + q[1], base + q[2],
                        base + q[0], base + q[2], base + q[3]]


# 몸통: 수평 박스 (z: -0.40 엉덩이 → +0.40 가슴)
add_box("body", (-0.14, 0.34, -0.40), (0.14, 0.62, 0.40), splits=4)
# 목+머리: 가슴 앞 위쪽
add_box("head", (-0.10, 0.52, 0.40), (0.10, 0.88, 0.66), splits=2)
# 주둥이
add_box("head", (-0.06, 0.58, 0.66), (0.06, 0.74, 0.90), splits=1)
# 꼬리: 엉덩이 뒤 위쪽
add_box("tail", (-0.03, 0.50, -0.72), (0.03, 0.66, -0.40), splits=2)
# 다리 4개: 지면(0)→몸통 밑(0.38), 앞다리 z=+0.30, 뒷다리 z=-0.30
for side, sx in (("L", 1), ("R", -1)):
    add_box(f"leg_front_{side}", (sx * 0.14 - 0.05, 0.0, 0.25),
            (sx * 0.14 + 0.05, 0.38, 0.35), splits=3, axis=1)
    add_box(f"leg_rear_{side}", (sx * 0.14 - 0.05, 0.0, -0.35),
            (sx * 0.14 + 0.05, 0.38, -0.25), splits=3, axis=1)

# GLB 조립 (정적 메시, 스킨 없음 — reconstruct 단계 출력과 동일 조건)
bin_data = bytearray()
gltf = {"asset": {"version": "2.0", "generator": "quadruped-dog-test"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "Dog", "mesh": 0}],
        "buffers": [{"byteLength": 0}], "bufferViews": [], "accessors": []}
pos_blob = b"".join(struct.pack("<fff", *p) for p in positions)
pv = worker._append_view(gltf, bin_data, pos_blob)
mins = [min(p[k] for p in positions) for k in range(3)]
maxs = [max(p[k] for p in positions) for k in range(3)]
gltf["accessors"].append({"bufferView": pv, "componentType": 5126,
                          "count": len(positions), "type": "VEC3",
                          "min": mins, "max": maxs})
idx_blob = b"".join(struct.pack("<H", i) for i in indices)
iv = worker._append_view(gltf, bin_data, idx_blob)
gltf["accessors"].append({"bufferView": iv, "componentType": 5123,
                          "count": len(indices), "type": "SCALAR"})
gltf["meshes"] = [{"name": "DogMesh", "primitives": [
    {"attributes": {"POSITION": 0}, "indices": 1}]}]

dog_glb = os.path.join(OUT, "dog.glb")
worker._write_glb(gltf, bin_data, dog_glb)
print(f"[1] 개 메시 생성: {dog_glb}")
print(f"    버텍스 {len(positions)} / 삼각형 {len(indices)//3}")
print(f"    치수: 폭(X) {maxs[0]-mins[0]:.2f}m × 높이(Y) {maxs[1]-mins[1]:.2f}m × 전장(Z) {maxs[2]-mins[2]:.2f}m")

# --- 2. auto_rig 실행 --------------------------------------------------------
rigged_glb = os.path.join(OUT, "dog-rigged.glb")
ok = worker.auto_rig(dog_glb, rigged_glb)
print(f"\n[2] auto_rig() 실행: {'성공(스킨 생성됨)' if ok else '실패(패스스루)'}")

rg, rbin = worker._read_glb(rigged_glb)
node_by_name = {n.get("name"): i for i, n in enumerate(rg.get("nodes", []))}
world = worker._rig_rest_world(rg, node_by_name)

# 스케일 정규화 배율 역산 (개 높이 0.88 < 1.2 → 표준 키로 확대됨)
rpos = worker._read_vec3(rg, rbin, 0, "POSITION")
new_h = max(p[1] for p in rpos) - min(p[1] for p in rpos)
scale = new_h / (maxs[1] - mins[1])
print(f"    스케일 정규화: {maxs[1]-mins[1]:.2f}m 개 → {new_h:.2f}m ({scale:.2f}배, 휴머노이드 표준 키 강제)")

print("\n    휴머노이드 조인트가 개 몸의 어디에 박혔는가 (개 원본 좌표계, m):")
ANATOMY = {  # 개 해부학 기준 실제 있어야 할 위치 설명
    "Hips": "골반(엉덩이, y0.48 z-0.30 부근)", "Spine": "등 중앙", "Chest": "가슴(어깨, z+0.30)",
    "Head": "머리(y0.70 z+0.55)", "LeftArm": "왼앞다리 상부", "LeftForeArm": "왼앞발",
    "LeftUpLeg": "왼뒷다리 상부", "LeftFoot": "왼뒷발(z-0.30)",
}
for name in ("Hips", "Spine", "Chest", "Head", "LeftArm", "LeftForeArm",
             "LeftUpLeg", "LeftLeg", "LeftFoot"):
    w = world.get(name)
    if not w:
        continue
    ox, oy, oz = (w[0] / scale, w[1] / scale, w[2] / scale)
    note = ANATOMY.get(name, "")
    print(f"      {name:13s} → ({ox:+.2f}, {oy:.2f}, {oz:+.2f})" + (f"   기대: {note}" if note else ""))

# --- 3. 부위별 지배 조인트 분포 ----------------------------------------------
prim = rg["meshes"][0]["primitives"][0]
attrs = prim["attributes"]
jnames = [rg["nodes"][j].get("name") for j in rg["skins"][0]["joints"]]


def read_vec4(acc_index, fmts):
    acc = rg["accessors"][acc_index]
    view = rg["bufferViews"][acc["bufferView"]]
    base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    fmt = fmts[acc["componentType"]]
    stride = view.get("byteStride", struct.calcsize(fmt) * 4)
    return [struct.unpack_from(f"<4{fmt}", rbin, base + i * stride)
            for i in range(acc["count"])]


vj = read_vec4(attrs["JOINTS_0"], {5121: "B", 5123: "H", 5125: "I"})
vw = read_vec4(attrs["WEIGHTS_0"], {5126: "f"})
dist = {}
for i, part in enumerate(part_of):
    slot = max(range(4), key=lambda s: vw[i][s])
    dom = jnames[vj[i][slot]]
    dist.setdefault(part, {}).setdefault(dom, 0)
    dist[part][dom] += 1

print("\n[3] 부위별 지배 조인트 분포 (버텍스 수):")
for part in ("body", "head", "tail", "leg_front_L", "leg_front_R", "leg_rear_L", "leg_rear_R"):
    d = sorted(dist.get(part, {}).items(), key=lambda kv: -kv[1])
    total = sum(c for _, c in d)
    tops = ", ".join(f"{n} {c}/{total}" for n, c in d[:3])
    print(f"      {part:13s} → {tops}")

# --- 4. 합성 걷기 클립 리타겟 + 검증 게이트 -----------------------------------
# HY-Motion이 반환하는 형식과 동일한 SMPL-H 22조인트 로컬 쿼터니언 걷기 합성:
# 다리 X축 스윙 ±28°, 팔 반대 위상 ±20°, 루트 +Z 전진 1.2m/2s.
SMPL_JOINTS = ["Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
               "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
               "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
               "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"]


def qx(deg):
    r = math.radians(deg) / 2
    return [math.sin(r), 0.0, 0.0, math.cos(r)]


frames, fps = 40, 20.0
quats, trans = [], []
for f in range(frames):
    ph = math.sin(2 * math.pi * f / frames * 2)  # 2보폭
    frame_q = [[0.0, 0.0, 0.0, 1.0] for _ in SMPL_JOINTS]
    ji = {n: k for k, n in enumerate(SMPL_JOINTS)}
    frame_q[ji["L_Hip"]] = qx(28 * ph)
    frame_q[ji["R_Hip"]] = qx(-28 * ph)
    frame_q[ji["L_Knee"]] = qx(max(0.0, -20 * ph))
    frame_q[ji["R_Knee"]] = qx(max(0.0, 20 * ph))
    frame_q[ji["L_Shoulder"]] = qx(-20 * ph)
    frame_q[ji["R_Shoulder"]] = qx(20 * ph)
    quats.append(frame_q)
    trans.append([0.0, 0.0, 1.2 * f / (frames - 1)])

payload = {"model": "synthetic-walk", "motions": [
    {"id": "walk", "fps": fps, "joints": SMPL_JOINTS, "quats": quats, "trans": trans}]}
animated_glb = os.path.join(OUT, "dog-animated.glb")
baked = worker.bake_animation(rigged_glb, payload, animated_glb)
print(f"\n[4] 휴머노이드 걷기 리타겟: {baked}클립 베이킹 → {animated_glb}")

report = validator.validate(animated_glb if baked else rigged_glb)
print(f"\n[5] 렌더링 정상성 게이트(validate_character): {'PASS' if report['ok'] else 'FAIL'}")
for section in ("upright", "hierarchy", "deformation", "arm_pose", "skinning"):
    r = report[section]
    print(f"      {section:12s}: {'ok' if r['ok'] else 'FAIL'}")
    for issue in r.get("issues", []):
        print(f"        - {issue}")
sk = report["skinning"]
if sk.get("edge_stretch_p99") is not None:
    print(f"      (엣지 신장률 p99={sk['edge_stretch_p99']}, max={sk['edge_stretch_max']}, "
          f"한계 {validator.MAX_EDGE_STRETCH_P99})")

print("\n결론 요약은 stdout 리포트 참고. 산출물:", OUT)
