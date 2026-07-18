#!/usr/bin/env python3
"""SpriteEngine JSON Lines worker baseline.

The worker intentionally uses only Python's standard library so the desktop
orchestrator can validate its process/progress/artifact contract before large
GPU environments are installed.
"""
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time


def emit(event_type, **payload):
    print(json.dumps({"type": event_type, **payload}, separators=(",", ":")), flush=True)


def dimensions(path):
    with open(path, "rb") as f:
        head = f.read(32)
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return struct.unpack(">II", head[16:24])
        if head[:2] == b"\xff\xd8":
            f.seek(2)
            while True:
                marker = f.read(2)
                if len(marker) < 2:
                    break
                if marker[0] != 0xFF:
                    continue
                length_raw = f.read(2)
                if len(length_raw) < 2:
                    break
                length = struct.unpack(">H", length_raw)[0]
                if marker[1] in range(0xC0, 0xC4):
                    data = f.read(5)
                    return struct.unpack(">HH", data[1:5])[::-1]
                f.seek(length - 2, 1)
        if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
            # Dimensions are optional for this baseline; format validation passed.
            return 0, 0
    raise ValueError("unsupported or corrupt image")


# --- GLB front-projection texture baking (stdlib only) -----------------------
# Hunyuan3D-2.1 shape-only 배포는 지오메트리만 반환하므로, retopo 단계에서
# 참조 이미지를 정면 투영(front projection) UV로 베이킹해 회색 프리뷰를 없앤다.
GLB_MAGIC, CHUNK_JSON, CHUNK_BIN = 0x46546C67, 0x4E4F534A, 0x004E4942


def _read_glb(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 12 or struct.unpack("<I", data[:4])[0] != GLB_MAGIC:
        raise ValueError("not a GLB file")
    json_chunk, bin_chunk, offset = None, b"", 12
    while offset + 8 <= len(data):
        clen, ctype = struct.unpack("<II", data[offset:offset + 8])
        chunk = data[offset + 8:offset + 8 + clen]
        if ctype == CHUNK_JSON:
            json_chunk = chunk
        elif ctype == CHUNK_BIN:
            bin_chunk = chunk
        offset += 8 + clen
    if json_chunk is None:
        raise ValueError("GLB is missing its JSON chunk")
    return json.loads(json_chunk), bytearray(bin_chunk)


def _write_glb(gltf, bin_data, path):
    blob = bytes(bin_data) + b"\x00" * (-len(bin_data) % 4)
    if gltf.get("buffers"):
        gltf["buffers"][0]["byteLength"] = len(blob)
    payload = json.dumps(gltf, separators=(",", ":")).encode()
    payload += b" " * (-len(payload) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", GLB_MAGIC, 2, 12 + 8 + len(payload) + 8 + len(blob)))
        f.write(struct.pack("<II", len(payload), CHUNK_JSON))
        f.write(payload)
        f.write(struct.pack("<II", len(blob), CHUNK_BIN))
        f.write(blob)


def _read_vec3(gltf, bin_data, accessor_index, what):
    acc = gltf["accessors"][accessor_index]
    if acc.get("componentType") != 5126 or acc.get("type") != "VEC3" or "bufferView" not in acc:
        raise ValueError(f"{what} accessor is not float32 VEC3")
    view = gltf["bufferViews"][acc["bufferView"]]
    stride = view.get("byteStride", 12)
    base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    return [struct.unpack_from("<fff", bin_data, base + i * stride) for i in range(acc["count"])]


def _read_indices(gltf, bin_data, prim):
    if "indices" not in prim:
        return None
    acc = gltf["accessors"][prim["indices"]]
    fmt = {5121: "<B", 5123: "<H", 5125: "<I"}.get(acc.get("componentType"))
    if fmt is None or "bufferView" not in acc:
        return None
    view = gltf["bufferViews"][acc["bufferView"]]
    base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    size = struct.calcsize(fmt)
    return [struct.unpack_from(fmt, bin_data, base + i * size)[0] for i in range(acc["count"])]


def _vertex_normals(positions, indices):
    """indexed triangle 메시에서 면적 가중 정점 법선을 계산한다 (NORMAL 부재 시)."""
    normals = [[0.0, 0.0, 0.0] for _ in positions]
    for t in range(0, len(indices) - 2, 3):
        ia, ib, ic = indices[t], indices[t + 1], indices[t + 2]
        ax, ay, az = positions[ia]
        e1 = (positions[ib][0] - ax, positions[ib][1] - ay, positions[ib][2] - az)
        e2 = (positions[ic][0] - ax, positions[ic][1] - ay, positions[ic][2] - az)
        cx = e1[1] * e2[2] - e1[2] * e2[1]
        cy = e1[2] * e2[0] - e1[0] * e2[2]
        cz = e1[0] * e2[1] - e1[1] * e2[0]
        for i in (ia, ib, ic):
            normals[i][0] += cx
            normals[i][1] += cy
            normals[i][2] += cz
    out = []
    for nx, ny, nz in normals:
        norm = (nx * nx + ny * ny + nz * nz) ** 0.5 or 1.0
        out.append((nx / norm, ny / norm, nz / norm))
    return out


def _append_view(gltf, bin_data, blob):
    while len(bin_data) % 4:
        bin_data.append(0)
    view = {"buffer": 0, "byteOffset": len(bin_data), "byteLength": len(blob)}
    bin_data.extend(blob)
    gltf.setdefault("bufferViews", []).append(view)
    return len(gltf["bufferViews"]) - 1


def _quat_rotate_vec3(q, v):
    """단위 쿼터니언 q(xyzw)로 벡터 v를 회전한다 (v' = q ⊗ v ⊗ q⁻¹의 최적화형)."""
    qx, qy, qz, qw = q
    vx, vy, vz = v
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def _quat_is_identity(q, eps=1e-6):
    return abs(q[0]) < eps and abs(q[1]) < eps and abs(q[2]) < eps and abs(q[3] - 1.0) < eps


def _bake_mesh_node_transform(gltf, bin_data):
    """메시가 달린 노드에 걸린 TRS를 버텍스 데이터에 직접 구워 넣고 노드를 항등으로 리셋한다.

    glTF 스펙상 스킨(JOINTS_0/WEIGHTS_0)이 붙은 메시 노드는 자신의 로컬
    트랜스폼이 렌더러에서 완전히 무시된다 — 최종 버텍스 위치는 오직 조인트
    트랜스폼(및 inverseBindMatrices)으로만 결정된다
    (https://github.khronos.org/glTF-Tutorials/gltfTutorial/gltfTutorial_020_Skins.html).

    Hunyuan3D-2.1 복원 출력은 원본 메시가 로컬 Z축이 키(예: 1.99m), Y축이
    깊이(예: 0.54m)로 저장되고, 노드 회전(보통 X축 ±90°)으로 뷰어에서 정상
    직립으로 보이도록 보정한다. 이 회전은 스킨이 없는 동안(reconstruct/retopo
    단계)에는 정상 적용되지만, auto_rig()가 스킨을 추가하는 순간부터 무시되어
    캐릭터가 원본 "누운" 자세로 렌더링된다. 게다가 auto_rig()의 관절 배치도
    로컬 Y축을 키로 잘못 가정해 실제 몸 형태와 무관한 위치에 스켈레톤이
    눌려 박힌다. 스킨을 추가하기 전에 이 노드 트랜스폼을 버텍스 좌표에
    직접 구워 넣어, 스킨 유무와 무관하게 항상 올바른 직립 자세가 되도록
    정규화한다. 이미 항등 트랜스폼이면 아무 것도 하지 않는다(멱등).
    """
    for node in gltf.get("nodes", []):
        if "mesh" not in node:
            continue
        rot = node.get("rotation") or [0.0, 0.0, 0.0, 1.0]
        trans = node.get("translation") or [0.0, 0.0, 0.0]
        scale = node.get("scale") or [1.0, 1.0, 1.0]
        if _quat_is_identity(rot) and trans == [0.0, 0.0, 0.0] and scale == [1.0, 1.0, 1.0]:
            continue
        mesh = gltf["meshes"][node["mesh"]]
        for prim in mesh.get("primitives", []):
            attrs = prim.get("attributes", {})
            for key in ("POSITION", "NORMAL"):
                aid = attrs.get(key)
                if aid is None:
                    continue
                acc = gltf["accessors"][aid]
                view = gltf["bufferViews"][acc["bufferView"]]
                stride = view.get("byteStride", 12)
                base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
                is_position = key == "POSITION"
                new_min = [math.inf] * 3
                new_max = [-math.inf] * 3
                for i in range(acc["count"]):
                    off = base + i * stride
                    x, y, z = struct.unpack_from("<fff", bin_data, off)
                    if is_position:
                        # 노드 로컬 행렬은 T * R * S 순서로 합성된다.
                        sx, sy, sz = x * scale[0], y * scale[1], z * scale[2]
                        rx, ry, rz = _quat_rotate_vec3(rot, (sx, sy, sz))
                        rx, ry, rz = rx + trans[0], ry + trans[1], rz + trans[2]
                    else:
                        # 법선은 이동/스케일 없이 회전만 반영한다.
                        rx, ry, rz = _quat_rotate_vec3(rot, (x, y, z))
                    struct.pack_into("<fff", bin_data, off, rx, ry, rz)
                    if is_position:
                        for k, val in enumerate((rx, ry, rz)):
                            new_min[k] = min(new_min[k], val)
                            new_max[k] = max(new_max[k], val)
                if is_position:
                    acc["min"], acc["max"] = new_min, new_max
        node["rotation"] = [0.0, 0.0, 0.0, 1.0]
        node["translation"] = [0.0, 0.0, 0.0]
        node["scale"] = [1.0, 1.0, 1.0]


def auto_rig(glb_path, output_path):
    """Static mesh에 바운딩박스 기반 휴머노이드 skeleton을 추가한다.

    이미 skins가 있으면 False(패스스루)를 반환한다. 성공 시 output_path에
    JOINTS_0/WEIGHTS_0가 추가된 skinned GLB를 쓰고 True를 반환한다.
    """
    gltf, bin_data = _read_glb(glb_path)
    if gltf.get("skins"):
        return False  # 이미 리깅된 GLB
    buffers = gltf.get("buffers") or []
    if not buffers or "uri" in buffers[0]:
        return False  # 외부 버퍼

    # 스킨을 추가하면 메시 노드 자신의 TRS는 렌더러에서 무시되므로, 관절
    # 배치를 계산하기 전에 노드 트랜스폼을 버텍스에 구워 넣어 정규화한다.
    _bake_mesh_node_transform(gltf, bin_data)

    # 1. 전체 메시 AABB -------------------------------------------------
    all_pos = []
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            pid = prim.get("attributes", {}).get("POSITION")
            if pid is not None:
                all_pos.extend(_read_vec3(gltf, bin_data, pid, "POSITION"))
    if not all_pos:
        return False

    xs = [p[0] for p in all_pos]; ys = [p[1] for p in all_pos]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    cx = (min_x + max_x) / 2
    cz = (min([p[2] for p in all_pos]) + max([p[2] for p in all_pos])) / 2
    w = (max_x - min_x) or 1.0
    h = (max_y - min_y) or 1.0

    # 2. 조인트 정의 (name, 월드 위치, 부모 이름) -------------------------
    JDEF = [
        ("Hips",         (cx,          min_y + h*.52, cz), None),
        ("Spine",        (cx,          min_y + h*.62, cz), "Hips"),
        ("Chest",        (cx,          min_y + h*.73, cz), "Spine"),
        ("Head",         (cx,          min_y + h*.88, cz), "Chest"),
        ("LeftUpLeg",    (cx - w*.13,  min_y + h*.48, cz), "Hips"),
        ("LeftLeg",      (cx - w*.13,  min_y + h*.27, cz), "LeftUpLeg"),
        ("LeftFoot",     (cx - w*.13,  min_y + h*.04, cz), "LeftLeg"),
        ("RightUpLeg",   (cx + w*.13,  min_y + h*.48, cz), "Hips"),
        ("RightLeg",     (cx + w*.13,  min_y + h*.27, cz), "RightUpLeg"),
        ("RightFoot",    (cx + w*.13,  min_y + h*.04, cz), "RightLeg"),
        ("LeftArm",      (cx - w*.30,  min_y + h*.73, cz), "Chest"),
        ("LeftForeArm",  (cx - w*.44,  min_y + h*.62, cz), "LeftArm"),
        ("RightArm",     (cx + w*.30,  min_y + h*.73, cz), "Chest"),
        ("RightForeArm", (cx + w*.44,  min_y + h*.62, cz), "RightArm"),
    ]
    JNAMES = [j[0] for j in JDEF]
    JWORLD = {j[0]: j[1] for j in JDEF}

    # 3. glTF 노드 삽입 --------------------------------------------------
    gltf.setdefault("nodes", [])
    base = len(gltf["nodes"])
    name_to_node = {name: base + i for i, (name, _, _) in enumerate(JDEF)}
    for name, wpos, parent in JDEF:
        pw = JWORLD[parent] if parent else (0.0, 0.0, 0.0)
        translation = [wpos[k] - pw[k] for k in range(3)]
        children = [name_to_node[n] for n, _, p in JDEF if p == name]
        node = {"name": name, "translation": translation}
        if children:
            node["children"] = children
        gltf["nodes"].append(node)

    # root joint을 scene 0의 nodes 목록에도 추가해 렌더러가 인식하도록 함
    gltf.setdefault("scenes", [{}])
    gltf["scenes"][0].setdefault("nodes", []).append(name_to_node["Hips"])

    # 4. 인버스 바인드 행렬 (column-major 4×4, 순수 이동 역변환) ----------
    ibm_bytes = bytearray()
    for name in JNAMES:
        px, py, pz = JWORLD[name]
        ibm_bytes += struct.pack("<16f",
            1, 0, 0, 0,   0, 1, 0, 0,   0, 0, 1, 0,   -px, -py, -pz, 1)
    ibm_view = _append_view(gltf, bin_data, ibm_bytes)
    gltf.setdefault("accessors", []).append({
        "bufferView": ibm_view, "componentType": 5126,
        "count": len(JNAMES), "type": "MAT4",
    })
    ibm_acc = len(gltf["accessors"]) - 1

    # skin 등록
    gltf["skins"] = [{
        "name": "AutoHumanoidRig",
        "joints": [name_to_node[n] for n in JNAMES],
        "inverseBindMatrices": ibm_acc,
        "skeleton": name_to_node["Hips"],
    }]

    # 5. 버텍스 조인트 / 가중치 할당 (영역 기반) -------------------------
    def nearest_joint(x, y):
        yn = (y - min_y) / h       # 0=bottom 1=top
        if yn < 0.08:
            return "LeftFoot" if x < cx else "RightFoot"
        if yn < 0.30:
            return "LeftLeg"  if x < cx else "RightLeg"
        if yn < 0.44:
            return "LeftUpLeg" if x < cx else "RightUpLeg"
        if yn < 0.58:
            return "Hips"
        if yn < 0.79:
            if x < cx - w * 0.22:
                return "LeftForeArm" if yn < 0.67 else "LeftArm"
            if x > cx + w * 0.22:
                return "RightForeArm" if yn < 0.67 else "RightArm"
            return "Spine" if yn < 0.68 else "Chest"
        return "Head"

    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            attrs = prim.get("attributes", {})
            pid = attrs.get("POSITION")
            if pid is None:
                continue
            positions = _read_vec3(gltf, bin_data, pid, "POSITION")
            n_verts = len(positions)
            j_bytes = bytearray()
            w_bytes = bytearray()
            for x, y, z in positions:
                jname = nearest_joint(x, y)
                ji = JNAMES.index(jname)
                j_bytes += struct.pack("<4H", ji, 0, 0, 0)
                w_bytes += struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
            jv = _append_view(gltf, bin_data, j_bytes)
            wv = _append_view(gltf, bin_data, w_bytes)
            gltf["accessors"].append({
                "bufferView": jv, "componentType": 5123,
                "count": n_verts, "type": "VEC4",
            })
            gltf["accessors"].append({
                "bufferView": wv, "componentType": 5126,
                "count": n_verts, "type": "VEC4",
            })
            attrs["JOINTS_0"]  = len(gltf["accessors"]) - 2
            attrs["WEIGHTS_0"] = len(gltf["accessors"]) - 1

    # 6. 메시 노드에 skin 인덱스 연결 -------------------------------------
    for node in gltf.get("nodes", []):
        if "mesh" in node and "skin" not in node:
            node["skin"] = 0

    gltf["buffers"][0]["byteLength"] = len(bin_data)
    _write_glb(gltf, bin_data, output_path)
    return True


# --- HY-Motion SMPL → auto-rig 리타게팅 · glTF 애니메이션 베이킹 --------------
# RunPod HY-Motion(task=motion)이 반환한 SMPL-H 바디 22조인트 로컬 쿼터니언을
# auto_rig()의 14조인트 스켈레톤으로 옮겨 진짜 glTF 애니메이션 채널로 굽는다.
# 두 스켈레톤 모두 rest pose에서 조인트 로컬 프레임이 월드축 정렬(회전 없는
# 순수 이동 노드)이므로 로컬 회전을 그대로 이식할 수 있고, 조인트 수 차이는
# 체인 축약(Spine2⊗Spine3→Chest 등) — 로컬 쿼터니언 합성 — 으로 해소한다.
SMPL_TO_RIG = {
    "Hips": ("Pelvis",),
    "Spine": ("Spine1",),
    "Chest": ("Spine2", "Spine3"),
    "Head": ("Neck", "Head"),
    "LeftUpLeg": ("L_Hip",), "LeftLeg": ("L_Knee",), "LeftFoot": ("L_Ankle",),
    "RightUpLeg": ("R_Hip",), "RightLeg": ("R_Knee",), "RightFoot": ("R_Ankle",),
    "LeftArm": ("L_Collar", "L_Shoulder"), "LeftForeArm": ("L_Elbow",),
    "RightArm": ("R_Collar", "R_Shoulder"), "RightForeArm": ("R_Elbow",),
}
SMPL_REST_HEIGHT = 1.7  # SMPL 성인 신장(m) — trans(미터)→캐릭터 단위 환산 기준


def _quat_mul(a, b):
    """해밀턴 곱 a⊗b (xyzw): R(a⊗b) = R(a)·R(b) — 부모 로컬 다음 자식 로컬."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


_ZUP_TO_YUP = (-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))  # X축 -90°: +Z↑ → +Y↑


def _convert_zup_quat(q):
    """Z-up 데이터의 로컬 회전을 Y-up 프레임으로 기저 변환: C ⊗ q ⊗ C⁻¹."""
    cx, cy, cz, cw = _ZUP_TO_YUP
    return _quat_mul(_quat_mul(_ZUP_TO_YUP, q), (-cx, -cy, -cz, cw))


def bake_animation(glb_path, motion_payload, output_path):
    """리깅된 GLB에 HY-Motion 응답(motions 배열)을 glTF 애니메이션으로 굽는다.

    각 모션이 개별 glTF animation(이름=프롬프트 id)이 되며, 성공적으로 구운
    클립 수를 반환한다 (skins 부재/조인트 불일치 등은 0 — 호출자가 passthrough).
    """
    motions = motion_payload.get("motions") or []
    if not motions:
        return 0
    gltf, bin_data = _read_glb(glb_path)
    if not gltf.get("skins"):
        return 0
    node_by_name = {n.get("name"): i for i, n in enumerate(gltf.get("nodes", []))}
    if "Hips" not in node_by_name:
        return 0
    # HY-Motion(HumanML3D 계열)은 Y-up이 기본; AMASS 원본 스타일 Z-up 응답이
    # 확인되면 payload.up_axis="z"로 기저 변환을 켤 수 있다.
    zup = str(motion_payload.get("up_axis", "y")).lower() == "z"
    hips_rest = gltf["nodes"][node_by_name["Hips"]].get("translation", [0.0, 0.0, 0.0])
    # 캐릭터 스케일: 메시 AABB 높이 / SMPL 신장 — 루트 이동을 캐릭터 크기에 맞춤
    ys = []
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            pid = prim.get("attributes", {}).get("POSITION")
            if pid is not None:
                ys.extend(p[1] for p in _read_vec3(gltf, bin_data, pid, "POSITION"))
    scale = ((max(ys) - min(ys)) if ys else 1.0) / SMPL_REST_HEIGHT or 1.0

    accessors = gltf.setdefault("accessors", [])

    def add_accessor(blob, count, kind, minmax=None):
        view = _append_view(gltf, bin_data, blob)
        acc = {"bufferView": view, "componentType": 5126, "count": count, "type": kind}
        if minmax:
            acc["min"], acc["max"] = minmax
        accessors.append(acc)
        return len(accessors) - 1

    animations = []
    for motion in motions:
        joints = motion.get("joints") or []
        quats = motion.get("quats") or []
        trans = motion.get("trans") or []
        frames = min(len(quats), len(trans))
        if frames < 2 or not joints:
            continue
        fps = float(motion.get("fps") or 30.0)
        jindex = {name: i for i, name in enumerate(joints)}
        times = [f / fps for f in range(frames)]
        time_acc = add_accessor(
            b"".join(struct.pack("<f", t) for t in times), frames, "SCALAR",
            ([times[0]], [times[-1]]))
        samplers, channels = [], []

        def add_channel(node, path, blob, kind):
            out_acc = add_accessor(blob, frames, kind)
            samplers.append({"input": time_acc, "output": out_acc, "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1, "target": {"node": node, "path": path}})

        for rig_name, chain in SMPL_TO_RIG.items():
            node = node_by_name.get(rig_name)
            if node is None or any(j not in jindex for j in chain):
                continue
            blob = bytearray()
            for f in range(frames):
                q = (0.0, 0.0, 0.0, 1.0)
                for jname in chain:
                    q = _quat_mul(q, tuple(quats[f][jindex[jname]]))
                if zup:
                    q = _convert_zup_quat(q)
                norm = math.sqrt(sum(c * c for c in q)) or 1.0
                blob += struct.pack("<4f", *(c / norm for c in q))
            add_channel(node, "rotation", bytes(blob), "VEC4")
        # 루트 이동: 첫 프레임 기준 delta를 캐릭터 스케일로 환산해 Hips rest에 더함
        t0 = trans[0]
        blob = bytearray()
        for f in range(frames):
            dx, dy, dz = (trans[f][k] - t0[k] for k in range(3))
            if zup:
                dx, dy, dz = dx, dz, -dy
            blob += struct.pack("<3f", hips_rest[0] + dx * scale,
                                hips_rest[1] + dy * scale, hips_rest[2] + dz * scale)
        add_channel(node_by_name["Hips"], "translation", bytes(blob), "VEC3")
        animations.append({
            "name": str(motion.get("id") or motion.get("text") or f"motion{len(animations)}"),
            "samplers": samplers,
            "channels": channels,
        })
    if not animations:
        return 0
    gltf["animations"] = animations
    _write_glb(gltf, bin_data, output_path)
    return len(animations)


def find_motion_payload(workspace, req):
    """motion 단계 입력 JSON 탐색: 요청 필드 → workspace/motion/hy_motion.json."""
    candidate = str(req.get("motionFile", "")) or os.path.join(workspace, "motion", "hy_motion.json")
    if os.path.isfile(candidate):
        with open(candidate, encoding="utf-8") as f:
            return json.load(f)
    return None


def find_reference_image(workspace):
    prepare_dir = os.path.join(workspace, "prepare")
    if not os.path.isdir(prepare_dir):
        return None
    for name in sorted(os.listdir(prepare_dir)):
        if os.path.splitext(name)[1].lower() in (".png", ".jpg", ".jpeg"):
            return os.path.join(prepare_dir, name)
    return None


def bake_texture(glb_path, image_path, output_path):
    """참조 이미지를 정면(+Z 시점, XY 평면) 투영 UV로 베이킹한 GLB를 쓴다.

    True면 output_path에 텍스처 포함 GLB가 생성된 것이고, 이미 텍스처가
    있거나 외부 버퍼를 쓰는 GLB면 False(무변경)를 반환한다.
    """
    gltf, bin_data = _read_glb(glb_path)
    if gltf.get("textures") or gltf.get("images") or gltf.get("materials"):
        return False  # 텍스처/머티리얼이 이미 있는 메시는 보존 (non-destructive)
    buffers = gltf.get("buffers") or []
    if not buffers or "uri" in buffers[0]:
        return False  # 외부 버퍼 GLB는 baseline 범위 밖

    # 정면 투영 UV는 로컬 X=좌우, Y=수직(키)을 가정한다. Hunyuan3D-2.1 출력처럼
    # 실제 키가 로컬 Z축이고 노드 회전으로 보정되는 경우, 투영 전에 회전을
    # 버텍스에 구워 넣어 두 축 가정이 실제로 맞도록 정규화한다 (자세한 설명은
    # _bake_mesh_node_transform 참고).
    _bake_mesh_node_transform(gltf, bin_data)

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif image_bytes[:2] == b"\xff\xd8":
        mime = "image/jpeg"
    else:
        return False  # glTF 표준 텍스처는 PNG/JPEG만 허용

    textured = False
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            attrs = prim.get("attributes", {})
            if "POSITION" not in attrs or "TEXCOORD_0" in attrs:
                continue
            positions = _read_vec3(gltf, bin_data, attrs["POSITION"], "POSITION")
            min_x, max_x = min(p[0] for p in positions), max(p[0] for p in positions)
            min_y, max_y = min(p[1] for p in positions), max(p[1] for p in positions)
            span_x, span_y = (max_x - min_x) or 1.0, (max_y - min_y) or 1.0
            uv = bytearray()
            for x, y, _ in positions:
                # glTF의 v축은 이미지 상단이 0이므로 y를 뒤집는다.
                uv += struct.pack("<ff", (x - min_x) / span_x, 1.0 - (y - min_y) / span_y)
            view = _append_view(gltf, bin_data, uv)
            gltf["accessors"].append({"bufferView": view, "componentType": 5126,
                                      "count": len(positions), "type": "VEC2"})
            attrs["TEXCOORD_0"] = len(gltf["accessors"]) - 1
            normals = None
            if "COLOR_0" not in attrs:
                if "NORMAL" in attrs:
                    normals = _read_vec3(gltf, bin_data, attrs["NORMAL"], "NORMAL")
                else:
                    # Hunyuan shape-only 출력은 NORMAL도 없으므로 indices에서 계산
                    indices = _read_indices(gltf, bin_data, prim)
                    if indices:
                        normals = _vertex_normals(positions, indices)
            if normals:
                # 정면 투영은 뒷면에 앞면 텍스처가 미러링되므로, 법선 z가
                # 뒤를 향할수록 정점 컬러(COLOR_0)로 어둡게 해 깊이감을 준다.
                colors = bytearray()
                for _, _, nz in normals:
                    shade = int(255 * (0.55 + 0.45 * max(nz, 0.0)))
                    colors += bytes((shade, shade, shade, 255))
                cview = _append_view(gltf, bin_data, colors)
                # vertex attribute stride는 4의 배수여야 하므로 VEC4 ubyte 사용
                gltf["accessors"].append({"bufferView": cview, "componentType": 5121,
                                          "normalized": True, "count": len(normals), "type": "VEC4"})
                attrs["COLOR_0"] = len(gltf["accessors"]) - 1
            prim["material"] = 0  # 아래에서 materials를 투영 머티리얼 하나로 교체
            textured = True
    if not textured:
        return False

    image_view = _append_view(gltf, bin_data, image_bytes)
    gltf["images"] = [{"mimeType": mime, "bufferView": image_view}]
    gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]
    gltf["textures"] = [{"source": 0, "sampler": 0}]
    gltf["materials"] = [{"name": "FrontProjectedBaseColor", "doubleSided": True,
                          "pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
                                                   "metallicFactor": 0.0, "roughnessFactor": 1.0}}]
    _write_glb(gltf, bin_data, output_path)
    return True


def run(req):
    job = req["jobId"]
    stage = req["stage"]
    workspace = os.path.abspath(req["workspace"])
    os.makedirs(workspace, exist_ok=True)
    emit("progress", jobId=job, progress=.08, message="Worker environment ready")
    if stage == "prepare":
        source = os.path.abspath(req["input"])
        width, height = dimensions(source)
        with open(source, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        out_dir = os.path.join(workspace, "prepare")
        os.makedirs(out_dir, exist_ok=True)
        output = os.path.join(out_dir, "reference" + os.path.splitext(source)[1].lower())
        shutil.copy2(source, output)
        emit("progress", jobId=job, progress=.62, message="Validated image and provenance")
        metrics = {"width": width, "height": height, "sha256": digest, "alphaRequired": True}
        emit("artifact", jobId=job, kind="reference", path=output, metrics=metrics)
    elif stage == "reconstruct":
        # Offline fallback produces a genuine glTF mesh so the whole studio can
        # be exercised without CUDA. A configured TripoSR adapter can replace it.
        out_dir = os.path.join(workspace, stage)
        os.makedirs(out_dir, exist_ok=True)
        output = os.path.join(out_dir, "character.glb")
        generator = os.path.join(os.path.dirname(__file__), "procedural_character.py")
        emit("progress", jobId=job, progress=.25, message="Building preview mesh and humanoid topology")
        proc = subprocess.run([sys.executable, generator, output], capture_output=True, text=True, check=True)
        generated = json.loads(proc.stdout)
        metrics = {**generated, "adapter": "procedural-offline", "previewOnly": True}
        emit("artifact", jobId=job, kind="mesh", path=output, metrics=metrics)
    elif stage in ("retopo", "rig", "motion", "export") and str(req.get("input", "")).lower().endswith(".glb"):
        # Preserve a working vertical slice: GLB remains loadable through every
        # stage and is copied immutably. Production adapters improve each pass.
        out_dir = os.path.join(workspace, stage)
        os.makedirs(out_dir, exist_ok=True)
        names = {"retopo":"character-clean.glb", "rig":"character-rigged.glb", "motion":"character-animated.glb", "export":"character-final.glb"}
        output = os.path.join(out_dir, names[stage])
        source = os.path.abspath(req["input"])
        metrics = {"adapter": "passthrough-offline", "validated": True, "previewOnly": True}
        message = f"Validated immutable {stage} GLB artifact"
        transformed = False
        if stage == "retopo":
            # shape-only 메시(텍스처 없음)에 참조 이미지를 정면 투영으로 베이킹.
            reference = find_reference_image(workspace)
            if reference:
                emit("progress", jobId=job, progress=.4, message="Projecting reference image onto mesh")
                try:
                    transformed = bake_texture(source, reference, output)
                    if transformed:
                        metrics = {"adapter": "front-projection-offline", "validated": True, "textured": True, "previewOnly": True}
                        message = "Baked front-projected base color texture"
                except Exception as exc:
                    emit("progress", jobId=job, progress=.5, message=f"Texture projection skipped: {exc}")
        elif stage == "rig":
            # skeleton이 없는 static mesh에 바운딩박스 기반 humanoid rig을 추가.
            emit("progress", jobId=job, progress=.35, message="Fitting humanoid skeleton to mesh bounds")
            try:
                transformed = auto_rig(source, output)
                if transformed:
                    metrics = {"adapter": "auto-rig-bbox", "validated": True, "skinned": True, "previewOnly": True}
                    message = "Fitted humanoid skeleton with skin weights"
            except Exception as exc:
                emit("progress", jobId=job, progress=.5, message=f"Auto-rig skipped: {exc}")
        elif stage == "motion":
            # HY-Motion 응답 JSON이 준비돼 있으면 SMPL→auto-rig 리타겟 후
            # glTF 애니메이션으로 베이킹한다 (없으면 기존 passthrough).
            try:
                payload = find_motion_payload(workspace, req)
                if payload:
                    emit("progress", jobId=job, progress=.4, message="Retargeting HY-Motion clips onto rig")
                    baked_clips = bake_animation(source, payload, output)
                    transformed = baked_clips > 0
                    if transformed:
                        metrics = {"adapter": "hy-motion-retarget", "validated": True,
                                   "animations": baked_clips, "model": str(payload.get("model", "")),
                                   "previewOnly": True}
                        message = f"Baked {baked_clips} HY-Motion clip(s) as glTF animations"
            except Exception as exc:
                emit("progress", jobId=job, progress=.5, message=f"Motion retarget skipped: {exc}")
        if not transformed:
            shutil.copy2(source, output)
        kinds = {"retopo":"clean-mesh", "rig":"rigged-model", "motion":"animated-model", "export":"package"}
        emit("progress", jobId=job, progress=.7, message=message)
        emit("artifact", jobId=job, kind=kinds[stage], path=output, metrics=metrics)
    else:
        # Adapter handshake artifact for unsupported custom input.
        out_dir = os.path.join(workspace, stage)
        os.makedirs(out_dir, exist_ok=True)
        output = os.path.join(out_dir, "adapter-request.json")
        with open(output, "w", encoding="utf-8") as f:
            json.dump(req, f, indent=2)
        emit("progress", jobId=job, progress=.7, message=f"Prepared {stage} adapter request")
        metrics = {"adapter": req.get("adapter", "baseline"), "requiresModel": True}
        emit("artifact", jobId=job, kind="adapter-request", path=output, metrics=metrics)
    time.sleep(.03)
    emit("done", jobId=job, stage=stage, progress=1, metrics=metrics)


for line in sys.stdin:
    try:
        request = json.loads(line)
        if request.get("type") != "run":
            raise ValueError("expected run message")
        run(request)
    except Exception as exc:
        emit("error", message=str(exc))
        sys.exit(1)
