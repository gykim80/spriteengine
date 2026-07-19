#!/usr/bin/env python3
"""SpriteEngine JSON Lines worker baseline.

The worker intentionally uses only Python's standard library so the desktop
orchestrator can validate its process/progress/artifact contract before large
GPU environments are installed.
"""
import hashlib
import importlib.util
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


# 복원 실패 유형: 캐릭터가 얇은 바닥 판(base slab) 위에 서 있는 메시
# (실측 2026-07-20: gladiator 3장 전부, vampire v1). 하단 밴드가 XZ 전폭
# (스팬 비율 1.00)·16×16 그리드 점유율 0.88~1.00으로 꽉 찬 슬래브가 깔리고,
# 그 위 캐릭터 자체는 정상이다. 정상 humanoid 하단 밴드는 발 수준(스팬
# ≤0.89, 점유율 ≤0.37), 4족은 네 발(점유율 ~0.02)이라 갭이 커서 안전하게
# 분리된다. 캐릭터는 구제 가능하므로 후보 기각 대신 슬래브만 잘라낸다.
BASE_SLAB_BAND = 0.015      # 검출 밴드 높이 (전체 높이 비율)
BASE_SLAB_SCAN_BANDS = 8    # 하단 12%까지 스캔 (vampire v1은 밴드 2~3이 슬래브)
BASE_SLAB_MIN_SPAN = 0.8    # 슬래브 판정: 밴드 XZ 스팬 / 전체 스팬
BASE_SLAB_MIN_FILL = 0.5    # 슬래브 판정: 그리드 점유 셀 비율
BASE_SLAB_GRID = 16
# 슬래브 제거 후 재정규화 배율 상한. 슬래브가 정규화 볼륨을 차지해 캐릭터가
# 미니어처로 축소된 복원(실측 gladiator: 키 0.67 → x2.96 업스케일 필요)은
# 버텍스 11,483(정상 대비 43%)·텍셀 밀도 저하로 진흙 인형 같은 저품질이 된다.
# 임계 초과 구제는 기각하고 원본 이미지 재생성을 유도한다 (얇은 받침대처럼
# 소폭 보정으로 끝나는 경우만 구제 허용).
BASE_SLAB_MAX_RESCUE_SCALE = 1.25


class BasePlaneRescueError(ValueError):
    """슬래브 제거 후 남은 캐릭터가 저해상도 미니어처라 구제를 기각함."""


# 저해상도 진흙(mud) 복원 기각 — face_count=40000 요청 기준 정상 Hunyuan
# 출력은 정점 23,876~29,531개(등록 22종 실측 2026-07-20). 실측 실패
# (gladiator, 사용자 신고 "완전 심각"): 11,483 verts — 디테일이 뭉개진 진흙
# 품질인데 직립/파편/슬래브 게이트를 모두 통과했다. 정상 최소값의 75%를
# 하한으로 두면 정상(23.9k)과 진흙(11.5k) 사이 갭이 2배 이상이라 안전하다.
RECON_MIN_VERTICES = 18000


class LowResolutionMeshError(ValueError):
    """복원 메시가 진흙 품질(정점 수 하한 미달) — 원본 이미지 재생성 필요."""


def total_vertex_count(glb_path):
    """GLB의 전체 mesh POSITION 정점 수 (accessor count 합)."""
    gltf, _ = _read_glb(glb_path)
    return sum(gltf["accessors"][prim["attributes"]["POSITION"]].get("count", 0)
               for mesh in gltf.get("meshes", [])
               for prim in mesh.get("primitives", [])
               if "POSITION" in prim.get("attributes", {}))


def _component_size(component_type):
    return {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}[component_type]


def _type_width(type_name):
    return {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[type_name]


def strip_base_plane(glb_path, output_path):
    """복원 메시 하단의 바닥 판(base slab)을 검출해 잘라낸다.

    검출은 노드 TRS 적용 월드 좌표에서, 제거는 로컬 버텍스 데이터에서 수행
    (노드 트랜스폼은 유지하되, 제거 후 남은 캐릭터를 Hunyuan 정규화 규약
    (최장축 ~1.99·원점 중심)으로 되돌리도록 노드 scale/translation을 보정 —
    슬래브가 정규화 기준을 차지해 캐릭터가 미니어처로 축소된 것을 복구).
    슬래브가 없으면 0을 반환하고 파일을 쓰지 않는다 (no-op, 멱등).
    반환값: 제거된 버텍스 수.
    """
    gltf, bin_data = _read_glb(glb_path)
    prims = []  # (node, prim, local_positions, world_positions)
    for node in gltf.get("nodes", []):
        if "mesh" not in node or "skin" in node:
            continue
        rot = node.get("rotation") or [0.0, 0.0, 0.0, 1.0]
        trans = node.get("translation") or [0.0, 0.0, 0.0]
        scale = node.get("scale") or [1.0, 1.0, 1.0]
        identity = _quat_is_identity(rot) and trans == [0.0, 0.0, 0.0] and scale == [1.0, 1.0, 1.0]
        for prim in gltf["meshes"][node["mesh"]].get("primitives", []):
            pid = prim.get("attributes", {}).get("POSITION")
            if pid is None:
                continue
            local = _read_vec3(gltf, bin_data, pid, "POSITION")
            if identity:
                world = local
            else:
                world = []
                for p in local:
                    p = (p[0] * scale[0], p[1] * scale[1], p[2] * scale[2])
                    p = _quat_rotate_vec3(rot, p)
                    world.append((p[0] + trans[0], p[1] + trans[1], p[2] + trans[2]))
            prims.append((node, prim, local, world))
    all_world = [p for _, _, _, world in prims for p in world]
    if len(all_world) < 100:
        return 0
    xs = [p[0] for p in all_world]; ys = [p[1] for p in all_world]; zs = [p[2] for p in all_world]
    min_x, span_x = min(xs), (max(xs) - min(xs)) or 1e-9
    min_z, span_z = min(zs), (max(zs) - min(zs)) or 1e-9
    min_y, h = min(ys), (max(ys) - min(ys)) or 1e-9
    band_h = BASE_SLAB_BAND * h
    slab_top_band = -1
    for b in range(BASE_SLAB_SCAN_BANDS):
        lo, hi = min_y + b * band_h, min_y + (b + 1) * band_h
        pts = [(p[0], p[2]) for p in all_world if lo <= p[1] < hi]
        if len(pts) < 32:
            continue
        bxs = [p[0] for p in pts]; bzs = [p[1] for p in pts]
        ratio_x = (max(bxs) - min(bxs)) / span_x
        ratio_z = (max(bzs) - min(bzs)) / span_z
        cells = {(min(BASE_SLAB_GRID - 1, int((x - min_x) / span_x * BASE_SLAB_GRID)),
                  min(BASE_SLAB_GRID - 1, int((z - min_z) / span_z * BASE_SLAB_GRID)))
                 for x, z in pts}
        fill = len(cells) / float(BASE_SLAB_GRID ** 2)
        if ratio_x >= BASE_SLAB_MIN_SPAN and ratio_z >= BASE_SLAB_MIN_SPAN and fill >= BASE_SLAB_MIN_FILL:
            slab_top_band = b
    if slab_top_band < 0:
        return 0
    cut_y = min_y + (slab_top_band + 1) * band_h

    removed = 0
    kept_world = []
    for node, prim, local, world in prims:
        keep = [i for i, p in enumerate(world) if p[1] >= cut_y]
        kept_world.extend(world[i] for i in keep)
        if len(keep) == len(world):
            continue
        removed += len(world) - len(keep)
        keep_set = set(keep)
        remap = {old: new for new, old in enumerate(keep)}
        # 모든 버텍스 속성(POSITION/NORMAL/TEXCOORD…)을 동일 필터로 재작성.
        for key, aid in list(prim.get("attributes", {}).items()):
            acc = gltf["accessors"][aid]
            elem = _component_size(acc["componentType"]) * _type_width(acc["type"])
            view = gltf["bufferViews"][acc["bufferView"]]
            stride = view.get("byteStride", elem)
            base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
            blob = bytearray()
            for i in keep:
                off = base + i * stride
                blob += bin_data[off:off + elem]
            new_acc = {"bufferView": _append_view(gltf, bin_data, bytes(blob)),
                       "byteOffset": 0, "componentType": acc["componentType"],
                       "count": len(keep), "type": acc["type"]}
            if key == "POSITION":
                kept = [local[i] for i in keep]
                new_acc["min"] = [min(p[k] for p in kept) for k in range(3)]
                new_acc["max"] = [max(p[k] for p in kept) for k in range(3)]
            gltf["accessors"].append(new_acc)
            prim["attributes"][key] = len(gltf["accessors"]) - 1
        # 세 정점이 모두 남은 삼각형만 보존, 인덱스는 uint32로 통일.
        idx = _read_indices(gltf, bin_data, prim)
        if idx is None:
            idx = list(range(len(world)))
        new_idx = []
        for t in range(0, len(idx) - 2, 3):
            tri = (idx[t], idx[t + 1], idx[t + 2])
            if tri[0] in keep_set and tri[1] in keep_set and tri[2] in keep_set:
                new_idx.extend((remap[tri[0]], remap[tri[1]], remap[tri[2]]))
        blob = struct.pack(f"<{len(new_idx)}I", *new_idx)
        gltf["accessors"].append({"bufferView": _append_view(gltf, bin_data, blob),
                                  "byteOffset": 0, "componentType": 5125,
                                  "count": len(new_idx), "type": "SCALAR"})
        prim["indices"] = len(gltf["accessors"]) - 1
    if not removed:
        return 0
    # 재정규화: 슬래브(예: 2×2 판)가 정규화 볼륨을 차지해 그 위 캐릭터가
    # 미니어처로 축소된 경우(실측 gladiator: 슬래브 제거 후 키 0.67), 남은
    # 메시를 Hunyuan 규약(최장축 STANDARD_CHARACTER_HEIGHT, 원점 중심)으로
    # 되돌린다. 균일 스케일은 노드 회전과 가환이므로 노드 TRS로만 보정한다:
    # world'(v) = T' + s·R·(S∘v),  T' = s·(T − bbox중심) → 중심이 원점으로.
    bounds = [(min(p[k] for p in kept_world), max(p[k] for p in kept_world)) for k in range(3)]
    size = max(hi - lo for lo, hi in bounds) or 1e-9
    s = STANDARD_CHARACTER_HEIGHT / size
    if s > BASE_SLAB_MAX_RESCUE_SCALE:
        # 파일은 쓰지 않고 기각한다 (원본 무변경) — 호출자가 후보 제외/작업
        # 실패로 처리해 이미지 재생성을 유도한다.
        raise BasePlaneRescueError(
            f"base slab rescue needs x{s:.2f} upscale (limit x{BASE_SLAB_MAX_RESCUE_SCALE}) — "
            "remaining character is a low-resolution miniature; regenerate the source image")
    center = [(lo + hi) / 2.0 for lo, hi in bounds]
    seen = set()
    for node, _, _, _ in prims:
        if id(node) in seen:  # 노드 하나에 primitive 여러 개여도 TRS 보정은 1회만
            continue
        seen.add(id(node))
        trans = node.get("translation") or [0.0, 0.0, 0.0]
        scale = node.get("scale") or [1.0, 1.0, 1.0]
        node["translation"] = [s * (t - c) for t, c in zip(trans, center)]
        node["scale"] = [s * sc for sc in scale]
    _write_glb(gltf, bin_data, output_path)
    return removed


# Hunyuan3D-2.1 정상 출력의 캐릭터 키 (실측 24종 전부 ~1.987). 복원이 간혹
# 표준에서 크게 벗어난 스케일로 나오면(실측: cowboy 0.494 — 1/4 크기),
# HY-Motion 루트 이동 리타게팅 스케일도 캐릭터 키에 비례해 함께 줄어들어
# 이동 모션이 제자리걸음처럼 퇴화한다. 키가 정상 범위를 벗어난 경우에만
# 표준 키로 정규화한다 (정상 출력은 no-op — 기존 산출물 불변).
STANDARD_CHARACTER_HEIGHT = 1.987
NORMAL_HEIGHT_RANGE = (1.2, 3.0)
# 4족 동물은 키가 아니라 체장(코~꼬리, Z 범위)이 대표 치수다. 중형견 전장
# ~1.4m를 표준으로, 인간 키 기준(1.2~3.0)을 4족에 적용하면 어깨높이 0.6m대의
# 정상 개가 2배 이상 거인화되므로(실측: 0.88m 개 → 2.26배) 축과 범위를 분리한다.
STANDARD_QUADRUPED_LENGTH = 1.4
NORMAL_LENGTH_RANGE = (0.5, 3.0)

def _core_extent(vals):
    """p10~p90 코어 범위 — A-pose 팔끝·꼬리끝 같은 극단값의 영향을 배제한다."""
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * 0.9))] - s[int(len(s) * 0.1)]


def _classify_body_type(all_pos):
    """버텍스 분포로 humanoid/quadruped를 판별한다.

    직립 캐릭터는 키(Y)가, 4족 동물은 체장(Z)이 코어 최장축이다.
    _bake_mesh_node_transform() 이후(Y축=수직 보장) 좌표를 전제로 한다.

    체장>키만으로는 "누워서 복원된 휴머노이드"(Hunyuan3D 실패 사례:
    barbarian/cowboy)가 4족으로 오분류돼 upright 게이트를 우회한다. 그래서
    4족 확증 조건을 더한다 — 서 있는 4족은 다리 4기둥만 지면에 닿아 지면
    근접 버텍스가 앞/뒤 두 클러스터로 갈리고 그 사이(배 밑)는 비지만, 누운
    몸통은 등/배가 전장에 걸쳐 연속으로 지면에 닿는다. 확증 실패 시
    humanoid로 두어 기존 upright 검사(누움 검출)가 잡아내게 한다.
    """
    # 1차 게이트: 코어 체장(Z) > 코어 키(Y). 실측 — 직립 휴머노이드 27종은
    # 키가 깊이의 3배 이상이라(보폭 자세 포함) 절대 넘지 않고, 개(1.6배)는
    # 물론 체장≈키인 말 체형도 코어 기준으로는 Z가 남는다(다리가 가늘어
    # y 질량이 몸통에 몰리므로 코어 키 < 총 높이). 최종 확증은 아래 배 밑
    # 갭 검사가 담당하므로 여기서는 최장축 여부만 거른다.
    cy = _core_extent([p[1] for p in all_pos]) or 1e-9
    cz = _core_extent([p[2] for p in all_pos])
    if cz <= cy:
        return "humanoid"
    ys = [p[1] for p in all_pos]
    zs = [p[2] for p in all_pos]
    min_y, h = min(ys), (max(ys) - min(ys)) or 1.0
    zc = (min(zs) + max(zs)) / 2
    length = (max(zs) - min(zs)) or 1.0
    ground = [p for p in all_pos if p[1] < min_y + 0.15 * h]  # 지면 근접(발/접지면)
    front = sorted(p[2] for p in ground if p[2] >= zc)
    rear = sorted(p[2] for p in ground if p[2] < zc)
    if not front or not rear:
        return "humanoid"  # 접지가 한쪽뿐 — 4족 자세가 아님
    front_med, rear_med = front[len(front) // 2], rear[len(rear) // 2]
    span = front_med - rear_med
    if span < 0.25 * length:
        return "humanoid"  # 앞/뒤 접지 클러스터가 충분히 벌어지지 않음
    # 배 밑 갭: 두 클러스터 중앙값 사이 가운데 40% 밴드에 접지 버텍스가
    # 거의 없어야 한다 (누운 몸통은 이 밴드가 접지 버텍스로 채워진다).
    lo, hi = rear_med + 0.3 * span, rear_med + 0.7 * span
    belly = sum(1 for p in ground if lo <= p[2] <= hi)
    return "quadruped" if belly <= 0.1 * len(ground) else "humanoid"


def _normalize_character_scale(gltf, bin_data, body_type="humanoid"):
    """캐릭터 대표 치수가 정상 범위 밖이면 표준 치수로 균등 스케일한다.

    대표 치수는 체형별로 다르다 — 휴머노이드는 키(Y 범위), 4족은 체장(Z 범위).
    _bake_mesh_node_transform() 이후(Y축=키 보장) 호출을 전제로 한다.
    법선은 균등 스케일에 불변이므로 POSITION만 수정한다. 적용 배율을 반환한다.
    """
    axis, target, (lo, hi) = ((2, STANDARD_QUADRUPED_LENGTH, NORMAL_LENGTH_RANGE)
                              if body_type == "quadruped"
                              else (1, STANDARD_CHARACTER_HEIGHT, NORMAL_HEIGHT_RANGE))
    pos_accessors = []
    seen = set()
    ys = []
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            aid = prim.get("attributes", {}).get("POSITION")
            if aid is None or aid in seen:
                continue
            seen.add(aid)
            pos_accessors.append(aid)
            ys.extend(p[axis] for p in _read_vec3(gltf, bin_data, aid, "POSITION"))
    if not ys:
        return 1.0
    height = max(ys) - min(ys)
    if height <= 0 or lo <= height <= hi:
        return 1.0
    s = target / height
    for aid in pos_accessors:
        acc = gltf["accessors"][aid]
        view = gltf["bufferViews"][acc["bufferView"]]
        stride = view.get("byteStride", 12)
        base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        for i in range(acc["count"]):
            off = base + i * stride
            x, y, z = struct.unpack_from("<fff", bin_data, off)
            struct.pack_into("<fff", bin_data, off, x * s, y * s, z * s)
        if "min" in acc:
            acc["min"] = [v * s for v in acc["min"]]
        if "max" in acc:
            acc["max"] = [v * s for v in acc["max"]]
    return s


_FORE_ALLOWED = {
    "LeftForeArm": frozenset(("LeftForeArm", "LeftArm")),
    "RightForeArm": frozenset(("RightForeArm", "RightArm")),
}
_ARM_CHAIN = frozenset(("LeftArm", "LeftForeArm", "RightArm", "RightForeArm"))
_LEG_CHAIN = frozenset(("LeftUpLeg", "LeftLeg", "LeftFoot",
                        "RightUpLeg", "RightLeg", "RightFoot"))


def _cut_fused_bridges(gltf, bin_data, prim, doms):
    """융합 브리지 삼각형 절단.

    복원 메시는 rest에서 손이 허벅지/가슴에 닿아(실측 2.5~5cm, A-pose)
    표면이 물리적으로 weld되는 경우가 있다. 이런 삼각형은 팔 지배 영역과
    비인접 체인 영역(허벅지·가슴·반대쪽 팔)을 직접 이어, 팔을 드는 순간
    고무막처럼 늘어나는 웨빙과 "손이 몸에 붙는" 증상을 만든다 — 웨이트로는
    해결 불가(어느 쪽에 주든 반대쪽이 찢어짐)라 지오메트리를 잘라야 한다.

    두 규칙으로 삼각형을 인덱스에서 제거한다 (절단 수 반환):
    (a) 전완 지배 버텍스 + {같은쪽 전완·상완} 외 지배 버텍스 공존
        — 손↔허벅지·가슴·반대팔 융합 (실측: 손-허벅지 2.5cm weld)
    (b) 팔체인 지배 + 다리체인 지배 버텍스 공존 — 긴 코트/케이프
        캐릭터에서 상완 지배 소매가 다리 지오메트리에 weld되는 경우
        (실측: knight/pirate/robot flykick 웨빙 17~277엣지)
    """
    idx_acc = prim.get("indices")
    if idx_acc is None:
        return 0
    acc = gltf["accessors"][idx_acc]
    view = gltf["bufferViews"][acc["bufferView"]]
    base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    fmt = {5121: "B", 5123: "H", 5125: "I"}[acc["componentType"]]
    idx = struct.unpack_from(f"<{acc['count']}{fmt}", bin_data, base)
    kept = bytearray()
    kept_count = 0
    cut = 0
    for t in range(0, len(idx) - 2, 3):
        tri = idx[t:t + 3]
        names = [doms[v] for v in tri]
        fore_bridge = any(n in _FORE_ALLOWED
                          and any(m not in _FORE_ALLOWED[n] for m in names)
                          for n in names)
        armleg_bridge = (any(n in _ARM_CHAIN for n in names)
                         and any(n in _LEG_CHAIN for n in names))
        if fore_bridge or armleg_bridge:
            cut += 1
            continue
        kept += struct.pack(f"<3{fmt}", *tri)
        kept_count += 3
    if cut:
        nv = _append_view(gltf, bin_data, kept)
        gltf["accessors"].append({
            "bufferView": nv, "componentType": acc["componentType"],
            "count": kept_count, "type": "SCALAR",
        })
        prim["indices"] = len(gltf["accessors"]) - 1
    return cut


def _quadruped_faces_backward(all_pos):
    """4족 메시가 −Z를 향하는지(머리가 min_z 쪽인지) 감지한다.

    복원 메시의 진행 방향은 보장되지 않는다 — 개가 −Z를 향하면 머리/꼬리가
    뒤바뀐 채 리깅되고 SMPL 보행 리타게팅(+Z 전진)이 뒤로 걷게 되는데,
    스켈레톤이 좌우 대칭이라 검증 게이트로는 잡히지 않는다.

    판별은 척추선(0.55h, _quadruped_joint_layout의 spine_y와 동일) 위
    버텍스 질량 비교다: 머리는 척추 위로 솟은 부피가 크고(두개골·목)
    꼬리는 가늘어서, 전방 1/3과 후방 1/3에서 y > min_y + 0.55h 버텍스
    수를 세면 머리 쪽이 항상 많다. 후방이 더 많으면 −Z 방향으로 판정한다.
    """
    ys = [p[1] for p in all_pos]; zs = [p[2] for p in all_pos]
    min_y = min(ys)
    h = (max(ys) - min_y) or 1.0
    min_z, max_z = min(zs), max(zs)
    zc = (min_z + max_z) / 2
    length = (max_z - min_z) or 1.0
    y_high = min_y + 0.55 * h
    front = sum(1 for p in all_pos if p[2] > zc + 0.15 * length and p[1] > y_high)
    rear = sum(1 for p in all_pos if p[2] < zc - 0.15 * length and p[1] > y_high)
    return rear > front


def _rewrite_mesh_vec3(gltf, bin_data, fn):
    """모든 POSITION/NORMAL 버텍스를 fn(x,y,z)→(x',y',z')로 재기록한다 (in-place).

    yaw 회전류(축 치환·부호 반전, det=+1) 전용 — winding order를 보존하므로
    인덱스는 손대지 않고, min/max는 코너에 fn을 적용해 성분별로 재계산한다.
    """
    seen = set()
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            attrs = prim.get("attributes", {})
            for name in ("POSITION", "NORMAL"):
                aid = attrs.get(name)
                if aid is None or aid in seen:
                    continue
                seen.add(aid)
                acc = gltf["accessors"][aid]
                view = gltf["bufferViews"][acc["bufferView"]]
                stride = view.get("byteStride", 12)
                base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
                for i in range(acc["count"]):
                    off = base + i * stride
                    p = struct.unpack_from("<fff", bin_data, off)
                    struct.pack_into("<fff", bin_data, off, *fn(*p))
                if "min" in acc and "max" in acc:
                    a, b = fn(*acc["min"]), fn(*acc["max"])
                    acc["min"] = [min(a[k], b[k]) for k in range(3)]
                    acc["max"] = [max(a[k], b[k]) for k in range(3)]


def _flip_mesh_yaw180(gltf, bin_data):
    """메시를 Y축 기준 180° 회전한다 (x, z 부호 반전, in-place)."""
    _rewrite_mesh_vec3(gltf, bin_data, lambda x, y, z: (-x, y, -z))


def _quadruped_joint_layout(all_pos):
    """4족 메시에서 수평 척추 + 다리 4체인 + 머리/꼬리 스켈레톤 배치를 실측한다.

    조인트 명명은 휴머노이드와 공유한다 — 앞다리=Arm 체인, 뒷다리=Leg 체인.
    이렇게 하면 HY-Motion SMPL 리타게팅(SMPL_TO_RIG)이 인간 팔 스윙을
    앞다리에, 다리 스윙을 뒷다리에 그대로 이식해 대각 보행(trot)이 되고,
    좌우 대칭 검증(EXPECTED_LEFT_RIGHT_PAIRS)도 재사용된다.

    배치는 bbox 비율 하드코딩이 아니라 실측 클러스터 기반이다: 하단 40%
    버텍스를 전/후로 나눠 앞·뒷다리 z 위치를 얻고, 앞다리보다 전방 버텍스
    원심으로 머리를, 뒷다리보다 후방 버텍스 원심으로 꼬리를 배치한다.
    반환: (JDEF, BONE_CHILD, tips)
    """
    xs = [p[0] for p in all_pos]; ys = [p[1] for p in all_pos]; zs = [p[2] for p in all_pos]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    cx = (min_x + max_x) / 2
    cz = (min_z + max_z) / 2
    h = (max_y - min_y) or 1.0
    length = (max_z - min_z) or 1.0

    def _median(vals):
        s = sorted(vals)
        return s[len(s) // 2]

    # 다리 클러스터: 하단 40% 버텍스를 z 중앙 기준 앞/뒤로 분할
    low = [p for p in all_pos if p[1] < min_y + 0.4 * h]
    if len(low) < 8:
        low = sorted(all_pos, key=lambda p: p[1])[:max(8, len(all_pos) // 5)]
    front = [p for p in low if p[2] >= cz]
    rear = [p for p in low if p[2] < cz]
    front_z = _median([p[2] for p in front]) if front else cz + 0.25 * length
    rear_z = _median([p[2] for p in rear]) if rear else cz - 0.25 * length
    leg_x = _median([abs(p[0] - cx) for p in low]) or 0.25 * (max_x - min_x)

    spine_y = min_y + 0.55 * h            # 수평 척추 라인 (몸통 중심 높이)
    top_y = min_y + 0.40 * h              # 다리 상부 (몸통 밑면)
    mid_y = min_y + 0.22 * h              # 무릎/팔꿈치
    foot_y = min_y + 0.06 * h             # 발목

    # 머리: 앞다리보다 전방 버텍스 원심 (개·말 등은 머리가 몸통 위-앞)
    head_pts = [p for p in all_pos if p[2] > front_z + 0.08 * length]
    if head_pts:
        head = (cx, sum(p[1] for p in head_pts) / len(head_pts),
                sum(p[2] for p in head_pts) / len(head_pts))
        nose = (cx, head[1], max(p[2] for p in head_pts))
    else:
        head = (cx, min_y + 0.8 * h, front_z + 0.15 * length)
        nose = (cx, head[1], max_z)

    JDEF = [
        ("Hips",         (cx, spine_y, rear_z), None),
        ("Spine",        (cx, spine_y, (front_z + rear_z) / 2), "Hips"),
        ("Chest",        (cx, spine_y, front_z), "Spine"),
        ("Head",         head, "Chest"),
        ("LeftUpLeg",    (cx + leg_x, top_y, rear_z), "Hips"),
        ("LeftLeg",      (cx + leg_x, mid_y, rear_z), "LeftUpLeg"),
        ("LeftFoot",     (cx + leg_x, foot_y, rear_z), "LeftLeg"),
        ("RightUpLeg",   (cx - leg_x, top_y, rear_z), "Hips"),
        ("RightLeg",     (cx - leg_x, mid_y, rear_z), "RightUpLeg"),
        ("RightFoot",    (cx - leg_x, foot_y, rear_z), "RightLeg"),
        ("LeftArm",      (cx + leg_x, top_y, front_z), "Chest"),
        ("LeftForeArm",  (cx + leg_x, mid_y, front_z), "LeftArm"),
        ("RightArm",     (cx - leg_x, top_y, front_z), "Chest"),
        ("RightForeArm", (cx - leg_x, mid_y, front_z), "RightArm"),
    ]
    BONE_CHILD = {
        "Hips": "Spine", "Spine": "Chest", "Chest": "Head",
        "LeftUpLeg": "LeftLeg", "LeftLeg": "LeftFoot",
        "RightUpLeg": "RightLeg", "RightLeg": "RightFoot",
        "LeftArm": "LeftForeArm", "RightArm": "RightForeArm",
    }
    tips = {
        "Head": nose,
        "LeftFoot": (cx + leg_x, min_y, rear_z),
        "RightFoot": (cx - leg_x, min_y, rear_z),
        "LeftForeArm": (cx + leg_x, min_y, front_z),
        "RightForeArm": (cx - leg_x, min_y, front_z),
    }
    # 꼬리: 뒷다리보다 후방 버텍스가 있으면 Tail 조인트 추가 (없으면 생략)
    tail_pts = [p for p in all_pos if p[2] < rear_z - 0.08 * length]
    if tail_pts:
        ty = sum(p[1] for p in tail_pts) / len(tail_pts)
        tz = sum(p[2] for p in tail_pts) / len(tail_pts)
        JDEF.append(("Tail", (cx, ty, tz), "Hips"))
        tips["Tail"] = (cx, ty, min(p[2] for p in tail_pts))
    return JDEF, BONE_CHILD, tips


def auto_rig(glb_path, output_path):
    """Static mesh에 실측 기반 skeleton을 추가한다 (휴머노이드/4족 자동 판별).

    코어 축 비율로 체형을 판별해 — 직립 메시는 휴머노이드 14조인트, 체장(Z)이
    키(Y)보다 긴 메시는 4족 스켈레톤(수평 척추, 앞다리=Arm 체인, 뒷다리=Leg
    체인, 실측 다리 클러스터 배치)을 받는다.
    이미 skins가 있으면 False(패스스루)를 반환한다. 성공 시 output_path에
    JOINTS_0/WEIGHTS_0가 추가된 skinned GLB를 쓰고 판별된 체형 문자열
    ("humanoid"/"quadruped", truthy)을 반환한다.
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

    # 1. 전체 메시 버텍스 → 체형 판별 → 표준 치수 정규화 -------------------
    all_pos = []
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            pid = prim.get("attributes", {}).get("POSITION")
            if pid is not None:
                all_pos.extend(_read_vec3(gltf, bin_data, pid, "POSITION"))
    if not all_pos:
        return False
    # 측방향 복원 4족(체장이 X축) 정렬: 코어 X가 최장축이면 yaw −90° 회전
    # 후보를 만들어 보고, 배 밑 갭 확인까지 통과할 때만 버퍼를 실제로
    # 회전한다 — T-pose 팔 벌린 휴머노이드는 확인 단계에서 걸러져 원본이
    # 보존된다.
    cx_ext = _core_extent([p[0] for p in all_pos])
    cy_ext = _core_extent([p[1] for p in all_pos]) or 1e-9
    cz_ext = _core_extent([p[2] for p in all_pos])
    if cx_ext > cz_ext and cx_ext > cy_ext:
        rotated = [(-p[2], p[1], p[0]) for p in all_pos]  # yaw −90°: +X → +Z
        if _classify_body_type(rotated) == "quadruped":
            _rewrite_mesh_vec3(gltf, bin_data, lambda x, y, z: (-z, y, x))
            all_pos = rotated
    body_type = _classify_body_type(all_pos)
    # 복원 스케일 이상치(표준 치수 대비 수배 차이)는 모션 루트 이동까지
    # 왜곡하므로, 관절 배치 전에 체형별 표준 치수로 정규화한다.
    s = _normalize_character_scale(gltf, bin_data, body_type)
    if s != 1.0:
        all_pos = [(p[0] * s, p[1] * s, p[2] * s) for p in all_pos]
    # 4족이 −Z를 향하면 관절 배치 전에 메시를 +Z 방향으로 돌려 세운다 —
    # 안 그러면 머리/꼬리가 뒤바뀐 리깅이 되고 보행 모션이 뒤로 걷는다.
    if body_type == "quadruped" and _quadruped_faces_backward(all_pos):
        _flip_mesh_yaw180(gltf, bin_data)
        all_pos = [(-p[0], p[1], -p[2]) for p in all_pos]

    xs = [p[0] for p in all_pos]; ys = [p[1] for p in all_pos]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    cx = (min_x + max_x) / 2
    cz = (min([p[2] for p in all_pos]) + max([p[2] for p in all_pos])) / 2
    w = (max_x - min_x) or 1.0
    h = (max_y - min_y) or 1.0

    # 2. 조인트 정의 (name, 월드 위치, 부모 이름) -------------------------
    # 좌우 명명은 SMPL/glTF 휴머노이드 규약(캐릭터가 +Z를 향할 때 왼쪽=+X)을
    # 따른다. 반대로 배치하면 리타게팅 시 SMPL 좌팔 회전(수평→매달림,
    # rot(Z,−75°)류)이 −X 팔에 적용돼 정확히 미러 — 팔이 위로 꺾여 머리 옆에
    # 붙는 증상이 난다 (실측: idle 팔 elevation 기대 −75° ↔ 버그 +75°).
    if body_type == "quadruped":
        JDEF, BONE_CHILD, tips = _quadruped_joint_layout(all_pos)
    else:
        # 팔을 벌린 전체 bbox 폭(w)을 어깨 폭으로 직접 사용하면 손끝 폭의 60%가
        # 어깨 폭이 되어 비정상적인 역삼각 체형이 된다. 인체 어깨 관절 간격은
        # 신장의 약 22~32%이므로, 손끝 폭 기반 추정치를 신장 28%로 상한한다.
        # 팔꿈치는 기존 bbox 추정치를 유지하되 어깨보다 바깥에 있도록 보장한다.
        shoulder_half = min(w * .30, h * .14)
        elbow_half = max(shoulder_half + h * .10, min(w * .44, h * .32))
        JDEF = [
            ("Hips",         (cx,          min_y + h*.52, cz), None),
            ("Spine",        (cx,          min_y + h*.62, cz), "Hips"),
            ("Chest",        (cx,          min_y + h*.73, cz), "Spine"),
            ("Head",         (cx,          min_y + h*.88, cz), "Chest"),
            ("LeftUpLeg",    (cx + w*.13,  min_y + h*.48, cz), "Hips"),
            ("LeftLeg",      (cx + w*.13,  min_y + h*.27, cz), "LeftUpLeg"),
            ("LeftFoot",     (cx + w*.13,  min_y + h*.04, cz), "LeftLeg"),
            ("RightUpLeg",   (cx - w*.13,  min_y + h*.48, cz), "Hips"),
            ("RightLeg",     (cx - w*.13,  min_y + h*.27, cz), "RightUpLeg"),
            ("RightFoot",    (cx - w*.13,  min_y + h*.04, cz), "RightLeg"),
            ("LeftArm",      (cx + shoulder_half, min_y + h*.73, cz), "Chest"),
            ("LeftForeArm",  (cx + elbow_half,    min_y + h*.62, cz), "LeftArm"),
            ("RightArm",     (cx - shoulder_half, min_y + h*.73, cz), "Chest"),
            ("RightForeArm", (cx - elbow_half,    min_y + h*.62, cz), "RightArm"),
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

    # skin 등록 — 이름이 체형 마커를 겸한다 (검증기가 upright 기준 분기에 사용)
    gltf["skins"] = [{
        "name": "AutoQuadrupedRig" if body_type == "quadruped" else "AutoHumanoidRig",
        "joints": [name_to_node[n] for n in JNAMES],
        "inverseBindMatrices": ibm_acc,
        "skeleton": name_to_node["Hips"],
    }]

    # 5. 버텍스 조인트/가중치 할당 — 본 세그먼트 거리 기반 2-조인트 블렌딩.
    # 이전의 x/y 임계값 영역 방식은 어깨 경계에서 Chest↔Arm 배정이 급격히
    # 갈려 애니메이션 시 팔이 몸통에서 찢어져 보였다. 각 조인트가 지배하는
    # 본(조인트→자식 조인트, 말단은 연장 팁)까지의 거리로 배정하고 가까운 두
    # 본을 역제곱 거리로 블렌딩해 경계를 연속적으로 만든다.
    # (4족은 BONE_CHILD/tips가 _quadruped_joint_layout에서 실측으로 나온다.)
    if body_type != "quadruped":
        BONE_CHILD = {
            "Hips": "Spine", "Spine": "Chest", "Chest": "Head",
            "LeftUpLeg": "LeftLeg", "LeftLeg": "LeftFoot",
            "RightUpLeg": "RightLeg", "RightLeg": "RightFoot",
            "LeftArm": "LeftForeArm", "RightArm": "RightForeArm",
        }
        tips = {
            "Head": (cx, max_y, cz),
            "LeftFoot": (JWORLD["LeftFoot"][0], min_y, JWORLD["LeftFoot"][2]),
            "RightFoot": (JWORLD["RightFoot"][0], min_y, JWORLD["RightFoot"][2]),
        }
        for side in ("Left", "Right"):
            a, f = JWORLD[side + "Arm"], JWORLD[side + "ForeArm"]
            tips[side + "ForeArm"] = tuple(f[k] + (f[k] - a[k]) for k in range(3))  # 1차: 직선 연장

    def _bones(t):
        return [(ji, JWORLD[name], t.get(name) or JWORLD[BONE_CHILD[name]])
                for ji, name in enumerate(JNAMES)]

    def _seg_dist2(p, a, b):
        ab = [b[k] - a[k] for k in range(3)]
        ap = [p[k] - a[k] for k in range(3)]
        denom = sum(c * c for c in ab) or 1e-12
        t = max(0.0, min(1.0, sum(ap[k] * ab[k] for k in range(3)) / denom))
        return sum((ap[k] - t * ab[k]) ** 2 for k in range(3))

    # 전완 팁 실측 보정(2-pass): bbox 배치는 팔이 어깨→손까지 직선이라
    # 가정하지만 실측 캐릭터는 rest에서 팔꿈치가 36~64° 굽어 있어, 직선
    # 연장 팁으로는 손 지오메트리가 전완 본 거리장에서 멀어져 허벅지 본에
    # 지배권을 뺏긴다(A-pose 손끝-허벅지 4~13cm). 1차 지배 배정으로 전완
    # 지배 버텍스의 말단 원심을 구해 전완 본 세그먼트를 실제 손 방향으로
    # 다시 놓는다.
    # (4족은 전완=앞다리가 수직 하강 세그먼트로 팁이 이미 지면 실측이라 불필요.)
    bones = _bones(tips)
    if body_type != "quadruped":
        dom = [min(bones, key=lambda b: _seg_dist2(p, b[1], b[2]))[0] for p in all_pos]
        for side in ("Left", "Right"):
            fj = JNAMES.index(side + "ForeArm")
            fpos = JWORLD[side + "ForeArm"]
            mine = [p for p, d in zip(all_pos, dom) if d == fj]
            if len(mine) < 8:
                continue
            far = sorted(mine, key=lambda p: -sum((p[k] - fpos[k]) ** 2 for k in range(3)))
            far = far[:max(1, len(far) // 10)]
            c = [sum(p[k] for p in far) / len(far) for k in range(3)]
            if sum((c[k] - fpos[k]) ** 2 for k in range(3)) > 1e-8:
                # 원심을 15% 지나치게 연장해 손끝까지 세그먼트가 닿도록 함
                tips[side + "ForeArm"] = tuple(fpos[k] + 1.15 * (c[k] - fpos[k]) for k in range(3))
        bones = _bones(tips)

    # 체인 인접 블렌딩 제한: 2번째 본은 지배 본과 스켈레톤에서 인접(부모/
    # 자식)한 본만 허용한다. 순수 거리 기반이던 이전 방식은 손 버텍스의
    # 21~53%가 UpLeg 보조 웨이트를 받아(실측 4캐릭터), 팔을 들거나 몸을
    # 숙이면 손이 다리에 앵커링된 채 늘어나는 웨빙·"손이 몸에 붙는" 증상의
    # 주범이었다.
    ADJACENT = {name: set() for name in JNAMES}
    for name, _, parent in JDEF:
        if parent:
            ADJACENT[name].add(parent)
            ADJACENT[parent].add(name)

    def joint_weights(p):
        """지배 조인트 + 인접 본 중 최근접 1개와 정규화 가중치 — (j1, j2, w1, w2)."""
        dists = [(_seg_dist2(p, a, b), ji) for ji, a, b in bones]
        d1, j1 = min(dists)
        if d1 < 1e-12:
            return j1, 0, 1.0, 0.0
        allowed = ADJACENT[JNAMES[j1]]
        d2, j2 = min((d, ji) for d, ji in dists if JNAMES[ji] in allowed)
        w1, w2 = 1.0 / d1, 1.0 / d2  # d는 거리 제곱 — 역제곱 가중
        return j1, j2, w1 / (w1 + w2), w2 / (w1 + w2)

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
            doms = []
            for pos in positions:
                j1, j2, w1, w2 = joint_weights(pos)
                doms.append(JNAMES[j1])
                j_bytes += struct.pack("<4H", j1, j2, 0, 0)
                w_bytes += struct.pack("<4f", w1, w2, 0.0, 0.0)
            if body_type != "quadruped":
                # 융합 브리지 절단은 인체 A-pose 체인(손↔허벅지 등) 전제 —
                # 4족은 다리 4개가 몸통에 정상적으로 붙어 있어 오히려 절단 위험.
                _cut_fused_bridges(gltf, bin_data, prim, doms)
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
    return body_type


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


def _quat_conj(q):
    return (-q[0], -q[1], -q[2], q[3])


def _quat_from_to(u, v):
    """단위벡터 u→v 최단호 회전 쿼터니언 (xyzw)."""
    d = sum(u[k] * v[k] for k in range(3))
    if d > 1.0 - 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    if d < -1.0 + 1e-9:
        # 정반대 방향: u에 수직인 아무 축으로 180°
        axis = (0.0, 1.0, 0.0) if abs(u[0]) > 0.9 else (1.0, 0.0, 0.0)
        c = (u[1] * axis[2] - u[2] * axis[1],
             u[2] * axis[0] - u[0] * axis[2],
             u[0] * axis[1] - u[1] * axis[0])
        n = math.sqrt(sum(x * x for x in c)) or 1.0
        return (c[0] / n, c[1] / n, c[2] / n, 0.0)
    c = (u[1] * v[2] - u[2] * v[1],
         u[2] * v[0] - u[0] * v[2],
         u[0] * v[1] - u[1] * v[0])
    q = (c[0], c[1], c[2], 1.0 + d)
    n = math.sqrt(sum(x * x for x in q)) or 1.0
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def _rig_rest_world(gltf, node_by_name):
    """Hips에서 자식 체인을 따라 각 조인트의 rest 월드 위치를 계산한다
    (auto_rig 조인트는 회전 없는 순수 이동 노드 — 이동 누적으로 충분)."""
    world = {}
    stack = [(node_by_name["Hips"], (0.0, 0.0, 0.0))]
    while stack:
        idx, pw = stack.pop()
        node = gltf["nodes"][idx]
        t = node.get("translation", [0.0, 0.0, 0.0])
        w = (pw[0] + t[0], pw[1] + t[1], pw[2] + t[2])
        if node.get("name"):
            world[node["name"]] = w
        for child in node.get("children", []):
            stack.append((child, w))
    return world


def _mesh_forearm_dirs(gltf, bin_data, world):
    """스킨 웨이트로 전완 지배 버텍스를 찾아 실측 전완 방향을 구한다.

    반환: {"Left": 단위벡터, "Right": 단위벡터} — 전완 조인트에서 지배
    버텍스 말단 원심(원거리 10%)으로 향하는 방향. 실측 불가 시 키 없음.
    """
    skins = gltf.get("skins") or []
    if not skins:
        return {}
    joints = skins[0]["joints"]
    jname = [gltf["nodes"][j].get("name") for j in joints]
    prim = next((p for m in gltf.get("meshes", []) for p in m.get("primitives", [])
                 if "JOINTS_0" in p.get("attributes", {})), None)
    if prim is None:
        return {}
    attrs = prim["attributes"]
    pos = _read_vec3(gltf, bin_data, attrs["POSITION"], "POSITION")

    def read_vec4(acc_index, fmt_map):
        acc = gltf["accessors"][acc_index]
        view = gltf["bufferViews"][acc["bufferView"]]
        base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        fmt = fmt_map[acc["componentType"]]
        stride = view.get("byteStride", struct.calcsize(fmt) * 4)
        return [struct.unpack_from(f"<4{fmt}", bin_data, base + i * stride)
                for i in range(acc["count"])]

    vj = read_vec4(attrs["JOINTS_0"], {5121: "B", 5123: "H", 5125: "I"})
    vw = read_vec4(attrs["WEIGHTS_0"], {5126: "f"})
    dirs = {}
    for side in ("Left", "Right"):
        fore = world.get(side + "ForeArm")
        if not fore:
            continue
        mine = []
        for i in range(len(pos)):
            slot = max(range(4), key=lambda s: vw[i][s])
            if vj[i][slot] < len(jname) and jname[vj[i][slot]] == side + "ForeArm":
                mine.append(pos[i])
        if len(mine) < 8:
            continue
        far = sorted(mine, key=lambda p: -sum((p[k] - fore[k]) ** 2 for k in range(3)))
        far = far[:max(1, len(far) // 10)]
        c = [sum(p[k] for p in far) / len(far) for k in range(3)]
        v = [c[k] - fore[k] for k in range(3)]
        n = math.sqrt(sum(x * x for x in v))
        if n > 1e-6:
            dirs[side] = [x / n for x in v]
    return dirs


def _arm_rest_deltas(gltf, bin_data, node_by_name):
    """SMPL rest(T-pose, 팔 수평)와 rig rest(A-pose, 팔 대각선)의 차이 보정.

    SMPL 로컬 회전을 그대로 이식하면 rest 각도 차이만큼 팔이 추가로 꺾인다.
    각 팔 조인트에 대해 D = (수평 팔 방향 → rig rest 팔 방향) 최단호 회전을
    구해, 리타겟 시 q' = D_parent ⊗ q ⊗ D⁻¹로 보정한다 (두 스켈레톤 모두
    rest 조인트 로컬 프레임이 월드축 정렬이라 이 형태로 충분).

    상완 delta는 리그 조인트 축(Arm→ForeArm)으로 구하고, 전완 delta는 메시
    실측 전완 방향(_mesh_forearm_dirs)으로 구한다 — 실제 캐릭터는 rest에서
    팔꿈치가 굽어 있어(실측 36~64°) 전완을 상완의 직선 연속으로 가정하면
    팔꿈치 이하 월드 방향이 그만큼 틀어져 손이 몸통으로 파고든다.
    반환: (delta, delta_parent) — 조인트명 → 쿼터니언(xyzw), 없으면 항등 취급.
    """
    world = _rig_rest_world(gltf, node_by_name)
    fore_dirs = _mesh_forearm_dirs(gltf, bin_data, world)
    delta, delta_parent = {}, {}
    for side in ("Left", "Right"):
        arm, fore = world.get(side + "Arm"), world.get(side + "ForeArm")
        if not arm or not fore:
            continue
        v = [fore[k] - arm[k] for k in range(3)]
        n = math.sqrt(sum(c * c for c in v))
        if n < 1e-9:
            continue
        v = [c / n for c in v]
        u = (1.0 if v[0] >= 0 else -1.0, 0.0, 0.0)  # T-pose: 같은 쪽 수평 바깥 방향
        d = _quat_from_to(u, v)
        delta[side + "Arm"] = d
        vf = fore_dirs.get(side)
        # 전완 실측 방향이 있으면 그것으로, 없으면 상완 연속 가정으로 폴백
        delta[side + "ForeArm"] = _quat_from_to(u, vf) if vf else d
        delta_parent[side + "ForeArm"] = d   # 부모(Arm)의 delta
    return delta, delta_parent


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
    # SMPL rest(T-pose) ↔ rig rest(A-pose) 팔 각도 차이 보정용 rest-delta
    arm_delta, arm_delta_parent = _arm_rest_deltas(gltf, bin_data, node_by_name)
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
                dj = arm_delta.get(rig_name)
                if dj:
                    q = _quat_mul(q, _quat_conj(dj))
                    dp = arm_delta_parent.get(rig_name)
                    if dp:
                        q = _quat_mul(dp, q)
                norm = math.sqrt(sum(c * c for c in q)) or 1.0
                blob += struct.pack("<4f", *(c / norm for c in q))
            add_channel(node, "rotation", bytes(blob), "VEC4")
        # 4족 꼬리 스웨이: Tail은 SMPL에 대응 관절이 없어 그대로 두면 모든
        # 모션에서 꼬리가 강직 상태다. 걸음 표준 주기(~1.25Hz)의 yaw 사인
        # 스웨이에 2배 주파수·절반 이하 진폭의 pitch 바운스를 합성해
        # 자연스러운 흔들림을 만든다 (자체 리깅 4족 마커가 있을 때만).
        tail_node = node_by_name.get("Tail")
        if tail_node is not None and gltf["skins"][0].get("name") == "AutoQuadrupedRig":
            blob = bytearray()
            for f in range(frames):
                yaw = math.radians(15.0) * math.sin(2 * math.pi * 1.25 * times[f])
                pitch = math.radians(6.0) * math.sin(2 * math.pi * 2.5 * times[f])
                q = _quat_mul((0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)),
                              (math.sin(pitch / 2), 0.0, 0.0, math.cos(pitch / 2)))
                blob += struct.pack("<4f", *q)
            add_channel(tail_node, "rotation", bytes(blob), "VEC4")
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


def neutralize_material(glb_path, output_path):
    """Hunyuan3D 텍스처 출력의 광택 과다 재질을 중화한 GLB를 쓴다.

    실측(2026-07-20, 구/신 캐릭터 공통): Hunyuan3D-2.1 텍스처 파이프라인은
    KHR_materials_specular specularColorFactor=[2,2,2](스펙큘러 2배)에
    metallicFactor 미지정(glTF 기본값 1.0 = 완전 금속) + metallicRoughness
    텍스처(roughness 평균 ~0.43-0.62)를 함께 내보내, 표준 3광 뷰어 조명에서
    전신이 광택 플라스틱 인형처럼 보이고 밝은 하이라이트 블롭(부유 파편처럼
    보임)까지 만든다. baseColor 텍스처만 남기고 specular 확장 제거,
    metallic 0, roughness 1.0으로 중화한다 (bake_texture의 오프라인
    투영 머티리얼과 동일 기준). 변경이 있으면 True, no-op이면 False(멱등).
    """
    gltf, bin_data = _read_glb(glb_path)
    materials = gltf.get("materials") or []
    changed = False
    for mat in materials:
        ext = mat.get("extensions") or {}
        if "KHR_materials_specular" in ext:
            del ext["KHR_materials_specular"]
            if not ext:
                mat.pop("extensions", None)
            changed = True
        pbr = mat.setdefault("pbrMetallicRoughness", {})
        if pbr.get("metallicFactor") != 0.0:
            pbr["metallicFactor"] = 0.0
            changed = True
        if pbr.get("roughnessFactor") != 1.0:
            pbr["roughnessFactor"] = 1.0
            changed = True
        if "metallicRoughnessTexture" in pbr:
            # 텍스처 참조만 제거한다 (이미지 바이트는 남지만 파괴적 재작성보다
            # 안전 — 기등록 GLB 인플레이스 교정에도 같은 경로를 쓴다).
            del pbr["metallicRoughnessTexture"]
            changed = True
    if not changed:
        return False
    # 더 이상 쓰지 않는 확장 선언 정리 (검증기의 extensionsUsed 일관성 유지)
    if not any("KHR_materials_specular" in (m.get("extensions") or {}) for m in materials):
        for key in ("extensionsUsed", "extensionsRequired"):
            lst = gltf.get(key)
            if lst and "KHR_materials_specular" in lst:
                lst.remove("KHR_materials_specular")
                if not lst:
                    gltf.pop(key, None)
    _write_glb(gltf, bin_data, output_path)
    return True


def render_check(path):
    """tools/matrix 검증기로 GLB 렌더링 정상성 평가 — 리포트 dict.

    직립/관절계층/클립변형/팔자세/스키닝을 실측해 "누워서 리깅됨",
    "팔이 위로 꺾임" 같은 버그가 앱 파이프라인을 통과하지 못하게 막는다.
    검증기 파일이 없는 배포 환경(예: RunPod 컨테이너)에서는 None을 반환해
    게이트를 건너뛴다. 앱은 임베드된 검증기를 워커와 같은 runtime 디렉토리에
    추출하므로 형제 경로를 먼저, repo 체크아웃 경로를 다음으로 찾는다.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    vpath = next((p for p in (os.path.join(here, "validate_character.py"),
                              os.path.join(here, "..", "tools", "matrix", "validate_character.py"))
                  if os.path.exists(p)), None)
    if vpath is None:
        return None
    spec = importlib.util.spec_from_file_location("validate_character", vpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.validate(path)


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
            # 저해상도 진흙 복원은 어떤 후처리로도 품질을 살릴 수 없으므로
            # 파이프라인 진입 전에 명시 실패시켜 이미지 재생성을 유도한다
            # (실측: gladiator 11,483 verts — 사용자 신고 "완전 심각").
            # 오프라인 procedural 프리뷰(character.glb, 12 verts)는 의도적
            # 저폴리이므로 실제 Hunyuan 복원 산출물에만 적용한다.
            if os.path.basename(source) == "hunyuan3d21.glb":
                verts = total_vertex_count(source)
                if verts < RECON_MIN_VERTICES:
                    raise LowResolutionMeshError(
                        f"reconstruction is low-res mud quality ({verts} vertices "
                        f"< {RECON_MIN_VERTICES}) — regenerate the source image")
            # 복원 메시가 바닥 판(base slab) 위에 서 있으면 슬래브만 잘라
            # 캐릭터를 구제한다 (정상 메시는 no-op — 실측: gladiator/vampire-v1).
            stripped = 0
            try:
                stripped = strip_base_plane(source, output)
            except BasePlaneRescueError:
                # 미니어처 구제는 저해상도(진흙 외형) 결과만 남으므로 조용한
                # passthrough 대신 작업을 명시 실패시켜 이미지 재생성을 유도한다.
                raise
            except Exception as exc:
                emit("progress", jobId=job, progress=.3, message=f"Base-plane strip skipped: {exc}")
            if stripped:
                transformed = True
                source = output  # 이후 텍스처 베이킹은 슬래브 제거본을 입력으로
                metrics = {"adapter": "base-plane-strip", "validated": True,
                           "strippedVertices": stripped, "previewOnly": True}
                message = f"Removed base plane ({stripped} vertices) from reconstruction"
            # shape-only 메시(텍스처 없음)에 참조 이미지를 정면 투영으로 베이킹.
            reference = find_reference_image(workspace)
            if reference:
                emit("progress", jobId=job, progress=.4, message="Projecting reference image onto mesh")
                try:
                    baked = bake_texture(source, reference, output)
                    if baked:
                        transformed = True
                        metrics = {"adapter": "front-projection-offline", "validated": True, "textured": True, "previewOnly": True}
                        if stripped:
                            metrics["strippedVertices"] = stripped
                        message = "Baked front-projected base color texture"
                except Exception as exc:
                    emit("progress", jobId=job, progress=.5, message=f"Texture projection skipped: {exc}")
            # Hunyuan 텍스처 출력의 광택 과다 재질(specular 2.0 + metallic 1.0)을
            # 중화 — 뷰어에서 플라스틱 인형처럼 보이는 문제의 근본 수정
            # (실측: nurse/gladiator, 구/신 캐릭터 공통).
            try:
                neutralized = neutralize_material(output if transformed else source, output)
            except Exception as exc:
                neutralized = False
                emit("progress", jobId=job, progress=.55, message=f"Material neutralize skipped: {exc}")
            if neutralized:
                if not transformed:
                    metrics = {"adapter": "material-neutralize", "validated": True, "previewOnly": True}
                    message = "Neutralized reconstruction material (specular/metallic)"
                transformed = True
                metrics["materialNeutralized"] = True
        elif stage == "rig":
            # skeleton이 없는 static mesh에 실측 기반 rig 추가 (체형 자동 판별:
            # 직립 → humanoid 14조인트, 체장>키 → quadruped 수평 척추 스켈레톤).
            emit("progress", jobId=job, progress=.35, message="Fitting auto-detected skeleton to mesh")
            try:
                transformed = auto_rig(source, output)  # "humanoid"/"quadruped"/False
                if transformed:
                    metrics = {"adapter": "auto-rig-bbox", "validated": True, "skinned": True,
                               "bodyType": transformed, "previewOnly": True}
                    message = f"Fitted {transformed} skeleton with skin weights"
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
        if transformed and stage in ("rig", "motion"):
            # 렌더링 정상성 게이트: 변환 실패는 위에서 passthrough로 계속하지만,
            # 변환이 "성공"했는데 결과가 몬스터(누움/팔 꺾임/웨빙)면 조용히
            # 통과시키지 않고 stage를 실패시켜 사용자에게 드러낸다.
            emit("progress", jobId=job, progress=.65, message="Running render-validity checks")
            check = render_check(output)
            if check is not None:
                metrics["renderValid"] = check["ok"]
                if not check["ok"]:
                    issues = [i for sec in ("upright", "hierarchy", "legs", "deformation",
                                            "arm_pose", "skinning", "hands")
                              for i in check.get(sec, {}).get("issues", [])]
                    metrics["renderIssues"] = issues
                    raise RuntimeError(f"{stage} render validation failed: " + "; ".join(issues))
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
