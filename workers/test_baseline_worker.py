#!/usr/bin/env python3
"""bake_texture(GLB front-projection baking) 단위 테스트.

baseline_worker.py는 모듈 레벨에서 stdin 루프를 돌기 때문에, 빈 stdin으로
바꿔치기한 뒤 임포트한다 (루프가 즉시 종료되어 부작용 없음).
"""
import importlib.util
import io
import json
import os
import struct
import sys
import tempfile
import unittest
import zlib

sys.stdin = io.StringIO("")
_spec = importlib.util.spec_from_file_location(
    "baseline_worker", os.path.join(os.path.dirname(__file__), "baseline_worker.py"))
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)


def make_png():
    def chunk(typ, data):
        return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", zlib.crc32(typ + data))
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\x00\x00")
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def make_glb(path, with_material=False, with_normals=True):
    """최소 사각형 GLB. with_normals=False면 Hunyuan shape-only 출력처럼
    POSITION+indices만 가진다 (법선은 워커가 직접 계산해야 함)."""
    positions = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
    normals = [(0, 0, 1), (0, 0, 1), (0, 0, -1), (0, 0, -1)]
    blob = bytearray()
    for v in positions:
        blob += struct.pack("<fff", *v)
    attributes = {"POSITION": 0}
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": 0}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 48}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"}],
        "meshes": [{"primitives": [{"attributes": attributes}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    if with_normals:
        offset = len(blob)
        for v in normals:
            blob += struct.pack("<fff", *v)
        gltf["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": 48})
        gltf["accessors"].append({"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"})
        attributes["NORMAL"] = 1
    else:
        # 반시계(+z를 향하는) 삼각형 두 개로 사각형 구성
        offset = len(blob)
        indices = (0, 1, 2, 0, 2, 3)
        blob += struct.pack("<6H", *indices)
        gltf["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": 12})
        gltf["accessors"].append({"bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR"})
        gltf["meshes"][0]["primitives"][0]["indices"] = 1
    gltf["buffers"][0]["byteLength"] = len(blob)
    if with_material:
        gltf["materials"] = [{"name": "Authored"}]
        gltf["meshes"][0]["primitives"][0]["material"] = 0
    worker._write_glb(gltf, blob, path)


class BakeTextureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="spriteengine-test-")
        self.addCleanup(self.tmp.cleanup)
        self.glb = os.path.join(self.tmp.name, "in.glb")
        self.out = os.path.join(self.tmp.name, "out.glb")
        self.png = os.path.join(self.tmp.name, "ref.png")
        with open(self.png, "wb") as f:
            f.write(make_png())

    def read_uv(self, gltf, bin_data, accessor_index):
        acc = gltf["accessors"][accessor_index]
        view = gltf["bufferViews"][acc["bufferView"]]
        base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        return [struct.unpack_from("<ff", bin_data, base + i * 8) for i in range(acc["count"])]

    def test_bakes_texture_uv_and_backface_shade(self):
        make_glb(self.glb)
        self.assertTrue(worker.bake_texture(self.glb, self.png, self.out))
        gltf, bin_data = worker._read_glb(self.out)
        self.assertEqual(len(gltf["images"]), 1)
        self.assertEqual(gltf["images"][0]["mimeType"], "image/png")
        self.assertEqual(len(gltf["textures"]), 1)
        self.assertEqual(gltf["materials"][0]["name"], "FrontProjectedBaseColor")
        attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
        # UV: (-1,-1)→(0,1), (1,1)→(1,0) — v축은 이미지 상단이 0
        uv = self.read_uv(gltf, bin_data, attrs["TEXCOORD_0"])
        self.assertEqual(uv[0], (0.0, 1.0))
        self.assertEqual(uv[2], (1.0, 0.0))
        # COLOR_0: 앞면(nz=1)은 255, 뒷면(nz=-1)은 0.55 음영
        acc = gltf["accessors"][attrs["COLOR_0"]]
        self.assertEqual((acc["componentType"], acc["type"], acc.get("normalized")), (5121, "VEC4", True))
        view = gltf["bufferViews"][acc["bufferView"]]
        base = view["byteOffset"]
        front, back = bin_data[base], bin_data[base + 8]
        self.assertEqual(front, 255)
        self.assertEqual(back, int(255 * 0.55))

    def test_computes_normals_when_missing(self):
        # Hunyuan shape-only 출력: POSITION+indices만 존재 → 법선 계산 후 음영
        make_glb(self.glb, with_normals=False)
        self.assertTrue(worker.bake_texture(self.glb, self.png, self.out))
        gltf, bin_data = worker._read_glb(self.out)
        attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
        self.assertIn("COLOR_0", attrs)
        acc = gltf["accessors"][attrs["COLOR_0"]]
        base = gltf["bufferViews"][acc["bufferView"]]["byteOffset"]
        # 모든 정점이 +z를 향하는 평면이므로 전부 최대 밝기여야 함
        for i in range(acc["count"]):
            self.assertEqual(bin_data[base + i * 4], 255)

    def test_glb_structure_stays_valid(self):
        make_glb(self.glb)
        worker.bake_texture(self.glb, self.png, self.out)
        data = open(self.out, "rb").read()
        magic, version, total = struct.unpack("<III", data[:12])
        self.assertEqual((magic, version, total), (worker.GLB_MAGIC, 2, len(data)))
        gltf, bin_data = worker._read_glb(self.out)
        self.assertEqual(gltf["buffers"][0]["byteLength"], len(bin_data))
        for view in gltf["bufferViews"]:
            self.assertLessEqual(view.get("byteOffset", 0) + view["byteLength"], len(bin_data))

    def test_preserves_authored_materials(self):
        make_glb(self.glb, with_material=True)
        self.assertFalse(worker.bake_texture(self.glb, self.png, self.out))
        self.assertFalse(os.path.exists(self.out))

    def test_rejects_non_png_jpeg_reference(self):
        make_glb(self.glb)
        webp = os.path.join(self.tmp.name, "ref.webp")
        with open(webp, "wb") as f:
            f.write(b"RIFF\x00\x00\x00\x00WEBPVP8 ")
        self.assertFalse(worker.bake_texture(self.glb, webp, self.out))

    def test_find_reference_image(self):
        ws = self.tmp.name
        os.makedirs(os.path.join(ws, "prepare"), exist_ok=True)
        self.assertIsNone(worker.find_reference_image(ws))
        ref = os.path.join(ws, "prepare", "reference.png")
        with open(ref, "wb") as f:
            f.write(make_png())
        self.assertEqual(worker.find_reference_image(ws), ref)


def make_static_glb(path):
    """스켈레톤 없는 단순 직사각형 GLB — auto_rig 입력용."""
    positions = [(-0.3, 0.0, -0.3), (0.3, 0.0, -0.3), (0.3, 1.8, -0.3), (-0.3, 1.8, -0.3),
                 (-0.3, 0.0,  0.3), (0.3, 0.0,  0.3), (0.3, 1.8,  0.3), (-0.3, 1.8,  0.3)]
    faces = [0,1,2, 0,2,3, 4,5,6, 4,6,7, 0,4,7, 0,7,3, 1,5,6, 1,6,2, 0,1,5, 0,5,4, 3,2,6, 3,6,7]
    buf = bytearray()
    for v in positions:
        buf += struct.pack("<fff", *v)
    idx_off = len(buf)
    for i in faces:
        buf += struct.pack("<H", i)
    gltf = {
        "asset": {"version": "2.0"}, "scene": 0,
        "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(buf)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 96},
            {"buffer": 0, "byteOffset": 96, "byteLength": len(buf) - 96},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": len(faces), "type": "SCALAR"},
        ],
    }
    worker._write_glb(gltf, buf, path)


class AutoRigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="spriteengine-autorig-")
        self.addCleanup(self.tmp.cleanup)
        self.glb = os.path.join(self.tmp.name, "static.glb")
        self.out = os.path.join(self.tmp.name, "rigged.glb")

    def test_adds_skin_and_joints(self):
        make_static_glb(self.glb)
        self.assertTrue(worker.auto_rig(self.glb, self.out))
        gltf, _ = worker._read_glb(self.out)
        self.assertEqual(len(gltf.get("skins", [])), 1)
        skin = gltf["skins"][0]
        self.assertEqual(skin["name"], "AutoHumanoidRig")
        self.assertEqual(len(skin["joints"]), 14)

    def test_adds_joints0_and_weights0(self):
        make_static_glb(self.glb)
        worker.auto_rig(self.glb, self.out)
        gltf, bin_data = worker._read_glb(self.out)
        attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
        self.assertIn("JOINTS_0", attrs)
        self.assertIn("WEIGHTS_0", attrs)
        # WEIGHTS_0: 2-조인트 블렌딩 — 합=1, 첫 슬롯이 지배(>=0.5) 가중치
        acc = gltf["accessors"][attrs["WEIGHTS_0"]]
        view = gltf["bufferViews"][acc["bufferView"]]
        base = view["byteOffset"]
        for i in range(acc["count"]):
            w = struct.unpack_from("<4f", bin_data, base + i * 16)
            self.assertAlmostEqual(sum(w), 1.0, places=5)
            self.assertGreaterEqual(w[0], 0.5 - 1e-6)
            self.assertEqual(w[2:], (0.0, 0.0))

    def test_glb_structure_valid(self):
        make_static_glb(self.glb)
        worker.auto_rig(self.glb, self.out)
        data = open(self.out, "rb").read()
        magic, version, total = struct.unpack("<III", data[:12])
        self.assertEqual((magic, version, total), (worker.GLB_MAGIC, 2, len(data)))
        gltf, bin_data = worker._read_glb(self.out)
        self.assertEqual(gltf["buffers"][0]["byteLength"], len(bin_data))
        for view in gltf["bufferViews"]:
            self.assertLessEqual(view.get("byteOffset", 0) + view["byteLength"], len(bin_data))

    def test_passthrough_when_already_skinned(self):
        # procedural_character.py 출력처럼 이미 skin이 있으면 False 반환
        make_static_glb(self.glb)
        worker.auto_rig(self.glb, self.out)  # first pass adds skin
        out2 = os.path.join(self.tmp.name, "rigged2.glb")
        self.assertFalse(worker.auto_rig(self.out, out2))
        self.assertFalse(os.path.exists(out2))

    def test_weight_regions(self):
        """하체 버텍스는 하체 조인트, 상체는 상체 조인트에 할당된다."""
        make_static_glb(self.glb)
        worker.auto_rig(self.glb, self.out)
        gltf, bin_data = worker._read_glb(self.out)
        skin = gltf["skins"][0]
        joint_nodes = skin["joints"]
        node_name = {n: gltf["nodes"][n]["name"] for n in joint_nodes}
        attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
        pos_acc = gltf["accessors"][attrs["POSITION"]]
        positions = worker._read_vec3(gltf, bin_data, attrs["POSITION"], "POSITION")
        acc = gltf["accessors"][attrs["JOINTS_0"]]
        view = gltf["bufferViews"][acc["bufferView"]]
        base = view["byteOffset"]
        for i, (x, y, z) in enumerate(positions):
            ji = struct.unpack_from("<H", bin_data, base + i * 8)[0]
            jname = node_name[joint_nodes[ji]]
            if y < 0.1 * 1.8:
                self.assertIn("Foot", jname, f"vertex at y={y:.2f} should be Foot, got {jname}")
            elif y > 0.8 * 1.8:
                self.assertEqual(jname, "Head", f"vertex at y={y:.2f} should be Head, got {jname}")


Q_NEG90X = [-0.7071067811865475, 0.0, 0.0, 0.7071067811865476]  # X축 -90°: 로컬 Z-up → Y-up


def make_rotated_static_glb(path, rotation=Q_NEG90X):
    """Hunyuan3D-2.1 출력 재현: 로컬 Z가 키(1.8m), Y는 깊이(0.6m)이고,
    노드 회전으로 뷰어에서 Y-up 직립으로 보정되는 GLB (버그 회귀 테스트용)."""
    positions = [(-0.3, -0.3, 0.0), (0.3, -0.3, 0.0), (0.3, -0.3, 1.8), (-0.3, -0.3, 1.8),
                 (-0.3,  0.3, 0.0), (0.3,  0.3, 0.0), (0.3,  0.3, 1.8), (-0.3,  0.3, 1.8)]
    faces = [0,1,2, 0,2,3, 4,5,6, 4,6,7, 0,4,7, 0,7,3, 1,5,6, 1,6,2, 0,1,5, 0,5,4, 3,2,6, 3,6,7]
    buf = bytearray()
    for v in positions:
        buf += struct.pack("<fff", *v)
    idx_off = len(buf)
    for i in faces:
        buf += struct.pack("<H", i)
    gltf = {
        "asset": {"version": "2.0"}, "scene": 0,
        "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "rotation": list(rotation)}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(buf)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": idx_off},
            {"buffer": 0, "byteOffset": idx_off, "byteLength": len(buf) - idx_off},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": len(faces), "type": "SCALAR"},
        ],
    }
    worker._write_glb(gltf, buf, path)


class NodeTransformBakeTest(unittest.TestCase):
    """스킨을 추가하면 메시 노드 자신의 트랜스폼이 렌더러에서 무시되어(glTF 스펙),
    Hunyuan3D-2.1 출력이 "누워서" 리깅되던 버그의 회귀 테스트."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="spriteengine-nodebake-")
        self.addCleanup(self.tmp.cleanup)
        self.glb = os.path.join(self.tmp.name, "rotated.glb")
        self.out = os.path.join(self.tmp.name, "rigged.glb")
        make_rotated_static_glb(self.glb)

    def test_auto_rig_resets_node_rotation(self):
        self.assertTrue(worker.auto_rig(self.glb, self.out))
        gltf, _ = worker._read_glb(self.out)
        node = next(n for n in gltf["nodes"] if "mesh" in n)
        self.assertEqual(node.get("rotation"), [0.0, 0.0, 0.0, 1.0])

    def test_auto_rig_bakes_true_height_into_y_axis(self):
        worker.auto_rig(self.glb, self.out)
        gltf, bin_data = worker._read_glb(self.out)
        attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
        positions = worker._read_vec3(gltf, bin_data, attrs["POSITION"], "POSITION")
        ys = [p[1] for p in positions]
        # 회전을 굽지 않으면 로컬 Y(깊이) 범위는 0.6에 불과하다 — 반드시 실제
        # 키(1.8, 로컬 Z였던 축)가 Y축 범위가 되어야 한다.
        self.assertAlmostEqual(max(ys) - min(ys), 1.8, places=4)

    def test_weight_regions_match_true_upright_pose(self):
        """구운 좌표 기준으로 발/머리 조인트가 실제 몸 형태에 맞게 배정되는지 확인
        (버그 상태에서는 눌린 0.6m 범위 안에서 잘못 배정된다)."""
        worker.auto_rig(self.glb, self.out)
        gltf, bin_data = worker._read_glb(self.out)
        skin = gltf["skins"][0]
        joint_nodes = skin["joints"]
        node_name = {n: gltf["nodes"][n]["name"] for n in joint_nodes}
        attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
        positions = worker._read_vec3(gltf, bin_data, attrs["POSITION"], "POSITION")
        acc = gltf["accessors"][attrs["JOINTS_0"]]
        view = gltf["bufferViews"][acc["bufferView"]]
        base = view["byteOffset"]
        for i, (x, y, z) in enumerate(positions):
            ji = struct.unpack_from("<H", bin_data, base + i * 8)[0]
            jname = node_name[joint_nodes[ji]]
            if y < 0.1 * 1.8:
                self.assertIn("Foot", jname, f"vertex at y={y:.2f} should be Foot, got {jname}")
            elif y > 0.8 * 1.8:
                self.assertEqual(jname, "Head", f"vertex at y={y:.2f} should be Head, got {jname}")

    def test_bake_texture_projects_using_true_upright_axis(self):
        """retopo 단계 텍스처 투영도 회전 보정 후 좌표를 사용해야 한다."""
        png = os.path.join(self.tmp.name, "ref.png")
        with open(png, "wb") as f:
            f.write(make_png())
        out = os.path.join(self.tmp.name, "textured.glb")
        self.assertTrue(worker.bake_texture(self.glb, png, out))
        gltf, bin_data = worker._read_glb(out)
        node = next(n for n in gltf["nodes"] if "mesh" in n)
        self.assertEqual(node.get("rotation"), [0.0, 0.0, 0.0, 1.0])
        attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
        positions = worker._read_vec3(gltf, bin_data, attrs["POSITION"], "POSITION")
        ys = [p[1] for p in positions]
        self.assertAlmostEqual(max(ys) - min(ys), 1.8, places=4)


SMPLH_JOINTS = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
    "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist",
]
Q_ID = [0.0, 0.0, 0.0, 1.0]
Q_90X = [0.7071067811865475, 0.0, 0.0, 0.7071067811865476]


def make_motion(frames=3, spine_bend=False, trans=None):
    """HY-Motion 응답 스키마의 합성 모션 페이로드."""
    quats = []
    for _ in range(frames):
        row = [list(Q_ID) for _ in SMPLH_JOINTS]
        if spine_bend:
            row[SMPLH_JOINTS.index("Spine2")] = list(Q_90X)
            row[SMPLH_JOINTS.index("Spine3")] = list(Q_90X)
        quats.append(row)
    return {"motions": [{
        "id": "test-clip", "text": "test", "fps": 30, "frames": frames,
        "joints": SMPLH_JOINTS, "quats": quats,
        "trans": trans or [[0.0, 0.0, 0.0]] * frames,
    }]}


class BakeAnimationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="spriteengine-anim-")
        self.addCleanup(self.tmp.cleanup)
        static = os.path.join(self.tmp.name, "static.glb")
        self.rigged = os.path.join(self.tmp.name, "rigged.glb")
        self.out = os.path.join(self.tmp.name, "animated.glb")
        make_static_glb(static)
        worker.auto_rig(static, self.rigged)

    def read_floats(self, gltf, bin_data, acc_index, width):
        acc = gltf["accessors"][acc_index]
        view = gltf["bufferViews"][acc["bufferView"]]
        base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        return [struct.unpack_from(f"<{width}f", bin_data, base + i * 4 * width)
                for i in range(acc["count"])]

    def channel_output(self, gltf, anim, node_name, path):
        node = next(i for i, n in enumerate(gltf["nodes"]) if n.get("name") == node_name)
        ch = next(c for c in anim["channels"]
                  if c["target"] == {"node": node, "path": path})
        return anim["samplers"][ch["sampler"]]["output"]

    def test_bakes_channels_for_all_rig_joints(self):
        self.assertEqual(worker.bake_animation(self.rigged, make_motion(), self.out), 1)
        gltf, _ = worker._read_glb(self.out)
        anim = gltf["animations"][0]
        self.assertEqual(anim["name"], "test-clip")
        # 14 조인트 회전 + Hips 이동 = 15 채널
        self.assertEqual(len(anim["channels"]), 15)
        rotations = [c for c in anim["channels"] if c["target"]["path"] == "rotation"]
        self.assertEqual(len(rotations), 14)

    def test_chain_collapse_composes_quaternions(self):
        # Spine2(90°X) ⊗ Spine3(90°X) → Chest는 180°X = (1,0,0,0)
        payload = make_motion(spine_bend=True)
        worker.bake_animation(self.rigged, payload, self.out)
        gltf, bin_data = worker._read_glb(self.out)
        acc = self.channel_output(gltf, gltf["animations"][0], "Chest", "rotation")
        for q in self.read_floats(gltf, bin_data, acc, 4):
            self.assertAlmostEqual(abs(q[0]), 1.0, places=5)
            self.assertAlmostEqual(q[3], 0.0, places=5)

    def test_root_translation_scaled_to_character(self):
        # 메시 높이 1.8 / SMPL 1.7 스케일로 delta가 Hips rest에 더해진다
        trans = [[0.0, 0.0, 0.0], [0.17, 0.0, 0.0], [0.34, 0.0, 0.0]]
        worker.bake_animation(self.rigged, make_motion(trans=trans), self.out)
        gltf, bin_data = worker._read_glb(self.out)
        acc = self.channel_output(gltf, gltf["animations"][0], "Hips", "translation")
        vals = self.read_floats(gltf, bin_data, acc, 3)
        rest = gltf["nodes"][next(i for i, n in enumerate(gltf["nodes"]) if n.get("name") == "Hips")]["translation"]
        self.assertAlmostEqual(vals[0][0], rest[0], places=5)
        self.assertAlmostEqual(vals[1][0], rest[0] + 0.17 * 1.8 / 1.7, places=5)
        self.assertAlmostEqual(vals[2][1], rest[1], places=5)

    def test_zup_conversion(self):
        # Z-up의 +Y 전진은 Y-up 프레임에서 -Z가 된다: C·(0,1,0) = (0,0,-1)
        trans = [[0.0, 0.0, 0.0], [0.0, 1.7, 0.0]]
        payload = make_motion(frames=2, trans=trans)
        payload["up_axis"] = "z"
        worker.bake_animation(self.rigged, payload, self.out)
        gltf, bin_data = worker._read_glb(self.out)
        acc = self.channel_output(gltf, gltf["animations"][0], "Hips", "translation")
        vals = self.read_floats(gltf, bin_data, acc, 3)
        rest = gltf["nodes"][next(i for i, n in enumerate(gltf["nodes"]) if n.get("name") == "Hips")]["translation"]
        self.assertAlmostEqual(vals[1][2], rest[2] - 1.8, places=5)
        self.assertAlmostEqual(vals[1][1], rest[1], places=5)

    def test_time_accessor_has_min_max(self):
        worker.bake_animation(self.rigged, make_motion(frames=30), self.out)
        gltf, _ = worker._read_glb(self.out)
        anim = gltf["animations"][0]
        inp = gltf["accessors"][anim["samplers"][0]["input"]]
        self.assertEqual(inp["min"], [0.0])
        self.assertAlmostEqual(inp["max"][0], 29 / 30, places=5)

    def test_returns_zero_without_skin(self):
        static = os.path.join(self.tmp.name, "unrigged.glb")
        make_static_glb(static)
        self.assertEqual(worker.bake_animation(static, make_motion(), self.out), 0)
        self.assertFalse(os.path.exists(self.out))

    def test_glb_structure_valid(self):
        worker.bake_animation(self.rigged, make_motion(), self.out)
        data = open(self.out, "rb").read()
        magic, version, total = struct.unpack("<III", data[:12])
        self.assertEqual((magic, version, total), (worker.GLB_MAGIC, 2, len(data)))
        gltf, bin_data = worker._read_glb(self.out)
        self.assertEqual(gltf["buffers"][0]["byteLength"], len(bin_data))
        for view in gltf["bufferViews"]:
            self.assertLessEqual(view.get("byteOffset", 0) + view["byteLength"], len(bin_data))

    def test_arm_rest_delta_lifts_apose_arm_to_tpose_on_identity(self):
        """"팔 꺾임" 버그 회귀: SMPL rest는 T-pose(팔 수평)이므로 항등 쿼터니언
        프레임에서 rig의 A-pose 대각선 팔이 수평으로 보정되어야 한다 (보정이
        없으면 SMPL 회전이 A-pose 위에 중첩 적용되어 팔이 과도하게 꺾인다)."""
        worker.bake_animation(self.rigged, make_motion(), self.out)
        gltf, bin_data = worker._read_glb(self.out)
        node_by_name = {n.get("name"): i for i, n in enumerate(gltf["nodes"])}
        world = worker._rig_rest_world(gltf, node_by_name)
        for side, sign in (("Left", -1.0), ("Right", 1.0)):
            arm, fore = world[side + "Arm"], world[side + "ForeArm"]
            v = [fore[k] - arm[k] for k in range(3)]
            n = sum(c * c for c in v) ** 0.5
            v = [c / n for c in v]
            self.assertLess(v[1], -0.5, "픽스처의 rest 팔은 대각선 아래여야 함")
            acc = self.channel_output(gltf, gltf["animations"][0], side + "Arm", "rotation")
            q = self.read_floats(gltf, bin_data, acc, 4)[0]
            rotated = worker._quat_rotate_vec3(q, v)
            self.assertAlmostEqual(rotated[1], 0.0, places=4, msg=f"{side} arm should become horizontal")
            self.assertAlmostEqual(rotated[0], sign, places=4)

    def test_arm_rest_delta_keeps_forearm_chain_consistent(self):
        """ForeArm에는 q' = D_arm ⊗ q ⊗ D⁻¹ 보정 — 항등 프레임에서 부모와 합성 시
        팔뚝 월드 방향도 수평이 되어야 한다."""
        worker.bake_animation(self.rigged, make_motion(), self.out)
        gltf, bin_data = worker._read_glb(self.out)
        node_by_name = {n.get("name"): i for i, n in enumerate(gltf["nodes"])}
        world = worker._rig_rest_world(gltf, node_by_name)
        anim = gltf["animations"][0]
        arm_q = self.read_floats(gltf, bin_data, self.channel_output(gltf, anim, "LeftArm", "rotation"), 4)[0]
        fore_q = self.read_floats(gltf, bin_data, self.channel_output(gltf, anim, "LeftForeArm", "rotation"), 4)[0]
        arm, fore = world["LeftArm"], world["LeftForeArm"]
        v = [fore[k] - arm[k] for k in range(3)]
        n = sum(c * c for c in v) ** 0.5
        v = [c / n for c in v]  # 팔뚝 rest 방향(픽스처에선 상완과 같은 대각선)
        combined = worker._quat_mul(tuple(arm_q), tuple(fore_q))
        rotated = worker._quat_rotate_vec3(combined, v)
        self.assertAlmostEqual(rotated[1], 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
