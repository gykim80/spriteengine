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


def make_static_glb(path, scale=1.0):
    """스켈레톤 없는 단순 직사각형 GLB — auto_rig 입력용."""
    positions = [(-0.3, 0.0, -0.3), (0.3, 0.0, -0.3), (0.3, 1.8, -0.3), (-0.3, 1.8, -0.3),
                 (-0.3, 0.0,  0.3), (0.3, 0.0,  0.3), (0.3, 1.8,  0.3), (-0.3, 1.8,  0.3)]
    positions = [(x * scale, y * scale, z * scale) for x, y, z in positions]
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


def make_bent_arm_glb(path):
    """실측 캐릭터형 합성 휴머노이드 — 팔꿈치가 굽고(전완이 앞+아래+안쪽)
    손이 허벅지 근처(A-pose)에 오는 GLB. 웨이트 블리딩/전완 delta 회귀용.

    반환: (버텍스 수, {"L": 손 클러스터 인덱스들, "R": ...})
    """
    pts = []
    hand_idx = {"L": [], "R": []}
    for y in (0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7):  # 몸통 기둥
        for sx in (-1, 1):
            for sz in (-1, 1):
                pts.append((0.15 * sx, y, 0.1 * sz))
    for y in (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85):  # 다리
        for sx in (-1, 1):
            for sz in (-1, 1):
                pts.append((0.13 * sx, y, 0.06 * sz))
    # 팔: 어깨→팔꿈치 직선 + 팔꿈치→손 굽은 곡선(앞/아래/안쪽) + 손 클러스터
    for label, s in (("R", -1), ("L", 1)):
        sh, el, hand = (0.45 * s, 1.35, 0.0), (0.50 * s, 1.10, 0.0), (0.30 * s, 0.80, 0.22)
        for i in range(5):
            t = i / 4.0
            pts.append((sh[0] + t * (el[0] - sh[0]), sh[1] + t * (el[1] - sh[1]), 0.0))
        for i in range(1, 20):
            t = i / 19.0
            pts.append((el[0] + t * (hand[0] - el[0]),
                        el[1] + t * (hand[1] - el[1]),
                        el[2] + t * (hand[2] - el[2])))
        for dx in (-0.02, 0.0, 0.02):
            for dy in (-0.03, 0.0, 0.03):
                hand_idx[label].append(len(pts))
                pts.append((hand[0] + dx, hand[1] + dy, hand[2]))
    pts.append((0.0, 0.0, 0.0))   # 발 바닥 (min_y=0)
    pts.append((0.0, 1.8, 0.0))   # 머리 꼭대기 (max_y=1.8)

    faces = []
    for i in range(0, len(pts) - 2):
        faces += [i, i + 1, i + 2]
    buf = bytearray()
    for v in pts:
        buf += struct.pack("<fff", *v)
    pos_len = len(buf)
    for i in faces:
        buf += struct.pack("<H", i)
    gltf = {
        "asset": {"version": "2.0"}, "scene": 0,
        "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(buf)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": pos_len},
            {"buffer": 0, "byteOffset": pos_len, "byteLength": len(buf) - pos_len},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(pts), "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": len(faces), "type": "SCALAR"},
        ],
    }
    worker._write_glb(gltf, buf, path)
    return len(pts), hand_idx


class BentArmRigTest(unittest.TestCase):
    """'손이 몸에 붙는' 버그 회귀: 웨이트 체인 인접 제한 + 전완 delta 실측."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="spriteengine-bentarm-")
        src = os.path.join(cls.tmp.name, "bent.glb")
        cls.out = os.path.join(cls.tmp.name, "rigged.glb")
        cls.n_pts, cls.hand_idx = make_bent_arm_glb(src)
        assert worker.auto_rig(src, cls.out)
        cls.gltf, cls.bin_data = worker._read_glb(cls.out)
        skin = cls.gltf["skins"][0]
        cls.jname = [cls.gltf["nodes"][j]["name"] for j in skin["joints"]]
        attrs = cls.gltf["meshes"][0]["primitives"][0]["attributes"]
        ja = cls.gltf["accessors"][attrs["JOINTS_0"]]
        cls.jbase = cls.gltf["bufferViews"][ja["bufferView"]]["byteOffset"]
        wa = cls.gltf["accessors"][attrs["WEIGHTS_0"]]
        cls.wbase = cls.gltf["bufferViews"][wa["bufferView"]]["byteOffset"]
        cls.node_by_name = {n.get("name"): i for i, n in enumerate(cls.gltf["nodes"])}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _vert_joints(self, i):
        j1, j2 = struct.unpack_from("<2H", self.bin_data, self.jbase + i * 8)
        w = struct.unpack_from("<4f", self.bin_data, self.wbase + i * 16)
        return self.jname[j1], self.jname[j2], w

    def test_blend_only_with_adjacent_chain_joints(self):
        """보조 웨이트는 지배 조인트의 부모/자식하고만 허용 — 특히 팔↔다리
        블렌드(손이 다리에 앵커링돼 웨빙되는 주범)는 0이어야 한다."""
        parent = {}
        for i, nd in enumerate(self.gltf["nodes"]):
            for c in nd.get("children", []):
                parent[c] = i
        arm = {"LeftArm", "LeftForeArm", "RightArm", "RightForeArm"}
        leg = {"LeftUpLeg", "LeftLeg", "LeftFoot",
               "RightUpLeg", "RightLeg", "RightFoot"}
        for i in range(self.n_pts):
            n1, n2, w = self._vert_joints(i)
            if w[1] <= 0:
                continue
            i1, i2 = self.node_by_name[n1], self.node_by_name[n2]
            self.assertTrue(parent.get(i1) == i2 or parent.get(i2) == i1,
                            f"vertex {i}: non-adjacent blend {n1}<->{n2}")
            self.assertFalse((n1 in arm and n2 in leg) or (n1 in leg and n2 in arm),
                             f"vertex {i}: arm-leg blend {n1}<->{n2}")

    def test_hand_cluster_dominated_by_forearm(self):
        """굽은 팔 끝 손 클러스터는 허벅지가 아니라 ForeArm이 지배해야 한다
        (2-pass 팁 실측 보정 인변량)."""
        for label, side in (("L", "Left"), ("R", "Right")):
            for i in self.hand_idx[label]:
                n1, _, _ = self._vert_joints(i)
                self.assertEqual(n1, side + "ForeArm",
                                 f"hand vertex {i} dominated by {n1}")

    def test_fused_bridge_triangles_cut(self):
        """전완 지배 버텍스와 비인접 체인(다리·몸통·반대팔) 지배 버텍스를
        함께 가진 삼각형(융합 웨빙의 원인)은 인덱스에서 제거돼야 한다."""
        prim = self.gltf["meshes"][0]["primitives"][0]
        acc = self.gltf["accessors"][prim["indices"]]
        view = self.gltf["bufferViews"][acc["bufferView"]]
        base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        fmt = {5121: "B", 5123: "H", 5125: "I"}[acc["componentType"]]
        idx = struct.unpack_from(f"<{acc['count']}{fmt}", self.bin_data, base)
        self.assertGreater(len(idx), 0)
        allowed = {"LeftForeArm": {"LeftForeArm", "LeftArm"},
                   "RightForeArm": {"RightForeArm", "RightArm"}}
        arm = {"LeftArm", "LeftForeArm", "RightArm", "RightForeArm"}
        leg = {"LeftUpLeg", "LeftLeg", "LeftFoot",
               "RightUpLeg", "RightLeg", "RightFoot"}
        doms = [self._vert_joints(i)[0] for i in range(self.n_pts)]
        for t in range(0, len(idx) - 2, 3):
            names = [doms[v] for v in idx[t:t + 3]]
            self.assertFalse(any(n in arm for n in names)
                             and any(n in leg for n in names),
                             f"arm-leg bridge triangle survived: {names}")
            for n in names:
                if n in allowed:
                    self.assertTrue(all(m in allowed[n] for m in names),
                                    f"bridge triangle survived: {names}")

    def test_forearm_delta_measured_from_mesh(self):
        """전완 rest-delta는 상완 연속 가정이 아니라 메시 실측 방향이어야
        한다 — 픽스처 전완은 아래+앞+안쪽으로 굽어 있고 좌우 미러다."""
        dirs = worker._mesh_forearm_dirs(
            self.gltf, self.bin_data,
            worker._rig_rest_world(self.gltf, self.node_by_name))
        self.assertEqual(set(dirs), {"Left", "Right"})
        for side, sx in (("Left", 1.0), ("Right", -1.0)):
            v = dirs[side]
            self.assertLess(v[0] * sx, 0.0, f"{side} forearm must point inward")
            self.assertLess(v[1], -0.5, f"{side} forearm must point down")
            self.assertGreater(v[2], 0.2, f"{side} forearm must point forward")
        deltas, dparents = worker._arm_rest_deltas(
            self.gltf, self.bin_data, self.node_by_name)
        for side in ("Left", "Right"):
            da, df = deltas[side + "Arm"], deltas[side + "ForeArm"]
            self.assertTrue(any(abs(da[k] - df[k]) > 1e-3 for k in range(4)),
                            f"{side}: ForeArm delta must differ from Arm delta")
            self.assertEqual(dparents[side + "ForeArm"], da)


class AutoRigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="spriteengine-autorig-")
        self.addCleanup(self.tmp.cleanup)
        self.glb = os.path.join(self.tmp.name, "static.glb")
        self.out = os.path.join(self.tmp.name, "rigged.glb")

    def test_adds_skin_and_joints(self):
        make_static_glb(self.glb)
        # 반환값은 판별된 체형 문자열(truthy) — rig 스테이지 bodyType 메트릭에 실린다
        self.assertEqual(worker.auto_rig(self.glb, self.out), "humanoid")
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

    def test_left_joints_on_positive_x_smpl_convention(self):
        """"팔 위로 꺾임" 버그 회귀: SMPL/glTF 휴머노이드 규약상 캐릭터가 +Z를
        향할 때 Left* 조인트는 +X 쪽이어야 한다. 반대(−X)면 리타게팅 시 SMPL
        좌팔 회전이 미러로 적용돼 팔이 위로 접힌다."""
        make_static_glb(self.glb)
        worker.auto_rig(self.glb, self.out)
        gltf, _ = worker._read_glb(self.out)
        node_by_name = {n.get("name"): i for i, n in enumerate(gltf["nodes"])}
        world = worker._rig_rest_world(gltf, node_by_name)
        for name, w in world.items():
            if name.startswith("Left"):
                self.assertGreater(w[0], 0.0, f"{name} must be on +X, got {w[0]:.3f}")
            elif name.startswith("Right"):
                self.assertLess(w[0], 0.0, f"{name} must be on -X, got {w[0]:.3f}")

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


class ScaleNormalizeTest(unittest.TestCase):
    """복원 스케일 이상치(예: cowboy 키 0.494 — 표준 1.987의 1/4)가 리깅 시
    표준 키로 정규화되는지의 회귀 테스트. HY-Motion 루트 이동 리타게팅이
    캐릭터 키에 비례하므로, 정규화 없이는 이동 모션이 제자리걸음으로 퇴화한다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="spriteengine-scale-")
        self.addCleanup(self.tmp.cleanup)
        self.glb = os.path.join(self.tmp.name, "static.glb")
        self.out = os.path.join(self.tmp.name, "rigged.glb")

    def mesh_height(self, path):
        gltf, bin_data = worker._read_glb(path)
        attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
        ys = [p[1] for p in worker._read_vec3(gltf, bin_data, attrs["POSITION"], "POSITION")]
        return max(ys) - min(ys)

    def test_undersized_mesh_normalized_to_standard_height(self):
        make_static_glb(self.glb, scale=0.25)  # 키 0.45 — cowboy 이상치 재현
        self.assertTrue(worker.auto_rig(self.glb, self.out))
        self.assertAlmostEqual(self.mesh_height(self.out),
                               worker.STANDARD_CHARACTER_HEIGHT, places=4)

    def test_normal_mesh_untouched(self):
        make_static_glb(self.glb)  # 키 1.8 — 정상 범위
        worker.auto_rig(self.glb, self.out)
        self.assertAlmostEqual(self.mesh_height(self.out), 1.8, places=5)

    def test_joints_follow_normalized_scale(self):
        make_static_glb(self.glb, scale=0.25)
        worker.auto_rig(self.glb, self.out)
        gltf, _ = worker._read_glb(self.out)
        hips = next(n for n in gltf["nodes"] if n.get("name") == "Hips")
        # Hips는 키의 52% 지점 — 정규화된 키 기준이어야 한다.
        self.assertAlmostEqual(hips["translation"][1],
                               worker.STANDARD_CHARACTER_HEIGHT * 0.52, places=3)


def make_quadruped_glb(path, scale=1.0, reverse=False, sideways=False, tail=True,
                       stretch=(1.0, 1.0, 1.0), missing_leg=None):
    """개 형태의 4족 GLB — 수평 몸통(Z 장축) + 다리 4기둥 + 머리/꼬리.

    reverse=True면 Y축 180° 회전(x,z 부호 반전) — 머리가 −Z를 향하는
    역방향 복원 메시를 흉내 낸다. sideways=True면 추가로 yaw 90° 회전해
    체장이 X축을 따르는 측방향 복원 메시를 흉내 낸다. tail=False면 꼬리가
    복원에서 뭉개져 사라진 메시를, stretch는 축별 비균등 비례(말=키 큰
    체형 등)를 흉내 낸다. missing_leg("front_L"/"rear_R" 등)을 주면 해당
    다리 기둥을 생략해 옆모습 복원의 가려진 다리 소실을 흉내 낸다.
    반환: (positions, part_of) — part_of[i]는 버텍스 i의 부위 라벨.
    """
    pts = []
    part_of = []

    def add(part, p):
        part_of.append(part)
        x, y, z = (p[k] * scale * stretch[k] for k in range(3))
        if reverse:
            x, z = -x, -z
        if sideways:
            x, z = z, -x  # yaw +90°: 머리(+Z)가 +X를 향한다
        pts.append((x, y, z))

    for z in (-0.40, -0.20, 0.0, 0.20, 0.40):          # 몸통 (등 y0.62 / 배 y0.38)
        for sx in (-1, 1):
            for y in (0.38, 0.62):
                add("body", (0.14 * sx, y, z))
    for z in (0.45, 0.58, 0.70):                       # 머리 (전방 위)
        for sx in (-1, 1):
            for y in (0.55, 0.85):
                add("head", (0.10 * sx, y, z))
    if tail:
        for z in (-0.70, -0.58, -0.45):                # 꼬리 (후방)
            for sx in (-1, 1):
                add("tail", (0.03 * sx, 0.62, z))
    for part, zc in (("front", 0.30), ("rear", -0.30)):  # 다리 4기둥
        for sx in (-1, 1):
            side = "L" if sx > 0 else "R"
            if f"{part}_{side}" == missing_leg:
                continue
            for y in (0.0, 0.12, 0.25, 0.38):
                for dz in (-0.05, 0.05):
                    add(f"leg_{part}_{side}", (0.14 * sx, y, zc + dz))

    faces = []
    for i in range(len(pts) - 2):
        faces += [i, i + 1, i + 2]
    buf = bytearray()
    for v in pts:
        buf += struct.pack("<fff", *v)
    pos_len = len(buf)
    for i in faces:
        buf += struct.pack("<H", i)
    gltf = {
        "asset": {"version": "2.0"}, "scene": 0,
        "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(buf)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": pos_len},
            {"buffer": 0, "byteOffset": pos_len, "byteLength": len(buf) - pos_len},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(pts), "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": len(faces), "type": "SCALAR"},
        ],
    }
    worker._write_glb(gltf, buf, path)
    return pts, part_of


class QuadrupedRigTest(unittest.TestCase):
    """4족 보행 리깅 회귀: 체형 판별 → 수평 척추 + 앞다리=Arm/뒷다리=Leg 체인."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="spriteengine-quadruped-")
        src = os.path.join(cls.tmp.name, "dog.glb")
        cls.out = os.path.join(cls.tmp.name, "rigged.glb")
        cls.pts, cls.part_of = make_quadruped_glb(src)
        assert worker.auto_rig(src, cls.out) == "quadruped"
        cls.gltf, cls.bin_data = worker._read_glb(cls.out)
        cls.skin = cls.gltf["skins"][0]
        cls.jname = [cls.gltf["nodes"][j]["name"] for j in cls.skin["joints"]]
        cls.node_by_name = {n.get("name"): i for i, n in enumerate(cls.gltf["nodes"])}
        cls.world = worker._rig_rest_world(cls.gltf, cls.node_by_name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_classifier(self):
        self.assertEqual(worker._classify_body_type(self.pts), "quadruped")
        # 직립 박스(휴머노이드)는 그대로 humanoid여야 한다
        box = [(x, y, z) for x in (-0.3, 0.3) for y in (0.0, 1.8) for z in (-0.3, 0.3)]
        self.assertEqual(worker._classify_body_type(box), "humanoid")

    def test_classifier_rejects_lying_humanoid(self):
        """누워서 복원된 휴머노이드(barbarian/cowboy 실패 사례)는 체장>키라도
        4족으로 오분류되면 안 된다 — 몸통이 전장에 걸쳐 지면에 닿아 배 밑
        갭이 없으므로 humanoid 유지 → 기존 upright 게이트가 누움을 잡는다."""
        lying = [(x, y, round(0.1 * i - 0.9, 2))
                 for i in range(19) for x in (-0.3, 0.3) for y in (0.0, 0.3)]
        self.assertEqual(worker._classify_body_type(lying), "humanoid")

    def test_skin_marker_and_tail(self):
        self.assertEqual(self.skin["name"], "AutoQuadrupedRig")
        self.assertIn("Tail", self.jname)   # 꼬리 버텍스가 있으므로 Tail 조인트 생성
        self.assertEqual(len(self.jname), 15)

    def test_horizontal_spine_and_leg_placement(self):
        w = self.world
        # 척추는 수평: Hips(후방)→Chest(전방)로 z가 증가, y는 몸통 높이로 동일
        self.assertLess(w["Hips"][2], w["Spine"][2])
        self.assertLess(w["Spine"][2], w["Chest"][2])
        self.assertAlmostEqual(w["Hips"][1], w["Chest"][1], places=5)
        # 머리는 몸통 앞·위
        self.assertGreater(w["Head"][2], w["Chest"][2])
        self.assertGreater(w["Head"][1], w["Chest"][1])
        # 앞다리(Arm 체인)는 전방 z, 뒷다리(Leg 체인)는 후방 z
        self.assertGreater(w["LeftArm"][2], 0.0)
        self.assertLess(w["LeftUpLeg"][2], 0.0)
        # SMPL 규약: Left*는 +X (리타게팅 미러 방지)
        for name, pos in w.items():
            if name.startswith("Left"):
                self.assertGreater(pos[0], 0.0, f"{name} must be on +X")
            elif name.startswith("Right"):
                self.assertLess(pos[0], 0.0, f"{name} must be on -X")

    def _dominant(self, i):
        attrs = self.gltf["meshes"][0]["primitives"][0]["attributes"]
        acc = self.gltf["accessors"][attrs["JOINTS_0"]]
        base = self.gltf["bufferViews"][acc["bufferView"]]["byteOffset"]
        return self.jname[struct.unpack_from("<H", self.bin_data, base + i * 8)[0]]

    def test_leg_weights_split_front_rear(self):
        """같은 쪽 앞·뒷다리가 서로 다른 체인에 배정된다 — 휴머노이드 리깅을
        강제하면 둘 다 한 Leg 체인에 눌려 붙던 버그의 회귀."""
        for i, part in enumerate(self.part_of):
            dom = self._dominant(i)
            if part.startswith("leg_front"):
                self.assertIn("Arm", dom, f"front leg vertex {i} bound to {dom}")
            elif part.startswith("leg_rear"):
                self.assertIn("Leg" if "Leg" in dom else "Foot", dom,
                              f"rear leg vertex {i} bound to {dom}")
                self.assertNotIn("Arm", dom)
            elif part == "tail":
                self.assertIn(dom, ("Tail", "Hips"), f"tail vertex {i} bound to {dom}")
            side = "L" if self.pts[i][0] > 0.05 else ("R" if self.pts[i][0] < -0.05 else None)
            if side and part.startswith("leg_"):
                self.assertTrue(dom.startswith("Left" if side == "L" else "Right"),
                                f"{part} vertex {i} crossed sides to {dom}")

    def test_backward_facing_dog_flipped(self):
        """−Z를 향해 복원된 개는 리깅 전에 자동으로 180° 플립되어야 한다 —
        안 그러면 머리/꼬리가 뒤바뀐 리깅이 되고 보행 모션이 뒤로 걷는데,
        스켈레톤이 좌우 대칭이라 검증 게이트로는 잡히지 않는다."""
        rev = os.path.join(self.tmp.name, "reversed.glb")
        rev_out = os.path.join(self.tmp.name, "reversed-rigged.glb")
        pts, part_of = make_quadruped_glb(rev, reverse=True)
        # 방향 감지기: 역방향 개만 backward, 정방향 개는 forward
        self.assertTrue(worker._quadruped_faces_backward(pts))
        self.assertFalse(worker._quadruped_faces_backward(self.pts))
        self.assertEqual(worker.auto_rig(rev, rev_out), "quadruped")
        gltf, bin_data = worker._read_glb(rev_out)
        attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
        pos = worker._read_vec3(gltf, bin_data, attrs["POSITION"], "POSITION")
        head_z = [pos[i][2] for i, p in enumerate(part_of) if p == "head"]
        tail_z = [pos[i][2] for i, p in enumerate(part_of) if p == "tail"]
        self.assertGreater(min(head_z), 0.0)  # 플립 후 머리 지오메트리는 +Z(전방)
        self.assertLess(max(tail_z), 0.0)     # 꼬리는 −Z(후방)
        node_by_name = {n.get("name"): i for i, n in enumerate(gltf["nodes"])}
        world = worker._rig_rest_world(gltf, node_by_name)
        self.assertGreater(world["Head"][2], world["Chest"][2])  # 머리 조인트도 전방
        self.assertLess(world["Hips"][2], world["Chest"][2])

    def test_sideways_dog_realigned(self):
        """체장이 X축으로 복원된 개는 yaw 90° 정렬 후 리깅되어야 한다 —
        정렬 없이는 분류기가 Z 체장만 보므로 휴머노이드로 오분류되어
        upright 게이트에서 작업 전체가 실패한다. reverse 조합은 정렬 후
        역방향 감지·플립까지 연쇄 동작하는지 검증한다."""
        for rev in (False, True):
            with self.subTest(reverse=rev):
                src = os.path.join(self.tmp.name, f"side-{rev}.glb")
                out = os.path.join(self.tmp.name, f"side-{rev}-rigged.glb")
                _, part_of = make_quadruped_glb(src, sideways=True, reverse=rev)
                self.assertEqual(worker.auto_rig(src, out), "quadruped")
                gltf, bin_data = worker._read_glb(out)
                attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
                pos = worker._read_vec3(gltf, bin_data, attrs["POSITION"], "POSITION")
                xs = [p[0] for p in pos]; zs = [p[2] for p in pos]
                self.assertGreater(max(zs) - min(zs), max(xs) - min(xs),
                                   "body length must be realigned to Z")
                head_z = [pos[i][2] for i, p in enumerate(part_of) if p == "head"]
                self.assertGreater(min(head_z), 0.0)  # 머리는 최종적으로 +Z(전방)

    def test_sideways_lying_humanoid_not_rotated(self):
        """X축으로 길게 누운 휴머노이드는 yaw 정렬 후보에서 배 밑 갭 확인에
        걸려 회전 없이 humanoid로 남아야 한다 — 그래야 기존 upright 게이트가
        누움을 잡는다."""
        lying_x = [(round(0.1 * i - 0.9, 2), y, x)
                   for i in range(19) for x in (-0.3, 0.3) for y in (0.0, 0.3)]
        rotated = [(-p[2], p[1], p[0]) for p in lying_x]
        self.assertEqual(worker._classify_body_type(rotated), "humanoid")

    def test_tail_sway_baked_into_animation(self):
        """Tail은 SMPL에 대응 관절이 없어 모션 베이킹 시 강직 상태였다 —
        걸음 주기 스웨이 채널이 합성되어 회전 값이 실제로 변해야 한다."""
        out = os.path.join(self.tmp.name, "dog-anim.glb")
        self.assertEqual(worker.bake_animation(self.out, make_motion(frames=30), out), 1)
        gltf, bin_data = worker._read_glb(out)
        node_by_name = {n.get("name"): i for i, n in enumerate(gltf["nodes"])}
        anim = gltf["animations"][0]
        ch = [c for c in anim["channels"]
              if c["target"]["node"] == node_by_name["Tail"]
              and c["target"]["path"] == "rotation"]
        self.assertEqual(len(ch), 1, "quadruped motion must include a Tail channel")
        acc = gltf["accessors"][anim["samplers"][ch[0]["sampler"]]["output"]]
        view = gltf["bufferViews"][acc["bufferView"]]
        base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        qy = [struct.unpack_from("<4f", bin_data, base + 16 * f)[1]
              for f in range(acc["count"])]
        self.assertGreater(max(qy) - min(qy), 0.05,
                           "tail must actually sway, not stay rigid")

    def test_horse_proportions_classified_and_rigged(self):
        """체장≈키인 말 체형(다리 긴 4족)은 예전 1.25배 체장/키 비율 게이트에
        걸려 휴머노이드로 오분류되던 케이스 — 코어 최장축 검사 + 배 밑 갭
        확증으로 4족 리깅을 받아야 한다."""
        src = os.path.join(self.tmp.name, "horse.glb")
        out = os.path.join(self.tmp.name, "horse-rigged.glb")
        pts, _ = make_quadruped_glb(src, stretch=(1.0, 1.5, 1.0))
        h = max(p[1] for p in pts) - min(p[1] for p in pts)
        length = max(p[2] for p in pts) - min(p[2] for p in pts)
        self.assertLess(length, 1.25 * h, "fixture must be horse-proportioned")
        self.assertEqual(worker._classify_body_type(pts), "quadruped")
        self.assertEqual(worker.auto_rig(src, out), "quadruped")

    def test_tailless_dog_rigs_and_animates(self):
        """복원에서 꼬리가 뭉개져 사라진 개도 4족 리깅과 모션 베이킹이
        성립해야 한다 — Tail 조인트 유무와 무관하게 크래시 없이 동작."""
        src = os.path.join(self.tmp.name, "tailless.glb")
        rigged = os.path.join(self.tmp.name, "tailless-rigged.glb")
        animated = os.path.join(self.tmp.name, "tailless-animated.glb")
        make_quadruped_glb(src, tail=False)
        self.assertEqual(worker.auto_rig(src, rigged), "quadruped")
        self.assertEqual(
            worker.bake_animation(rigged, make_motion(frames=10), animated), 1)

    def test_length_scale_normalization(self):
        """4족은 키(Y)가 아니라 체장(Z) 기준으로 정규화 — 0.88m 개가 휴머노이드
        표준 키 1.99m로 2.3배 거인화되던 버그의 회귀."""
        tiny = os.path.join(self.tmp.name, "tiny.glb")
        tiny_out = os.path.join(self.tmp.name, "tiny-rigged.glb")
        make_quadruped_glb(tiny, scale=0.2)  # 체장 0.28m < 정상 하한 0.5m
        self.assertTrue(worker.auto_rig(tiny, tiny_out))
        gltf, bin_data = worker._read_glb(tiny_out)
        attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
        zs = [p[2] for p in worker._read_vec3(gltf, bin_data, attrs["POSITION"], "POSITION")]
        self.assertAlmostEqual(max(zs) - min(zs), worker.STANDARD_QUADRUPED_LENGTH, places=4)
        # 정상 체장(1.4m)은 손대지 않는다
        gltf2, bin2 = worker._read_glb(self.out)
        attrs2 = gltf2["meshes"][0]["primitives"][0]["attributes"]
        zs2 = [p[2] for p in worker._read_vec3(gltf2, bin2, attrs2["POSITION"], "POSITION")]
        self.assertAlmostEqual(max(zs2) - min(zs2), 1.4, places=5)


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
        for side, sign in (("Left", 1.0), ("Right", -1.0)):
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

    def test_smpl_hanging_arms_stay_hanging(self):
        """"더 괴물이 되었는데" 버그 회귀: HY-Motion idle의 팔은 rot(Z,∓75°)류
        (좌팔 −, 우팔 +)로 T-pose 수평에서 아래로 매달린다. 리깅 좌우가 SMPL
        규약과 뒤집혀 있으면 같은 회전이 반대쪽 팔에 적용돼 정확히 미러 —
        팔 elevation이 +75°(위로 꺾임)가 된다. 구운 뒤 팔 월드 방향이
        반드시 아래(y<−0.8)여야 한다."""
        import math
        payload = make_motion()
        half = math.radians(75.0) / 2.0
        for row in payload["motions"][0]["quats"]:
            row[SMPLH_JOINTS.index("L_Shoulder")] = [0.0, 0.0, -math.sin(half), math.cos(half)]
            row[SMPLH_JOINTS.index("R_Shoulder")] = [0.0, 0.0, math.sin(half), math.cos(half)]
        worker.bake_animation(self.rigged, payload, self.out)
        gltf, bin_data = worker._read_glb(self.out)
        node_by_name = {n.get("name"): i for i, n in enumerate(gltf["nodes"])}
        world = worker._rig_rest_world(gltf, node_by_name)
        anim = gltf["animations"][0]
        for side in ("Left", "Right"):
            # 상위 체인(Hips/Spine/Chest)은 항등이므로 Arm 채널 쿼터니언이 곧
            # 팔의 월드 회전 — rest 본 방향에 적용해 최종 방향을 얻는다.
            q = self.read_floats(gltf, bin_data,
                                 self.channel_output(gltf, anim, side + "Arm", "rotation"), 4)[0]
            arm, fore = world[side + "Arm"], world[side + "ForeArm"]
            v = [fore[k] - arm[k] for k in range(3)]
            n = sum(c * c for c in v) ** 0.5
            rotated = worker._quat_rotate_vec3(tuple(q), [c / n for c in v])
            self.assertLess(rotated[1], -0.8,
                            f"{side} arm must hang down, got dir={rotated}")

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


class RenderGateTest(unittest.TestCase):
    """워커 rig/motion 단계의 렌더링 정상성 게이트 회귀 테스트.

    변환이 "성공"했어도 결과가 몬스터(누움/팔 꺾임/웨빙)면 stage를 실패시켜
    잘못된 캐릭터가 앱 파이프라인을 조용히 통과하지 못하게 한다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="spriteengine-gate-")
        self.addCleanup(self.tmp.cleanup)
        self.glb = os.path.join(self.tmp.name, "static.glb")
        make_static_glb(self.glb)

    def _run(self, stage, input_path):
        events = []
        original = worker.emit
        worker.emit = lambda t, **p: events.append({"type": t, **p})
        try:
            worker.run({"jobId": "gate-test", "stage": stage,
                        "workspace": self.tmp.name, "input": input_path})
        finally:
            worker.emit = original
        return events

    def test_rig_stage_reports_render_valid(self):
        events = self._run("rig", self.glb)
        art = next(e for e in events if e["type"] == "artifact")
        self.assertTrue(art["metrics"].get("skinned"))
        self.assertTrue(art["metrics"].get("renderValid"),
                        f"valid auto-rig must pass the gate: {art['metrics']}")

    def test_rig_stage_quadruped_orientations_pass_gate(self):
        """역방향·측방향 복원 개도 프로덕션 rig 스테이지에서 방향 보정을 거쳐
        검증 게이트를 통과해야 한다 — bodyType 메트릭과 renderValid까지
        워커 프로토콜 전체 경로로 확인한다."""
        cases = ({"reverse": True}, {"sideways": True},
                 {"reverse": True, "sideways": True})
        for kw in cases:
            with self.subTest(**kw):
                dog = os.path.join(self.tmp.name,
                                   "dog-" + "-".join(sorted(kw)) + ".glb")
                make_quadruped_glb(dog, **kw)
                events = self._run("rig", dog)
                art = next(e for e in events if e["type"] == "artifact")
                self.assertEqual(art["metrics"].get("bodyType"), "quadruped")
                self.assertTrue(art["metrics"].get("renderValid"),
                                f"oriented dog must pass gates: {art['metrics']}")

    def test_rig_stage_rejects_three_legged_dog(self):
        """옆모습 복원에서 가려진 다리가 소실된 개는 rig 게이트가 차단해야
        한다 — 다리 기둥 3개/사분면 미달을 legs 검사가 실측으로 잡는다."""
        dog = os.path.join(self.tmp.name, "dog-3legs.glb")
        make_quadruped_glb(dog, missing_leg="rear_L")
        with self.assertRaises(RuntimeError) as ctx:
            self._run("rig", dog)
        self.assertIn("leg", str(ctx.exception).lower())

    def test_gate_fails_stage_on_invalid_render(self):
        bad = {"ok": False}
        for sec in ("upright", "hierarchy", "legs", "deformation", "arm_pose", "skinning"):
            bad[sec] = {"ok": True, "issues": []}
        bad["arm_pose"] = {"ok": False, "issues": ["arms point upward (test)"]}
        original = worker.render_check
        worker.render_check = lambda path: bad
        try:
            with self.assertRaises(RuntimeError) as ctx:
                self._run("rig", self.glb)
        finally:
            worker.render_check = original
        self.assertIn("arms point upward", str(ctx.exception))


class LegQualityTest(unittest.TestCase):
    """멀티시드 복원 후보 선별용 leg_quality 점수 회귀 테스트.

    원시 Hunyuan 복원 GLB는 Z-up 정점 + 메시 노드 +90° X 회전으로 Y-up을
    만든다 — 노드 변환을 무시하면 업축이 틀어져 4기둥 탐지가 전부 깨진다
    (실측 회귀: dog2 원시 복원이 columns=1로 실격 처리됐던 버그)."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "validate_character",
            os.path.join(os.path.dirname(__file__), "..", "tools", "matrix",
                         "validate_character.py"))
        cls.vc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.vc)
        cls.tmp = tempfile.TemporaryDirectory(prefix="spriteengine-legq-")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def _write_points_glb(path, pts, rotation=None):
        """POSITION만 가진 최소 GLB (노드 회전 옵션)."""
        buf = bytearray()
        for v in pts:
            buf += struct.pack("<fff", *v)
        node = {"mesh": 0}
        if rotation:
            node["rotation"] = rotation
        mins = [min(p[k] for p in pts) for k in range(3)]
        maxs = [max(p[k] for p in pts) for k in range(3)]
        gltf = {
            "asset": {"version": "2.0"}, "scene": 0,
            "scenes": [{"nodes": [0]}], "nodes": [node],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
            "buffers": [{"byteLength": len(buf)}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(buf)}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(pts),
                           "type": "VEC3", "min": mins, "max": maxs}],
        }
        worker._write_glb(gltf, buf, path)

    def test_clean_quadruped_scores_positive(self):
        glb = os.path.join(self.tmp.name, "clean.glb")
        make_quadruped_glb(glb)
        q = self.vc.leg_quality(glb)
        self.assertEqual(q["columns"], 4)
        self.assertEqual(q["quadrants"], 4)
        self.assertGreaterEqual(q["score"], 0.0, q)

    def test_missing_leg_disqualified(self):
        glb = os.path.join(self.tmp.name, "3legs.glb")
        make_quadruped_glb(glb, missing_leg="front_L")
        q = self.vc.leg_quality(glb)
        self.assertEqual(q["score"], -1.0, q)

    def test_web_between_legs_lowers_score(self):
        """다리쌍 사이를 채운 웹(막) 지오메트리는 점수를 깎아야 한다 —
        실측 회귀: 웹 결함 개(dog2)가 정상 후보와 분리 지표 동률이라
        선별이 불가능했던 문제를 web 밀도 페널티가 해소한다."""
        clean = os.path.join(self.tmp.name, "web-clean.glb")
        pts, _ = make_quadruped_glb(clean)
        webbed = os.path.join(self.tmp.name, "webbed.glb")
        # 접지 밴드(y<0.15h≈0.13) 위쪽만 채운다 — 지면까지 닿는 웹은 기둥
        # 병합으로 이미 실격(-1.0) 처리되므로 페널티 경로가 잡을 대상은
        # 공중에 뜬 막이다.
        web = [(x / 100.0, y / 100.0, z / 100.0)
               for x in (-4, 0, 4) for y in (15, 20, 25)
               for z in (28, 30, 32)]  # 앞다리쌍(x=±0.14, z≈0.30) 사이 막
        self._write_points_glb(webbed, pts + web)
        qc = self.vc.leg_quality(clean)
        qw = self.vc.leg_quality(webbed)
        self.assertEqual(qc["front_web"], 0.0, qc)
        self.assertGreater(qw["front_web"], 0.0, qw)
        self.assertLess(qw["score"], qc["score"])

    def test_zup_node_rotation_matches_yup(self):
        """+90° X 노드 회전(원시 Hunyuan 복원)이 적용된 Z-up 데이터도
        Y-up 데이터와 동일한 점수를 받아야 한다."""
        yup = os.path.join(self.tmp.name, "yup.glb")
        pts, _ = make_quadruped_glb(yup)
        zup = os.path.join(self.tmp.name, "zup.glb")
        # world = R_x(+90°)·data ⇒ data = (x, z, -y)
        self._write_points_glb(zup, [(x, z, -y) for x, y, z in pts],
                               rotation=[0.7071068, 0.0, 0.0, 0.7071068])
        qy = self.vc.leg_quality(yup)
        qz = self.vc.leg_quality(zup)
        self.assertEqual(qz["columns"], 4, qz)
        self.assertAlmostEqual(qz["score"], qy["score"], places=3)


class StripBasePlaneTest(unittest.TestCase):
    """복원 메시 하단 바닥 판(base slab) 제거 회귀 테스트.

    실측(2026-07-20): gladiator 3장·vampire v1은 캐릭터가 2×2 슬래브 위에
    선 채 복원됐고, 슬래브가 정규화 볼륨을 차지해 캐릭터가 키 0.67의
    미니어처로 축소됐다. strip_base_plane은 슬래브를 잘라내고 남은
    캐릭터를 Hunyuan 규약(최장축 ~1.987, 원점 중심)으로 재정규화한다."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "validate_character",
            os.path.join(os.path.dirname(__file__), "..", "tools", "matrix",
                         "validate_character.py"))
        cls.vc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.vc)
        cls.tmp = tempfile.TemporaryDirectory(prefix="spriteengine-slab-")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # 기둥(캐릭터 대역) 버텍스 수 — 4열 × 41행
    COLUMN_VERTS = 4 * 41
    SLAB_VERTS = 21 * 21

    @staticmethod
    def _make_glb(path, with_slab=True, rotation=None, pillar_y0=0.06, pillar_dy=0.044):
        """좁은 수직 기둥(캐릭터) + 옵션 전폭 슬래브를 가진 indexed GLB.

        rotation을 주면 데이터를 Z-up으로 저장해 원시 Hunyuan 복원처럼
        노드 회전(+90° X)으로 Y-up이 되는 경우를 재현한다.
        기본 치수는 기둥 높이 1.76(슬래브 스팬 2.0 대비 구제 배율 ~1.13 —
        허용 범위). pillar_dy를 줄이면 gladiator형 미니어처(기각 대상)가 된다."""
        world = []
        tris = []
        # 기둥: x ∈ {-0.05,-0.02,0.02,0.05}, y = pillar_y0 + k·pillar_dy, z=0
        cols = (-0.05, -0.02, 0.02, 0.05)
        for k in range(41):
            y = pillar_y0 + k * pillar_dy
            for x in cols:
                world.append((x, y, 0.0))
        for k in range(40):
            for c in range(3):
                a = 4 * k + c
                tris += [(a, a + 1, a + 5), (a, a + 5, a + 4)]
        if with_slab:
            base = len(world)
            for i in range(21):
                for j in range(21):
                    world.append((-1.0 + i * 0.1, 0.0, -1.0 + j * 0.1))
            for i in range(20):
                for j in range(20):
                    a = base + i * 21 + j
                    tris += [(a, a + 1, a + 22), (a, a + 22, a + 21)]
        if rotation:
            # world = R_x(+90°)·local ⇒ local = (x, z, -y)
            pts = [(x, z, -y) for x, y, z in world]
        else:
            pts = world
        buf = bytearray()
        for v in pts:
            buf += struct.pack("<fff", *v)
        idx_off = len(buf)
        flat = [i for t in tris for i in t]
        buf += struct.pack(f"<{len(flat)}I", *flat)
        node = {"mesh": 0}
        if rotation:
            node["rotation"] = rotation
        gltf = {
            "asset": {"version": "2.0"}, "scene": 0,
            "scenes": [{"nodes": [0]}], "nodes": [node],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
            "buffers": [{"byteLength": len(buf)}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": idx_off},
                            {"buffer": 0, "byteOffset": idx_off, "byteLength": len(flat) * 4}],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": len(pts), "type": "VEC3",
                 "min": [min(p[k] for p in pts) for k in range(3)],
                 "max": [max(p[k] for p in pts) for k in range(3)]},
                {"bufferView": 1, "componentType": 5125, "count": len(flat), "type": "SCALAR"},
            ],
        }
        worker._write_glb(gltf, buf, path)

    def _world_bounds(self, path):
        gltf, bin_data = worker._read_glb(path)
        pos = self.vc._world_positions(gltf, bin_data)
        return pos, [(min(p[k] for p in pos), max(p[k] for p in pos)) for k in range(3)]

    def test_strips_slab_and_renormalizes(self):
        glb = os.path.join(self.tmp.name, "slab.glb")
        self._make_glb(glb, with_slab=True)
        removed = worker.strip_base_plane(glb, glb)
        self.assertEqual(removed, self.SLAB_VERTS)
        pos, bounds = self._world_bounds(glb)
        self.assertEqual(len(pos), self.COLUMN_VERTS)
        # Hunyuan 규약 재정규화: 최장축 = 표준 키, bbox 중심 = 원점
        size = max(hi - lo for lo, hi in bounds)
        self.assertAlmostEqual(size, worker.STANDARD_CHARACTER_HEIGHT, places=4)
        for lo, hi in bounds:
            self.assertAlmostEqual((lo + hi) / 2.0, 0.0, places=4)
        # 인덱스는 기둥 삼각형만 남고 전부 유효 범위여야 한다
        gltf, bin_data = worker._read_glb(glb)
        idx = worker._read_indices(gltf, bin_data, gltf["meshes"][0]["primitives"][0])
        self.assertEqual(len(idx), 40 * 3 * 2 * 3)
        self.assertLess(max(idx), self.COLUMN_VERTS)
        # 멱등성: 슬래브가 사라졌으므로 재호출은 no-op
        self.assertEqual(worker.strip_base_plane(glb, glb), 0)

    def test_zup_node_rotation(self):
        """원시 Hunyuan 복원(Z-up 정점 + 노드 +90° X 회전)에서도 월드 기준으로
        슬래브를 검출·제거하고 재정규화해야 한다."""
        glb = os.path.join(self.tmp.name, "slab-zup.glb")
        self._make_glb(glb, with_slab=True, rotation=[0.7071068, 0.0, 0.0, 0.7071068])
        removed = worker.strip_base_plane(glb, glb)
        self.assertEqual(removed, self.SLAB_VERTS)
        pos, bounds = self._world_bounds(glb)
        size = max(hi - lo for lo, hi in bounds)
        self.assertAlmostEqual(size, worker.STANDARD_CHARACTER_HEIGHT, places=4)
        # 재정규화 후에도 월드 업축은 Y (기둥의 최장축이 Y여야 함)
        self.assertAlmostEqual(bounds[1][1] - bounds[1][0], size, places=4)

    def test_noop_without_slab(self):
        """슬래브 없는 정상 메시는 바이트 하나 건드리지 않아야 한다
        (실측: vampire-v2/nurse 정상 후보 no-op 확인)."""
        glb = os.path.join(self.tmp.name, "clean.glb")
        self._make_glb(glb, with_slab=False)
        before = open(glb, "rb").read()
        self.assertEqual(worker.strip_base_plane(glb, glb), 0)
        self.assertEqual(open(glb, "rb").read(), before)

    def test_rejects_miniature_rescue(self):
        """슬래브 제거 후 캐릭터가 미니어처(구제 배율 초과)면 기각해야 한다
        (실측 gladiator: 키 0.67 → x2.96 업스케일 시 버텍스 43%·텍셀 밀도
        저하로 진흙 품질 — 구제 대신 이미지 재생성이 정답). 원본은 무변경."""
        glb = os.path.join(self.tmp.name, "miniature.glb")
        self._make_glb(glb, with_slab=True, pillar_y0=0.02, pillar_dy=0.015)  # 기둥 높이 0.6
        before = open(glb, "rb").read()
        with self.assertRaises(worker.BasePlaneRescueError):
            worker.strip_base_plane(glb, glb)
        self.assertEqual(open(glb, "rb").read(), before)


class NeutralizeMaterialTest(unittest.TestCase):
    """Hunyuan 텍스처 출력의 광택 과다 재질 중화 회귀 테스트
    (실측: specularColorFactor [2,2,2] + metallic 기본 1.0 + MR 텍스처)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="spriteengine-test-")
        self.addCleanup(self.tmp.cleanup)
        self.glb = os.path.join(self.tmp.name, "in.glb")

    def _make_hunyuan_like_glb(self, path):
        make_glb(path, with_material=True)
        gltf, bin_data = worker._read_glb(path)
        gltf["extensionsUsed"] = ["KHR_materials_specular"]
        gltf["materials"][0] = {
            "name": "Material_0",
            "extensions": {"KHR_materials_specular": {"specularColorFactor": [2.0, 2.0, 2.0]}},
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicRoughnessTexture": {"index": 1},
                # metallicFactor 미지정 = glTF 기본 1.0 (완전 금속)
            },
        }
        worker._write_glb(gltf, bin_data, path)

    def test_neutralizes_specular_and_metallic(self):
        self._make_hunyuan_like_glb(self.glb)
        self.assertTrue(worker.neutralize_material(self.glb, self.glb))
        gltf, _ = worker._read_glb(self.glb)
        mat = gltf["materials"][0]
        self.assertNotIn("extensions", mat)
        self.assertNotIn("KHR_materials_specular", gltf.get("extensionsUsed", []))
        pbr = mat["pbrMetallicRoughness"]
        self.assertEqual(pbr["metallicFactor"], 0.0)
        self.assertEqual(pbr["roughnessFactor"], 1.0)
        self.assertNotIn("metallicRoughnessTexture", pbr)
        # baseColor 텍스처는 보존
        self.assertEqual(pbr["baseColorTexture"], {"index": 0})

    def test_idempotent_noop_after_neutralize(self):
        self._make_hunyuan_like_glb(self.glb)
        worker.neutralize_material(self.glb, self.glb)
        before = open(self.glb, "rb").read()
        self.assertFalse(worker.neutralize_material(self.glb, self.glb))
        self.assertEqual(open(self.glb, "rb").read(), before)

    def test_noop_on_offline_projection_material(self):
        """bake_texture가 만든 오프라인 투영 재질(metallic 0/roughness 1)은
        이미 중화 상태이므로 no-op이어야 한다."""
        make_glb(self.glb, with_material=False)
        gltf, bin_data = worker._read_glb(self.glb)
        gltf["materials"] = [{"name": "FrontProjectedBaseColor",
                              "pbrMetallicRoughness": {"metallicFactor": 0.0,
                                                       "roughnessFactor": 1.0}}]
        worker._write_glb(gltf, bin_data, self.glb)
        before = open(self.glb, "rb").read()
        self.assertFalse(worker.neutralize_material(self.glb, self.glb))
        self.assertEqual(open(self.glb, "rb").read(), before)


class MudGateTest(unittest.TestCase):
    """retopo 저해상도 진흙(mud) 복원 기각 게이트 회귀 테스트.

    실측(2026-07-20 gladiator, 사용자 신고 "완전 심각"): 11,483 verts 진흙
    복원이 직립/파편/슬래브 게이트를 전부 통과해 앱 파이프라인에 등록됐다.
    게이트는 실제 Hunyuan 산출물(hunyuan3d21.glb)에만 적용하고, 의도적
    저폴리인 오프라인 procedural 프리뷰(character.glb)는 예외로 둔다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="spriteengine-mud-")
        self.addCleanup(self.tmp.cleanup)

    def _run_retopo(self, input_path):
        events = []
        original = worker.emit
        worker.emit = lambda t, **p: events.append({"type": t, **p})
        try:
            worker.run({"jobId": "mud-test", "stage": "retopo",
                        "workspace": self.tmp.name, "input": input_path})
        finally:
            worker.emit = original
        return events

    def _count_glb(self, path, count):
        """POSITION accessor count만 가진 최소 GLB (게이트는 count 합만 본다)."""
        gltf = {"asset": {"version": "2.0"},
                "buffers": [{"byteLength": 0}],
                "accessors": [{"componentType": 5126, "count": count, "type": "VEC3"}],
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}]}
        worker._write_glb(gltf, b"", path)

    def test_total_vertex_count_sums_position_accessors(self):
        glb = os.path.join(self.tmp.name, "quad.glb")
        make_glb(glb)
        self.assertEqual(worker.total_vertex_count(glb), 4)

    def test_low_res_hunyuan_rejected(self):
        """실측 진흙 정점 수(11,483)의 hunyuan3d21.glb는 retopo가 명시 실패."""
        glb = os.path.join(self.tmp.name, "hunyuan3d21.glb")
        self._count_glb(glb, 11483)
        with self.assertRaises(worker.LowResolutionMeshError) as ctx:
            self._run_retopo(glb)
        self.assertIn("11483", str(ctx.exception))

    def test_normal_res_hunyuan_passes_gate(self):
        """정상 최소 실측(23,876 verts)은 게이트를 통과해 artifact까지 도달."""
        glb = os.path.join(self.tmp.name, "hunyuan3d21.glb")
        self._count_glb(glb, 23876)
        events = self._run_retopo(glb)
        self.assertTrue(any(e["type"] == "artifact" for e in events))

    def test_procedural_preview_exempt(self):
        """오프라인 프리뷰(character.glb, 저폴리)는 이름으로 게이트 면제."""
        glb = os.path.join(self.tmp.name, "character.glb")
        make_glb(glb)  # 4 verts — 게이트 대상이면 즉시 기각될 정점 수
        events = self._run_retopo(glb)
        self.assertTrue(any(e["type"] == "artifact" for e in events))


if __name__ == "__main__":
    unittest.main()
