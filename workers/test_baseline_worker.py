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
        # WEIGHTS_0 accessor: each vertex has one dominant weight = 1.0
        acc = gltf["accessors"][attrs["WEIGHTS_0"]]
        view = gltf["bufferViews"][acc["bufferView"]]
        base = view["byteOffset"]
        for i in range(acc["count"]):
            w0 = struct.unpack_from("<f", bin_data, base + i * 16)[0]
            self.assertAlmostEqual(w0, 1.0, places=5)

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


if __name__ == "__main__":
    unittest.main()
