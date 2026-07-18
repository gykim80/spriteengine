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


def make_glb(path, with_material=False):
    """POSITION+NORMAL만 가진 최소 사각형 GLB (Hunyuan shape-only 출력 모사)."""
    positions = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
    normals = [(0, 0, 1), (0, 0, 1), (0, 0, -1), (0, 0, -1)]
    blob = bytearray()
    for v in positions + normals:
        blob += struct.pack("<fff", *v)
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 48},
            {"buffer": 0, "byteOffset": 48, "byteLength": 48},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
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


if __name__ == "__main__":
    unittest.main()
