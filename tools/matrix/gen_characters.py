#!/usr/bin/env python3
"""gpt-image-2로 3D 복원용 캐릭터 원본 이미지 10종 생성.

Hunyuan3D-2.1 입력에 유리한 조건: 단일 전신 캐릭터, 정면, A-pose,
깨끗한 밝은 배경, 선명한 실루엣.

키: OPENAI_API_KEY 환경변수 (없으면 ~/.davinci/app.db의 image_gen_openai_key 폴백)
출력: $MATRIX_ROOT/characters/*.png (기본 /tmp/spriteengine_matrix)
"""
import argparse
import base64
import json
import os
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("MATRIX_ROOT", "/tmp/spriteengine_matrix"))
OUT = ROOT / "characters"
OUT.mkdir(parents=True, exist_ok=True)


def load_key():
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"].strip()
    db = Path.home() / ".davinci/app.db"
    row = sqlite3.connect(str(db)).execute(
        "SELECT value FROM app_config WHERE key='image_gen_openai_key'").fetchone()
    if not row:
        sys.exit("OPENAI_API_KEY가 없고 davinci 설정에서도 키를 찾지 못했습니다")
    return row[0].strip()


key = load_key()

# 세트 2: 앱 프로덕션 경로 검증용 신규 10종 (--set 2 또는 CHARACTER_SET=2)
CHARACTERS_SET2 = {
    "samurai":    "a samurai warrior in dark red lacquered armor with a katana sheathed at the hip",
    "boxer":      "a muscular boxer wearing red gloves, blue shorts and boxing shoes",
    "ballerina":  "a ballerina in a light pink tutu and pointe shoes with hair in a bun",
    "zombie":     "a cartoonish zombie office worker with torn shirt and green skin",
    "superhero":  "a superhero in a blue suit with a yellow cape and emblem on the chest",
    "detective":  "a detective in a beige trench coat and fedora hat holding nothing",
    "scientist":  "a scientist in a white lab coat with safety goggles on the forehead",
    "farmer":     "a friendly farmer in denim overalls, plaid shirt and straw hat",
    "idol":       "a k-pop idol dancer in a stylish silver stage outfit",
    "barbarian":  "a barbarian warrior with a fur loincloth, leather straps and a big axe on the back",
}

CHARACTERS = {
    "knight":     "a fantasy knight in silver plate armor with a blue cape",
    "wizard":     "an old wizard with a long white beard, purple robe and pointed hat",
    "robot":      "a friendly humanoid robot with white and orange panels",
    "astronaut":  "an astronaut in a white space suit with orange visor accents",
    "ninja":      "a ninja in a black outfit with a red scarf",
    "pirate":     "a pirate captain with a tricorn hat, red coat and boots",
    "chef":       "a cheerful chef in white uniform and tall chef hat",
    "firefighter": "a firefighter in yellow protective gear and helmet",
    "viking":     "a viking warrior with braided beard, horned helmet and fur cloak",
    "cyberpunk":  "a cyberpunk woman with neon blue jacket and glowing visor",
}

TEMPLATE = ("Full body 3D game character concept art of {desc}. Single character, "
            "standing straight in a relaxed A-pose with arms slightly away from the body, "
            "facing the camera directly, feet visible, whole body in frame. "
            "Stylized PBR game-asset look, clean plain light gray studio background, "
            "soft even lighting, no shadows on background, no text, no watermark.")


def generate(name, desc):
    body = json.dumps({
        "model": "gpt-image-2", "prompt": TEMPLATE.format(desc=desc),
        "size": "1024x1536", "quality": "medium", "n": 1,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/images/generations", data=body, method="POST",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    png = base64.b64decode(data["data"][0]["b64_json"])
    target = OUT / f"{name}.png"
    target.write_bytes(png)
    print(f"{name}: {len(png)} bytes -> {target}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["1", "2"], default=os.environ.get("CHARACTER_SET", "1"))
    ap.add_argument("--only", help="쉼표로 구분한 캐릭터 이름만 생성 (예: samurai,boxer)")
    args = ap.parse_args()
    pool = CHARACTERS if args.set == "1" else CHARACTERS_SET2
    if args.only:
        wanted = [n.strip() for n in args.only.split(",") if n.strip()]
        pool = {n: pool[n] for n in wanted if n in pool}

    failures = {}
    for name, desc in pool.items():
        if (OUT / f"{name}.png").exists():
            print(f"{name}: exists, skipping", flush=True)
            continue
        for attempt in (1, 2):
            try:
                generate(name, desc)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"{name}: attempt {attempt} failed: {exc}", flush=True)
                failures[name] = str(exc)
                time.sleep(5)
        else:
            continue
        failures.pop(name, None)

    if failures:
        print("FAILED:", json.dumps(failures, indent=2))
        sys.exit(1)
    print("ALL_CHARACTERS_OK")


if __name__ == "__main__":
    main()
