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
    "superhero":  "a masked vigilante hero in a matte green and black tactical suit with a utility belt, no cape, no logo",
    "detective":  "a detective in a beige trench coat and fedora hat holding nothing",
    "scientist":  "a scientist in a white lab coat with safety goggles on the forehead",
    "farmer":     "a friendly farmer in denim overalls, plaid shirt and straw hat",
    "idol":       "a k-pop idol dancer in a stylish silver stage outfit",
    # 주의: 몸통에서 멀리 뻗는 소품(등의 큰 도끼, 넓은 모자 챙)은 실루엣을
    # 옆으로 퍼뜨려 Hunyuan3D 복원이 누운 형태로 나오는 실패를 유발했다.
    # 소품을 몸에 밀착시키는 프롬프트로 조정 (실측: barbarian/cowboy 복원 실패).
    "barbarian":  "a barbarian warrior with a fur loincloth, leather straps, iron bracers and war paint, no weapons",
    "cowboy":     "a cowboy in a brown leather vest, jeans, a snug narrow-brim cowboy hat and leather boots, no weapons",
    "clown":      "a cheerful circus clown in a colorful polka-dot outfit with a red nose and curly wig",
    "wrestler":   "a professional wrestler in a spandex singlet with a championship belt around the waist",
    "monk":       "a shaolin monk in an orange and yellow robe with prayer beads around the wrist",
    "mechanic":   "a mechanic in navy blue coveralls with a tool belt and a grease-stained cap",
    # 신규 5종 (2026-07-20): 실측 학습된 제약 반영 — 몸에서 멀리 뻗는 소품 금지
    # (barbarian/cowboy 복원 실패), 케이프 금지(superhero), 챙은 snug/narrow만.
    "vampire":    "an elegant vampire in a fitted black victorian suit with a dark red vest and slicked-back hair, no cape",
    # 주의: "teal scrubs + stethoscope" 계열은 서로 다른 이미지 3장 모두
    # Hunyuan3D가 상단부 파편(최장축 0.3~0.6, 중심 오프셋 1.5~3.1)만 복원하는
    # 체계적 실패 → 청진기를 빼고 단순한 전신 유니폼 묘사로 조정 (실측 2026-07-20).
    "nurse":      "a friendly nurse in a teal short-sleeve scrub top and matching scrub pants with white sneakers",
    # 주의: 맨살 위주 gladiator 묘사는 gpt-image-2 safety(output 단계)에 차단됨
    # (실측 400 moderation_blocked) → 붉은 튜닉을 입은 로마 병사풍으로 조정.
    "gladiator":  "a roman gladiator warrior wearing a red tunic under a bronze chest plate, leather arm guards and sandals, no weapons, no helmet",
    "skater":     "a skateboarder in a fitted hoodie, beanie and knee pads, no skateboard",
    "explorer":   "a jungle explorer in a khaki shirt, cargo pants and a snug narrow-brim boonie hat",
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
            "Natural human anatomy with moderate shoulder width, relaxed sloping shoulders, "
            "anatomically correct separated hands, five clearly formed fingers on each hand, "
            "palms angled slightly toward the thighs, no clenched fists, no fused fingers. "
            "Stylized PBR game-asset look, clean plain light gray studio background, "
            "soft even lighting, no shadows on background, no text, no watermark.")

# 세트 q: 4족(quadruped) 리깅 경로 검증용 (--set q 또는 CHARACTER_SET=q).
# 복원 결과가 X축 체장으로 나와도 auto-rig의 측방향 정렬(yaw 90°)·머리 방향
# 감지(180°) 경로가 정규화한다.
CHARACTERS_QUADRUPED = {
    "dog":   "a friendly medium-sized shiba dog with orange and cream fur and a curled tail",
    "dog2":  "a friendly medium-sized shiba dog with orange and cream fur and a curled tail",
    "dog3":  "a friendly medium-sized shiba dog with orange and cream fur and a curled tail",
    "dog4":  "a friendly medium-sized shiba dog with orange and cream fur and a curled tail",
    "horse": "a sturdy brown horse with a dark mane and tail",
    "cat":   "a gray tabby cat with a long straight tail",
}

# A-pose/팔 문구는 휴머노이드 전용이라 4족은 별도 템플릿을 쓴다.
# 실측 회귀(dog): 정측면 뷰는 반대편 다리가 완전히 가려져 Hunyuan3D 복원에서
# 가려진 다리의 형상·텍스처가 뭉개졌다 → 3/4 시점으로 네 다리가 모두 서로
# 떨어져 보이게 강제한다 (validate_character의 legs 게이트가 결손을 실측 차단).
QUADRUPED_TEMPLATE = ("Full body 3D game animal concept art of {desc}. Single animal, "
                      "standing naturally on all four legs, seen from a front three-quarter "
                      "view so that all four legs are clearly separate and fully visible "
                      "with visible gaps between them, no leg hidden or overlapping, "
                      "tail visible, whole body in frame. Stylized PBR game-asset look, "
                      "clean plain light gray studio background, soft even lighting, "
                      "no shadows on background, no text, no watermark.")


def generate(name, desc):
    # 멀티이미지 후보(name-v2, name-v3 …)도 원본과 같은 템플릿을 쓴다.
    base = name.split("-v", 1)[0]
    tpl = QUADRUPED_TEMPLATE if base in CHARACTERS_QUADRUPED else TEMPLATE
    # 4족은 몸이 가로로 길어 가로형 캔버스가 잘림 없이 담긴다.
    size = "1536x1024" if base in CHARACTERS_QUADRUPED else "1024x1536"
    body = json.dumps({
        "model": "gpt-image-2", "prompt": tpl.format(desc=desc),
        "size": size, "quality": "medium", "n": 1,
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
    ap.add_argument("--set", choices=["1", "2", "q"], default=os.environ.get("CHARACTER_SET", "1"))
    ap.add_argument("--only", help="쉼표로 구분한 캐릭터 이름만 생성 (예: samurai,boxer)")
    args = ap.parse_args()
    pool = {"1": CHARACTERS, "2": CHARACTERS_SET2, "q": CHARACTERS_QUADRUPED}[args.set]
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
