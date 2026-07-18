#!/usr/bin/env python3
"""HY-Motion-1.0 텍스트→모션 생성 서브프로세스 (RunPod 워커 내부 전용).

handler.py가 task=="motion" 요청을 받으면 이 스크립트를 별도 프로세스로 실행한다.
프로세스 격리 이유:
  1) 의존성 격리 — HY-Motion은 transformers 4.53(Qwen3 지원)이 필요하지만
     Hunyuan3D 스택은 다른 버전을 쓴다. PYTHONPATH 오버레이(pydeps311_motion 우선)를
     이 프로세스에만 적용한다.
  2) VRAM 회수 — Qwen3-8B 텍스트 인코더(~16GB)가 생성 후 프로세스 종료와 함께
     완전히 해제되어 같은 워커의 shape/paint 파이프라인과 공존할 수 있다.

프로토콜: stdin으로 JSON 요청 1개 → stdout 마지막 줄에 JSON 응답 1줄.
  요청:  {"prompts": [{"id": "run", "text": "...", "duration": 4.0}, ...],
          "seed": 42, "cfg_scale": 5.0}
  응답:  {"motions": [{"id", "fps", "frames", "joints", "quats", "trans", ...}],
          "errors": {id: message}}

quats는 SMPL-H 바디 22조인트의 로컬 회전(xyzw, 프레임×조인트×4),
trans는 루트 이동(프레임×3, 미터). Rh(글로벌 루트 방향)는 poses[:, :3]에
포함되어 있으므로 Pelvis 회전이 곧 루트 방향이다.
"""
import json
import math
import os
import sys

# SMPL-H 바디 조인트 순서 (hymotion smplh2woodfbx.SMPLH_JOINT2NUM의 앞 22개)
SMPLH_BODY_JOINTS = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
    "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist",
]


def axis_angle_to_quat(x, y, z):
    """axis-angle (rad) → 쿼터니언 (x, y, z, w)."""
    angle = math.sqrt(x * x + y * y + z * z)
    if angle < 1e-8:
        return (0.0, 0.0, 0.0, 1.0)
    s = math.sin(angle / 2.0) / angle
    return (x * s, y * s, z * s, math.cos(angle / 2.0))


def generate(runtime, item, seed, cfg_scale):
    text = str(item.get("text", "")).strip()
    duration = min(max(float(item.get("duration", 4.0)), 1.0), 10.0)
    if not text:
        raise ValueError("empty prompt text")
    # output_format="dict" — FBX SDK 없이 (html, [], model_output) 3-튜플 반환.
    _, _, model_output = runtime.generate_motion(
        text=text,
        seeds_csv=str(seed),
        duration=duration,
        cfg_scale=cfg_scale,
        output_format="dict",
        output_dir="/tmp/hymotion_out",
        output_filename=str(item.get("id", "motion")),
    )
    from hymotion.pipeline.body_model import construct_smpl_data_dict

    smpl = construct_smpl_data_dict(
        model_output["rot6d"][0].clone(), model_output["transl"][0].clone()
    )
    poses = smpl["poses"]  # (frames, 52*3) axis-angle — 앞 22*3이 바디
    trans = smpl["trans"]  # (frames, 3) 미터
    frames = int(smpl["num_frames"])
    quats = []
    for f in range(frames):
        row = poses[f]
        quats.append([
            axis_angle_to_quat(float(row[j * 3]), float(row[j * 3 + 1]), float(row[j * 3 + 2]))
            for j in range(len(SMPLH_BODY_JOINTS))
        ])
    return {
        "id": item.get("id", "motion"),
        "text": text,
        "fps": int(smpl.get("mocap_framerate", 30)),
        "frames": frames,
        "joints": SMPLH_BODY_JOINTS,
        "quats": quats,
        "trans": [[float(v) for v in trans[f]] for f in range(frames)],
    }


def main():
    req = json.loads(sys.stdin.read())
    prompts = req.get("prompts") or []
    seed = int(req.get("seed", 42))
    cfg_scale = float(req.get("cfg_scale", 5.0))
    root = os.getenv("HYMOTION_ROOT", "/runpod-volume/hymotion10")
    os.chdir(root)  # config/stats 상대경로가 repo 루트를 가정한다

    from hymotion.utils.t2m_runtime import T2MRuntime

    # 주의: ckpt_name을 생략하면 T2MRuntime 기본값 "latest.ckpt"(cwd 상대)가 되는데,
    # load_in_demo는 파일이 없으면 warnings.warn만 찍고 랜덤 초기화 가중치로
    # 진행한다. 모델이 adaLN-Zero 초기화라 조건화 gate가 전부 0 → 텍스트가
    # 완전히 무시되고 같은 seed/duration이면 프롬프트와 무관하게 동일한 모션이
    # 나온다. 반드시 절대경로를 넘기고 존재를 선검증한다.
    ckpt = os.path.join(root, "ckpts/tencent/HY-Motion-1.0-Lite/latest.ckpt")
    if not os.path.isfile(ckpt):
        sys.stdout.write("\n" + json.dumps({
            "motions": [], "errors": {"_runtime": f"checkpoint missing: {ckpt}"},
        }) + "\n")
        return
    runtime = T2MRuntime(
        config_path=os.path.join(root, "ckpts/tencent/HY-Motion-1.0-Lite/config.yml"),
        ckpt_name=ckpt,
        disable_prompt_engineering=True,  # LLM 리라이터 없이 원문 프롬프트 사용
    )
    motions, errors = [], {}
    for item in prompts:
        try:
            motions.append(generate(runtime, item, seed, cfg_scale))
        except Exception as exc:  # noqa: BLE001 — 프롬프트별 독립 실패 허용
            errors[str(item.get("id", "?"))] = f"{type(exc).__name__}: {exc}"
    # stdout에는 hymotion 로그가 섞이므로 마지막 한 줄로 구분 출력한다.
    sys.stdout.write("\n" + json.dumps({"motions": motions, "errors": errors}) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
