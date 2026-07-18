// 모션 스펙 → THREE.AnimationClip 합성기.
// 리깅이 없는 static mesh(Hunyuan3D 출력 등)도 root transform 연출로 움직일 수 있도록
// actor 루트의 position/quaternion/scale 키프레임 트랙을 파라메트릭으로 생성한다.
// three.js를 import하므로 CharacterViewport(three 청크) 안에서만 사용한다.
import * as THREE from 'three';
import type {MotionActionId, MotionSpec} from './motionScript';

const FPS = 30;

type Pose = {x: number; y: number; z: number; pitch: number; yaw: number; roll: number; scale: number};
// 프리미티브 시작 시점의 누적 상태 (이전 동작이 이동시킨 위치를 이어받는다)
type Cursor = {x: number; y: number; z: number; yaw: number};

const easeOut = (u: number) => 1 - (1 - u) * (1 - u);
const easeInOut = (u: number) => (u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2);
const bell = (u: number) => Math.sin(Math.PI * u); // 0→1→0

type Primitive = {
  duration: number;
  /** u∈[0,1]에서 cursor 기준 상대 pose */
  sample: (u: number, c: Cursor) => Pose;
  /** 종료 시 cursor에 커밋되는 변위 */
  commit: (c: Cursor) => void;
};

const base = (c: Cursor): Pose => ({x: c.x, y: c.y, z: c.z, pitch: 0, yaw: c.yaw, roll: 0, scale: 1});

const PRIMITIVES: Record<MotionActionId, Primitive> = {
  fly: {
    duration: 1.3,
    sample: (u, c) => ({...base(c), y: c.y + 1.2 * easeOut(u), z: c.z - 0.9 * easeInOut(u), pitch: -0.45 * bell(u)}),
    commit: c => { c.y += 1.2; c.z -= 0.9; },
  },
  kick: {
    duration: 0.9,
    // 0~0.35 젖히기(windup) → 0.35~0.7 전방 타격 → 복귀
    sample: (u, c) => {
      const wind = Math.min(u / 0.35, 1);
      const strike = u < 0.35 ? 0 : Math.min((u - 0.35) / 0.35, 1);
      const recover = u < 0.7 ? 0 : (u - 0.7) / 0.3;
      const pitch = 0.4 * easeOut(wind) - 0.95 * easeOut(strike) + 0.55 * easeInOut(recover);
      return {...base(c), z: c.z - 0.55 * easeOut(strike) + 0.25 * easeInOut(recover), y: c.y + 0.12 * bell(strike), pitch, roll: 0.1 * bell(strike)};
    },
    commit: c => { c.z -= 0.3; },
  },
  punch: {
    duration: 0.7,
    sample: (u, c) => {
      const strike = easeOut(Math.min(u / 0.4, 1));
      const recover = u < 0.4 ? 0 : easeInOut((u - 0.4) / 0.6);
      return {...base(c), z: c.z - 0.35 * strike + 0.35 * recover, yaw: c.yaw + 0.35 * strike - 0.35 * recover, pitch: -0.15 * bell(u)};
    },
    commit: () => {},
  },
  jump: {
    duration: 1.1,
    sample: (u, c) => {
      const squash = u < 0.12 ? 1 - 0.12 * bell(u / 0.12) : u > 0.85 ? 1 - 0.1 * bell((u - 0.85) / 0.15) : 1 + 0.08 * bell((u - 0.12) / 0.73);
      return {...base(c), y: c.y + 1.0 * bell(easeInOut(u)), scale: squash};
    },
    commit: () => {},
  },
  spin: {
    duration: 1.0,
    sample: (u, c) => ({...base(c), yaw: c.yaw + Math.PI * 2 * easeInOut(u), y: c.y + 0.08 * bell(u)}),
    commit: () => {},
  },
  dash: {
    duration: 0.8,
    sample: (u, c) => ({...base(c), z: c.z - 1.3 * easeOut(u), pitch: -0.3 * bell(u)}),
    commit: c => { c.z -= 1.3; },
  },
  run: {
    duration: 1.6,
    sample: (u, c) => ({...base(c), z: c.z - 1.4 * u, y: c.y + 0.07 * Math.abs(Math.sin(u * Math.PI * 6)), roll: 0.06 * Math.sin(u * Math.PI * 6), pitch: -0.12 * Math.min(1, 4 * u, 4 * (1 - u))}),
    commit: c => { c.z -= 1.4; },
  },
  walk: {
    duration: 1.8,
    sample: (u, c) => ({...base(c), z: c.z - 0.9 * u, y: c.y + 0.04 * Math.abs(Math.sin(u * Math.PI * 4)), roll: 0.05 * Math.sin(u * Math.PI * 4)}),
    commit: c => { c.z -= 0.9; },
  },
  wave: {
    duration: 1.2,
    sample: (u, c) => ({...base(c), roll: 0.16 * Math.sin(u * Math.PI * 4) * bell(u), y: c.y + 0.03 * bell(u)}),
    commit: () => {},
  },
  bow: {
    duration: 1.4,
    sample: (u, c) => ({...base(c), pitch: -0.7 * bell(easeInOut(u))}),
    commit: () => {},
  },
  dance: {
    duration: 1.6,
    sample: (u, c) => ({...base(c), y: c.y + 0.12 * Math.abs(Math.sin(u * Math.PI * 4)), yaw: c.yaw + 0.45 * Math.sin(u * Math.PI * 4), roll: 0.1 * Math.sin(u * Math.PI * 8)}),
    commit: () => {},
  },
  idle: {
    duration: 2.0,
    sample: (u, c) => ({...base(c), y: c.y + 0.04 * Math.sin(u * Math.PI * 4), scale: 1 + 0.01 * Math.sin(u * Math.PI * 4)}),
    commit: () => {},
  },
};

const SETTLE_DURATION = 0.7;

/** MotionSpec을 루프 가능한 AnimationClip으로 컴파일한다. 트랙은 mixer root(actor)를 대상으로 한다. */
export function buildMotionClip(spec: MotionSpec): THREE.AnimationClip {
  const times: number[] = [];
  const positions: number[] = [];
  const quaternions: number[] = [];
  const scales: number[] = [];
  const euler = new THREE.Euler();
  const quat = new THREE.Quaternion();
  const cursor: Cursor = {x: 0, y: 0, z: 0, yaw: 0};

  let t = 0;
  const push = (time: number, p: Pose) => {
    times.push(time);
    positions.push(p.x, p.y, p.z);
    euler.set(p.pitch, p.yaw, p.roll, 'YXZ');
    quat.setFromEuler(euler);
    quaternions.push(quat.x, quat.y, quat.z, quat.w);
    scales.push(p.scale, p.scale, p.scale);
  };

  // tempo("빠르게"/"천천히")는 프리미티브 길이를 줄이거나 늘려 전체 속도를 바꾼다
  const tempo = spec.tempo > 0 ? spec.tempo : 1;
  for (const action of spec.actions) {
    const prim = PRIMITIVES[action.id];
    const dur = prim.duration / tempo;
    const repeat = Math.max(1, action.repeat || 1);
    // "두 번 발차기" → 커서를 커밋하며 프리미티브를 연속 반복
    for (let r = 0; r < repeat; r++) {
      const steps = Math.max(2, Math.round(dur * FPS));
      const start: Cursor = {...cursor};
      for (let i = 0; i < steps; i++) push(t + (i / steps) * dur, prim.sample(i / steps, start));
      prim.commit(cursor);
      t += dur;
    }
  }

  // 마무리: 원점 복귀 (yaw는 가장 가까운 정면으로) → LoopRepeat 시 자연스럽게 이어진다
  const settleFrom: Cursor = {...cursor};
  const settleYawTarget = Math.round(cursor.yaw / (Math.PI * 2)) * Math.PI * 2;
  const settleSteps = Math.max(2, Math.round(SETTLE_DURATION * FPS));
  for (let i = 0; i <= settleSteps; i++) {
    const u = easeInOut(i / settleSteps);
    push(t + (i / settleSteps) * SETTLE_DURATION, {
      x: settleFrom.x * (1 - u),
      y: settleFrom.y * (1 - u),
      z: settleFrom.z * (1 - u),
      yaw: settleFrom.yaw + (settleYawTarget - settleFrom.yaw) * u,
      pitch: 0, roll: 0, scale: 1,
    });
  }
  t += SETTLE_DURATION;

  return new THREE.AnimationClip(spec.clipName, t, [
    new THREE.VectorKeyframeTrack('.position', times, positions),
    new THREE.QuaternionKeyframeTrack('.quaternion', times, quaternions),
    new THREE.VectorKeyframeTrack('.scale', times, scales),
  ]);
}
