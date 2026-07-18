// 모션 스펙 → THREE.AnimationClip 합성기.
// 리깅이 없는 static mesh(Hunyuan3D 출력 등)는 actor root transform으로,
// SkinnedMesh에는 per-bone QuaternionKeyframeTrack을 추가해 관절 변형을 구현한다.
// three.js를 import하므로 CharacterViewport(three 청크) 안에서만 사용한다.
import * as THREE from 'three';
import type {MotionActionId, MotionSpec} from './motionScript';

const FPS = 30;
const DEG = Math.PI / 180;

type Pose = {x: number; y: number; z: number; pitch: number; yaw: number; roll: number; scale: number};
type Cursor = {x: number; y: number; z: number; yaw: number};

const easeOut   = (u: number) => 1 - (1 - u) * (1 - u);
const easeInOut = (u: number) => (u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2);
const bell      = (u: number) => Math.sin(Math.PI * u);

type Primitive = {
  duration: number;
  sample: (u: number, c: Cursor) => Pose;
  commit: (c: Cursor) => void;
};

const base = (c: Cursor): Pose => ({x: c.x, y: c.y, z: c.z, pitch: 0, yaw: c.yaw, roll: 0, scale: 1});

const PRIMITIVES: Record<MotionActionId, Primitive> = {
  fly:   { duration: 1.3, sample: (u,c) => ({...base(c), y: c.y+1.2*easeOut(u), z: c.z-0.9*easeInOut(u), pitch: -0.45*bell(u)}), commit: c=>{c.y+=1.2;c.z-=0.9;} },
  kick:  { duration: 0.9, sample: (u,c) => { const wind=Math.min(u/0.35,1); const strike=u<0.35?0:Math.min((u-0.35)/0.35,1); const recover=u<0.7?0:(u-0.7)/0.3; const pitch=0.4*easeOut(wind)-0.95*easeOut(strike)+0.55*easeInOut(recover); return {...base(c),z:c.z-0.55*easeOut(strike)+0.25*easeInOut(recover),y:c.y+0.12*bell(strike),pitch,roll:0.1*bell(strike)};}, commit: c=>{c.z-=0.3;} },
  punch: { duration: 0.7, sample: (u,c) => { const strike=easeOut(Math.min(u/0.4,1)); const recover=u<0.4?0:easeInOut((u-0.4)/0.6); return {...base(c),z:c.z-0.35*strike+0.35*recover,yaw:c.yaw+0.35*strike-0.35*recover,pitch:-0.15*bell(u)};}, commit:()=>{} },
  jump:  { duration: 1.1, sample: (u,c) => { const squash=u<0.12?1-0.12*bell(u/0.12):u>0.85?1-0.1*bell((u-0.85)/0.15):1+0.08*bell((u-0.12)/0.73); return {...base(c),y:c.y+1.0*bell(easeInOut(u)),scale:squash};}, commit:()=>{} },
  spin:  { duration: 1.0, sample: (u,c) => ({...base(c), yaw:c.yaw+Math.PI*2*easeInOut(u), y:c.y+0.08*bell(u)}), commit:()=>{} },
  dash:  { duration: 0.8, sample: (u,c) => ({...base(c), z:c.z-1.3*easeOut(u), pitch:-0.3*bell(u)}), commit:c=>{c.z-=1.3;} },
  run:   { duration: 1.6, sample: (u,c) => ({...base(c), z:c.z-1.4*u, y:c.y+0.07*Math.abs(Math.sin(u*Math.PI*6)), roll:0.06*Math.sin(u*Math.PI*6), pitch:-0.12*Math.min(1,4*u,4*(1-u))}), commit:c=>{c.z-=1.4;} },
  walk:  { duration: 1.8, sample: (u,c) => ({...base(c), z:c.z-0.9*u, y:c.y+0.04*Math.abs(Math.sin(u*Math.PI*4)), roll:0.05*Math.sin(u*Math.PI*4)}), commit:c=>{c.z-=0.9;} },
  wave:  { duration: 1.2, sample: (u,c) => ({...base(c), roll:0.16*Math.sin(u*Math.PI*4)*bell(u), y:c.y+0.03*bell(u)}), commit:()=>{} },
  bow:   { duration: 1.4, sample: (u,c) => ({...base(c), pitch:-0.7*bell(easeInOut(u))}), commit:()=>{} },
  dance: { duration: 1.6, sample: (u,c) => ({...base(c), y:c.y+0.12*Math.abs(Math.sin(u*Math.PI*4)), yaw:c.yaw+0.45*Math.sin(u*Math.PI*4), roll:0.1*Math.sin(u*Math.PI*8)}), commit:()=>{} },
  idle:  { duration: 2.0, sample: (u,c) => ({...base(c), y:c.y+0.04*Math.sin(u*Math.PI*4), scale:1+0.01*Math.sin(u*Math.PI*4)}), commit:()=>{} },
};

const SETTLE_DURATION = 0.7;

// ── Bone animation ────────────────────────────────────────────────────────────

export type BoneRole =
  | 'hips' | 'spine' | 'spine1' | 'neck' | 'head'
  | 'leftUpperLeg' | 'leftLowerLeg' | 'leftFoot'
  | 'rightUpperLeg' | 'rightLowerLeg' | 'rightFoot'
  | 'leftUpperArm' | 'leftLowerArm'
  | 'rightUpperArm' | 'rightLowerArm';

export type BoneMap = Partial<Record<BoneRole, string>>;

type BoneRot = {x?: number; y?: number; z?: number};
type BonePose = Partial<Record<BoneRole, BoneRot>>;
type BoneTimeline = [number, BonePose][];

// 중립 포즈: BonePose와 BoneRot 양쪽에서 사용하므로 intersection으로 선언
const N = {} as BonePose & BoneRot;

const BONE_TIMELINES: Partial<Record<MotionActionId, BoneTimeline>> = {
  kick:  [[0,N],[.20,{hips:{x:-5},rightUpperLeg:{x:25},rightLowerLeg:{x:-15}}],[.55,{hips:{x:8,z:-4},rightUpperLeg:{x:-65},rightLowerLeg:{x:20},rightFoot:{x:-10}}],[.80,{hips:{x:-3},rightUpperLeg:{x:-20},rightLowerLeg:{x:5}}],[1,N]],
  punch: [[0,N],[.30,{spine:{x:-5,z:5},rightUpperArm:{x:-20,z:-30}}],[.55,{hips:{y:12},spine:{x:-8,z:8},rightUpperArm:{x:-15,z:-55},rightLowerArm:{x:10}}],[.80,{spine:{x:-3},rightUpperArm:{x:-10,z:-25}}],[1,N]],
  jump:  [[0,N],[.15,{hips:{x:8},spine:{x:5},leftUpperLeg:{x:25},rightUpperLeg:{x:25},leftLowerLeg:{x:-20},rightLowerLeg:{x:-20}}],[.40,{hips:{x:-5},spine:{x:-3},leftUpperLeg:{x:-15},rightUpperLeg:{x:-15},leftUpperArm:{x:20,z:30},rightUpperArm:{x:20,z:-30}}],[.70,{hips:{x:5},leftUpperLeg:{x:15},rightUpperLeg:{x:15},leftLowerLeg:{x:-15},rightLowerLeg:{x:-15}}],[.90,{hips:{x:10},spine:{x:8},leftUpperLeg:{x:20},rightUpperLeg:{x:20},leftLowerLeg:{x:-15},rightLowerLeg:{x:-15}}],[1,N]],
  fly:   [[0,N],[.30,{spine:{x:-12},hips:{x:-8},leftUpperArm:{x:10,z:45},rightUpperArm:{x:10,z:-45},leftLowerArm:{x:-15},rightLowerArm:{x:-15}}],[.70,{spine:{x:-15},hips:{x:-10},leftUpperArm:{x:8,z:40},rightUpperArm:{x:8,z:-40}}],[1,N]],
  wave:  [[0,N],[.20,{rightUpperArm:{x:-15,z:-55},rightLowerArm:{x:-30}}],[.50,{rightUpperArm:{x:-10,z:-60},rightLowerArm:{x:-10}}],[.70,{rightUpperArm:{x:-20,z:-55},rightLowerArm:{x:-40}}],[1,N]],
  bow:   [[0,N],[.35,{spine:{x:30},spine1:{x:20},hips:{x:-10}}],[.65,{spine:{x:35},spine1:{x:25},hips:{x:-12},head:{x:10}}],[1,N]],
  dance: [[0,N],[.25,{hips:{z:12},spine:{z:-8},leftUpperArm:{x:-10,z:30},rightUpperArm:{x:-10,z:-30}}],[.50,{hips:{y:10},spine:{z:0},leftUpperArm:{x:-20,z:35},rightUpperArm:{x:-20,z:-35}}],[.75,{hips:{z:-12},spine:{z:8},leftUpperArm:{x:-10,z:20},rightUpperArm:{x:-10,z:-20}}],[1,N]],
  spin:  [[0,N],[.25,{leftUpperArm:{x:5,z:42},rightUpperArm:{x:5,z:-42}}],[.50,{leftUpperArm:{x:8,z:45},rightUpperArm:{x:8,z:-45}}],[.75,{leftUpperArm:{x:5,z:42},rightUpperArm:{x:5,z:-42}}],[1,N]],
  run:   [[0,N],[.25,{spine:{x:-8},leftUpperLeg:{x:-40},rightUpperLeg:{x:35},leftUpperArm:{x:35,z:10},rightUpperArm:{x:-35,z:-10}}],[.50,{spine:{x:-5}}],[.75,{spine:{x:-8},leftUpperLeg:{x:35},rightUpperLeg:{x:-40},leftUpperArm:{x:-35,z:10},rightUpperArm:{x:35,z:-10}}],[1,N]],
  idle:  [[0,{spine:{x:-2}}],[.50,{spine:{x:3},hips:{y:2}}],[1,{spine:{x:-2}}]],
};

const BONE_PATTERNS: Record<BoneRole, RegExp[]> = {
  hips:          [/^hips?$/i, /pelvis/i, /^mixamorigHips/i],
  spine:         [/^spine$/i, /^spine_?0?1$/i, /^mixamorigSpine$/i],
  spine1:        [/^spine_?1$/i, /^spine_?0?2$/i, /^chest$/i, /^upperchest/i, /^mixamorigSpine1$/i, /^mixamorigChest$/i],
  neck:          [/^neck/i, /^mixamorigNeck/i],
  head:          [/^head$/i, /^mixamorigHead$/i],
  leftUpperLeg:  [/^leftupleg$/i, /^leftupperleg$/i, /left.*upper.*leg/i, /leftthigh/i, /^mixamorigLeftUpLeg$/i],
  leftLowerLeg:  [/^leftleg$/i, /^leftlowerleg$/i, /^mixamorigLeftLeg$/i],
  leftFoot:      [/^leftfoot$/i, /leftankle/i, /^mixamorigLeftFoot$/i],
  rightUpperLeg: [/^rightupleg$/i, /^rightupperleg$/i, /right.*upper.*leg/i, /rightthigh/i, /^mixamorigRightUpLeg$/i],
  rightLowerLeg: [/^rightleg$/i, /^rightlowerleg$/i, /^mixamorigRightLeg$/i],
  rightFoot:     [/^rightfoot$/i, /rightankle/i, /^mixamorigRightFoot$/i],
  leftUpperArm:  [/^leftarm$/i, /^leftupperarm$/i, /^mixamorigLeftArm$/i],
  leftLowerArm:  [/^leftforearm$/i, /^leftlowerarm$/i, /^mixamorigLeftForeArm$/i],
  rightUpperArm: [/^rightarm$/i, /^rightupperarm$/i, /^mixamorigRightArm$/i],
  rightLowerArm: [/^rightforearm$/i, /^rightlowerarm$/i, /^mixamorigRightForeArm$/i],
};

export function detectBoneMap(boneNames: string[]): BoneMap {
  const map: BoneMap = {};
  for (const [role, patterns] of Object.entries(BONE_PATTERNS) as [BoneRole, RegExp[]][]) {
    map[role] = patterns.reduce<string | undefined>((found, pat) => found ?? boneNames.find(n => pat.test(n)), undefined);
  }
  return map;
}

function buildBoneTracks(spec: MotionSpec, boneMap: BoneMap): THREE.QuaternionKeyframeTrack[] {
  const tempo = spec.tempo > 0 ? spec.tempo : 1;
  const usedRoles = new Set<BoneRole>();
  for (const action of spec.actions) {
    const tl = BONE_TIMELINES[action.id];
    if (tl) tl.forEach(([,pose]) => (Object.keys(pose) as BoneRole[]).forEach(r => usedRoles.add(r)));
  }
  if (!usedRoles.size) return [];

  const timesMap = new Map<BoneRole, number[]>();
  const quatsMap = new Map<BoneRole, number[]>();
  for (const role of usedRoles) { if (!boneMap[role]) continue; timesMap.set(role,[]); quatsMap.set(role,[]); }

  const euler = new THREE.Euler(); const quat = new THREE.Quaternion();
  const pushKey = (role: BoneRole, t: number, rot: BoneRot) => {
    const ts = timesMap.get(role); const qs = quatsMap.get(role); if (!ts||!qs) return;
    euler.set((rot.x??0)*DEG,(rot.y??0)*DEG,(rot.z??0)*DEG,'XYZ'); quat.setFromEuler(euler);
    ts.push(t); qs.push(quat.x,quat.y,quat.z,quat.w);
  };

  let t = 0;
  for (const action of spec.actions) {
    const dur = PRIMITIVES[action.id].duration / tempo;
    const repeat = Math.max(1, action.repeat||1);
    const tl = BONE_TIMELINES[action.id];
    for (let r=0; r<repeat; r++) {
      for (const role of usedRoles) {
        if (!boneMap[role]) continue;
        if (!tl) { pushKey(role,t,N); pushKey(role,t+dur,N); }
        else { for (const [u,pose] of tl) pushKey(role, t+u*dur, pose[role]??N); }
      }
      t += dur;
    }
  }
  for (const role of usedRoles) {
    if (!boneMap[role]) continue;
    pushKey(role,t,N); pushKey(role,t+SETTLE_DURATION,N);
  }

  return (Array.from(usedRoles) as BoneRole[]).flatMap(role => {
    const boneName = boneMap[role]; if (!boneName) return [];
    const ts = timesMap.get(role)!; const qs = quatsMap.get(role)!;
    if (ts.length < 2) return [];
    return [new THREE.QuaternionKeyframeTrack(`${boneName}.quaternion`, ts, qs)];
  });
}

// ── Main export ───────────────────────────────────────────────────────────────

export function buildMotionClip(spec: MotionSpec, boneMap?: BoneMap): THREE.AnimationClip {
  const times: number[] = []; const positions: number[] = []; const quaternions: number[] = []; const scales: number[] = [];
  const euler = new THREE.Euler(); const quat = new THREE.Quaternion();
  const cursor: Cursor = {x:0, y:0, z:0, yaw:0};
  let t = 0;
  const push = (time: number, p: Pose) => {
    times.push(time); positions.push(p.x,p.y,p.z);
    euler.set(p.pitch,p.yaw,p.roll,'YXZ'); quat.setFromEuler(euler);
    quaternions.push(quat.x,quat.y,quat.z,quat.w); scales.push(p.scale,p.scale,p.scale);
  };
  const tempo = spec.tempo>0?spec.tempo:1;
  for (const action of spec.actions) {
    const prim=PRIMITIVES[action.id]; const dur=prim.duration/tempo; const repeat=Math.max(1,action.repeat||1);
    for (let r=0; r<repeat; r++) {
      const steps=Math.max(2,Math.round(dur*FPS)); const start:Cursor={...cursor};
      for (let i=0; i<steps; i++) push(t+(i/steps)*dur,prim.sample(i/steps,start));
      prim.commit(cursor); t+=dur;
    }
  }
  const sf:Cursor={...cursor}; const syt=Math.round(cursor.yaw/(Math.PI*2))*Math.PI*2;
  const ss=Math.max(2,Math.round(SETTLE_DURATION*FPS));
  for (let i=0; i<=ss; i++) { const u=easeInOut(i/ss); push(t+(i/ss)*SETTLE_DURATION,{x:sf.x*(1-u),y:sf.y*(1-u),z:sf.z*(1-u),yaw:sf.yaw+(syt-sf.yaw)*u,pitch:0,roll:0,scale:1}); }
  t+=SETTLE_DURATION;

  const rootTracks = [
    new THREE.VectorKeyframeTrack('.position',times,positions),
    new THREE.QuaternionKeyframeTrack('.quaternion',times,quaternions),
    new THREE.VectorKeyframeTrack('.scale',times,scales),
  ];
  const boneTracks = boneMap&&Object.values(boneMap).some(Boolean) ? buildBoneTracks(spec,boneMap) : [];
  return new THREE.AnimationClip(spec.clipName, t, [...rootTracks, ...boneTracks]);
}
