// 텍스트 연출 AnimationClip을 모델에 bake해 GLB 바이너리로 내보낸다.
// three.js를 import하므로 MotionPanel에서 dynamic import로만 로드해
// 메인 번들에 three가 포함되지 않도록 한다.
import * as THREE from 'three';
import {GLTFLoader} from 'three/examples/jsm/loaders/GLTFLoader.js';
import {GLTFExporter} from 'three/examples/jsm/exporters/GLTFExporter.js';
import {buildMotionClip} from './motionClip';
import type {MotionSpec} from './motionScript';

// Uint8Array → base64 (call stack 한계를 피하기 위해 chunk 단위 변환)
function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

/** modelUrl의 GLB에 연출 clip을 추가해 GLB(base64)로 bake한다. embedded clip은 유지된다. */
export async function bakeAnimatedGLBBase64(modelUrl: string, spec: MotionSpec): Promise<string> {
  const res = await fetch(modelUrl);
  if (!res.ok) throw new Error(`모델을 읽을 수 없습니다 (${res.status})`);
  const buffer = await res.arrayBuffer();
  const gltf = await new GLTFLoader().parseAsync(buffer, '');

  // viewport와 동일한 정규화(높이 2.75, 바닥 접지)를 적용해
  // 연출 clip의 이동량이 미리보기와 같은 비율로 보이도록 한다.
  const model = gltf.scene;
  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const scale = 2.75 / Math.max(size.y, .01);
  model.scale.setScalar(scale);
  model.position.set(-center.x * scale, -box.min.y * scale, -center.z * scale);

  // 연출 clip은 actor 루트를 대상으로 한다 (viewport 재생 구조와 동일)
  const actor = new THREE.Group();
  actor.name = 'actor-root';
  actor.add(model);

  const clip = buildMotionClip(spec);
  const out = await new GLTFExporter().parseAsync(actor, {
    binary: true,
    animations: [...gltf.animations, clip],
  });
  return bytesToBase64(new Uint8Array(out as ArrayBuffer));
}
