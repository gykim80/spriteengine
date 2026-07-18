import {useEffect,useRef} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls.js';
import {GLTFLoader} from 'three/examples/jsm/loaders/GLTFLoader.js';
import {buildMotionClip} from './motion/motionClip';
import type {MotionSpec} from './motion/motionScript';

export type PlaybackState={duration:number,time:number,clips:string[],active:string,loaded:boolean,error?:string};
type Props={playing:boolean;clip:string;speed:number;time:number;showSkeleton:boolean;modelUrl?:string;motion?:MotionSpec|null;onState?:(state:PlaybackState)=>void};

export default function CharacterViewport({playing,clip,speed,time,showSkeleton,modelUrl='/models/Soldier.glb',motion=null,onState}:Props){
 const host=useRef<HTMLDivElement>(null); const stateRef=useRef({playing,clip,speed,time,showSkeleton,motion,onState});
 useEffect(()=>{stateRef.current={playing,clip,speed,time,showSkeleton,motion,onState}},[playing,clip,speed,time,showSkeleton,motion,onState]);
 useEffect(()=>{
  const el=host.current!; const scene=new THREE.Scene(); scene.background=new THREE.Color(0xeef1f6); scene.fog=new THREE.Fog(0xeef1f6,7,16);
  const camera=new THREE.PerspectiveCamera(38,Math.max(el.clientWidth,1)/Math.max(el.clientHeight,1),.05,100); camera.position.set(3.2,1.9,5.4);
  const renderer=new THREE.WebGLRenderer({antialias:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.setSize(el.clientWidth,el.clientHeight); renderer.shadowMap.enabled=true; renderer.shadowMap.type=THREE.PCFShadowMap; renderer.outputColorSpace=THREE.SRGBColorSpace; el.appendChild(renderer.domElement);
  scene.add(new THREE.HemisphereLight(0xffffff,0xdfe6f0,2.3)); const key=new THREE.DirectionalLight(0xfff4e8,3.2); key.position.set(3,6,4); key.castShadow=true; key.shadow.mapSize.set(2048,2048); scene.add(key); const rim=new THREE.DirectionalLight(0xbcd0ff,1.6);rim.position.set(-4,3,-4);scene.add(rim);
  const floor=new THREE.Mesh(new THREE.PlaneGeometry(24,24),new THREE.MeshStandardMaterial({color:0xe7ebf2,roughness:1})); floor.rotation.x=-Math.PI/2; floor.receiveShadow=true; scene.add(floor); const grid=new THREE.GridHelper(24,48,0xb9c3d4,0xd8dfeb);grid.position.y=.004;scene.add(grid);
  const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.target.set(0,1,0);controls.minDistance=2;controls.maxDistance=10;
  let mixer:THREE.AnimationMixer|undefined, action:THREE.AnimationAction|undefined, model:THREE.Object3D|undefined, helper:THREE.SkeletonHelper|undefined, clips:THREE.AnimationClip[]=[]; let active=''; let duration=0; let disposed=false;
  // 텍스트 연출로 합성된 clip과, 마지막으로 컴파일한 spec (참조 비교로 재컴파일 감지)
  let directed:THREE.AnimationClip|undefined; let lastMotion:MotionSpec|null=null;
  const report=(error?:string)=>stateRef.current.onState?.({duration,time:action?.time||0,clips:clips.map(c=>c.name),active,loaded:!!model,error});
  new GLTFLoader().load(modelUrl,gltf=>{if(disposed)return;model=gltf.scene; const box=new THREE.Box3().setFromObject(model);const size=box.getSize(new THREE.Vector3());const center=box.getCenter(new THREE.Vector3());const scale=2.75/Math.max(size.y,.01);model.scale.setScalar(scale);model.position.set(-center.x*scale,-box.min.y*scale,-center.z*scale);model.traverse(o=>{if((o as THREE.Mesh).isMesh){const m=o as THREE.Mesh;m.castShadow=true;m.receiveShadow=true}});const actor=new THREE.Group();actor.name='actor-root';actor.add(model);scene.add(actor);clips=[...gltf.animations];mixer=new THREE.AnimationMixer(actor);helper=new THREE.SkeletonHelper(model);helper.visible=stateRef.current.showSkeleton;const skeletonMaterial=helper.material as THREE.LineBasicMaterial;skeletonMaterial.depthTest=false;skeletonMaterial.transparent=true;skeletonMaterial.opacity=.8;scene.add(helper);report();},undefined,e=>report(`Model load failed: ${String(e)}`));
  const resize=()=>{camera.aspect=Math.max(el.clientWidth,1)/Math.max(el.clientHeight,1);camera.updateProjectionMatrix();renderer.setSize(el.clientWidth,el.clientHeight)};const ro=new ResizeObserver(resize);ro.observe(el);const timer=new THREE.Timer();timer.connect(document);let frame=0,lastReport=0;
  const draw=(now:number)=>{frame=requestAnimationFrame(draw);timer.update(now);const s=stateRef.current;controls.update();if(helper)helper.visible=s.showSkeleton;
  // 텍스트 연출 spec이 바뀌면 directed clip을 교체한다 (해제 시 제거)
  const m=s.motion??null;if(mixer&&m!==lastMotion){lastMotion=m;if(directed){mixer.uncacheClip(directed);clips=clips.filter(c=>c!==directed);if(active===directed.name){action?.stop();action=undefined;active='';duration=0}}directed=m?buildMotionClip(m):undefined;if(directed)clips=[...clips,directed];report()}
  if(mixer&&clips.length){let wanted=clips.find(c=>c.name.toLowerCase()===s.clip.toLowerCase())||clips.find(c=>c.name.toLowerCase().includes(s.clip.toLowerCase()))||clips[0];if(wanted.name!==active){const next=mixer.clipAction(wanted);next.reset().setEffectiveTimeScale(s.speed).setEffectiveWeight(1).play();if(action&&action!==next)action.crossFadeTo(next,.28,true);action=next;active=wanted.name;duration=wanted.duration;report()}action!.paused=!s.playing;action!.setEffectiveTimeScale(s.speed);if(Math.abs((action!.time||0)-s.time)>.12&&!s.playing)action!.time=Math.min(s.time,duration);mixer.update(s.playing?timer.getDelta():0);if(now-lastReport>100){lastReport=now;report()}}renderer.render(scene,camera)};draw(performance.now());
  return()=>{disposed=true;cancelAnimationFrame(frame);timer.disconnect();ro.disconnect();controls.dispose();mixer?.stopAllAction();scene.traverse(o=>{const m=o as THREE.Mesh;if(m.geometry)m.geometry.dispose();const mats=Array.isArray(m.material)?m.material:[m.material];mats.filter(Boolean).forEach(x=>(x as THREE.Material).dispose())});renderer.dispose();if(renderer.domElement.parentElement===el)el.removeChild(renderer.domElement)};
 },[modelUrl]);
 return <div className="three-host" ref={host}/>;
}
