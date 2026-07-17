import {useEffect,useRef} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls.js';
import {GLTFLoader} from 'three/examples/jsm/loaders/GLTFLoader.js';

export type PlaybackState={duration:number,time:number,clips:string[],active:string,loaded:boolean,error?:string};
type Props={playing:boolean;clip:string;speed:number;time:number;showSkeleton:boolean;modelUrl?:string;onState?:(state:PlaybackState)=>void};

export default function CharacterViewport({playing,clip,speed,time,showSkeleton,modelUrl='/models/Soldier.glb',onState}:Props){
 const host=useRef<HTMLDivElement>(null); const stateRef=useRef({playing,clip,speed,time,showSkeleton,onState});
 useEffect(()=>{stateRef.current={playing,clip,speed,time,showSkeleton,onState}},[playing,clip,speed,time,showSkeleton,onState]);
 useEffect(()=>{
  const el=host.current!; const scene=new THREE.Scene(); scene.background=new THREE.Color(0x181b20); scene.fog=new THREE.Fog(0x181b20,7,16);
  const camera=new THREE.PerspectiveCamera(38,Math.max(el.clientWidth,1)/Math.max(el.clientHeight,1),.05,100); camera.position.set(3.2,1.9,5.4);
  const renderer=new THREE.WebGLRenderer({antialias:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.setSize(el.clientWidth,el.clientHeight); renderer.shadowMap.enabled=true; renderer.shadowMap.type=THREE.PCFSoftShadowMap; renderer.outputColorSpace=THREE.SRGBColorSpace; el.appendChild(renderer.domElement);
  scene.add(new THREE.HemisphereLight(0xe6efff,0x33231d,2.1)); const key=new THREE.DirectionalLight(0xffddc9,3.4); key.position.set(3,6,4); key.castShadow=true; key.shadow.mapSize.set(2048,2048); scene.add(key); const rim=new THREE.DirectionalLight(0x769cff,1.8);rim.position.set(-4,3,-4);scene.add(rim);
  const floor=new THREE.Mesh(new THREE.PlaneGeometry(24,24),new THREE.MeshStandardMaterial({color:0x171a1e,roughness:1})); floor.rotation.x=-Math.PI/2; floor.receiveShadow=true; scene.add(floor); const grid=new THREE.GridHelper(24,48,0x505760,0x292e34);grid.position.y=.004;scene.add(grid);
  const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.target.set(0,1,0);controls.minDistance=2;controls.maxDistance=10;
  let mixer:THREE.AnimationMixer|undefined, action:THREE.AnimationAction|undefined, model:THREE.Object3D|undefined, helper:THREE.SkeletonHelper|undefined, clips:THREE.AnimationClip[]=[]; let active=''; let duration=0; let disposed=false;
  const report=(error?:string)=>stateRef.current.onState?.({duration,time:action?.time||0,clips:clips.map(c=>c.name),active,loaded:!!model,error});
  new GLTFLoader().load(modelUrl,gltf=>{if(disposed)return;model=gltf.scene; const box=new THREE.Box3().setFromObject(model);const size=box.getSize(new THREE.Vector3());const center=box.getCenter(new THREE.Vector3());const scale=2.75/Math.max(size.y,.01);model.scale.setScalar(scale);model.position.set(-center.x*scale,-box.min.y*scale,-center.z*scale);model.traverse(o=>{if((o as THREE.Mesh).isMesh){const m=o as THREE.Mesh;m.castShadow=true;m.receiveShadow=true}});scene.add(model);clips=gltf.animations;mixer=new THREE.AnimationMixer(model);helper=new THREE.SkeletonHelper(model);helper.visible=stateRef.current.showSkeleton;const skeletonMaterial=helper.material as THREE.LineBasicMaterial;skeletonMaterial.depthTest=false;skeletonMaterial.transparent=true;skeletonMaterial.opacity=.8;scene.add(helper);report();},undefined,e=>report(`Model load failed: ${String(e)}`));
  const resize=()=>{camera.aspect=Math.max(el.clientWidth,1)/Math.max(el.clientHeight,1);camera.updateProjectionMatrix();renderer.setSize(el.clientWidth,el.clientHeight)};const ro=new ResizeObserver(resize);ro.observe(el);const clock=new THREE.Clock();let frame=0,lastReport=0;
  const draw=(now:number)=>{frame=requestAnimationFrame(draw);const s=stateRef.current;controls.update();if(helper)helper.visible=s.showSkeleton;if(mixer&&clips.length){let wanted=clips.find(c=>c.name.toLowerCase()===s.clip.toLowerCase())||clips.find(c=>c.name.toLowerCase().includes(s.clip.toLowerCase()))||clips[0];if(wanted.name!==active){const next=mixer.clipAction(wanted);next.reset().setEffectiveTimeScale(s.speed).setEffectiveWeight(1).play();if(action&&action!==next)action.crossFadeTo(next,.28,true);action=next;active=wanted.name;duration=wanted.duration;report()}action!.paused=!s.playing;action!.setEffectiveTimeScale(s.speed);if(Math.abs((action!.time||0)-s.time)>.12&&!s.playing)action!.time=Math.min(s.time,duration);mixer.update(s.playing?clock.getDelta():0);if(now-lastReport>100){lastReport=now;report()}}else clock.getDelta();renderer.render(scene,camera)};draw(performance.now());
  return()=>{disposed=true;cancelAnimationFrame(frame);ro.disconnect();controls.dispose();mixer?.stopAllAction();scene.traverse(o=>{const m=o as THREE.Mesh;if(m.geometry)m.geometry.dispose();const mats=Array.isArray(m.material)?m.material:[m.material];mats.filter(Boolean).forEach(x=>(x as THREE.Material).dispose())});renderer.dispose();if(renderer.domElement.parentElement===el)el.removeChild(renderer.domElement)};
 },[modelUrl]);
 return <div className="three-host" ref={host}/>;
}
