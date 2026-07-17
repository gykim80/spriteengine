import {useEffect,useRef} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls.js';

export default function CharacterViewport({animated=false}:{animated?:boolean}){
 const host=useRef<HTMLDivElement>(null);
 useEffect(()=>{
  const el=host.current!; const scene=new THREE.Scene(); scene.background=new THREE.Color(0x181b20); scene.fog=new THREE.Fog(0x181b20,5,12);
  const camera=new THREE.PerspectiveCamera(38,el.clientWidth/el.clientHeight,.1,100); camera.position.set(3,2.1,5.2);
  const renderer=new THREE.WebGLRenderer({antialias:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.setSize(el.clientWidth,el.clientHeight); renderer.shadowMap.enabled=true; el.appendChild(renderer.domElement);
  scene.add(new THREE.HemisphereLight(0xdce7ff,0x30231e,2.2)); const key=new THREE.DirectionalLight(0xffd4bd,3); key.position.set(3,5,4); key.castShadow=true; scene.add(key);
  const floor=new THREE.Mesh(new THREE.PlaneGeometry(20,20),new THREE.MeshStandardMaterial({color:0x191c20,roughness:1})); floor.rotation.x=-Math.PI/2; floor.position.y=-1.82; floor.receiveShadow=true; scene.add(floor);
  const grid=new THREE.GridHelper(20,40,0x4c525a,0x292e34); grid.position.y=-1.81; scene.add(grid);
  const root=new THREE.Group(); scene.add(root); const mat=new THREE.MeshStandardMaterial({color:0x72777c,roughness:.63,metalness:.08});
  const part=(g:THREE.BufferGeometry,p:[number,number,number],r:[number,number,number]=[0,0,0])=>{const m=new THREE.Mesh(g,mat);m.position.set(...p);m.rotation.set(...r);m.castShadow=true;root.add(m);return m};
  part(new THREE.SphereGeometry(.34,32,24),[0,1.18,0]); part(new THREE.CapsuleGeometry(.48,.85,8,20),[0,.35,0]);
  part(new THREE.CapsuleGeometry(.14,.9,8,16),[-.65,.35,0],[0,0,-.12]); part(new THREE.CapsuleGeometry(.14,.9,8,16),[.65,.35,0],[0,0,.12]);
  const legL=part(new THREE.CapsuleGeometry(.17,1.05,8,16),[-.25,-.9,0]); const legR=part(new THREE.CapsuleGeometry(.17,1.05,8,16),[.25,-.9,0]);
  const bones=new THREE.Group(); const boneMat=new THREE.LineBasicMaterial({color:0xff6841}); const points=[new THREE.Vector3(0,-1.45,0),new THREE.Vector3(0,-.45,0),new THREE.Vector3(0,.35,0),new THREE.Vector3(0,1.18,0)]; bones.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),boneMat)); [[0,-.45,0],[-.65,.35,0],[.65,.35,0],[-.25,-1.45,0],[.25,-1.45,0]].forEach(p=>{const j=new THREE.Mesh(new THREE.SphereGeometry(.035,12,8),new THREE.MeshBasicMaterial({color:0xffb199}));j.position.set(...p as [number,number,number]);bones.add(j)}); root.add(bones);
  const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.target.set(0,0,0);controls.minDistance=3;controls.maxDistance=9;
  const resize=()=>{camera.aspect=el.clientWidth/el.clientHeight;camera.updateProjectionMatrix();renderer.setSize(el.clientWidth,el.clientHeight)}; const ro=new ResizeObserver(resize);ro.observe(el);
  let frame=0,start=performance.now(); const draw=(t:number)=>{frame=requestAnimationFrame(draw); controls.update(); if(animated){root.position.y=Math.sin((t-start)/280)*.025;legL.rotation.x=Math.sin((t-start)/350)*.18;legR.rotation.x=-legL.rotation.x} renderer.render(scene,camera)};draw(start);
  return()=>{cancelAnimationFrame(frame);ro.disconnect();controls.dispose();renderer.dispose();el.removeChild(renderer.domElement)};
 },[animated]);
 return <div className="three-host" ref={host}/>;
}
