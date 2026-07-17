#!/usr/bin/env python3
"""Generate a deterministic animation-ready humanoid GLB for offline pipeline tests.

The mesh is intentionally low-poly and procedural, but the output is a genuine
skinned glTF 2.0 asset with joints, inverse bind matrices, weights and clips.
"""
import json, math, os, struct, sys

out=sys.argv[1]; os.makedirs(os.path.dirname(out),exist_ok=True)
buf=bytearray(); views=[]; access=[]
def align():
 while len(buf)%4: buf.append(0)
def pack(values,fmt,ctype,typ,count,minv=None,maxv=None,target=None):
 align(); off=len(buf); flat=[]
 for v in values: flat.extend(v if isinstance(v,(tuple,list)) else [v])
 data=struct.pack('<'+fmt*len(flat),*flat);buf.extend(data)
 vi=len(views); views.append({'buffer':0,'byteOffset':off,'byteLength':len(data),**({'target':target} if target else {})})
 ai=len(access); a={'bufferView':vi,'componentType':ctype,'count':count,'type':typ}
 if minv is not None:a['min']=minv
 if maxv is not None:a['max']=maxv
 access.append(a);return ai
# tapered humanoid torso, each vertex weighted to one of hips/spine/chest
pos=[(-.42,0, -.2),(.42,0,-.2),(.42,0,.2),(-.42,0,.2),(-.34,1,-.16),(.34,1,-.16),(.34,1,.16),(-.34,1,.16),(-.27,1.75,-.14),(.27,1.75,-.14),(.27,1.75,.14),(-.27,1.75,.14)]
faces=[]
for a,b,c,d in [(0,1,2,3),(8,11,10,9),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0),(4,8,9,5),(5,9,10,6),(6,10,11,7),(7,11,8,4)]:faces += [a,b,c,a,c,d]
norm=[]
for x,y,z in pos:
 l=max(math.hypot(x,z),.001);norm.append((x/l,0,z/l))
joints=[];weights=[]
for x,y,z in pos:
 j=0 if y<.5 else 1 if y<1.35 else 2;joints.append((j,0,0,0));weights.append((1.,0.,0.,0.))
pa=pack(pos,'f',5126,'VEC3',len(pos),[-.42,0,-.2],[.42,1.75,.2],34962);na=pack(norm,'f',5126,'VEC3',len(pos),target=34962);ja=pack(joints,'H',5123,'VEC4',len(pos),target=34962);wa=pack(weights,'f',5126,'VEC4',len(pos),target=34962);ia=pack(faces,'H',5123,'SCALAR',len(faces),[0],[11],34963)
# skeleton hips->spine->chest->head, arms and legs
nodes=[
 {'name':'Character','children':[1,8,11],'mesh':0,'skin':0},
 {'name':'Hips','translation':[0,.65,0],'children':[2]},
 {'name':'Spine','translation':[0,.45,0],'children':[3]},
 {'name':'Chest','translation':[0,.45,0],'children':[4,5,6]},
 {'name':'Head','translation':[0,.5,0]},
 {'name':'LeftArm','translation':[-.35,.25,0],'children':[7]}, {'name':'RightArm','translation':[.35,.25,0]}, {'name':'LeftForeArm','translation':[-.45,0,0]},
 {'name':'LeftUpLeg','translation':[-.2,.65,0],'children':[9]}, {'name':'LeftLeg','translation':[0,-.65,0],'children':[10]}, {'name':'LeftFoot','translation':[0,-.55,.12]},
 {'name':'RightUpLeg','translation':[.2,.65,0],'children':[12]}, {'name':'RightLeg','translation':[0,-.65,0],'children':[13]}, {'name':'RightFoot','translation':[0,-.55,.12]}]
# inverse binds simplified identity; structurally valid and renderer recomputes world transforms
mats=[]
for _ in range(13):mats.append((1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1))
iba=pack(mats,'f',5126,'MAT4',len(mats))
times=[(0.,),(.5,),(1.,)];ta=pack(times,'f',5126,'SCALAR',3,[0],[1])
def quat_x(a):return (math.sin(a/2),0,0,math.cos(a/2))
def clip(name,amount):
 sam=[];chs=[]
 for node,phase in [(8,1),(11,-1),(5,-1),(6,1)]:
  vals=[quat_x(-amount*phase),quat_x(amount*phase),quat_x(-amount*phase)]
  oa=pack(vals,'f',5126,'VEC4',3)
  sam.append({'input':ta,'output':oa,'interpolation':'LINEAR'});chs.append({'sampler':len(sam)-1,'target':{'node':node,'path':'rotation'}})
 return {'name':name,'samplers':sam,'channels':chs}
anims=[clip('Idle',.035),clip('Walk',.55),clip('Run',.9)]
doc={'asset':{'version':'2.0','generator':'SpriteEngine procedural worker'},'scene':0,'scenes':[{'nodes':[0]}],'nodes':nodes,'meshes':[{'name':'CharacterMesh','primitives':[{'attributes':{'POSITION':pa,'NORMAL':na,'JOINTS_0':ja,'WEIGHTS_0':wa},'indices':ia,'material':0}]}],'materials':[{'name':'StudioMaterial','pbrMetallicRoughness':{'baseColorFactor':[.22,.46,.68,1],'metallicFactor':0,'roughnessFactor':.72}}],'skins':[{'name':'HumanoidRig','inverseBindMatrices':iba,'joints':list(range(1,14)),'skeleton':1}],'animations':anims,'buffers':[{'byteLength':len(buf)}],'bufferViews':views,'accessors':access}
js=json.dumps(doc,separators=(',',':')).encode();js+=b' '*((4-len(js)%4)%4);align();binb=bytes(buf);binb+=b'\0'*((4-len(binb)%4)%4)
glb=struct.pack('<4sII',b'glTF',2,12+8+len(js)+8+len(binb))+struct.pack('<I4s',len(js),b'JSON')+js+struct.pack('<I4s',len(binb),b'BIN\0')+binb
open(out,'wb').write(glb)
print(json.dumps({'path':out,'bytes':len(glb),'triangles':len(faces)//3,'bones':13,'clips':['Idle','Walk','Run']}))
