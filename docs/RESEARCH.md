# SpriteEngine — Research & Product Plan

> 한 장의 캐릭터 이미지를 animation-ready 3D asset으로 변환하는 local-first Wails studio.

## 1. 결론

단일 foundation model에 전 과정을 맡기지 않는다. **입력 정규화 → multi-view/mesh 생성 → geometry 정리 → rig/skin → motion retarget → export/QA**를 교체 가능한 worker pipeline으로 구성한다. Wails는 project orchestration과 UI를 담당하고, GPU inference는 격리된 Python worker 또는 remote provider가 담당한다.

### 권장 baseline

| 단계 | MVP 기본값 | 대안 | 이유 |
|---|---|---|---|
| Segmentation | rembg / SAM 2 | BiRefNet | 입력 배경 제거와 alpha matte |
| Single image → mesh | TripoSR | InstantMesh, Hunyuan3D-2, TRELLIS | 빠른 local baseline과 고품질 provider 분리 |
| Character 특화 | CharacterGen | PIFuHD, ICON, ECON, PSHuman | humanoid prior 및 의복 형상 |
| Mesh repair | Blender headless + trimesh | Open3D, PyMeshLab | remesh, normals, decimation, UV 자동화 |
| Auto-rig | Make-It-Animatable | UniRig/HumanRig, RigNet | skeleton+skinning을 animation-ready 형태로 생성 |
| 표준 body | SMPL-X | SMPL, STAR | humanoid topology/landmark 기준 |
| Motion | MDM | MotionGPT, MoMask | text-to-motion 및 HumanML3D 생태계 |
| Retarget | Blender constraints | R2ET, Rokoko tooling | 표준 humanoid bone mapping |
| Runtime preview | Three.js | Babylon.js | React UI 내 GLB/skinning/animation 지원 |
| Export | GLB 우선 | FBX, USD/USDZ, VRM | 웹/게임엔진 호환성과 material 보존 |

## 2. 핵심 GitHub 조사

### Image-to-3D / reconstruction

- [Zero123++](https://github.com/SUDO-AI-3D/zero123plus) — consistent multi-view diffusion, Apache-2.0. 직접 mesh를 완성하는 도구가 아니라 view prior이므로 reconstruction backend와 결합한다.
- [TripoSR](https://github.com/VAST-AI-Research/TripoSR) — fast single-image reconstruction, MIT. MVP local adapter 1순위.
- [InstantMesh](https://github.com/TencentARC/InstantMesh) — sparse-view LRM 기반 textured mesh, Apache-2.0.
- [OpenLRM](https://github.com/3DTopia/OpenLRM) — open LRM implementation, Apache-2.0.
- [Wonder3D](https://github.com/xxlong0/Wonder3D) — cross-domain normal/RGB multi-view diffusion, MIT.
- [One-2-3-45](https://github.com/One-2-3-45/One-2-3-45) — optimization-free mesh pipeline, Apache-2.0.
- [CRM](https://github.com/thu-ml/CRM) — convolutional reconstruction, ECCV 2024, MIT.
- [LGM](https://github.com/3DTopia/LGM) — multi-view Gaussian model, MIT.
- [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian) — efficient generative Gaussian splatting, MIT.
- [Shap-E](https://github.com/openai/shap-e) — image/text conditioned 3D baseline, MIT.
- [TRELLIS](https://github.com/microsoft/TRELLIS) — structured 3D latents, MIT; 품질 provider 후보.
- [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) — high-resolution shape/texture; custom license를 배포 전에 별도 검토.

### Human / character 특화

- [CharacterGen](https://github.com/zjp-shadow/CharacterGen) — single-image character generation with pose canonicalization, Apache-2.0.
- [PIFuHD](https://github.com/facebookresearch/pifuhd) — clothed human digitization의 고전적 baseline.
- [ICON](https://github.com/YuliangXiu/ICON) — normals 기반 clothed human reconstruction.
- [ECON](https://github.com/YuliangXiu/ECON) — normal integration 기반 explicit clothed human.
- [PSHuman](https://github.com/pengHTYX/PSHuman) — single-image 3D human 계열, MIT.
- [AniGS](https://github.com/aigc3d/AniGS) — single image animatable Gaussian avatar, CVPR 2025. mesh export와 별도 experimental track.

### Rigging / skinning

- [Make-It-Animatable](https://github.com/jasongzy/Make-It-Animatable) — animation-ready character authoring, MIT; 현재 최우선 통합 후보.
- [RigNet](https://github.com/zhan-xu/RigNet) — skeleton/skin prediction, GPL-3.0. 프로세스 격리 및 라이선스 영향 검토 필요.
- [HumanRig](https://github.com/c8241998/HumanRig) — humanoid large-scale rigging, MIT.
- [Neural Blend Shapes](https://github.com/PeizhuoLi/neural-blend-shapes) — rigging, skinning, blend shapes, SIGGRAPH 2021.
- [SMPL-X](https://github.com/vchoutas/smplx) — body/hands/face parametric reference. model/data license는 code license와 별도 확인.

### Motion / animation

- [MDM](https://github.com/GuyTevet/motion-diffusion-model) — text-conditioned human motion diffusion.
- [MotionGPT](https://github.com/OpenMotionLab/MotionGPT) — unified motion-language model.
- [MoMask](https://github.com/EricGuo5513/momask-codes) — masked modeling 기반 text-to-motion.
- [text-to-motion](https://github.com/EricGuo5513/text-to-motion) — CVPR 2022 motion generation baseline.
- [ACTOR](https://github.com/Mathux/ACTOR) — action-conditioned motion generation.
- [R2ET](https://github.com/Kebii/R2ET) — motion retargeting 연구 구현.
- [Rokoko Blender integration](https://github.com/Rokoko/rokoko-studio-live-blender) — live motion workflow 참고.

### Geometry / editor infrastructure

- [Blender](https://github.com/blender/blender), [trimesh](https://github.com/mikedh/trimesh), [Open3D](https://github.com/isl-org/Open3D), [PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab)
- [Three.js](https://github.com/mrdoob/three.js), [glTF Transform](https://github.com/donmccurdy/glTF-Transform)
- [Wails](https://github.com/wailsapp/wails) — Go desktop shell.

> GitHub stars와 activity는 탐색 우선순위일 뿐 기술 품질 또는 상업 사용 가능성을 보장하지 않는다. model weights, datasets, generated-output terms를 각각 검토해야 한다.

## 3. 주요 논문

| 논문 | 연도 | 링크 | 제품상 의미 |
|---|---:|---|---|
| Zero-1-to-3 | 2023 | [arXiv:2303.11328](https://arxiv.org/abs/2303.11328) | viewpoint-conditioned diffusion의 출발점 |
| Zero123++ | 2023 | [arXiv:2310.15110](https://arxiv.org/abs/2310.15110) | multi-view consistency 강화 |
| One-2-3-45 | 2023 | [arXiv](https://arxiv.org/search/?query=One-2-3-45&searchtype=title) | per-shape optimization 없는 빠른 mesh |
| Wonder3D | 2023 | [arXiv](https://arxiv.org/search/?query=Wonder3D&searchtype=title) | RGB/normal joint diffusion |
| TripoSR | 2024 | [arXiv](https://arxiv.org/search/?query=TripoSR&searchtype=title) | 빠른 feed-forward reconstruction |
| InstantMesh | 2024 | [arXiv](https://arxiv.org/search/?query=InstantMesh&searchtype=title) | sparse-view LRM mesh generation |
| CharacterGen | 2024 | [arXiv](https://arxiv.org/search/?query=CharacterGen&searchtype=title) | pose canonicalization을 포함한 character 특화 |
| RigNet | 2020 | [arXiv:2005.00559](https://arxiv.org/abs/2005.00559) | neural skeleton과 skinning |
| Neural Blend Shapes | 2021 | [project](https://github.com/PeizhuoLi/neural-blend-shapes) | deformation 품질과 blend shape |
| Human Motion Diffusion Model | 2022 | [arXiv:2209.14916](https://arxiv.org/abs/2209.14916) | generative motion baseline |
| MotionGPT | 2023 | [arXiv:2306.14795](https://arxiv.org/abs/2306.14795) | language-motion 통합 interface |
| Make-It-Animatable | 2024 | [arXiv:2411.18197](https://arxiv.org/abs/2411.18197) | arbitrary character animation-ready authoring |
| HumanRig | 2024 | [arXiv:2412.02317](https://arxiv.org/abs/2412.02317) | large-scale humanoid auto-rig dataset/method |
| ASMR | 2025 | [arXiv:2503.13579](https://arxiv.org/abs/2503.13579) | 2D generative prior 기반 skeleton/skin |
| Make-A-Character 2 | 2025 | [arXiv:2501.07870](https://arxiv.org/abs/2501.07870) | single image에서 animatable character 생성 |
| StdGEN | 2024 | [arXiv:2411.05738](https://arxiv.org/abs/2411.05738) | semantic-decomposed character generation |

### 추가 discovery index

- [Awesome 3D Gaussian Splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting) — Gaussian avatar/animation 계열 추적용 index.
- [Awesome 3D Gen](https://github.com/justimyhxu/awesome-3D-generation) — image/text-to-3D 논문과 구현 index.
- [Awesome Text-to-Motion](https://github.com/Zilize/awesome-text-to-motion) — motion generation benchmark와 구현 index.
- [Morig](https://arxiv.org/abs/2210.09463) — point-cloud motion-aware rigging.
- [Automated Body Structure Extraction](https://arxiv.org/abs/1705.05508) — arbitrary mesh skeleton extraction baseline.

> “모조리”는 시점에 따라 바뀌는 공개 생태계 특성상 유한 목록으로 보장할 수 없다. 위 목록은 제품 통합 가능성, 공개 code/weights, character relevance를 기준으로 선별한 registry이며 discovery index와 GitHub/arXiv 검색으로 지속 갱신한다.

## 4. Product architecture

```text
Wails desktop
├── React/Three.js: project browser, stage graph, rig editor, timeline, GLB viewer
├── Go orchestration
│   ├── project manifest + SQLite
│   ├── job state machine / cancellation / logs
│   ├── worker process manager
│   └── import/export and provider credentials
└── Python workers (isolated environments)
    ├── preprocess: segmentation, crop, pose check
    ├── reconstruct: TripoSR / InstantMesh / remote provider
    ├── geometry: Blender headless repair, retopo, UV
    ├── rig: Make-It-Animatable / HumanRig
    └── motion: generation, retarget, foot-lock, export
```

### Project manifest

각 stage는 immutable artifact와 provenance를 기록한다: `input hash`, `adapter/version`, `model/weight hash`, `parameters`, `seed`, `started/finished`, `stdout/stderr`, `output files`, `QA metrics`. 재실행 시 downstream만 invalidate한다.

### Worker protocol

초기에는 stdin/stdout JSON Lines가 단순하고 안정적이다.

```json
{"type":"run","jobId":"...","stage":"reconstruct","adapter":"triposr","input":".../source.png","options":{"resolution":512}}
{"type":"progress","jobId":"...","progress":0.42,"message":"Decoding mesh"}
{"type":"artifact","kind":"mesh","path":".../reconstruct/model.glb"}
{"type":"done","metrics":{"vertices":42108,"watertight":false}}
```

## 5. 품질 gate

- 입력: 한 명/전신 여부, 해상도, 가림, alpha, 정면 pose confidence.
- Mesh: disconnected components, non-manifold edge, inverted normal, triangle count, texture coverage.
- Rig: bone hierarchy, joint-inside-mesh 비율, left/right symmetry, weight sum, zero-weight vertices.
- Animation: foot skating, ground penetration, joint-limit violation, self-intersection, root drift.
- Export: GLB validator, texture embedding, animation duration, coordinate system, scale.

## 6. 실행 roadmap

### Phase 0 — 현재 prototype
- Wails v2 + React shell, stage-oriented UI, local persisted job model.
- 실제 연구 결과와 adapter architecture 문서화.

### Phase 1 — vertical slice
- image import/preview/hash, rembg worker.
- TripoSR adapter와 GLB artifact import.
- Three.js OrbitControls, material/normal inspection.
- cancellable process execution과 structured logs.

### Phase 2 — animation-ready
- Blender headless cleanup recipe.
- Make-It-Animatable adapter, skeleton overlay, bone remapping UI.
- bundled idle/walk/run motion, retarget, foot-lock QA.
- GLB export.

### Phase 3 — production studio
- InstantMesh/CharacterGen/TRELLIS/Hunyuan provider selection.
- rig weight paint correction, timeline, clips, transitions.
- FBX/USDZ/VRM export, presets for Unity/Unreal/Godot.
- GPU/VRAM estimator, model manager, resumable downloads.

## 7. 위험과 대응

1. **Single-view ambiguity**: 뒤쪽 texture/geometry는 추정이다. multi-view edit와 regenerate region UX 제공.
2. **생성 mesh topology**: rig 전에 quad-friendly retopo와 component merge가 필수.
3. **의복/머리 deformation**: body skeleton만으로 부족하므로 accessory bones, cloth/hair segmentation track 필요.
4. **GPU packaging**: model을 app binary에 넣지 않고 model manager와 worker env로 분리.
5. **라이선스**: code, weights, datasets, output rights를 registry에서 별도 필드로 관리. GPL component는 결합/배포 방식 법률 검토.
6. **Wails webview 한계**: inference는 UI thread 밖 process에서 수행하고 progress event만 전달.

## 8. 성공 지표

- supported GPU에서 input→rigged GLB median 5분 이하.
- standard test set의 90% 이상이 manual Blender repair 없이 열림.
- 3개 bundled motion에서 catastrophic skin collapse 5% 미만.
- stage 실패 후 재개 가능, deterministic provenance 100% 기록.
