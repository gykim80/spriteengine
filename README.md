# Rabbit

한 장의 캐릭터 이미지를 animation-ready 3D asset으로 변환하기 위한 local-first Wails studio입니다.

## 구현 범위

- Wails v2 + Go orchestration, React/TypeScript UI
- Three.js interactive skinned GLB preview와 실제 embedded motion playback
- Motion Lab clip browser, play/pause, timeline scrubbing, speed, crossfade, skeleton overlay
- local GLB/GLTF import 및 embedded animation 자동 탐색
- Itch.io, Quaternius, Mixamo, ActorCore license-aware source library
- native reference image import, workspace 복사, SHA-256 provenance
- `queued → ready → running → done|failed` pipeline state machine
- embedded Python JSON Lines worker 실행 및 Wails progress events
- immutable artifact/log manifest와 local persistence
- image cleanup → reconstruction → retopology → auto-rig → motion → export stage UI
- GitHub, 논문, license 및 품질 gate 조사

현재 local worker는 image provenance 검증부터 animated GLB export까지 6단계 artifact chain을 실행합니다. `Run full pipeline`으로 전 단계를 연속 실행하고, 생성된 GLB는 backend의 등록 artifact 검증을 거쳐 Motion Lab에 즉시 로드됩니다. 이 local path는 end-to-end workflow와 rig/motion/export를 완전히 검증하는 procedural fallback이며, photoreal reconstruction을 위해서는 TripoSR/InstantMesh 등의 별도 model weights adapter가 필요합니다. Motion Lab은 bundled skinned GLB와 pipeline output의 Idle/Walk/Run clip을 Three.js `AnimationMixer`로 실제 재생하며, 사용자가 가진 animated GLB/GLTF도 local import할 수 있습니다.

## 문서

- [Research와 product plan](docs/RESEARCH.md)
- [2025–2026 motion/rigging update](docs/RESEARCH_2026.md)
- [Runtime architecture와 worker protocol](docs/ARCHITECTURE.md)
- [Bundled Motion Lab asset provenance](docs/licenses/MOTION_DEMO_ASSET.md)

## 검증

```bash
cd frontend && npm install && npm run build
cd .. && go test ./... && go vet ./...
wails build
```

생성 앱은 `build/bin/Rabbit.app`입니다.

## 권장 production model stack

`rembg/SAM2 → TripoSR 또는 InstantMesh → Blender headless → Make-It-Animatable/HumanRig → Blender retarget → GLB`를 기본 경로로 사용하고, TRELLIS·Hunyuan3D는 고품질 provider로 격리합니다. Code license뿐 아니라 weights, datasets, output terms를 배포 전에 각각 검토해야 합니다.
