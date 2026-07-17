# SpriteEngine Studio

한 장의 캐릭터 이미지를 animation-ready 3D asset으로 변환하기 위한 local-first Wails studio입니다.

## 구현 범위

- Wails v2 + Go orchestration, React/TypeScript UI
- Three.js interactive character/skeleton preview와 motion preview
- native reference image import, workspace 복사, SHA-256 provenance
- `queued → ready → running → done|failed` pipeline state machine
- embedded Python JSON Lines worker 실행 및 Wails progress events
- immutable artifact/log manifest와 local persistence
- image cleanup → reconstruction → retopology → auto-rig → motion → export stage UI
- GitHub, 논문, license 및 품질 gate 조사

현재 baseline worker는 실제 이미지 형식과 provenance를 검증하고 prepared artifact를 생성합니다. GPU model이 필요한 stage는 adapter request artifact를 생성하여 pipeline integration을 검증하며, 가짜 3D 결과를 만들지 않습니다.

## 문서

- [Research와 product plan](docs/RESEARCH.md)
- [Runtime architecture와 worker protocol](docs/ARCHITECTURE.md)

## 검증

```bash
cd frontend && npm install && npm run build
cd .. && go test ./... && go vet ./...
wails build
```

생성 앱은 `build/bin/SpriteEngine Studio.app`입니다.

## 권장 production model stack

`rembg/SAM2 → TripoSR 또는 InstantMesh → Blender headless → Make-It-Animatable/HumanRig → Blender retarget → GLB`를 기본 경로로 사용하고, TRELLIS·Hunyuan3D는 고품질 provider로 격리합니다. Code license뿐 아니라 weights, datasets, output terms를 배포 전에 각각 검토해야 합니다.
