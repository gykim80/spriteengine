# GitHub evidence catalog

이 파일은 저장소 이름을 수동 나열한 문서가 아니라, authenticated `gh` + GitHub REST API로 검증한 snapshot의 설명이다.

- Machine-readable catalog: [`research/github-catalog.json`](../research/github-catalog.json)
- 수집 항목: canonical URL, description, stars, forks, open issues, latest push, archive 여부, SPDX license
- 범위: single-image 3D, multiview diffusion, human reconstruction, auto-rig/skinning, motion generation/retargeting, DCC/export/runtime
- 현재 snapshot: 42개 후보 중 API에서 36개 repository를 검증했다.

## 해석 원칙

1. GitHub stars는 품질 또는 production 적합성 보증이 아니다.
2. repository license와 model weight/data license는 별도로 확인해야 한다.
3. 논문 공개와 runnable code 공개는 동일하지 않다.
4. `error`가 기록된 후보는 rename/private/removal 가능성이 있어 architecture dependency로 채택하지 않는다.
5. 제품 adapter 채택 전 commit pin, checksum, GPU matrix, commercial-use review를 수행한다.

## 제품 shortlist

| Stage | Primary | Fallback / research |
|---|---|---|
| Prepare | rembg/SAM 계열 | BiRefNet 계열 |
| Reconstruction | TripoSR | InstantMesh, Stable Fast 3D, Hunyuan3D-2 |
| Character prior | CharacterGen | ECON, ICON, PIFuHD |
| Rig | UniRig | Make-It-Animatable, HumanRig, RigNet |
| Motion | HY-Motion | MDM, MotionDiffuse, MoMask, MotionGPT |
| Retarget | Blender | Skeleton-aware Motion Retargeting, STMC |
| Export | glTF-Transform | glTF-Blender-IO |

Catalog은 날짜가 포함된 evidence snapshot이며, 최신성 확인 시 API로 재생성해야 한다.
