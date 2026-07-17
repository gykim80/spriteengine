# 2025–2026 Character/Motion Research Update

조회일: 2026-07-17. GitHub search의 `updated_at`은 품질이나 publication date가 아니며, integration 전 code/weights/data/output license를 각각 확인한다.

## Recent motion generation and interaction

- [Tencent-Hunyuan/HY-Motion-1.0](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) — text-to-3D human motion foundation model; production provider 후보.
- [wzyabcas/InterAct](https://github.com/wzyabcas/InterAct) — human interaction motion generation 연구 track.
- [nv-tlabs/stmc](https://github.com/nv-tlabs/stmc) — multi-track timeline motion composition.
- [czh-98/STAR](https://github.com/czh-98/STAR) — skeleton-aware motion retargeting, Apache-2.0 표시.
- [mmlab-cv/skeleton-aware-motion-retargeting](https://github.com/mmlab-cv/skeleton-aware-motion-retargeting) — skeleton-aware retargeting 연구 구현.
- [eherr/motion_preprocessing_tool](https://github.com/eherr/motion_preprocessing_tool) — motion preprocessing, MIT 표시.
- [TalehMalikov/mdm-to-fbx](https://github.com/TalehMalikov/mdm-to-fbx) — MDM output에서 FBX workflow 참고.

## Recent rigging/avatar discovery

- [UniRig](https://github.com/VAST-AI-Research/UniRig) — unified automatic rigging의 핵심 integration 후보.
- [HumanRig](https://github.com/c8241998/HumanRig) — humanoid rigging dataset/method.
- [Make-It-Animatable](https://github.com/jasongzy/Make-It-Animatable) — arbitrary character skeleton/skinning.
- [haoz19/Automatic-Rigging](https://github.com/haoz19/Automatic-Rigging) — automatic rigging implementation, GitHub에서 MIT 표시.
- [AniGS](https://github.com/aigc3d/AniGS) — animatable Gaussian avatar, CVPR 2025.
- [DualdiffAvatar](https://github.com/guwan-xie/DualdiffAvatar) — recent avatar reconstruction discovery track.
- [Ultraman](https://github.com/yisuanwang/Ultraman) — single-image human avatar discovery track.

## Legal asset sources surfaced in Motion Lab

- [Itch.io 3D Characters](https://itch.io/game-assets/tag-3d/tag-characters) — creator별 license가 다르므로 asset page를 source of truth로 사용.
- [Itch.io Animation assets](https://itch.io/game-assets/tag-3d/tag-animation)
- [Quaternius](https://quaternius.com/) — CC0 표시 pack 중심이나 개별 pack 확인.
- [Mixamo](https://www.mixamo.com/) — Adobe account/service terms 적용; open-source asset으로 재배포하지 않는다.
- [ActorCore Free Motion](https://www.reallusion.com/actorcore/free-motion.html) — Reallusion terms 적용.
- Itch.io discovery examples: [Basic Motions Free](https://kevdev.itch.io/basic-motions-free), [CC0 Animations](https://maxparata.itch.io/cc0-animations), [PSX Rigged Character Bases](https://ink-ribbon.itch.io/psx-rigged-character-bases), [KayKit Adventurers](https://kaylousberg.itch.io/kaykit-adventurers).

## Platform decision

저작권이 불명확한 파일을 자동 scraping/bundling하지 않는다. Studio는 source library를 열고 사용자가 합법적으로 받은 GLB/GLTF를 local import한다. Embedded animation을 자동 탐색하고 Three.js `AnimationMixer`로 playback, scrubbing, speed control, crossfade, skeleton inspection을 제공한다. FBX/BVH는 다음 Blender worker adapter에서 GLB로 normalize한다.
