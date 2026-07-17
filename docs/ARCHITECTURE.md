# SpriteEngine Architecture

## Runtime boundaries

```text
React + Three.js WebView
        │ Wails bindings / events
Go orchestrator
        │ JSON Lines over stdin/stdout
isolated Python model workers
        │
immutable project artifacts
```

Go owns project state, stage transitions, process lifecycle and artifact provenance. Python owns image/model processing. A worker cannot mutate the project manifest directly.

## State machine

A stage transitions through `queued → ready → running → done|failed`. Only one `ready` stage is dispatched. Success unlocks the following stage; failure preserves prior immutable artifacts. Every dispatch writes logs and each artifact records stage, kind, path and metrics.

## Current executable adapter

`workers/baseline_worker.py` and `workers/procedural_character.py` are embedded in the Wails binary and materialized under the app config runtime directory. The prepare stage validates PNG/JPEG/WebP input, dimensions and SHA-256. The offline reconstruction fallback then emits a genuine glTF 2.0 binary containing a skinned humanoid mesh, 13-joint skeleton, inverse bind matrices, skin weights, and `Idle`/`Walk`/`Run` clips. Remaining offline stages copy and validate immutable GLB artifacts, so a complete pipeline can be executed and tested without CUDA. These preview artifacts are explicitly marked `previewOnly`; production model adapters replace them rather than misrepresenting procedural geometry as AI reconstruction.

## Model adapter contract

Input:

```json
{"type":"run","jobId":"id","stage":"reconstruct","adapter":"triposr","workspace":"/project","input":"/project/prepare/reference.png","options":{}}
```

Output events:

```json
{"type":"progress","progress":0.42,"message":"Decoding mesh"}
{"type":"artifact","kind":"mesh","path":"/project/reconstruct/model.glb","metrics":{"triangles":42000}}
{"type":"done","stage":"reconstruct","progress":1}
```

Errors are structured `error` events followed by a non-zero process exit. Progress events are forwarded to React using Wails events.

## Target adapters

- `prepare/rembg`: alpha matting and subject validation
- `reconstruct/triposr`: fast local GLB baseline
- `reconstruct/instantmesh`: multi-view quality option
- `retopo/blender`: headless repair, decimation, UV and normals
- `rig/make-it-animatable`: skeleton and skin weights
- `motion/blender`: humanoid retarget and foot lock
- `export/gltf-transform`: validation and packaging

Each model environment remains separately installable because Torch/CUDA dependencies conflict and model licenses differ.

## Project layout

```text
projects/<id>/
  source.png
  prepare/reference.png
  reconstruct/model.glb
  retopo/character.glb
  rig/character-rigged.glb
  motion/*.glb
  export/*.glb
```

`jobs.json` is the current manifest store. SQLite migration is planned when concurrent downloads and richer clip metadata require transactions.
