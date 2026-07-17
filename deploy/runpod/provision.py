#!/usr/bin/env python3
"""Idempotently create SpriteEngine's RunPod volume, template, and endpoint."""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

API = "https://rest.runpod.io/v1"


def request(key, method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(API + path, data=data, method=method)
    req.add_header("Authorization", "Bearer " + key)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"RunPod {method} {path}: HTTP {exc.code}: {detail}") from exc


def items(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("items", "data", "templates", "endpoints", "networkVolumes"):
            if isinstance(value.get(key), list):
                return value[key]
    return []


def find_named(key, path, name):
    return next((x for x in items(request(key, "GET", path)) if x.get("name") == name), None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Public Docker image, including immutable tag")
    p.add_argument("--region", default="US-CA-2")
    p.add_argument("--volume-gb", type=int, default=100)
    p.add_argument("--gpu", action="append", dest="gpus")
    p.add_argument("--output", default="runpod-deployment.json")
    args = p.parse_args()
    key = os.getenv("RUNPOD_API_KEY", "").strip()
    if not key:
        sys.exit("RUNPOD_API_KEY is required")
    gpus = args.gpus or ["NVIDIA A100 80GB PCIe", "NVIDIA H100 80GB HBM3", "NVIDIA L40S"]

    volume = find_named(key, "/networkvolumes", "spriteengine-model-cache")
    if not volume:
        volume = request(key, "POST", "/networkvolumes", {
            "name": "spriteengine-model-cache", "size": args.volume_gb, "dataCenterId": args.region,
        })

    template = find_named(key, "/templates", "spriteengine-hunyuan3d21")
    if not template:
        template = request(key, "POST", "/templates", {
            "name": "spriteengine-hunyuan3d21", "imageName": args.image,
            "isServerless": True, "containerDiskInGb": 100, "volumeInGb": 0,
            "volumeMountPath": "/runpod-volume", "env": {}, "ports": [],
        })

    endpoint = find_named(key, "/endpoints", "spriteengine-hunyuan3d21")
    if not endpoint:
        endpoint = request(key, "POST", "/endpoints", {
            "name": "spriteengine-hunyuan3d21", "templateId": template["id"],
            "networkVolumeId": volume["id"], "dataCenterIds": [args.region],
            "computeType": "GPU", "gpuTypeIds": gpus, "gpuCount": 1,
            "workersMin": 0, "workersMax": 1, "idleTimeout": 5,
            "scalerType": "QUEUE_DELAY", "scalerValue": 4,
            "executionTimeoutMs": 900000, "flashboot": True,
            "allowedCudaVersions": ["12.4", "12.5", "12.6", "12.7", "12.8"],
        })

    result = {
        "endpointId": endpoint["id"], "endpointName": endpoint.get("name"),
        "templateId": template["id"], "volumeId": volume["id"],
        "region": args.region, "image": args.image,
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
