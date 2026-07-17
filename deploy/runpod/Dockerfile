FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git git-lfs ninja-build python3 python3-dev python3-pip \
    libegl1 libegl1-mesa-dev libeigen3-dev libgl1 libglib2.0-0 libgles2-mesa-dev \
    libglvnd-dev libxrender1 libcgal-dev && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip && \
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
WORKDIR /opt
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git hunyuan && \
    pip install -r /opt/hunyuan/requirements.txt && pip install runpod
RUN cd /opt/hunyuan/hy3dpaint/custom_rasterizer && pip install -e . && \
    cd /opt/hunyuan/hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh

FROM build AS runtime
ENV PYOPENGL_PLATFORM=egl HF_HOME=/runpod-volume/huggingface \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface/hub \
    HUNYUAN_MODEL=tencent/Hunyuan3D-2.1
WORKDIR /opt/hunyuan
COPY deploy/runpod/handler.py /opt/hunyuan/spriteengine_handler.py
CMD ["python3", "-u", "/opt/hunyuan/spriteengine_handler.py"]
