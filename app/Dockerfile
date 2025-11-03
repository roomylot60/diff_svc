# app/Dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 기본 도구 및 오디오 툴
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git ffmpeg sox libsndfile1 \
    build-essential wget ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 워크스페이스
WORKDIR /workspace

# 파이썬 의존성: CUDA 12.1용 PyTorch + 오디오/ML 스택
# (필요 시 버전 고정: torch==2.3.x torchaudio==2.3.x)
RUN python3 -m pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchaudio \
 && pip install \
    numpy scipy soundfile librosa resampy \
    tqdm pyyaml tensorboard scikit-learn \
    fairseq==0.12.2 torchcrepe \
    numba llvmlite \
    matplotlib

# 로컬 요구사항(있다면)
COPY requirements.txt ./
RUN if [ -s requirements.txt ]; then pip install -r requirements.txt; fi

# 프로젝트 파일
COPY scripts ./scripts
COPY configs ./configs

# 캐시 디렉토리 (권한)
RUN mkdir -p /workspace/.cache/huggingface && chmod -R 777 /workspace/.cache

# 기본 진입: 쉘
CMD ["bash"]