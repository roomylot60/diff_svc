#!/usr/bin/env bash
# Conda 환경 설정
set -e
conda create -y -n svcover python=3.10
conda activate svcover

# PyTorch CUDA(환경에 맞는 버전 선택)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 오디오/신호
pip install numpy scipy soundfile librosa resampy

# 모델/학습 유틸
pip install tqdm pyyaml tensorboard scikit-learn

# ContentVec/HuBERT 추론용(필요 패키지)
pip install fairseq==0.12.2 torchcrepe

# RMVPE(Python 패키지 배포가 안 될 수 있으므로, 로컬 모듈로 둘 수도 있음)
# 사용 환경에 맞게 rmvpe 모듈 설치/클론 후 PYTHONPATH 추가
pip install numba

# 멜/보코더/시각화
pip install llvmlite