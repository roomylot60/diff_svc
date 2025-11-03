# 목표

* **모델 타입**: Diff-SVC (Singing Voice Conversion)
* **피처 셋**: ContentVec(콘텐츠) + RMVPE(F0) + Loudness(RMS)
* **오디오 스펙**: 44.1 kHz / hop 512 / win 1024 / n_mels 80~100
* **보코더**: HiFi-GAN(범용 사전학습)
* **GPU**: GTX 1660 Super (6GB) 기준, 오프라인 고품질 변환에 맞춘 설정

---

## 1) 프로젝트 구조

```
svcover/
  env/
  data/
    raw/                 # 원본 보컬(또는 UVR로 추출한 보컬)
    sr44100/             # 리샘플된 44.1k wav (mono)
  features/
    content/             # ContentVec 특징 npy [T, D_c]
    f0/                  # F0 npy [T,1]
    loud/                # loudness npy [T,1]
    stats/               # 표준화 통계(pkl)
  models/
    contentvec/          # ContentVec 가중치/필요 파일
    rmvpe/               # RMVPE 가중치
    hifigan/             # HiFi-GAN (universal) ckpt
    diffsvc/             # 학습된 Diff-SVC 디코더 ckpt
  configs/
    train_44100.yaml
  scripts/
    00_setup_env.sh
    01_resample_to_44100.py
    02_extract_contentvec.py
    03_extract_rmvpe_f0.py
    04_extract_loudness.py
    05_build_stats.py
    06_preprocess_for_training.py
    07_train_diffsvc.py
    08_infer_diffsvc.py
    09_vocode_hifigan.py
    10_mix_with_instrumental.py
  outputs/
    mels/
    wavs/
```

---

## 2) 환경 구성

### (1) Conda/venv 생성

```bash
bash scripts/00_setup_env.sh
```

`00_setup_env.sh` 예시:

```bash
#!/usr/bin/env bash
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
```

> *RMVPE는 레포를 클론해 `models/rmvpe/`에 가중치를 두고, `from rmvpe import RMVPE` 형태로 로컬 임포트하는 패턴이 일반적입니다. 필요 시 `PYTHONPATH`에 경로 추가.*

---

## 3) 오디오 표준화(44.1k/mono)

```bash
python scripts/01_resample_to_44100.py \
  --in_dir data/raw --out_dir data/sr44100 --sr 44100
```

`01_resample_to_44100.py` 요지:

```python
import argparse, soundfile as sf, librosa, os
from pathlib import Path

def main(in_dir, out_dir, sr):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in in_dir.rglob('*.wav'):
        y, sr_in = librosa.load(p, sr=None, mono=True)
        if sr_in != sr:
            y = librosa.resample(y, orig_sr=sr_in, target_sr=sr)
        sf.write(out_dir/p.name, y, sr)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir'); ap.add_argument('--out_dir'); ap.add_argument('--sr', type=int, default=44100)
    args = ap.parse_args(); main(args.in_dir, args.out_dir, args.sr)
```

---

## 4) 특징 추출

### (1) ContentVec (콘텐츠)

```bash
python scripts/02_extract_contentvec.py \
  --wav_dir data/sr44100 \
  --out_dir features/content \
  --hop 512
```

`02_extract_contentvec.py` 요지:

```python
# pseudo: ContentVec/HuBERT forward 후 [T, D_c] 저장
# 필요 시 fairseq + 사전학습 가중치 로드
```

### (2) RMVPE (F0)

```bash
python scripts/03_extract_rmvpe_f0.py \
  --wav_dir data/sr44100 \
  --out_dir features/f0 \
  --hop 512
```

`03_extract_rmvpe_f0.py` 요지:

```python
# pseudo: RMVPE로 f0 추출 → 무성구간 보간 → [T,1] 저장
```

### (3) Loudness (RMS dB 표준화)

```bash
python scripts/04_extract_loudness.py \
  --wav_dir data/sr44100 \
  --out_dir features/loud \
  --hop 512 --win 1024
```

`04_extract_loudness.py` 요지:

```python
# pseudo: frame별 RMS→dB→정규화 → [T,1] 저장
```

---

## 5) 표준화 통계/인덱스 빌드

```bash
python scripts/05_build_stats.py \
  --content_dir features/content --f0_dir features/f0 --loud_dir features/loud \
  --out_dir features/stats
```

* 각 피처별 μ/σ 저장(`.pkl`) → 학습/추론 동일 적용

---

## 6) Diff-SVC 학습 준비

```bash
python scripts/06_preprocess_for_training.py \
  --content_dir features/content --f0_dir features/f0 --loud_dir features/loud \
  --stats_dir features/stats --out_manifest configs/train_manifest.jsonl
```

* JSONL: `{"utt_id": ..., "content": path, "f0": path, "loud": path, "wav": wav_path}`

### 학습 설정(`configs/train_44100.yaml` 예시)

```yaml
sr: 44100
hop: 512
win: 1024
n_mels: 80
features:
  content_dim: 768   # ContentVec base 기준 예시
  f0: true
  loudness: true
  speaker_embed: false  # 단일 화자 세팅이면 false
f0:
  method: rmvpe
train:
  batch_size: 1
  accum_steps: 2
  steps: 200000
  lr: 2e-4
  amp: true
  ckpt_dir: models/diffsvc
data:
  manifest: configs/train_manifest.jsonl
loss:
  mel: 1.0
  stft: 1.0
  # 필요 시 추가 손실 항목

inference:
  steps: 120   # 역확산 스텝 (품질↔속도 균형)
```

---

## 7) 학습 실행(디코더; Diff-SVC)

```bash
python scripts/07_train_diffsvc.py \
  --config configs/train_44100.yaml
```

* 내부에서 `manifest`를 읽고, Content/F0/Loudness를 조건으로 **멜 스펙 복원** 학습
* VRAM 6GB 환경: `batch=1`, AMP(fp16) 권장

---

## 8) 추론(멜 생성)

```bash
python scripts/08_infer_diffsvc.py \
  --config configs/train_44100.yaml \
  --ckpt models/diffsvc/ckpt_best.pt \
  --content features/content/<file>.npy \
  --f0 features/f0/<file>.npy \
  --loud features/loud/<file>.npy \
  --out_mel outputs/mels/<file>.npy
```

* 입력 보컬을 다른 보이스로 변환하려면, **타깃 화자 설정**(speaker embed)을 활성화한 구성(멀티스피커)으로 학습해야 합니다. 단일 화자 변환이면 해당 임베딩을 고정하거나, 단일 타깃 디코더로 학습합니다.

---

## 9) 보코더 합성(HiFi-GAN)

```bash
python scripts/09_vocode_hifigan.py \
  --mel outputs/mels/<file>.npy \
  --ckpt models/hifigan/g_universal.pth \
  --sr 44100 \
  --out_wav outputs/wavs/<file>_conv.wav
```

* HiFi-GAN universal 사전학습 가중치 사용 (44.1k 호환 체크)
* 추론 시 AMP/ONNX로 최적화 가능

---

## 10) 반주와 믹스다운(선택)

```bash
python scripts/10_mix_with_instrumental.py \
  --vocal outputs/wavs/<file>_conv.wav \
  --inst data/instrumental/<file>_inst.wav \
  --out outputs/wavs/<file>_mix.wav \
  --gain_vocal -1.0 --gain_inst 0.0
```

---

## 11) 1660 Super에서의 팁

* **학습**: `batch_size=1`, `accum_steps=2~4`, `fp16` → VRAM 6GB 내
* **역확산 스텝**: 100~150 권장(품질↑). 느리면 80~100으로 타협
* **체크포인트**: 1~2k step 간격 저장, 중간 샘플링으로 품질 확인
* **캐시**: 모든 피처를 `.npy`로 캐시 → 학습/추론 반복 속도↑

---

## 12) 품질 점검 체크리스트

* ContentVec가 **언어·발음**을 잘 보존하는지 (자음/모음 뭉개짐 확인)
* RMVPE F0 곡선의 **연속성/무성구간 처리** (보간, 미디안 필터)
* Loudness 정규화로 **호흡/강세**가 과도하게 눌리지 않는지
* 멜→파형(HiFi-GAN)에서 **치찰음/고주파** 손실 여부
* 전체 체인 **SR/hop 일치** (44100/512)

---

## 13) 다음 단계(선택)

* 보코더 **부분 파인튜닝**(타깃 화자 데이터 10~30분) → 음색 일관성 향상
* 역확산 **가속 샘플러**(DDIM 등) 실험 → 속도 향상
* ContentVec vs PPG(Whisper-Ko) **A/B 테스트**

---

### 요약

> **ContentVec + RMVPE + 44.1 kHz/hop512 + HiFi-GAN** 조합으로, 1660 Super에서도 오프라인 고품질 SVC 구축이 가능합니다. 위 스크립트 골격과 설정으로 빠르게 엔드투엔드 파이프라인을 구성하신 뒤, 역확산 스텝/보코더/정규화 통계를 튜닝하시기 바랍니다.

---

## 14) 도커 컨테이너/컴포즈 구성 (GTX 1660 Super, CUDA)

**목표:** 한 개 이미지로 전처리·특징추출(ContentVec/RMVPE)·학습(Diff-SVC 디코더)·추론·보코더(HiFi-GAN)까지 실행.

### A) 디렉터리 전제(로컬)

```
svcover/
  Dockerfile
  docker-compose.yml
  requirements.txt
  scripts/               # 앞서 작성한 01~10 스크립트들
  data/ features/ models/ configs/ outputs/
```

### B) Dockerfile (CUDA 12.1 + PyTorch)

```dockerfile
# svcover/Dockerfile
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
```

### C) docker-compose.yml (GPU 할당 + 볼륨 마운트)

```yaml
# svcover/docker-compose.yml
version: "3.9"

services:
  svcover:
    build: .
    container_name: svcover
    ipc: host
    stdin_open: true
    tty: true
    working_dir: /workspace
    volumes:
      - ./:/workspace
      - ./models:/workspace/models
      - ./data:/workspace/data
      - ./features:/workspace/features
      - ./outputs:/workspace/outputs
    environment:
      - TZ=Asia/Seoul
      - HF_HOME=/workspace/.cache/huggingface
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    # 필요 시 컨테이너 시작과 동시에 특정 명령 실행 가능
    # command: ["bash", "-lc", "python scripts/07_train_diffsvc.py --config configs/train_44100.yaml"]
```

> **참고**: Docker Desktop(Windows) + WSL2 환경에서도 `--gpus all` 또는 위 `deploy.resources...devices` 방식으로 GPU 전달이 필요합니다. (Compose V2에서는 `device_requests`가 자동 매핑)

### D) 실행 순서 (컨테이너 내부 명령)

```bash
# 0) 빌드/실행
docker compose build
docker compose up -d

# 1) 컨테이너 접속
docker exec -it svcover bash

# 2) 44.1k 리샘플
python scripts/01_resample_to_44100.py --in_dir data/raw --out_dir data/sr44100 --sr 44100

# 3) ContentVec 특징
python scripts/02_extract_contentvec.py --wav_dir data/sr44100 --out_dir features/content --hop 512

# 4) RMVPE F0
python scripts/03_extract_rmvpe_f0.py --wav_dir data/sr44100 --out_dir features/f0 --hop 512

# 5) Loudness
python scripts/04_extract_loudness.py --wav_dir data/sr44100 --out_dir features/loud --hop 512 --win 1024

# 6) 통계
python scripts/05_build_stats.py --content_dir features/content --f0_dir features/f0 --loud_dir features/loud --out_dir features/stats

# 7) 학습 준비(매니페스트)
python scripts/06_preprocess_for_training.py \
  --content_dir features/content --f0_dir features/f0 --loud_dir features/loud \
  --stats_dir features/stats --out_manifest configs/train_manifest.jsonl

# 8) 학습 실행 (Diff-SVC 디코더)
python scripts/07_train_diffsvc.py --config configs/train_44100.yaml

# 9) 추론(멜 생성)
python scripts/08_infer_diffsvc.py \
  --config configs/train_44100.yaml \
  --ckpt models/diffsvc/ckpt_best.pt \
  --content features/content/<file>.npy \
  --f0 features/f0/<file>.npy \
  --loud features/loud/<file>.npy \
  --out_mel outputs/mels/<file>.npy

# 10) 보코더 합성
python scripts/09_vocode_hifigan.py \
  --mel outputs/mels/<file>.npy \
  --ckpt models/hifigan/g_universal.pth \
  --sr 44100 \
  --out_wav outputs/wavs/<file>_conv.wav

# 11) (선택) 반주와 믹스다운
python scripts/10_mix_with_instrumental.py \
  --vocal outputs/wavs/<file>_conv.wav \
  --inst data/instrumental/<file>_inst.wav \
  --out outputs/wavs/<file>_mix.wav
```

### E) GPU/성능 팁 (1660 Super)

* `python -m torch.utils.collect_env`로 CUDA 인식 확인
* 학습 시 `batch_size=1`, `amp(fp16)=on`, `accum_steps=2~4`
* 역확산 스텝: 100~150(품질), 느리면 80~100
* 보코더 ONNX 변환/FP16 추론으로 합성 속도 향상

### F) 모델 가중치 배치

* `models/contentvec/` : ContentVec 가중치(예: `checkpoint_best.pt`)
* `models/rmvpe/`      : RMVPE 가중치
* `models/hifigan/`    : `g_universal.pth` 등 범용 보코더
* `models/diffsvc/`    : 학습 결과(checkpoints)

### G) Makefile (선택, 로컬에서 단축 명령)

```makefile
build:
	docker compose build

up:
	docker compose up -d

sh:
	docker exec -it svcover bash

extract:
	docker exec -it svcover bash -lc "python scripts/02_extract_contentvec.py --wav_dir data/sr44100 --out_dir features/content --hop 512 && \
  python scripts/03_extract_rmvpe_f0.py --wav_dir data/sr44100 --out_dir features/f0 --hop 512 && \
  python scripts/04_extract_loudness.py --wav_dir data/sr44100 --out_dir features/loud --hop 512 --win 1024 && \
  python scripts/05_build_stats.py --content_dir features/content --f0_dir features/f0 --loud_dir features/loud --out_dir features/stats"

train:
	docker exec -it svcover bash -lc "python scripts/07_train_diffsvc.py --config configs/train_44100.yaml"

infer MEL?=outputs/mels/sample.npy CKPT?=models/diffsvc/ckpt_best.pt:
	docker exec -it svcover bash -lc "python scripts/08_infer_diffsvc.py --config configs/train_44100.yaml --ckpt $(CKPT) --content features/content/sample.npy --f0 features/f0/sample.npy --loud features/loud/sample.npy --out_mel $(MEL)"

vocode WAV?=outputs/wavs/sample_conv.wav MEL?=outputs/mels/sample.npy:
	docker exec -it svcover bash -lc "python scripts/09_vocode_hifigan.py --mel $(MEL) --ckpt models/hifigan/g_universal.pth --sr 44100 --out_wav $(WAV)"
```

> 이 구성으로 **하나의 컨테이너**에서 전처리→특징추출→학습→추론→보코더까지 모두 실행할 수 있습니다. 프로파일 분리가 필요하면 `profiles: [train|infer|vocoder]` 로 서비스를 쪼개면 됩니다.
