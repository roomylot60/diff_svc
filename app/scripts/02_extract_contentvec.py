#!/usr/bin/env python3
"""
두 가지 백엔드를 지원:
1. hubert_torchaudio(기본): torchaudio.pipelines.HUBERT_BASE
2. contentvec_fairseq: fairseq 체크포인트 기반 ContentVec

ContentVec(→ fairseq) 또는 HuBERT(torchaudio) 중 선택해 추출할 수 있게 했습니다.
기본값은 HuBERT(torchaudio) 로 안전 실행, --backend contentvec_fairseq 주면 ContentVec 사용.
"""
import argparse, os, json
from pathlib import Path
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from tqdm import tqdm

# ---- HuBERT via torchaudio (16 kHz) ----
def load_hubert_torchaudio(device):
    """
    input:
        device: torchaudio에 내장된 HuBERT Base 파이프라인을 로드
    output:
        model: 이미 eval() 모드로 전환된 HuBERT 추론 모델
        target_sr: 16000 (HuBERT는 16kHz 입력 기대)
    description:
        사전학습 모델을 곧바로 사용 가능; 별도 ckpt 경로 불필요
    """
    bundle = torchaudio.pipelines.HUBERT_BASE
    model = bundle.get_model().to(device).eval()
    target_sr = bundle.sample_rate  # 16000
    return model, target_sr

@torch.inference_mode()
def extract_hubert_features(model, wav_16k: torch.Tensor):
    """
    input:
        wav_16k: [1, T], float32
    output:
        model returns (features, _)
    description:
        T: sample_rate * duration = 오디오 길이에 따른 샘플 수
        T': convolution을 거쳐 축소된 프레임 수
    """
    features, _ = model(wav_16k)  # [1, T', 768]
    return features.squeeze(0).cpu().numpy()  # [T', 768]

# ---- ContentVec via fairseq (16 kHz) ----
def load_contentvec_fairseq(ckpt_path: str, device: str):
    """
    fairseq.checkpoint_utils.load_model_ensemble_and_task()로 ContentVec ckpt 로드
    output:
        model: ContentVec 인코더 (주로 Convolution + Transformer 구조)
        16000: ContentVec은 항상 16kHz 샘플링을 요구하므로, 리샘플링 기준을 명시적으로 반환
    """
    from fairseq import checkpoint_utils
    # fairseq 형태로 저장된 가중치를 로드 -> 모델 구조, 파라미터를 복원
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([ckpt_path], suffix="")
    model = models[0] # models: 1개 이상의 모델 -> models[0]을 선택
    model.to(device)
    model.eval()
    return model, 16000

@torch.inference_mode()
def extract_contentvec_features(model, wav_16k: torch.Tensor):
    """
    ContentVec의 일반적 추론 루틴(레포마다 조금씩 다르지만, 여기서는)
    1. model.feature_extractor(wav) → 저수준 특징
    2. model.feature_aggregator(z) → 시간축 임베딩
    output:
        B: 배치 크기 (1)
        C: feature 채널 수 (예: 512)
        T': 다운샘플된 시간 프레임 길이 (≈ T / 320) -> 약 10ms 단위에서 시간 프레임 수
        D: ContentVec의 hidden dimension (보통 768) -> 발음·언어적 특징 표현하는 임베딩 차원
    """
    # Adapted from typical contentvec usage: forward_features
    # Some forks use model.extract_features; we use generic interface.
    z = model.feature_extractor(wav_16k)     # [B, T] → [B, C, T']
    c = model.feature_aggregator(z)          # [B, C, T'] → [B, T', D]
    return c.squeeze(0).cpu().numpy()
    # .squeeze(0): 배치 차원 제거 (1개 샘플만 있으므로: [1, T', D] → [T', D])
    # .cup().numpy(): GPU 텐서를 CPU 메모리로 옮기고 numpy 배열로 변환해 저장.
    # .npy 파일은 시간축별 벡터 시퀀스로 저장됩니다:

def main():
    """
    --wav_dir: 입력 WAV 디렉토리(보통 data/sr44100/)
    --out_dir: 추출된 콘텐츠 임베딩을 저장할 디렉토리(예: features/content/)
    --hop: 후단 정렬 힌트(코드 내 직접 사용은 안 하지만, 메타 목적/호환성 위한 옵션)
    --backend: hubert_torchaudio(기본) | contentvec_fairseq
    --contentvec_ckpt: ContentVec ckpt 경로(백엔드가 ContentVec일 때 필수)
    --device: cuda 또는 cpu 자동 선택
    백엔드 초기화
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--hop", type=int, default=512, help="후단 정렬용 참조(여기서는 저장만)")
    ap.add_argument("--backend", choices=["hubert_torchaudio","contentvec_fairseq"],
                    default="hubert_torchaudio")
    ap.add_argument("--contentvec_ckpt", default="", help="ContentVec fairseq 체크포인트 경로")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    wav_dir, out_dir = Path(args.wav_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "hubert_torchaudio":
        model, req_sr = load_hubert_torchaudio(args.device)
        extractor = lambda y: extract_hubert_features(model, y)
    else:
        assert args.contentvec_ckpt, "--contentvec_ckpt 를 지정하세요."
        model, req_sr = load_contentvec_fairseq(args.contentvec_ckpt, args.device)
        extractor = lambda y: extract_contentvec_features(model, y)

    wavs = list(wav_dir.rglob("*.wav"))
    for p in tqdm(wavs, desc="Content features"):
        y, sr = librosa.load(p, sr=None, mono=True)
        if sr != req_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=req_sr, res_type="kaiser_best")
        y_t = torch.from_numpy(y).float().unsqueeze(0).to(args.device)  # [1, T]
        feats = extractor(y_t)  # [T', D]
        npy_path = out_dir / (p.stem + ".npy")
        np.save(npy_path, feats)

if __name__ == "__main__":
    main()
