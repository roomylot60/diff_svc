#!/usr/bin/env python3
"""
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
    bundle = torchaudio.pipelines.HUBERT_BASE
    model = bundle.get_model().to(device).eval()
    target_sr = bundle.sample_rate  # 16000
    return model, target_sr

@torch.inference_mode()
def extract_hubert_features(model, wav_16k: torch.Tensor):
    # wav_16k: [1, T], float32
    # model returns (features, _)
    features, _ = model(wav_16k)  # [1, T', 768]
    return features.squeeze(0).cpu().numpy()  # [T', 768]

# ---- ContentVec via fairseq (16 kHz) ----
def load_contentvec_fairseq(ckpt_path: str, device: str):
    from fairseq import checkpoint_utils
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([ckpt_path], suffix="")
    model = models[0]
    model.to(device)
    model.eval()
    return model, 16000

@torch.inference_mode()
def extract_contentvec_features(model, wav_16k: torch.Tensor):
    # Adapted from typical contentvec usage: forward_features
    # Some forks use model.extract_features; we use generic interface.
    z = model.feature_extractor(wav_16k)     # [B, C, T']
    c = model.feature_aggregator(z)          # [B, T', D]
    return c.squeeze(0).cpu().numpy()

def main():
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
