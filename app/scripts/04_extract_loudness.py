#!/usr/bin/env python3
"""
프레임 RMS → dB 변환 → 표준화는 통계 파일로 별도 처리(⑤에서 μ/σ 산출).
여기서는 raw dB 시계열을 저장합니다.
"""
import argparse
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

def frame_rms_db(y, sr, win, hop):
    # librosa.feature.rms는 윈도 개수 기반 RMS -> dB 변환
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=True)[0]  # [T]
    rms = np.maximum(rms, 1e-12)
    db = 20 * np.log10(rms)
    return db.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--win", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=512)
    args = ap.parse_args()

    wav_dir, out_dir = Path(args.wav_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = list(wav_dir.rglob("*.wav"))
    for p in tqdm(wavs, desc="Loudness (RMS dB)"):
        y, sr = librosa.load(p, sr=None, mono=True)
        db = frame_rms_db(y, sr, args.win, args.hop)  # [T]
        np.save(out_dir / (p.stem + ".npy"), db.reshape(-1, 1))

if __name__ == "__main__":
    main()
