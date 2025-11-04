#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
44.1kHz/mono WAV -> 로그멜(80bin) 타깃 생성
- 시간축: sr=44100, hop=512 기준 프레임 수로 맞춤
- 추후 디코더 학습의 ground truth로 사용
"""
import argparse, os, glob
import numpy as np
import soundfile as sf
import librosa

def compute_num_frames(n_samples: int, hop: int) -> int:
    return int(np.floor((n_samples - 1) / hop) + 1)

def wav_to_logmel(y, sr=44100, n_mels=80, n_fft=2048, hop_length=512, fmin=40, fmax=16000):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=1.0
    )  # [80, T]
    logmel = np.log(np.clip(S, 1e-8, None)).T.astype(np.float32)  # [T, 80]
    return logmel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True)       # e.g., data/sr44100
    ap.add_argument("--out_dir", required=True)       # e.g., features/mel
    ap.add_argument("--hop", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    wavs = sorted(glob.glob(os.path.join(args.wav_dir, "*.wav")))
    for i, wp in enumerate(wavs, 1):
        base = os.path.splitext(os.path.basename(wp))[0]
        y, sr = sf.read(wp)
        if y.ndim > 1: y = y.mean(axis=1)
        if sr != 44100:
            y = librosa.resample(y.astype(np.float32), sr, 44100, res_type="kaiser_best")
            sr = 44100
        y = y.astype(np.float32)
        logmel = wav_to_logmel(y, sr=sr, hop_length=args.hop)
        np.save(os.path.join(args.out_dir, f"{base}.npy"), logmel)
        print(f"[{i}/{len(wavs)}] mel -> {base}.npy  shape={logmel.shape}")

if __name__ == "__main__":
    main()
