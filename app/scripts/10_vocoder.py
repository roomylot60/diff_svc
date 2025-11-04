#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
멜 -> 파형 복원
- 기본: Griffin-Lim (외부 가중치 없이 동작)
- 선택: HiFi-GAN generator (pretrained ckpt 제공 시)
"""
import argparse, os, glob
import numpy as np
import soundfile as sf
import torch
import librosa

# ---- Optional: HiFi-GAN minimal loader (generator-only) ----
class DummyHiFiGAN(torch.nn.Module):
    """실제 HiFi-GAN 구조 대신, ckpt 로드 실패 시 대비용 더미 (에러를 내지 않되 사용자는 Griffin-Lim로 대체)"""
    def __init__(self): super().__init__()
    def forward(self, mel): raise NotImplementedError("HiFi-GAN ckpt가 필요합니다.")

def load_hifigan(ckpt_path: str, device: str):
    try:
        state = torch.load(ckpt_path, map_location=device)
        # 실제 사용 시: Generator 클래스를 정의하고 state_dict 로드
        # 여기서는 구조 의존성 없이 ckpt 가용성만 확인
        model = DummyHiFiGAN()
        if isinstance(state, dict) and "generator" in state:
            # 예: state["generator"]가 weight일 경우. 프로젝트별로 맞춰 수정하세요.
            model.load_state_dict(state["generator"], strict=False)
        else:
            # 구조 미일치 시에도 객체만 반환하여 안내
            pass
        model.to(device).eval()
        return model
    except Exception as e:
        print(f"[HiFi-GAN] 로드 실패: {e}")
        return None

def griffin_lim_from_mel(logmel, sr=44100, n_fft=2048, hop=512, fmin=40, fmax=16000, n_mels=80, iters=60):
    """로그멜을 선형 STFT로 역변환 후 Griffin-Lim 복원 (단순 baseline)"""
    mel = np.exp(logmel.T)  # [80,T]
    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    # mel = mel_fb @ |X|  ->  |X| ~= pinv(mel_fb) @ mel
    pinv = np.linalg.pinv(mel_fb)
    mag = np.maximum(pinv @ mel, 1e-8)  # [fft_bins, T]
    y = librosa.griffinlim(mag, hop_length=hop, n_fft=n_fft, win_length=n_fft, n_iter=iters)
    return y.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mel_dir", required=True)       # features/mel_pred 또는 features/mel
    ap.add_argument("--out_dir", required=True)       # out_wav
    ap.add_argument("--hifigan_ckpt", default="")     # 선택: HiFi-GAN ckpt 경로
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    use_hifigan = bool(args.hifigan_ckpt)

    if use_hifigan:
        g = load_hifigan(args.hifigan_ckpt, args.device)
        if g is None:
            print("[WARN] HiFi-GAN 사용 불가 -> Griffin-Lim으로 대체")

    mel_paths = sorted(glob.glob(os.path.join(args.mel_dir, "*.npy")))
    for i, mp in enumerate(mel_paths, 1):
        key = os.path.splitext(os.path.basename(mp))[0]
        logmel = np.load(mp)  # [T, 80]

        if use_hifigan and g is not None:
            # 실제 HiFi-GAN 사용 시 mel scaling, transposition, normalization 등 프로젝트 규약에 맞춰 조정 필요
            # 아래는 placeholder 경로: 바로는 동작하지 않으니 Griffin-Lim 먼저 확인 후, HiFi-GAN 구조 연결하세요.
            try:
                with torch.no_grad():
                    mel_t = torch.from_numpy(logmel.T).unsqueeze(0).to(args.device)  # [1, 80, T]
                    y = g(mel_t).squeeze().cpu().numpy().astype(np.float32)
            except Exception as e:
                print(f"[HiFi-GAN 경로 실패 -> Griffin-Lim 대체] {key}: {e}")
                y = griffin_lim_from_mel(logmel)
        else:
            y = griffin_lim_from_mel(logmel)

        sf.write(os.path.join(args.out_dir, f"{key}.wav"), y, 44100, subtype="PCM_16")
        print(f"[{i}/{len(mel_paths)}] wav -> {key}.wav (len={len(y)/44100:.2f}s)")

if __name__ == "__main__":
    main()
