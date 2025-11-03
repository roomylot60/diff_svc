#!/usr/bin/env python3
"""
기본은 RMVPE. 옵션으로 --backend crepe 주면 torchcrepe 사용.
무성구간 보간과 미디안 필터로 매끈한 곡선 출력.
"""
import argparse
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import torch

def interp_unvoiced(f0):
    f0 = f0.astype(np.float32)
    nz = np.nonzero(f0)[0]
    if len(nz) < 2:
        return f0
    f = np.interp(np.arange(len(f0)), nz, f0[nz])
    return f

def median_filter(x, k=3):
    if k <= 1: return x
    from scipy.signal import medfilt
    return medfilt(x, kernel_size=k)

def rmvpe_extract(y, sr, hop):
    # 기대: rmvpe 모듈이 설치되어 있고, 모델 가중치가 내부적으로 로딩 가능
    from rmvpe import RMVPE
    model = RMVPE(device="cuda" if torch.cuda.is_available() else "cpu")
    # RMVPE는 frame hop을 샘플 단위가 아니라 ms 기준 내부 변환 가능. 여기선 librosa frame처리 없이 호출.
    f0 = model.infer_from_audio(y, sr)  # Hz, shape [T]
    return f0

def crepe_extract(y, sr, hop):
    import torchcrepe
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y_t = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T]
    # frame_hop(s): hop/sr
    step_size = int(round(hop / sr * 1000))  # ms 단위 근사
    f0, pd = torchcrepe.predict(y_t, sr, step_size, fmin=50, fmax=1100, model="full", batch_size=1024, device=device, return_periodicity=True)
    f0 = torchcrepe.filter.median(torchcrepe.threshold.At(0.1)(f0, pd), 3)
    return f0.squeeze(0).detach().cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--hop", type=int, default=512)
    ap.add_argument("--backend", choices=["rmvpe","crepe"], default="rmvpe")
    ap.add_argument("--median", type=int, default=3)
    args = ap.parse_args()

    wav_dir, out_dir = Path(args.wav_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = list(wav_dir.rglob("*.wav"))
    for p in tqdm(wavs, desc=f"F0 ({args.backend})"):
        y, sr = librosa.load(p, sr=None, mono=True)
        if args.backend == "rmvpe":
            f0 = rmvpe_extract(y, sr, args.hop)
        else:
            f0 = crepe_extract(y, sr, args.hop)

        # 보간 + 미디안 필터
        f0 = interp_unvoiced(f0)
        if args.median > 1:
            f0 = median_filter(f0, k=args.median)

        # (선택) 로그-헐츠 변환은 후단에서 수행. 여기선 Hz 그대로 저장.
        np.save(out_dir / (p.stem + ".npy"), f0.astype(np.float32))

if __name__ == "__main__":
    main()
