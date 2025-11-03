#!/usr/bin/env python3
"""
오디오 표준화(44.1k/mono)
"""
import argparse, os
from pathlib import Path
import librosa, soundfile as sf
import numpy as np

# 감쇠 전용 정규화
# def peak_normalize(y: np.ndarray, peak: float = -1.0):
#     """
#     y: 오디오 신호(float32, -1.0~1.0 범위)
#     peak: 목표 음압(default -1.0dBFS)
#     """
#     if peak is None:  # no-op
#         return y
#     # peak in dBFS (e.g., -1.0 dB)
#     # convert to linear: -1.0dBFS -> 10^(-1/20)=0.891
#     lin_peak = 10 ** (peak / 20.0) # 오디오 신호는 dB로 표현 시 log scale 사용
#     max_abs = np.max(np.abs(y)) + 1e-12 # 입력 신호의 최대 절댓값 계산(1e-12는 안정화 상수)
#     if max_abs > lin_peak:
#         y = y * (lin_peak / max_abs) # 신호가 클 경우, 선형 비율 스케일링
#     return y

# 양방향 정규화(감쇠, 증폭)
def peak_normalize(y: np.ndarray, peak: float = -1.0, min_threshold: float = 0.05):
    """
    y: 오디오 신호(float32, -1.0~1.0)
    peak: 목표 피크(dBFS, 예: -1.0)
    min_threshold: 너무 작은 신호는 증폭하지 않기 위한 최소 레벨
    """
    if peak is None:
        return y

    lin_peak = 10 ** (peak / 20.0)
    max_abs = np.max(np.abs(y)) + 1e-12

    # 아주 작은 신호는 증폭 시 노이즈까지 커지므로 방지
    if max_abs < min_threshold:
        return y

    # 피크가 작거나 클 경우 모두 동일하게 스케일링
    y = y * (lin_peak / max_abs)
    return y

def process(in_path: Path, out_path: Path, target_sr: int, peak_db: float):
    y, sr_in = librosa.load(str(in_path), sr=None, mono=True)
    if sr_in != target_sr:
        y = librosa.resample(y, orig_sr=sr_in, target_sr=target_sr, res_type="kaiser_best")
    y = peak_normalize(y, peak_db)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y, target_sr, subtype="PCM_16")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="원본 wav 디렉토리")
    ap.add_argument("--out_dir", required=True, help="44.1kHz 리샘플 출력 디렉토리")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--peak_db", type=float, default=-1.0, help="피크 정규화 목표(dBFS). 끄려면 999 등 매우 큰 값")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    wavs = list(in_dir.rglob("*.wav"))
    for p in wavs:
        rel = p.relative_to(in_dir)
        out_p = out_dir / rel
        process(p, out_p, args.sr, args.peak_db)

if __name__ == "__main__":
    main()
