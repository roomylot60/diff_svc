#!/usr/bin/env python3
"""
오디오 표준화(44.1k/mono)
"""
import argparse, os
from pathlib import Path
import librosa, soundfile as sf
import numpy as np

def peak_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)

def loudness_dbfs(y: np.ndarray) -> float:
    """RMS 기반 dBFS (간단 추정). 무음 보호 포함."""
    rms = np.sqrt(np.mean(y**2) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)

def peak_normalize_bidir(y: np.ndarray, target_db: float = -1.0, min_threshold: float = 0.05) -> np.ndarray:
    """
    양방향 피크 정규화: 피크가 크면 감쇠, 작으면 증폭.
    너무 작은 신호(max_abs < min_threshold)는 증폭하지 않음(노이즈 보호).
    """
    lin_target = peak_to_linear(target_db)
    max_abs = float(np.max(np.abs(y)) + 1e-12)
    if max_abs < min_threshold:
        return y  # 노이즈 증폭 방지
    return y * (lin_target / max_abs)

def rms_normalize(y: np.ndarray, target_db: float = -20.0, min_threshold: float = 0.005) -> np.ndarray:
    """
    RMS(평균 레벨) 정규화. 무음에 가까운 신호는 증폭하지 않음.
    """
    rms = float(np.sqrt(np.mean(y**2) + 1e-12))
    if rms < min_threshold:
        return y
    current_db = 20.0 * np.log10(rms + 1e-12)
    gain_db = target_db - current_db
    gain = peak_to_linear(gain_db)
    return y * gain

def process_file(in_path: Path, out_path: Path, target_sr: int,
                 normalize: str, target_db: float, min_threshold: float,
                 peak_db_limit: float):
    # 1) 로드(모노 유지)
    y, sr_in = librosa.load(str(in_path), sr=None, mono=True)

    # 2) 리샘플
    if sr_in != target_sr:
        y = librosa.resample(y, orig_sr=sr_in, target_sr=target_sr, res_type="kaiser_best")

    # 3) 정규화
    if normalize == "peak":
        y = peak_normalize_bidir(y, target_db=target_db, min_threshold=min_threshold)
    elif normalize == "rms":
        y = rms_normalize(y, target_db=target_db, min_threshold=min_threshold)
        # RMS 정규화 후 안전 헤드룸 보장(선택): 피크가 너무 크면 감쇠
        lin_limit = peak_to_linear(peak_db_limit)
        max_abs = float(np.max(np.abs(y)) + 1e-12)
        if max_abs > lin_limit:
            y = y * (lin_limit / max_abs)
    else:
        # none: 아무 것도 안 함
        pass

    # 4) 저장(16-bit PCM)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y, target_sr, subtype="PCM_16")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="원본 wav 디렉토리")
    ap.add_argument("--out_dir", required=True, help="44.1kHz 출력 디렉토리")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--normalize", choices=["none","peak","rms"], default="peak",
                    help="정규화 방식: none/peak(양방향)/rms")
    ap.add_argument("--target_db", type=float, default=-1.0,
                    help="피크 또는 RMS 목표 dBFS (정규화 방식에 따라 해석)")
    ap.add_argument("--min_threshold", type=float, default=0.05,
                    help="증폭 금지 임계치(피크/무음 보호)")
    ap.add_argument("--peak_db_limit", type=float, default=-1.0,
                    help="RMS 정규화 후 최종 피크 상한(dBFS). 예: -1.0")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    wavs = list(in_dir.rglob("*.wav"))
    for p in wavs:
        rel = p.relative_to(in_dir)
        out_p = out_dir / rel
        process_file(
            p, out_p, target_sr=args.sr,
            normalize=args.normalize,
            target_db=args.target_db,
            min_threshold=args.min_threshold,
            peak_db_limit=args.peak_db_limit
        )

if __name__ == "__main__":
    main()

"""
# (기본) 양방향 피크 정규화: -1.0 dBFS로 통일
python scripts/01_resample_to_44100.py \
  --in_dir data/raw --out_dir data/sr44100 --normalize peak --target_db -1.0

# RMS 정규화로 평균 레벨을 -20 dBFS 부근으로 맞춘 뒤, 피크 상한 -1 dBFS 보장
python scripts/01_resample_to_44100.py \
  --in_dir data/raw --out_dir data/sr44100 --normalize rms --target_db -20 --peak_db_limit -1
"""