#!/usr/bin/env python3
"""
features/content/*.npy, features/f0/*.npy, features/loud/*.npy 를 훑어 각 피처별(차원별) 평균/표준편차(μ/σ) 를 산출해 features/stats/에 저장합니다.
(추론 시 동일 통계를 적용해 표준화
"""
import argparse, pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm

def stack_and_stats(npy_paths, axis=0, max_files=None):
    arrs = []
    for i, p in enumerate(tqdm(npy_paths, desc="stack")):
        if (max_files is not None) and (i >= max_files):
            break
        a = np.load(p)
        a = a.reshape(a.shape[0], -1)  # [T, D]
        arrs.append(a)
    X = np.concatenate(arrs, axis=0)  # [sum_T, D]
    mu = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1e-8
    return mu, std

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content_dir", required=True)
    ap.add_argument("--f0_dir", required=True)
    ap.add_argument("--loud_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit_files", type=int, default=None, help="빠른 통계를 위한 파일 수 제한")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Content stats
    c_paths = sorted(Path(args.content_dir).glob("*.npy"))
    c_mu, c_std = stack_and_stats(c_paths, max_files=args.limit_files)
    with open(out_dir / "content_stats.pkl", "wb") as f:
        pickle.dump({"mu": c_mu, "std": c_std}, f)

    # F0 stats (Hz → 보통 로그 변환 후 정규화. 여기서는 raw Hz 통계 제공)
    f_paths = sorted(Path(args.f0_dir).glob("*.npy"))
    f_mu, f_std = stack_and_stats(f_paths, max_files=args.limit_files)
    with open(out_dir / "f0_stats.pkl", "wb") as f:
        pickle.dump({"mu": f_mu, "std": f_std}, f)

    # Loudness stats (dB)
    l_paths = sorted(Path(args.loud_dir).glob("*.npy"))
    l_mu, l_std = stack_and_stats(l_paths, max_files=args.limit_files)
    with open(out_dir / "loud_stats.pkl", "wb") as f:
        pickle.dump({"mu": l_mu, "std": l_std}, f)

    print(f"[OK] Saved stats to {out_dir}")

if __name__ == "__main__":
    main()
