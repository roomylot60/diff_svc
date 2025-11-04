#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, glob, pickle
import numpy as np
import torch
from pathlib import Path
from 07_decoder_model import MelDecoder

def load_stats(stats_dir):
    import pickle
    with open(Path(stats_dir)/"content_stats.pkl","rb") as f: c = pickle.load(f)
    with open(Path(stats_dir)/"f0_stats.pkl","rb") as f: f0 = pickle.load(f)
    with open(Path(stats_dir)/"loud_stats.pkl","rb") as f: l = pickle.load(f)
    return c, f0, l

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content_dir", required=True)
    ap.add_argument("--f0_dir", required=True)
    ap.add_argument("--loud_dir", required=True)
    ap.add_argument("--stats_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)   # e.g., features/mel_pred
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    c_paths = sorted(Path(args.content_dir).glob("*.npy"))
    c_stats, f0_stats, l_stats = load_stats(args.stats_dir)

    ckpt = torch.load(args.ckpt, map_location=args.device)
    content_dim = ckpt["content_dim"]
    model = MelDecoder(content_dim=content_dim).to(args.device)
    model.load_state_dict(ckpt["model"]); model.eval()

    for i, cp in enumerate(c_paths, 1):
        key = cp.stem
        content = np.load(cp)
        f0 = np.load(Path(args.f0_dir)/f"{key}_f0_hz.npy")
        loud = np.load(Path(args.loud_dir)/f"{key}.npy")

        T = min(content.shape[0], f0.shape[0], loud.shape[0])
        content, f0, loud = content[:T], f0[:T].reshape(T,1), loud[:T]

        # 정규화(05의 통계 사용) :contentReference[oaicite:9]{index=9}
        content = (content - c_stats["mu"]) / c_stats["std"]
        f0n = (np.log(np.clip(f0,1e-6,None)) - f0_stats["mu"]) / f0_stats["std"]
        loudn = (loud - l_stats["mu"]) / l_stats["std"]

        with torch.no_grad():
            c_t = torch.from_numpy(content).float().unsqueeze(0).to(args.device)
            f_t = torch.from_numpy(f0n).float().unsqueeze(0).to(args.device)
            l_t = torch.from_numpy(loudn).float().unsqueeze(0).to(args.device)
            mel = model(c_t, f_t, l_t).squeeze(0).cpu().numpy().astype(np.float32)

        np.save(Path(args.out_dir)/f"{key}.npy", mel)
        print(f"[{i}/{len(c_paths)}] mel_pred -> {key}.npy  shape={mel.shape}")

if __name__ == "__main__":
    main()
