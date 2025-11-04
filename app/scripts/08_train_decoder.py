#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, glob, pickle
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from 07_decoder_model import MelDecoder

def load_stats(stats_dir):
    with open(Path(stats_dir)/"content_stats.pkl","rb") as f: c = pickle.load(f)
    with open(Path(stats_dir)/"f0_stats.pkl","rb") as f: f0 = pickle.load(f)
    with open(Path(stats_dir)/"loud_stats.pkl","rb") as f: l = pickle.load(f)
    return c, f0, l

class FeatDataset(Dataset):
    def __init__(self, content_dir, f0_dir, loud_dir, mel_dir, stats_dir):
        self.content_paths = sorted(Path(content_dir).glob("*.npy"))
        self.f0_dir = Path(f0_dir); self.loud_dir = Path(loud_dir); self.mel_dir = Path(mel_dir)
        self.c_stats, self.f0_stats, self.l_stats = load_stats(stats_dir)

    def __len__(self): return len(self.content_paths)

    def __getitem__(self, idx):
        cp = self.content_paths[idx]
        key = cp.stem
        content = np.load(cp)                     # [T, Dc]
        f0 = np.load(self.f0_dir/f"{key}_f0_hz.npy")    # [T]
        loud = np.load(self.loud_dir/f"{key}.npy")      # [T, 1] (04에서 저장) :contentReference[oaicite:7]{index=7}
        mel = np.load(self.mel_dir/f"{key}.npy")        # [T, 80]

        T = min(content.shape[0], f0.shape[0], loud.shape[0], mel.shape[0])
        content = content[:T]
        f0 = f0[:T].reshape(T,1)
        loud = loud[:T]
        mel = mel[:T]

        # 정규화
        c_mu, c_std = self.c_stats["mu"], self.c_stats["std"]
        content = (content - c_mu) / c_std

        # f0는 log-변환 후 통계(05에서 raw μ/σ 저장이지만, 여기서는 안정 위해 log 처리 후 μ/σ 재사용) :contentReference[oaicite:8]{index=8}
        f0_safe = np.clip(f0, 1e-6, None)
        f0_log = np.log(f0_safe)
        f0_mu, f0_std = self.f0_stats["mu"], self.f0_stats["std"]
        f0_norm = (f0_log - f0_mu) / f0_std

        l_mu, l_std = self.l_stats["mu"], self.l_stats["std"]
        loud_norm = (loud - l_mu) / l_std

        return (
            torch.from_numpy(content).float(),
            torch.from_numpy(f0_norm).float(),
            torch.from_numpy(loud_norm).float(),
            torch.from_numpy(mel).float()
        )

def collate(batch):
    # 길이 일치 전제(01~05/06에서 같은 hop=512 기준), 단순 스택
    C, F, L, M = zip(*batch)
    return torch.stack(C,0), torch.stack(F,0), torch.stack(L,0), torch.stack(M,0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content_dir", required=True)
    ap.add_argument("--f0_dir", required=True)
    ap.add_argument("--loud_dir", required=True)
    ap.add_argument("--mel_dir", required=True)
    ap.add_argument("--stats_dir", required=True)       # features/stats
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--save_path", default="mel_decoder.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ds = FeatDataset(args.content_dir, args.f0_dir, args.loud_dir, args.mel_dir, args.stats_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=2)

    # content 차원은 파일에서 결정
    sample_c = np.load(sorted(Path(args.content_dir).glob("*.npy"))[0])
    content_dim = sample_c.shape[1]
    model = MelDecoder(content_dim=content_dim, hidden=args.hidden).to(args.device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    l1 = nn.L1Loss(); l2 = nn.MSELoss()

    for ep in range(1, args.epochs+1):
        model.train(); total=0.0
        for content, f0n, loudn, mel in dl:
            content=content.to(args.device); f0n=f0n.to(args.device)
            loudn=loudn.to(args.device); mel=mel.to(args.device)
            pred = model(content, f0n, loudn)
            loss = 0.7*l1(pred, mel) + 0.3*l2(pred, mel)
            optim.zero_grad(); loss.backward(); optim.step()
            total += float(loss.item())*content.size(0)
        print(f"[{ep}/{args.epochs}] train_loss={total/len(ds):.4f}")
        torch.save({"model": model.state_dict(),
                    "content_dim": content_dim}, args.save_path)
    print(f"[OK] saved -> {args.save_path}")

if __name__ == "__main__":
    main()
