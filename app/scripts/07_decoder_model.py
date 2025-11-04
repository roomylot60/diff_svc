#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch, torch.nn as nn

class MelDecoder(nn.Module):
    """
    간단한 조건부 BiLSTM 디코더:
      x = concat([content, f0_norm, loud_norm]) -> BiLSTM(2층, hidden) -> Linear(80)
    """
    def __init__(self, content_dim=768, cond_dim=2, hidden=512, mel_bins=80, num_layers=2, dropout=0.1):
        super().__init__()
        in_dim = content_dim + cond_dim
        self.prenet = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.blstm = nn.LSTM(
            input_size=hidden, hidden_size=hidden//2, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.proj = nn.Linear(hidden, mel_bins)

    def forward(self, content, f0_norm, loud_norm):
        """
        content: [B, T, Dc]
        f0_norm: [B, T, 1]
        loud_norm: [B, T, 1]
        return mel: [B, T, 80]
        """
        x = torch.cat([content, f0_norm, loud_norm], dim=-1)
        x = self.prenet(x)
        x, _ = self.blstm(x)
        mel = self.proj(x)
        return mel
