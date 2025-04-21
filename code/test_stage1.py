#!/usr/bin/env python3
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler

from hybrid_two_stage import (
    Stage1Dataset,
    Stage1Detector,
    train_stage1,
    eval_stage1,
    LR1,
    DESIRED_N1,
    device,
)

# point this at your real processed data directory:
RAW_DIR = Path("/users/okalova/sleep/STAT-4830-GOALZ-project/data/processed_sleepedf")

if __name__ == "__main__":
    # load just a handful of windows so it runs in seconds
    ds = Stage1Dataset(RAW_DIR)
    ds.X = ds.X[:64]
    ds.Y = ds.Y[:64]

    # rebuild sampler weights over our tiny subset
    w = np.bincount(ds.Y.numpy(), minlength=2)
    w1 = (w[0] / (w[1] + 1e-6)) * (DESIRED_N1 / (1 - DESIRED_N1))
    sw = np.where(ds.Y.numpy() == 1, w1, 1.0)

    loader = DataLoader(
        ds,
        batch_size=8,
        sampler=WeightedRandomSampler(sw, len(sw), replacement=True),
    )

    model = Stage1Detector().to(device)
    opt   = optim.Adam(model.parameters(), lr=LR1)

    # single‐epoch smoke test
    loss = train_stage1(model, loader, opt)
    acc  = eval_stage1(model, loader)

    print(f"test →  loss={loss:.4f},  acc={acc:.4f}")
