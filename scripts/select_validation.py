# -*- coding: utf-8 -*-
"""Build a brightness-matched validation set from LOLv2-real Train.

The official Train and Test splits have different GT brightness distributions
(Train is much brighter).  We stratify 30 Train pairs by the Test GT brightness
quantiles, so that exposure-related hyper-parameters are selected on a
representative distribution.  Test images are only used for their brightness
statistics here, never for parameter fitting.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "lolv2"
OUT = ROOT / "results" / "val_matched.json"
N_BINS, PER_BIN = 5, 6


def gt_mean(path: Path) -> float:
    img = cv2.imread(str(path))
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


def main() -> None:
    test = sorted((DATA / "Test" / "GT").glob("*.png"))
    train = sorted((DATA / "Train" / "GT").glob("*.png"))
    test_means = np.array([gt_mean(p) for p in test])
    train_means = np.array([gt_mean(p) for p in train])
    edges = np.quantile(test_means, np.linspace(0, 1, N_BINS + 1))
    print("test GT mean", test_means.mean(), "quantile edges", edges)

    used: set[int] = set()
    chosen: list[dict] = []
    for b in range(N_BINS):
        lo, hi = edges[b], edges[b + 1]
        inside = test_means[(test_means >= lo) & (test_means <= hi)]
        targets = np.quantile(inside, np.linspace(0.1, 0.9, PER_BIN))
        for t in targets:
            order = np.argsort(np.abs(train_means - t))
            idx = next(int(i) for i in order if int(i) not in used)
            used.add(idx)
            chosen.append({
                "name": train[idx].name,
                "train_gt_mean": float(train_means[idx]),
                "target_mean": float(t),
                "bin": b,
            })

    val_in = DATA / "Val" / "Input"
    val_gt = DATA / "Val" / "GT"
    val_in.mkdir(parents=True, exist_ok=True)
    val_gt.mkdir(parents=True, exist_ok=True)
    for item in chosen:
        name = item["name"]
        shutil.copy2(DATA / "Train" / "Input" / name, val_in / name)
        shutil.copy2(DATA / "Train" / "GT" / name, val_gt / name)
    means = [x["train_gt_mean"] for x in chosen]
    print(f"selected {len(chosen)} pairs; GT mean {np.mean(means):.2f} ± {np.std(means):.2f}")
    OUT.write_text(json.dumps({
        "test_mean": float(test_means.mean()),
        "test_std": float(test_means.std()),
        "val_mean": float(np.mean(means)),
        "val_std": float(np.std(means)),
        "pairs": chosen,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
