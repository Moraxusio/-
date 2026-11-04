# -*- coding: utf-8 -*-
"""Select RetinexConfig on a validation subset of LOLv2-real Train.

The 100-pair test split is never used here. Run from the repository root:

    python scripts/tune_validation.py --limit 20
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RetinexConfig
from src.io_utils import imread
from src.metrics import compute_metrics
from src.retinex import enhance

DATA = ROOT / "data" / "lolv2" / "Train"
OUT = ROOT / "results" / "tuning.csv"
BEST = ROOT / "results" / "best_config.json"


def load_pairs(limit: int | None = None):
    pairs = []
    for p in sorted((DATA / "Input").glob("*.png")):
        gt = DATA / "GT" / p.name
        if gt.is_file():
            pairs.append((p, gt))
    if limit:
        pairs = pairs[:limit]
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    pairs = load_pairs(args.limit)
    print(f"validation pairs: {len(pairs)}")
    images = [(imread(p), imread(g)) for p, g in pairs]

    configs = []
    # Stage 1: global tone / chroma parameters.
    for go, sat, wb in itertools.product((0.08, 0.12, 0.16), (0.8, 0.9), (0.5,)):
        configs.append(RetinexConfig(gamma_offset=go, saturation=sat, white_balance=wb))
    # Stage 2: guided-filter radius and regularisation.
    for r in (8, 16, 32):
        configs.append(RetinexConfig(radius=r, eps=0.02))

    rows = []
    for idx, cfg in enumerate(configs, 1):
        vals = []
        for img, gt in images:
            out = enhance(img, cfg)
            m = compute_metrics(img, out, reference=gt)
            vals.append((m["psnr"], m["ssim"], m["loe"], m["colorfulness"]))
        a = np.array(vals)
        row = {
            "radius": cfg.radius,
            "eps": cfg.eps,
            "gamma_offset": cfg.gamma_offset,
            "saturation": cfg.saturation,
            "white_balance": cfg.white_balance,
            "psnr": a[:, 0].mean(),
            "ssim": a[:, 1].mean(),
            "loe": a[:, 2].mean(),
            "colorfulness": a[:, 3].mean(),
        }
        row["score"] = row["psnr"] + 20.0 * row["ssim"] - 2.0 * row["loe"]
        rows.append(row)
        print(f"[{idx:02d}/{len(configs)}] r={cfg.radius} eps={cfg.eps} "
              f"go={cfg.gamma_offset} sat={cfg.saturation} wb={cfg.white_balance} "
              f"PSNR={row['psnr']:.3f} SSIM={row['ssim']:.4f} LOE={row['loe']:.4f} "
              f"score={row['score']:.3f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rows.sort(key=lambda r: r["score"], reverse=True)
    best = rows[0]
    cfg = RetinexConfig(
        radius=int(best["radius"]),
        eps=float(best["eps"]),
        gamma_offset=float(best["gamma_offset"]),
        saturation=float(best["saturation"]),
        white_balance=float(best["white_balance"]),
    )
    with BEST.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    print("\nBest config saved to", BEST)
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    print("\nTop 3:")
    for r in rows[:3]:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
