# -*- coding: utf-8 -*-
"""Ablation study on the 100-pair LOLv2-real test split.

Each variant removes or replaces one component of the full model. Run after
selecting results/best_config.json on the validation split.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RetinexConfig
from src.io_utils import imread
from src.metrics import compute_metrics
from src.retinex import enhance

DATA = ROOT / "data" / "lolv2" / "Test"
OUT = ROOT / "results" / "ablation.csv"


def load_config() -> RetinexConfig:
    cfg = RetinexConfig()
    path = ROOT / "results" / "best_config.json"
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            for k, v in json.load(f).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
    return cfg


VARIANTS = {
    "Full": {},
    "w/o guided filter": {"illumination_smoother": "gaussian"},
    "fixed gamma": {"adaptive_gamma": False},
    "w/o chroma control": {"saturation": 1.0},
    "w/o white balance": {"white_balance": 0.0},
    "w/o exposure cap": {"exposure_cap": 0.0},
    "w/o denoising": {"denoise_h": 0.0},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    base = load_config()
    pairs = []
    for p in sorted((DATA / "Input").glob("*.png")):
        gt = DATA / "GT" / p.name
        if gt.is_file():
            pairs.append((p, gt))
    if args.limit:
        pairs = pairs[: args.limit]
    images = [(imread(p), imread(g)) for p, g in pairs]
    print(f"test pairs: {len(images)}")

    rows = []
    for name, overrides in VARIANTS.items():
        cfg = RetinexConfig(**{**base.to_dict(), **overrides})
        vals = []
        t0 = time.perf_counter()
        for img, gt in images:
            out = enhance(img, cfg)
            m = compute_metrics(img, out, reference=gt)
            vals.append((m["psnr"], m["ssim"], m["loe"], m["colorfulness"]))
        dt = (time.perf_counter() - t0) / max(len(images), 1) * 1000
        a = np.array(vals)
        row = {
            "variant": name,
            "psnr": a[:, 0].mean(),
            "psnr_std": a[:, 0].std(),
            "ssim": a[:, 1].mean(),
            "ssim_std": a[:, 1].std(),
            "loe": a[:, 2].mean(),
            "colorfulness": a[:, 3].mean(),
            "time_ms": dt,
        }
        rows.append(row)
        print(f"{name:20s} PSNR={row['psnr']:.3f} SSIM={row['ssim']:.4f} "
              f"LOE={row['loe']:.4f} time={row['time_ms']:.0f} ms", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (ROOT / "results" / "ablation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("saved", OUT)


if __name__ == "__main__":
    main()
