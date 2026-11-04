# -*- coding: utf-8 -*-
"""Select exposure-related hyper-parameters on the brightness-matched validation set."""
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

DATA = ROOT / "data" / "lolv2" / "Val"
OUT = ROOT / "results" / "tuning.csv"
BEST = ROOT / "results" / "best_config.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    pairs = []
    for p in sorted((DATA / "Input").glob("*.png")):
        gt = DATA / "GT" / p.name
        if gt.is_file():
            pairs.append((p, gt))
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"matched validation pairs: {len(pairs)}")
    images = [(imread(p), imread(g)) for p, g in pairs]

    configs = []
    for go, cap in itertools.product((0.06, 0.08, 0.10, 0.12), (0.40, 0.45, 0.50)):
        configs.append(RetinexConfig(gamma_offset=go, exposure_cap=cap))

    rows = []
    for idx, cfg in enumerate(configs, 1):
        vals = []
        for img, gt in images:
            out = enhance(img, cfg)
            m = compute_metrics(img, out, reference=gt)
            vals.append((m["psnr"], m["ssim"], m["loe"]))
        a = np.array(vals)
        row = {
            "gamma_offset": cfg.gamma_offset,
            "exposure_cap": cfg.exposure_cap,
            "psnr": a[:, 0].mean(),
            "ssim": a[:, 1].mean(),
            "loe": a[:, 2].mean(),
        }
        row["score"] = row["psnr"] + 20.0 * row["ssim"] - 2.0 * row["loe"]
        rows.append(row)
        print(f"[{idx:02d}/{len(configs)}] go={cfg.gamma_offset} cap={cfg.exposure_cap} "
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
        gamma_offset=float(best["gamma_offset"]),
        exposure_cap=float(best["exposure_cap"]),
    )
    BEST.write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nBest config saved to", BEST)
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    print("\nTop 3:")
    for r in rows[:3]:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
