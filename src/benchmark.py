# -*- coding: utf-8 -*-
"""Benchmark runner for LOLv2-real.

Example
-------
python -m src.benchmark --split Test --methods input,gamma,clahe,msrcr,lime,ours \
    --out results/metrics.csv --summary results/summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .baselines import METHOD_LABELS, make_method
from .config import RetinexConfig
from .io_utils import imread, imwrite
from .metrics import compute_metrics

METRIC_KEYS = ["psnr", "ssim", "loe", "colorfulness", "entropy", "avg_gradient", "time_ms"]


def discover_pairs(data_root: str | os.PathLike, split: str = "Test") -> list[dict]:
    """Return [{'name': ..., 'input': ..., 'gt': ...}, ...] sorted by name."""
    root = Path(data_root) / split
    input_dir = root / "Input"
    gt_dir = root / "GT"
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    pairs = []
    for path in sorted(input_dir.glob("*.png")):
        gt = gt_dir / path.name
        if not gt.is_file():
            continue
        pairs.append({"name": path.name, "input": str(path), "gt": str(gt)})
    if not pairs:
        raise FileNotFoundError(f"no pairs found under {root}")
    return pairs


def _load_existing(csv_path: Path) -> tuple[set[tuple[str, str]], list[dict]]:
    if not csv_path.is_file():
        return set(), []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    done = {(r["image"], r["method"]) for r in rows}
    return done, rows


def _summary(rows: list[dict]) -> dict:
    methods = []
    for m in sorted({r["method"] for r in rows}):
        subset = [r for r in rows if r["method"] == m]
        item = {"method": m, "label": METHOD_LABELS.get(m, m), "n": len(subset)}
        for key in METRIC_KEYS:
            vals = np.array([float(r[key]) for r in subset if r.get(key) not in ("", None)])
            if vals.size:
                item[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        methods.append(item)
    return {"methods": methods, "num_images": len({r["image"] for r in rows})}


def run_benchmark(data_root: str, split: str, methods: list[str], out_csv: str,
                  summary_json: str, limit: int | None = None,
                  save_dir: str | None = None, save_names: list[str] | None = None,
                  resume: bool = False, cfg: RetinexConfig | None = None) -> dict:
    pairs = discover_pairs(data_root, split)
    if limit:
        pairs = pairs[:limit]
    cfg = cfg or RetinexConfig()
    save_names = set(save_names or [])
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    done, rows = _load_existing(out_path) if resume else (set(), [])
    write_header = not (resume and out_path.is_file() and out_path.stat().st_size > 0)

    # Warm up imports and OpenCV's internal buffers so that the first timed
    # call is not dominated by one-off initialisation.
    if pairs:
        warm = np.full((64, 64, 3), 32, dtype=np.uint8)
        for method in methods:
            try:
                make_method(method, cfg)(warm)
            except Exception:
                pass

    with out_path.open("a" if resume else "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "method", "psnr", "ssim", "loe",
                           "colorfulness", "entropy", "avg_gradient", "time_ms"]
        )
        if write_header:
            writer.writeheader()
        f.flush()

        for idx, pair in enumerate(pairs, 1):
            name = pair["name"]
            img = imread(pair["input"])
            gt = imread(pair["gt"])
            for method in methods:
                if (name, method) in done:
                    continue
                fn = make_method(method, cfg)
                t0 = time.perf_counter()
                out = fn(img)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                m = compute_metrics(img, out, reference=gt if method != "input" else gt)
                row = {
                    "image": name,
                    "method": method,
                    "psnr": f"{m['psnr']:.4f}",
                    "ssim": f"{m['ssim']:.6f}",
                    "loe": f"{m['loe']:.6f}",
                    "colorfulness": f"{m['colorfulness']:.4f}",
                    "entropy": f"{m['entropy']:.4f}",
                    "avg_gradient": f"{m['avg_gradient']:.4f}",
                    "time_ms": f"{dt_ms:.2f}",
                }
                writer.writerow(row)
                f.flush()
                rows.append(row)
                if save_dir and (name in save_names or not save_names):
                    imwrite(Path(save_dir) / method / name, out)
            done.update((name, m) for m in methods)
            print(f"[{idx:03d}/{len(pairs):03d}] {name} done", flush=True)

    # Re-read to ensure a complete, ordered table when resuming.
    with out_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    summary = _summary(rows)
    summary["config"] = asdict(cfg)
    summary["split"] = split
    summary["method_names"] = methods
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run low-light enhancement benchmarks")
    parser.add_argument("--data", default="data/lolv2", help="dataset root")
    parser.add_argument("--split", default="Test", help="Test or Train")
    parser.add_argument("--methods", default="input,gamma,clahe,msrcr,lime,ours")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="results/metrics.csv")
    parser.add_argument("--summary", default="results/summary.json")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--save-names", default="", help="comma-separated image names")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", default=None, help="JSON file overriding RetinexConfig")
    args = parser.parse_args(argv)

    cfg = RetinexConfig()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            for k, v in json.load(f).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    save_names = [n.strip() for n in args.save_names.split(",") if n.strip()]
    run_benchmark(args.data, args.split, methods, args.out, args.summary,
                  limit=args.limit, save_dir=args.save_dir, save_names=save_names,
                  resume=args.resume, cfg=cfg)


if __name__ == "__main__":
    main()
