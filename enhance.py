# -*- coding: utf-8 -*-
"""Command-line interface for low-light image enhancement.

Examples
--------
python enhance.py input.jpg -o output.png --method ours
python enhance.py --input-dir photos/ --output-dir enhanced/ --method ours
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.baselines import make_method
from src.config import RetinexConfig
from src.io_utils import imread, imwrite


def load_config(path: str | None) -> RetinexConfig:
    cfg = RetinexConfig()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            for k, v in json.load(f).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
    return cfg


def enhance_one(path: Path, out_path: Path, method: str, cfg: RetinexConfig,
                compare: bool = False) -> float:
    img = imread(path)
    fn = make_method(method, cfg)
    t0 = time.perf_counter()
    out = fn(img)
    dt = (time.perf_counter() - t0) * 1000.0
    if compare:
        h = min(img.shape[0], out.shape[0])
        panel = np.hstack([img[:h], np.full((h, 8, 3), 255, np.uint8), out[:h]])
        imwrite(out_path, panel)
    else:
        imwrite(out_path, out)
    return dt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Low-light image enhancement (guided-filter Retinex)")
    parser.add_argument("input", nargs="?", help="input image path")
    parser.add_argument("-o", "--output", help="output image path")
    parser.add_argument("--input-dir", help="batch input directory")
    parser.add_argument("--output-dir", help="batch output directory")
    parser.add_argument("--method", default="ours",
                        choices=["ours", "lime", "gamma", "clahe", "msrcr"])
    parser.add_argument("--config", help="JSON file overriding RetinexConfig (ours only)")
    parser.add_argument("--compare", action="store_true", help="save input/output side by side")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    if args.input_dir:
        in_dir = Path(args.input_dir)
        out_dir = Path(args.output_dir or "outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted([p for p in in_dir.iterdir()
                        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}])
        if not files:
            raise SystemExit(f"no images found in {in_dir}")
        for i, p in enumerate(files, 1):
            out = out_dir / f"{p.stem}_{args.method}.png"
            dt = enhance_one(p, out, args.method, cfg, compare=args.compare)
            print(f"[{i:03d}/{len(files):03d}] {p.name} -> {out.name}  {dt:.0f} ms")
        return

    if not args.input:
        parser.error("provide an input image or --input-dir")
    in_path = Path(args.input)
    out_path = Path(args.output or f"outputs/{in_path.stem}_{args.method}.png")
    dt = enhance_one(in_path, out_path, args.method, cfg, compare=args.compare)
    print(f"{in_path} -> {out_path}  ({dt:.0f} ms)")


if __name__ == "__main__":
    main()
