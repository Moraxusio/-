# -*- coding: utf-8 -*-
"""Generate all figures used by the report and README."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import METHOD_LABELS, make_method
from src.config import RetinexConfig
from src.io_utils import imread
from src.metrics import psnr, ssim

DATA = ROOT / "data" / "lolv2" / "Test"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["input", "gamma", "clahe", "msrcr", "lime", "ours"]
COLORS = {
    "input": "#9aa0a6",
    "gamma": "#f2a900",
    "clahe": "#d97706",
    "msrcr": "#b91c1c",
    "lime": "#2563eb",
    "ours": "#059669",
}


def load_config() -> RetinexConfig:
    path = ROOT / "results" / "best_config.json"
    cfg = RetinexConfig()
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            for k, v in json.load(f).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
    return cfg


def read_metrics() -> list[dict]:
    path = ROOT / "results" / "metrics.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _save(fig, name: str, max_width: int = 1900) -> None:
    path = FIG / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        if im.width > max_width:
            h = round(im.height * max_width / im.width)
            im = im.resize((max_width, h), Image.Resampling.LANCZOS)
        im.save(path, optimize=True)
    except Exception as exc:
        print("post-process skipped:", exc)
    print("saved", path)


def qualitative() -> None:
    cfg = load_config()
    names = ["00690.png", "00695.png", "00699.png", "00780.png"]
    cols = [("input", "Input"), ("gt", "GT"), ("lime", "LIME"), ("ours", "Ours")]
    fig, axes = plt.subplots(len(names), len(cols), figsize=(3.55 * len(cols), 2.25 * len(names)))
    for r, name in enumerate(names):
        inp = imread(DATA / "Input" / name)
        gt = imread(DATA / "GT" / name)
        cache = {"input": inp, "gt": gt}
        for key in ("lime", "ours"):
            cache[key] = make_method(key, cfg)(inp)
        for c, (key, label) in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(cv2.cvtColor(cache[key], cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(label, fontsize=13, fontweight="bold", pad=8)
            if c >= 2:
                ax.set_xlabel(f"PSNR {psnr(cache[key], gt):.2f} / SSIM {ssim(cache[key], gt):.3f}",
                              fontsize=8, labelpad=2)
            if c == 0:
                ax.set_ylabel(name.replace(".png", ""), fontsize=9, rotation=0,
                              labelpad=32, va="center")
    plt.tight_layout()
    _save(fig, "qualitative.png")


def detail_crop() -> None:
    """Zoomed crop to show noise/detail differences."""
    cfg = load_config()
    name, (x, y, w, h) = "00695.png", (120, 150, 250, 180)
    inp = imread(DATA / "Input" / name)
    gt = imread(DATA / "GT" / name)
    outs = {k: make_method(k, cfg)(inp) for k in ("lime", "ours")}
    keys = [("input", "Input"), ("gt", "GT"), ("lime", "LIME"), ("ours", "Ours")]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 2.8))
    for ax, (key, label) in zip(axes, keys):
        im = inp if key == "input" else gt if key == "gt" else outs[key]
        crop = im[y:y + h, x:x + w]
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    _save(fig, "detail_crop.png")


def pipeline_diagram() -> None:
    steps = [
        ("Input\n$I$", "#e5e7eb"),
        ("Illumination\n$L=\\max_c I_c$", "#dbeafe"),
        ("Guided filter\n$L_f=\\mathrm{GF}(L)$", "#dbeafe"),
        ("Adaptive\nGamma\n$\\hat L=L_f^{\\gamma}$", "#dcfce7"),
        ("Reflectance\n$R=I/L_f$", "#fef3c7"),
        ("Recombine\n$R^{\\alpha}\\hat L$", "#dcfce7"),
        ("Chroma + WB\n$s,\\ \\mathrm{GW}$", "#fae8ff"),
        ("NLM denoise\n$\\mathrm{NLM}$", "#fee2e2"),
        ("Output\n$\\hat I$", "#e5e7eb"),
    ]
    fig, ax = plt.subplots(figsize=(15, 3.2))
    ax.set_xlim(0, 15.2); ax.set_ylim(0, 3.2); ax.axis("off")
    w, h, gap = 1.50, 1.15, 0.15
    for i, (text, color) in enumerate(steps):
        x = 0.2 + i * (w + gap)
        y = 1.0
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                             linewidth=1.0, edgecolor="#374151", facecolor=color)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, y + h / 2),
                                         (x + w + gap - 0.02, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=14,
                                         linewidth=1.2, color="#374151"))
    ax.text(0.2, 0.35, "The guided filter is O(N) and replaces the sparse linear solver of LIME; "
                       "chroma/WB and NLM suppress colour cast and noise.",
            fontsize=10, color="#4b5563")
    _save(fig, "pipeline.png")


def metric_plots() -> None:
    rows = read_metrics()
    if not rows:
        print("no metrics.csv; skip metric plots")
        return
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [COLORS[m] for m in methods]

    # mean bars
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, title in zip(axes, ("psnr", "ssim", "loe"),
                              ("PSNR (dB) $\\uparrow$", "SSIM $\\uparrow$", "LOE $\\downarrow$")):
        vals = [np.mean([float(r[key]) for r in rows if r["method"] == m]) for m in methods]
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}" if key != "ssim" else f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    _save(fig, "metric_bars.png")

    # box plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, key, title in zip(axes, ("psnr", "ssim", "loe"),
                              ("PSNR (dB) $\\uparrow$", "SSIM $\\uparrow$", "LOE $\\downarrow$")):
        data = [[float(r[key]) for r in rows if r["method"] == m] for m in methods]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        medianprops={"color": "black", "linewidth": 1.2})
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.75)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    _save(fig, "metric_boxes.png")

    # runtime (log scale)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    vals = [np.mean([float(r["time_ms"]) for r in rows if r["method"] == m]) for m in methods]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Time per image (ms, log scale)")
    ax.set_title("Runtime on CPU (600 x 400)")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.tick_params(axis="x", rotation=20)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.0f}",
                ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    _save(fig, "runtime.png")



def dataset_samples() -> None:
    names = ["00690.png", "00695.png", "00699.png"]
    fig, axes = plt.subplots(len(names), 2, figsize=(6.6, 2.1 * len(names)))
    for r, name in enumerate(names):
        inp = imread(DATA / "Input" / name)
        gt = imread(DATA / "GT" / name)
        for c, (im, label) in enumerate([(inp, "Input"), (gt, "GT")]):
            ax = axes[r, c]
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(label, fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(name.replace(".png", ""), fontsize=9, rotation=0,
                              labelpad=30, va="center")
    plt.tight_layout()
    _save(fig, "dataset_samples.png")


def ablation_plot() -> None:
    path = ROOT / "results" / "ablation.csv"
    if not path.is_file():
        print("no ablation.csv; skip ablation plot")
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    labels = [r["variant"] for r in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, key, title in zip(axes, ("psnr", "ssim"), ("PSNR (dB) $\\uparrow$", "SSIM $\\uparrow$")):
        vals = [float(r[key]) for r in rows]
        colors = ["#059669" if r["variant"] == "Full" else "#94a3b8" for r in rows]
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=0.25)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.2f}" if key == "psnr" else f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    _save(fig, "ablation.png")


def tuning_plot() -> None:
    path = ROOT / "results" / "tuning_matched.csv"
    if not path.is_file():
        path = ROOT / "results" / "tuning.csv"
    if not path.is_file():
        print("no tuning csv; skip tuning plot")
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    gammas = sorted({float(r["gamma_offset"]) for r in rows})
    caps = sorted({float(r["exposure_cap"]) for r in rows if "exposure_cap" in r})
    if not caps:
        print("tuning csv has no exposure_cap; skip")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, key, title, cmap in (
        (axes[0], "psnr", "PSNR (dB): gamma offset vs. exposure cap", "YlGn"),
        (axes[1], "ssim", "SSIM: gamma offset vs. exposure cap", "Blues"),
    ):
        mat = np.full((len(caps), len(gammas)), np.nan)
        for r in rows:
            i = caps.index(float(r["exposure_cap"]))
            j = gammas.index(float(r["gamma_offset"]))
            mat[i, j] = float(r[key])
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(gammas))); ax.set_xticklabels([f"{g:.2f}" for g in gammas])
        ax.set_yticks(range(len(caps))); ax.set_yticklabels([f"{c:.2f}" for c in caps])
        ax.set_xlabel("gamma offset"); ax.set_ylabel("exposure cap")
        ax.set_title(title, fontsize=11)
        for i in range(len(caps)):
            for j in range(len(gammas)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}" if key == "psnr" else f"{mat[i, j]:.3f}",
                            ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    _save(fig, "tuning.png")


def failure_cases() -> None:
    rows = read_metrics()
    if not rows:
        print("no metrics.csv; skip failure cases")
        return
    by_image: dict[str, dict] = {}
    for r in rows:
        by_image.setdefault(r["image"], {})[r["method"]] = r
    scored = []
    for name, pair in by_image.items():
        if "ours" in pair and "lime" in pair:
            scored.append((float(pair["ours"]["psnr"]) - float(pair["lime"]["psnr"]), name))
    scored.sort()
    names = [n for _, n in scored[:2]]
    cfg = load_config()
    fig, axes = plt.subplots(len(names), 4, figsize=(12.5, 2.6 * len(names)))
    if len(names) == 1:
        axes = np.array([axes])
    for r, name in enumerate(names):
        inp = imread(DATA / "Input" / name)
        gt = imread(DATA / "GT" / name)
        lime = make_method("lime", cfg)(inp)
        ours = make_method("ours", cfg)(inp)
        for c, (im, label) in enumerate([(inp, "Input"), (gt, "GT"), (lime, "LIME"), (ours, "Ours")]):
            ax = axes[r, c]
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            if r == 0:
                ax.set_title(label, fontsize=12, fontweight="bold")
            if c >= 2:
                ax.set_xlabel(f"PSNR {psnr(im, gt):.2f} / SSIM {ssim(im, gt):.3f}", fontsize=8)
            if c == 0:
                ax.set_ylabel(name.replace(".png", ""), fontsize=9, rotation=0,
                              labelpad=32, va="center")
    plt.tight_layout()
    _save(fig, "failure.png")


def main() -> None:
    pipeline_diagram()
    dataset_samples()
    qualitative()
    detail_crop()
    metric_plots()
    ablation_plot()
    tuning_plot()
    failure_cases()


if __name__ == "__main__":
    main()
