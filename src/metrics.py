# -*- coding: utf-8 -*-
"""Image-quality metrics used in the experiments.

PSNR and SSIM are full-reference metrics (enhanced vs. GT). LOE measures
lightness-order preservation (enhanced vs. original) and is a no-reference
naturalness proxy. Colourfulness, entropy and average gradient describe the
visual statistics of the output.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity as _ssim


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return float(_ssim(ga, gb, data_range=255))


def colorfulness(img: np.ndarray) -> float:
    b, g, r = [img[..., i].astype(np.float64) for i in range(3)]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                 + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def entropy(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = np.histogram(gray, bins=256, range=(0, 256))[0].astype(np.float64)
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def avg_gradient(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))


def _lightness(img: np.ndarray) -> np.ndarray:
    return np.max(img, axis=2).astype(np.float64) / 255.0


def loe(original: np.ndarray, enhanced: np.ndarray, size: int = 64) -> float:
    """Lightness Order Error (lower is better).

    For every pair of pixels we compare the relative lightness order in the
    original and enhanced images. The O(N^2) form is evaluated on a 64 x 64
    downsampled image, which is the usual practice for this metric.
    """
    a = _lightness(original)
    b = _lightness(enhanced)
    a = cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA).ravel()
    b = cv2.resize(b, (size, size), interpolation=cv2.INTER_AREA).ravel()
    order_a = a[:, None] >= a[None, :]
    order_b = b[:, None] >= b[None, :]
    rd = np.mean(order_a != order_b, axis=1)
    return float(rd.mean())


def compute_metrics(original: np.ndarray, enhanced: np.ndarray,
                    reference: np.ndarray | None = None) -> dict:
    out = {
        "loe": loe(original, enhanced),
        "colorfulness": colorfulness(enhanced),
        "entropy": entropy(enhanced),
        "avg_gradient": avg_gradient(enhanced),
    }
    if reference is not None:
        out["psnr"] = psnr(enhanced, reference)
        out["ssim"] = ssim(enhanced, reference)
    return out
