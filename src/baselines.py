# -*- coding: utf-8 -*-
"""Classical baselines and a thin wrapper around the vendored LIME code."""
from __future__ import annotations

import cv2
import numpy as np

from .config import RetinexConfig
from .retinex import enhance


def gamma_correct(img: np.ndarray, gamma: float = 0.45) -> np.ndarray:
    """Global gamma correction with a 256-entry LUT."""
    lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, lut)


def clahe(img: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE on the L channel of CIELAB."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _retinex_channel(channel: np.ndarray, sigma: float) -> np.ndarray:
    blur = cv2.GaussianBlur(channel, (0, 0), sigma)
    return np.log10(channel + 1.0) - np.log10(blur + 1.0)


def msrcr(img: np.ndarray, scales=(15, 80, 250), gain: float = 5.0,
          offset: float = 25.0, alpha: float = 125.0, beta: float = 46.0) -> np.ndarray:
    """Multi-scale Retinex with colour restoration (MSRCR).

    This is the standard implementation used in many low-light papers. It is
    kept intentionally un-tuned so that it can serve as a baseline.
    """
    x = img.astype(np.float64) + 1.0
    msr = np.mean([_retinex_channel(x, s) for s in scales], axis=0)
    total = np.sum(x, axis=2, keepdims=True) + 1.0
    crf = beta * (np.log10(alpha * x) - np.log10(total))
    out = gain * msr * crf + offset
    out = np.clip(out, 0, 255)
    for c in range(3):
        lo, hi = np.percentile(out[..., c], (1, 99))
        out[..., c] = (out[..., c] - lo) * 255.0 / max(hi - lo, 1e-6)
    return np.clip(out, 0, 255).astype(np.uint8)


def lime(img: np.ndarray, gamma: float = 0.6, lambda_: float = 0.15,
         sigma: int = 3) -> np.ndarray:
    """LIME baseline (illumination-map estimation with a sparse solver)."""
    from third_party.lime.exposure_enhancement import enhance_image_exposure

    return enhance_image_exposure(
        img, gamma=gamma, lambda_=lambda_, dual=False, sigma=sigma
    )


def dual(img: np.ndarray, gamma: float = 0.6, lambda_: float = 0.15,
         sigma: int = 3) -> np.ndarray:
    """DUAL baseline (under- and over-exposure fusion)."""
    from third_party.lime.exposure_enhancement import enhance_image_exposure

    return enhance_image_exposure(
        img, gamma=gamma, lambda_=lambda_, dual=True, sigma=sigma
    )


METHOD_LABELS = {
    "input": "Input",
    "gamma": "Gamma",
    "clahe": "CLAHE",
    "msrcr": "MSRCR",
    "lime": "LIME",
    "dual": "DUAL",
    "ours": "Ours",
}


def make_method(name: str, cfg: RetinexConfig | None = None):
    """Return a callable img_bgr -> img_bgr for a method name."""
    name = name.lower()
    if name == "input":
        return lambda im: im.copy()
    if name == "gamma":
        return gamma_correct
    if name == "clahe":
        return clahe
    if name == "msrcr":
        return msrcr
    if name == "lime":
        return lime
    if name == "dual":
        return dual
    if name == "ours":
        cfg = cfg or RetinexConfig()
        return lambda im: enhance(im, cfg)
    raise ValueError(f"unknown method: {name}")


def available_methods() -> list[str]:
    return ["input", "gamma", "clahe", "msrcr", "lime", "dual", "ours"]
