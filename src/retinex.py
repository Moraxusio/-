# -*- coding: utf-8 -*-
"""Guided-filter Retinex low-light enhancement.

Pipeline
--------
1. Estimate the illumination map L = max_c(I_c) and smooth it with a
   self-guided filter, which preserves strong edges but removes texture.
2. Compute the reflectance R = I / (L + eps).
3. Apply an adaptive gamma curve to L; darker inputs receive a stronger lift.
4. Recombine I_hat = R^alpha * L_hat.
5. Suppress over-saturation with a chroma control step, remove residual colour
   cast with a mild gray-world balance, and denoise with non-local means.

The implementation only uses NumPy/OpenCV and runs on a CPU.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import RetinexConfig


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 16,
                  eps: float = 0.02) -> np.ndarray:
    """Fast guided filter (He et al., 2013) with box-filter means.

    Parameters
    ----------
    guide : 2-D float array
        Guidance image. The filter preserves its edges.
    src : 2-D float array
        Input image to be filtered. Both arrays must have equal shape.
    radius : int
        Window radius; the box kernel is (2 * radius + 1) square.
    eps : float
        Regularisation strength. Larger values produce smoother output.

    Returns
    -------
    np.ndarray
        Filtered image, float64, same shape as src.
    """
    if guide.shape != src.shape:
        raise ValueError("guide and src must have the same shape")
    guide = guide.astype(np.float64, copy=False)
    src = src.astype(np.float64, copy=False)
    ksize = (2 * int(radius) + 1, 2 * int(radius) + 1)
    border = cv2.BORDER_REFLECT

    mean_i = cv2.boxFilter(guide, cv2.CV_64F, ksize, borderType=border)
    mean_p = cv2.boxFilter(src, cv2.CV_64F, ksize, borderType=border)
    corr_i = cv2.boxFilter(guide * guide, cv2.CV_64F, ksize, borderType=border)
    corr_ip = cv2.boxFilter(guide * src, cv2.CV_64F, ksize, borderType=border)

    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + float(eps))
    b = mean_p - a * mean_i

    mean_a = cv2.boxFilter(a, cv2.CV_64F, ksize, borderType=border)
    mean_b = cv2.boxFilter(b, cv2.CV_64F, ksize, borderType=border)
    return mean_a * guide + mean_b


def estimate_illumination(img_bgr: np.ndarray, cfg: RetinexConfig) -> np.ndarray:
    """Return the smoothed illumination map in [eps, 1]."""
    x = img_bgr.astype(np.float64) / 255.0
    light = np.max(x, axis=2)
    light = guided_filter(light, light, radius=cfg.radius, eps=cfg.eps)
    return np.clip(light, 1e-4, 1.0)


def adaptive_gamma(light: np.ndarray, cfg: RetinexConfig) -> float:
    """Choose gamma from the mean illumination.

    A dark image (small mean) gets a small gamma, i.e. a stronger lift.
    The offset adds a small global brightening on top of the adaptive term.
    """
    mean_l = float(np.clip(light.mean(), 1e-3, 0.999))
    gamma = float(np.log(cfg.gamma_target) / np.log(mean_l))
    gamma = float(np.clip(gamma, cfg.gamma_lo, cfg.gamma_hi))
    return gamma + float(cfg.gamma_offset)


def enhance(img_bgr: np.ndarray, cfg: RetinexConfig | None = None,
            return_illumination: bool = False):
    """Enhance a BGR uint8 image.

    Parameters
    ----------
    img_bgr : np.ndarray
        BGR uint8 image.
    cfg : RetinexConfig, optional
        Hyper-parameters. Defaults are used when omitted.
    return_illumination : bool
        If True, also return the illumination map as a float array.

    Returns
    -------
    np.ndarray or tuple
        Enhanced BGR uint8 image, optionally with the illumination map.
    """
    if cfg is None:
        cfg = RetinexConfig()
    if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must be an H x W x 3 BGR image")

    x = img_bgr.astype(np.float64) / 255.0
    light = np.max(x, axis=2)
    if cfg.illumination_smoother == "guided":
        light = guided_filter(light, light, radius=cfg.radius, eps=cfg.eps)
    elif cfg.illumination_smoother == "gaussian":
        sigma = max(cfg.radius / 2.0, 1.0)
        light = cv2.GaussianBlur(light, (0, 0), sigma)
    else:
        raise ValueError(f"unknown illumination smoother: {cfg.illumination_smoother}")
    light = np.clip(light, 1e-4, 1.0)

    reflectance = np.clip(x / light[..., None], 0.0, 1.0)

    gamma = adaptive_gamma(light, cfg) if cfg.adaptive_gamma else float(cfg.gamma_fixed)
    light_hat = np.power(light, gamma)

    # Highlight-safe normalisation: map a high percentile to a fixed level so
    # that very dark inputs do not become washed out after the gamma lift.
    hi = float(np.percentile(light_hat, cfg.highlight_percentile))
    if hi <= 1e-6:
        hi = 1.0
    light_hat = np.clip(light_hat / hi * cfg.highlight_scale, 0.0, 1.0)

    out = np.power(reflectance, cfg.alpha) * light_hat[..., None]

    # Chroma control: interpolate towards the per-pixel mean luminance. This
    # removes the strong colour amplification that plain Retinex produces.
    gray = out.mean(axis=2, keepdims=True)
    out = np.clip(gray + cfg.saturation * (out - gray), 0.0, 1.0)

    # Mild gray-world white balance (strength 0 disables it).
    if cfg.white_balance > 0:
        channel_mean = out.reshape(-1, 3).mean(axis=0)
        gain = (channel_mean.mean() / np.maximum(channel_mean, 1e-6)) ** cfg.white_balance
        out = np.clip(out * gain, 0.0, 1.0)

    out_u8 = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)

    if cfg.denoise_h > 0:
        out_u8 = cv2.fastNlMeansDenoisingColored(
            out_u8, None, float(cfg.denoise_h), float(cfg.denoise_h), 7, 21
        )

    if return_illumination:
        return out_u8, light
    return out_u8


def illumination_vis(light: np.ndarray) -> np.ndarray:
    """Convert a float illumination map to a displayable uint8 heatmap."""
    lo, hi = float(light.min()), float(light.max())
    if hi - lo < 1e-6:
        norm = np.zeros_like(light)
    else:
        norm = (light - lo) / (hi - lo)
    gray = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
