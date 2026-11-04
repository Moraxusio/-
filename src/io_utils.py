# -*- coding: utf-8 -*-
"""Unicode-safe image I/O helpers (important on Chinese Windows paths)."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


def imread(path: str | os.PathLike, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image from a path that may contain non-ASCII characters."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"cannot read image: {path}")
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f"cannot decode image: {path}")
    return img


def imwrite(path: str | os.PathLike, img: np.ndarray) -> None:
    """Write an image to a path that may contain non-ASCII characters."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        ext = ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"cannot encode image: {path}")
    buf.tofile(str(path))
