# -*- coding: utf-8 -*-
"""Configuration objects for the proposed method."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RetinexConfig:
    """Hyper-parameters of the guided-filter Retinex enhancer.

    The defaults were selected on a 30-pair validation split of LOLv2-real
    (see scripts/tune_validation.py); the 100-pair test split is untouched.
    """

    radius: int = 16
    eps: float = 0.02
    illumination_smoother: str = "guided"  # "guided" or "gaussian"
    adaptive_gamma: bool = True
    gamma_fixed: float = 0.6
    gamma_target: float = 0.45
    gamma_lo: float = 0.35
    gamma_hi: float = 0.85
    gamma_offset: float = 0.12
    alpha: float = 1.0
    saturation: float = 0.8
    white_balance: float = 0.5
    denoise_h: float = 4.0
    highlight_percentile: float = 99.5
    highlight_scale: float = 0.96

    def to_dict(self) -> dict:
        return asdict(self)
