# -*- coding: utf-8 -*-
"""Fill the numeric placeholders in README.md from results/summary.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))


def get(method: str, key: str) -> float:
    for item in summary["methods"]:
        if item["method"] == method:
            return float(item[key]["mean"])
    raise KeyError(method)


lime_psnr, lime_ssim, lime_time = get("lime", "psnr"), get("lime", "ssim"), get("lime", "time_ms")
ours_psnr, ours_ssim, ours_time = get("ours", "psnr"), get("ours", "ssim"), get("ours", "time_ms")
replacements = {
    "__PSNR_LIME__": f"{lime_psnr:.2f}",
    "__SSIM_LIME__": f"{lime_ssim:.4f}",
    "__TIME_LIME__": f"{lime_time:.0f}",
    "__PSNR_OURS__": f"{ours_psnr:.2f}",
    "__SSIM_OURS__": f"{ours_ssim:.4f}",
    "__TIME_OURS__": f"{ours_time:.0f}",
    "__DELTA_PSNR__": f"{ours_psnr - lime_psnr:+.2f}",
    "__SPEEDUP__": f"{lime_time / max(ours_time, 1e-6):.1f}",
}
path = ROOT / "README.md"
text = path.read_text(encoding="utf-8")
for key, value in replacements.items():
    text = text.replace(key, value)
path.write_text(text, encoding="utf-8")
print("README updated:", replacements)
