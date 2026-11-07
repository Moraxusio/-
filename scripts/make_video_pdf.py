# -*- coding: utf-8 -*-
"""Create the one-page PDF that accompanies the optional demo video."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / "report" / "视频预览.png"
OUT = ROOT / "report" / "可选视频说明.pdf"
W, H = 1240, 1754  # A4 at 150 dpi
FONT = r"C:\Windows\Fonts\msyh.ttc"
BOLD = r"C:\Windows\Fonts\msyhbd.ttc"

img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)


def f(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size=size)


d.text((90, 85), "可选视频说明", font=f(42, True), fill=(17, 24, 39))
d.text((90, 150), "姓名：樊恩旭    学号：20243984", font=f(21), fill=(55, 65, 81))
d.text((90, 190), "课程：图像处理    日期：2026 年 11 月", font=f(18), fill=(107, 114, 128))

preview = Image.open(PREVIEW).convert("RGB")
preview.thumbnail((1040, 585), Image.Resampling.LANCZOS)
px = (W - preview.width) // 2
py = 240
img.paste(preview, (px, py))
d.rectangle([px, py, px + preview.width, py + preview.height], outline=(209, 213, 219), width=2)

y = py + preview.height + 45
d.text((90, y), "视频文件：2.可选视频.mp4", font=f(22, True), fill=(17, 24, 39))
d.text((90, y + 40), "时长约 51 秒，1280×720，24 fps，无声；总大小约 10 MB。", font=f(18), fill=(55, 65, 81))
d.text((90, y + 80), "内容：标题 → 问题 → 方法流程 → 4 组前后对比 → 结果 → 复现。", font=f(18), fill=(55, 65, 81))
d.text((90, y + 120), "说明：因视频文件较大，演示视频单独放在同目录的 MP4 文件中；本页用于说明与预览。",
       font=f(17), fill=(107, 114, 128))

# footer
d.line([(90, H - 130), (W - 90, H - 130)], fill=(209, 213, 219), width=1)
d.text((90, H - 105), "Low-light Image Enhancement · Guided-Filter Retinex", font=f(15), fill=(107, 114, 128))
d.text((W - 90, H - 105), "1 / 1", font=f(15), fill=(107, 114, 128), anchor="ra")

img.save(OUT, "PDF", resolution=150.0)
print("saved", OUT, OUT.stat().st_size)
