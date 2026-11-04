# -*- coding: utf-8 -*-
"""Build a silent 1280x720 demo video from the final results.

Usage:
    python scripts/make_video.py
Output:
    report/演示视频.mp4
    report/视频封面.png
    report/视频预览.png
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RetinexConfig
from src.io_utils import imread, imwrite
from src.metrics import psnr, ssim
from src.retinex import enhance

W, H, FPS = 1280, 720, 24
BG = (10, 15, 28)
PANEL = (18, 26, 44)
TEXT = (235, 240, 248)
MUTED = (148, 163, 184)
GREEN = (16, 185, 129)
BLUE = (59, 130, 246)
AMBER = (245, 158, 11)
FONT_REG = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"

DATA = ROOT / "data" / "lolv2" / "Test"
OUT = ROOT / "report" / "演示视频.mp4"
POSTER = ROOT / "report" / "视频封面.png"
PREVIEW = ROOT / "report" / "视频预览.png"


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size=size)


def base_frame() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for x in range(0, W, 64):
        d.line([(x, 0), (x, H)], fill=(17, 24, 40), width=1)
    for y in range(0, H, 64):
        d.line([(0, y), (W, y)], fill=(17, 24, 40), width=1)
    d.rectangle([0, 0, 8, H], fill=GREEN)
    return img, d


def draw_text(d, xy, text, font, fill=TEXT, anchor="la", spacing=6):
    d.multiline_text(xy, text, font=font, fill=fill, anchor=anchor, spacing=spacing)


def fit_to_box(img_bgr: np.ndarray, box: tuple[int, int, int, int]) -> Image.Image:
    x, y, w, h = box
    im = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    scale = min(w / im.width, h / im.height)
    im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                   Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def paste_rounded(canvas: Image.Image, im: Image.Image,
                  box: tuple[int, int, int, int], radius: int = 14):
    x, y, w, h = box
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    canvas.paste(im, (x, y), mask)


def chip(d: ImageDraw.ImageDraw, xy, text, font, fg=TEXT, bg=(0, 0, 0)):
    x, y = xy
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.rectangle([x, y, x + tw + 18, y + th + 12], fill=bg)
    d.text((x + 9, y + 5), text, font=font, fill=fg)


def title_slide() -> Image.Image:
    img, d = base_frame()
    draw_text(d, (90, 190), "低光照图像增强", get_font(58, True), TEXT)
    draw_text(d, (94, 280), "Guided-Filter Retinex  ·  无需训练  ·  CPU 运行", get_font(26), MUTED)
    d.rectangle([94, 336, 520, 340], fill=GREEN)
    draw_text(d, (94, 380), "LOLv2-real  ·  100 对测试图像  ·  PSNR 19.76 dB", get_font(22), GREEN)
    draw_text(d, (94, 440), "图像处理课程大作业", get_font(18), MUTED)
    draw_text(d, (W - 90, H - 70), "before → after", get_font(18), MUTED, anchor="ra")
    return img


def problem_slide() -> Image.Image:
    img, d = base_frame()
    draw_text(d, (80, 50), "问题：真实暗图，亮度只有 GT 的约 1/6", get_font(34, True), TEXT)
    gap = 40
    box_w, box_h = 390, 260
    x1 = (W - 2 * box_w - gap) // 2
    y1 = 150
    inp = imread(DATA / "Input" / "00690.png")
    gt = imread(DATA / "GT" / "00690.png")
    paste_rounded(img, fit_to_box(inp, (0, 0, box_w, box_h)), (x1, y1, box_w, box_h))
    paste_rounded(img, fit_to_box(gt, (0, 0, box_w, box_h)), (x1 + box_w + gap, y1, box_w, box_h))
    chip(d, (x1 + 10, y1 + 10), "Input  mean 14.5", get_font(16), fg=TEXT, bg=(0, 0, 0))
    chip(d, (x1 + box_w + gap + 10, y1 + 10), "GT  mean 94.5", get_font(16), fg=TEXT, bg=(0, 0, 0))
    y2 = 470
    chips = ["暗部噪声被放大", "室内灯光偏色", "高光容易截断"]
    f = get_font(20)
    for i, s in enumerate(chips):
        x = 80 + i * 390
        d.rounded_rectangle([x, y2, x + 350, y2 + 62], radius=12, fill=PANEL)
        d.text((x + 18, y2 + 18), s, font=f, fill=TEXT)
    draw_text(d, (80, 570), "目标：提亮、保细节、抑噪声、不偏色", get_font(24), GREEN)
    return img


def pipeline_slide() -> Image.Image:
    img, d = base_frame()
    draw_text(d, (80, 45), "方法：引导滤波 Retinex + 曝光校准", get_font(34, True), TEXT)
    p = Image.open(ROOT / "results" / "figures" / "pipeline.png").convert("RGB")
    p.thumbnail((1120, 270), Image.Resampling.LANCZOS)
    px = (W - p.width) // 2
    img.paste(p, (px, 130))
    d.rectangle([px, 130, px + p.width, 130 + p.height], outline=(51, 65, 85), width=1)
    labels = ["自引导滤波 O(N)", "自适应 Gamma", "色度 + 白平衡", "曝光校准 + NLM"]
    for i, s in enumerate(labels):
        x = 80 + i * 290
        d.rounded_rectangle([x, 470, x + 270, 530], radius=12, fill=PANEL)
        d.text((x + 18, 487), s, font=get_font(18), fill=TEXT)
    draw_text(d, (80, 585), "核心改动：用盒式滤波替代 LIME 的稀疏线性求解，并补上曝光与噪声控制。",
              get_font(21), MUTED)
    return img


def compare_base(name: str, ps: float, ss: float, out_mode: bool) -> Image.Image:
    img, d = base_frame()
    draw_text(d, (80, 34), f"{name.replace('.png', '')}  ·  低光照增强", get_font(30, True), TEXT)
    chip(d, (W - 240, 30), "Ours" if out_mode else "Input", get_font(18),
         fg=BG if out_mode else TEXT, bg=GREEN if out_mode else PANEL)
    return img


def compare_slide(name: str, ps: float, ss: float):
    inp = imread(DATA / "Input" / name)
    gt = imread(DATA / "GT" / name)
    cfg = RetinexConfig(**json.loads((ROOT / "results" / "best_config.json").read_text(encoding="utf-8")))
    ours = enhance(inp, cfg)
    box = (220, 100, 840, 560)
    inp_box = fit_to_box(inp, box)
    out_box = fit_to_box(ours, box)
    # input hold
    img = compare_base(name, ps, ss, False)
    d = ImageDraw.Draw(img)
    paste_rounded(img, inp_box, box)
    draw_text(d, (80, 670), "Input  ·  低光照原图", get_font(20), MUTED)
    yield img, 0.8
    # wipe
    n = int(1.5 * FPS)
    for i in range(n):
        t = (i + 1) / n
        cut = int(box[2] * t)
        frame = compare_base(name, ps, ss, True)
        df = ImageDraw.Draw(frame)
        paste_rounded(frame, inp_box, box)
        crop = out_box.crop((0, 0, cut, box[3]))
        frame.paste(crop, (box[0], box[1]))
        df.rectangle([box[0] + cut - 2, box[1], box[0] + cut + 2, box[1] + box[3]], fill=GREEN)
        df.text((box[0] + 12, box[1] + 12), "Input", font=get_font(16), fill=TEXT)
        df.text((box[0] + cut + 10, box[1] + 12), "Ours", font=get_font(16), fill=GREEN)
        yield frame, 1.0 / FPS
    # output hold
    img = compare_base(name, ps, ss, True)
    d = ImageDraw.Draw(img)
    paste_rounded(img, out_box, box)
    draw_text(d, (box[0], 670), f"Ours  ·  PSNR {ps:.2f} dB  ·  SSIM {ss:.3f}", get_font(20), GREEN)
    yield img, 2.4


def results_slide() -> Image.Image:
    img, d = base_frame()
    draw_text(d, (80, 45), "结果：PSNR 最高，速度约为 LIME 的 6 倍", get_font(32, True), TEXT)
    rows = [
        ("方法", "PSNR (dB)", "SSIM", "时间"),
        ("LIME", "17.14", "0.6465", "3456 ms"),
        ("Gamma", "17.33", "0.7178", "0.6 ms"),
        ("Ours", "19.76", "0.6727", "588 ms"),
    ]
    x0, y0, row_h = 150, 150, 58
    cols = [0, 300, 520, 720]
    d.rounded_rectangle([x0 - 20, y0 - 20, 1130, y0 + row_h * len(rows) + 10], radius=16, fill=PANEL)
    for r, row in enumerate(rows):
        y = y0 + r * row_h
        if r == 3:
            d.rounded_rectangle([x0 - 8, y - 6, 1110, y + row_h - 10], radius=8,
                                fill=(9, 60, 45))
        for c, val in enumerate(row):
            f = get_font(22, bold=(r == 0 or (r == 3 and c == 0)))
            color = GREEN if (r == 3 and c > 0) else TEXT if r == 3 else MUTED if r == 0 else TEXT
            d.text((x0 + cols[c], y + 14), val, font=f, fill=color)
    cards = [
        ("+2.62 dB", "PSNR vs LIME", GREEN),
        ("5.9×", "速度快于 LIME", BLUE),
        ("0.6727", "SSIM（高于 LIME）", AMBER),
    ]
    for i, (big, small, color) in enumerate(cards):
        x = 110 + i * 370
        d.rounded_rectangle([x, 500, x + 330, 620], radius=14, fill=PANEL)
        d.text((x + 22, 515), big, font=get_font(34, True), fill=color)
        d.text((x + 22, 570), small, font=get_font(17), fill=MUTED)
    draw_text(d, (110, 655), "100 对测试图像 · 600×400 · CPU 单线程计时", get_font(18), MUTED)
    return img


def closing_slide() -> Image.Image:
    img, d = base_frame()
    draw_text(d, (90, 190), "复现与提交", get_font(44, True), TEXT)
    d.rectangle([94, 270, 450, 274], fill=GREEN)
    draw_text(d, (94, 320), "python app/gui.py", get_font(24), GREEN)
    draw_text(d, (94, 370), "python enhance.py 图片路径 -o 输出.png", get_font(20), MUTED)
    draw_text(d, (94, 440), "代码、报告、结果图表和复现脚本见提交包。", get_font(22), TEXT)
    draw_text(d, (94, 500), "LOLv2-real  ·  Guided-Filter Retinex  ·  2026", get_font(18), MUTED)
    draw_text(d, (W - 90, H - 70), "Thanks", get_font(22, True), GREEN, anchor="ra")
    return img


def main() -> None:
    csv_path = ROOT / "results" / "metrics.csv"
    metrics = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] == "ours":
                metrics[row["image"]] = (float(row["psnr"]), float(row["ssim"]))
    names = ["00780.png", "00699.png", "00695.png", "00690.png"]
    for n in names:
        if n not in metrics:
            raise SystemExit(f"missing metrics for {n}")

    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise SystemExit("VideoWriter failed to open")
    frames = 0

    def write(img: Image.Image, seconds: float):
        nonlocal frames
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for _ in range(int(round(seconds * FPS))):
            writer.write(arr)
            frames += 1

    t0 = time.perf_counter()
    poster = title_slide()
    poster.save(POSTER)
    write(poster, 4.0)
    write(problem_slide(), 7.0)
    write(pipeline_slide(), 7.0)

    preview_saved = False
    for n in names:
        ps, ss = metrics[n]
        for img, sec in compare_slide(n, ps, ss):
            write(img, sec)
            if not preview_saved and n == "00780.png" and sec > 1.0:
                img.save(PREVIEW)
                preview_saved = True

    write(results_slide(), 9.0)
    write(closing_slide(), 5.0)
    writer.release()
    print(f"wrote {OUT}  frames={frames}  time={time.perf_counter()-t0:.1f}s")
    print(f"poster={POSTER}  preview={PREVIEW}")


if __name__ == "__main__":
    main()
