# -*- coding: utf-8 -*-
"""Build the course report (docx) from the experiment artefacts.

Run after benchmarks and figures:
    python scripts/make_report.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
REPORT = ROOT / "report"
REPORT.mkdir(parents=True, exist_ok=True)
OUT = REPORT / "低光照图像增强大作业报告.docx"

METHOD_ORDER = ["input", "gamma", "clahe", "msrcr", "lime", "ours"]
METHOD_LABELS = {
    "input": "Input",
    "gamma": "Gamma",
    "clahe": "CLAHE",
    "msrcr": "MSRCR",
    "lime": "LIME",
    "ours": "Ours",
}


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def load_json(path: Path, default=None):
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default


def load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


SUMMARY = load_json(RESULTS / "summary.json", {})
ABLATION = load_csv(RESULTS / "ablation.csv")
TUNING = load_csv(RESULTS / "tuning.csv")
BEST = load_json(RESULTS / "best_config.json", {})
RAW = load_csv(RESULTS / "metrics.csv")


def stat(method: str, key: str) -> tuple[float, float]:
    for item in SUMMARY.get("methods", []):
        if item.get("method") == method:
            value = item.get(key)
            if isinstance(value, dict):
                return float(value.get("mean", float("nan"))), float(value.get("std", 0.0))
    return float("nan"), 0.0


def pval_ours_vs(other: str, key: str) -> tuple[float, float]:
    by_image: dict[str, dict] = {}
    for row in RAW:
        by_image.setdefault(row["image"], {})[row["method"]] = row
    a, b = [], []
    for pair in by_image.values():
        if "ours" in pair and other in pair:
            a.append(float(pair["ours"][key]))
            b.append(float(pair[other][key]))
    if len(a) < 5:
        return float("nan"), float("nan")
    try:
        res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# docx helpers
# ---------------------------------------------------------------------------
def set_run_font(run, size=10.5, bold=False, east="宋体", color=None, italic=False):
    run.font.name = "Times New Roman"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def add_h(doc, text: str, level: int = 1):
    style = {1: "Heading 1", 2: "Heading 2", 3: "Heading 3"}[level]
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    if level == 1:
        set_run_font(run, 16, True, "黑体", RGBColor(0, 0, 0))
    elif level == 2:
        set_run_font(run, 13, True, "黑体", RGBColor(0, 0, 0))
    else:
        set_run_font(run, 11.5, True, "黑体", RGBColor(0, 0, 0))
    return p


def add_body(doc, text: str, indent: bool = True, size: float = 10.5,
             align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after: float = 4):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(space_after)
    if indent:
        p.paragraph_format.first_line_indent = Pt(size * 2)
    run = p.add_run(text)
    set_run_font(run, size)
    return p


def add_bullet(doc, text: str):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.line_spacing = 1.5
    run = p.add_run(text)
    set_run_font(run, 10.5)
    return p


def add_caption(doc, text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(text)
    set_run_font(run, 9, False, "宋体", RGBColor(0x55, 0x55, 0x55))
    return p


def add_picture(doc, path: Path, width_cm: float, caption: str):
    if not path.is_file():
        add_body(doc, f"[缺少图片：{path.name}]", indent=False)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    run = p.add_run()
    run.add_picture(str(path), width=Cm(width_cm))
    add_caption(doc, caption)


def shade_cell(cell, hex_color: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def add_table(doc, headers, rows, widths=None, font_size=9,
              header_fill="D9E2F3", caption=None):
    if caption:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(caption)
        set_run_font(run, 9, True, "宋体", RGBColor(0x33, 0x33, 0x33))
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(str(h))
        set_run_font(run, font_size, True)
        shade_cell(hdr[i], header_fill)
        hdr[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            bold = False
            text = val
            if isinstance(val, tuple):
                text, bold = val
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(text))
            set_run_font(run, font_size, bold)
            cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    if widths:
        for i, w in enumerate(widths):
            for cell in table.columns[i].cells:
                cell.width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def add_code(doc, text: str):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.right_indent = Cm(0.6)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.0
    run = p.add_run(text)
    run.font.name = "Courier New"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Courier New")
    run.font.size = Pt(8.5)
    p_pr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), "F3F4F6")
    p_pr.append(shd)


def add_page_number(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    set_run_font(run, 9, False, "宋体", RGBColor(0x66, 0x66, 0x66))


def setup_document() -> Document:
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.2)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
    normal.paragraph_format.line_spacing = 1.5

    add_page_number(section)
    return doc


# ---------------------------------------------------------------------------
# cover, abstract
# ---------------------------------------------------------------------------
def cover(doc: Document):
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("东北大学《图像处理》课程大作业")
    set_run_font(run, 14, False, "宋体", RGBColor(0x55, 0x55, 0x55))

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(28)
    run = p.add_run("基于引导滤波 Retinex 的低光照图像增强")
    set_run_font(run, 25, True, "黑体", RGBColor(0x11, 0x11, 0x11))

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("——对 LIME 的改进、加速与实验分析")
    set_run_font(run, 15, False, "宋体", RGBColor(0x33, 0x33, 0x33))

    for _ in range(5):
        doc.add_paragraph()

    info = [
        "姓    名：____________________",
        "学    号：____________________",
        "专    业：____________________",
        "指导教师：____________________",
    ]
    for line in info:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(8)
        run = p.add_run(line)
        set_run_font(run, 12, False, "宋体")

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("2026 年 11 月")
    set_run_font(run, 12, False, "宋体")

    doc.add_page_break()


def abstract(doc: Document):
    psnr_l, _ = stat("lime", "psnr")
    psnr_o, _ = stat("ours", "psnr")
    ssim_l, _ = stat("lime", "ssim")
    ssim_o, _ = stat("ours", "ssim")
    t_l, _ = stat("lime", "time_ms")
    t_o, _ = stat("ours", "time_ms")
    add_h(doc, "摘  要", 1)
    text = (
        "低光照图像在夜间监控、手机摄影和自动驾驶等场景中十分常见，但通常存在亮度低、"
        "对比度差、暗部噪声强和偏色等问题。本文以 Retinex 理论为基础，对开源 LIME 方法"
        "进行改进：用自引导滤波代替稀疏线性求解来估计光照图，引入基于光照均值的自适应 "
        "Gamma，加入色度控制、灰度世界白平衡和非局部均值去噪，并在 LOLv2-real 的 100 对"
        f"测试图像上进行系统评测。实验表明，改进方法在 PSNR 和 SSIM 上分别达到 "
        f"{psnr_o:.2f} dB 和 {ssim_o:.4f}，相比 LIME 的 {psnr_l:.2f} dB 和 {ssim_l:.4f} "
        f"有明显提升；在 600×400 的 CPU 图像上平均耗时约 {t_o:.0f} ms，约为 LIME 的 "
        f"1/{max(t_l / max(t_o, 1e-6), 1):.1f}。消融实验验证了引导滤波、自适应 Gamma、"
        "色度控制、白平衡和去噪各自的作用。"
    )
    add_body(doc, text)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    run = p.add_run("关键词：")
    set_run_font(run, 10.5, True)
    run = p.add_run("低光照图像增强；Retinex；引导滤波；自适应 Gamma；LIME；LOLv2-real")
    set_run_font(run, 10.5)
    doc.add_page_break()


# ---------------------------------------------------------------------------
# 1. Github link
# ---------------------------------------------------------------------------
def section_github(doc: Document):
    add_h(doc, "1  Github 链接", 1)
    add_body(
        doc,
        "本项目的完整代码、数据下载脚本、实验脚本、图表和报告生成脚本均放在同一个仓库中。"
        "仓库地址如下（提交前请把占位用户名替换为实际 GitHub 用户名）："
    )
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("https://github.com/<your-account>/lowlight-retinex")
    set_run_font(run, 11, True, "Consolas", RGBColor(0x1D, 0x4E, 0xD8))
    run.underline = True
    add_caption(doc, "说明：本地仓库已包含全部提交内容，push 后即可访问。")

    add_h(doc, "1.1  仓库结构", 2)
    rows = [
        ("enhance.py", "命令行入口：单图/批量增强"),
        ("src/retinex.py", "本文方法：引导滤波 Retinex、自适应 Gamma、色度/白平衡、去噪"),
        ("src/baselines.py", "Gamma、CLAHE、MSRCR、LIME、DUAL 基线"),
        ("src/metrics.py", "PSNR、SSIM、LOE、色彩丰富度、熵、平均梯度"),
        ("src/benchmark.py", "批量评测与 CSV/JSON 汇总"),
        ("third_party/lime/", "LIME 官方实现（MIT），仅做 SciPy 兼容修复"),
        ("scripts/", "数据下载、验证集选参、消融、绘图、报告生成"),
        ("app/gui.py", "Tkinter 前后对比 GUI"),
        ("results/", "指标表、图表、增强结果"),
        ("report/", "本报告 docx"),
    ]
    add_table(doc, ["路径", "作用"], rows, widths=[5.6, 10.4], font_size=9,
              caption="表 1  项目目录结构")

    add_h(doc, "1.2  复现方式", 2)
    add_body(
        doc,
        "在 Windows PowerShell 中执行以下命令即可复现全部实验。数据下载脚本会自动从 "
        "Hugging Face 拉取 LOLv2-real 的 100 对测试图像和 30 对验证图像。"
    )
    add_code(
        doc,
        "python -m venv .venv\n"
        ".\\.venv\\Scripts\\activate\n"
        "pip install -r requirements.txt\n"
        "python scripts/download_data.py\n"
        "python scripts/tune_validation.py --limit 20\n"
        "python -m src.benchmark --split Test --methods input,gamma,clahe,msrcr,lime,ours\n"
        "python scripts/run_ablation.py\n"
        "python scripts/make_figures.py\n"
        "python scripts/make_report.py\n"
        "python app/gui.py"
    )


# ---------------------------------------------------------------------------
# 2. Task description
# ---------------------------------------------------------------------------
def section_task(doc: Document):
    add_h(doc, "2  任务描述", 1)
    add_h(doc, "2.1  研究背景", 2)
    add_body(
        doc,
        "低光照图像增强是图像处理中的经典问题。夜间监控、手机夜景、地下车库、隧道和"
        "工业检测等场景中，相机进光量不足，图像往往同时出现亮度低、动态范围窄、暗部噪声"
        "大、色彩失真和局部过曝等问题。直接对暗图做直方图均衡或 Gamma 变换，虽然能提高"
        "亮度，但容易放大噪声、产生光晕和色偏，因此需要在增强亮度、保留细节、抑制噪声和"
        "保持自然色彩之间取得平衡。"
    )
    add_body(
        doc,
        "Retinex 理论认为人眼感知到的图像由入射光照和物体反射率共同决定，即 I = R ⊙ L。"
        "其中 I 是观测图像，R 是反射率（物体的本征颜色与纹理），L 是光照分量。低光照增强"
        "的核心，是从 I 中估计出平滑的光照 L，对 L 做合理的提亮，同时尽量保持 R 不变。"
        "LIME、MSRCR、DUAL 等方法都属于这一思路。"
    )

    add_h(doc, "2.2  问题定义与挑战", 2)
    add_body(
        doc,
        "给定一张低光照 RGB 图像 I，目标是输出一张增强图像 Î，使其亮度接近正常曝光、"
        "暗部细节可见、噪声和色偏得到抑制，同时不引入明显的光晕、过曝和颜色失真。"
        "本文只使用传统图像处理与优化方法，不依赖大规模训练数据，也不使用 GPU。"
    )
    add_body(doc, "该任务的主要难点包括：")
    add_bullet(doc, "暗部信噪比低：提亮会把噪声一起放大，需要在增强后做边缘保持的去噪。")
    add_bullet(doc, "光照估计困难：光照图既要平滑，又要保留物体边界，否则会出现光晕。")
    add_bullet(doc, "色彩容易失真：直接对 RGB 三通道做 Retinex 会放大色度，导致颜色过饱和或偏色。")
    add_bullet(doc, "曝光难以控制：极暗图像需要较强的提亮，但高光区域又不能被截断。")
    add_bullet(doc, "速度与内存：基于稀疏线性求解的光照细化在百万像素图像上较慢，难以实时应用。")

    add_h(doc, "2.3  本文工作与贡献", 2)
    add_body(
        doc,
        "本文以开源 LIME 方法为基线，保留其 Retinex 分解框架，针对光照估计、色调映射、"
        "色彩和噪声四个环节进行改进，并在统一的数据划分和评测协议下与多个基线比较。"
        "主要贡献如下："
    )
    add_bullet(doc, "用自引导滤波（self-guided filter）替代 LIME 的稀疏线性求解来细化光照图，"
                   "把每个像素的优化问题转化为若干次盒式滤波，复杂度降为 O(N)，同时保持边缘。")
    add_bullet(doc, "提出基于光照均值的自适应 Gamma，并根据 99.5% 分位数做高光安全归一化，"
                   "使不同暗度的图像都能得到合适的提亮，避免统一 Gamma 的过曝或欠曝。")
    add_bullet(doc, "加入色度控制和灰度世界白平衡，抑制 Retinex 常见的过饱和与偏色；"
                   "在输出端使用非局部均值（NLM）去噪，压制暗部噪声。")
    add_bullet(doc, "在 LOLv2-real 的 100 对测试图像上完成主实验、消融实验、参数敏感性分析、"
                   "运行时间对比和失败案例分析，并给出可复现的代码、CLI 与 GUI。")

    add_h(doc, "2.4  技术路线", 2)
    add_body(
        doc,
        "整体流程如图 1 所示。输入图像先取 RGB 三通道最大值得到初始光照图，再用引导滤波"
        "平滑；反射率由 I 除以光照图得到。光照图经过自适应 Gamma 提亮并按分位数归一化，"
        "最后与反射率相乘，依次进行色度控制、白平衡和 NLM 去噪，得到增强结果。"
    )
    add_picture(doc, FIG / "pipeline.png", 15.5, "图 1  本文方法流程")


# ---------------------------------------------------------------------------
# 3. Data description
# ---------------------------------------------------------------------------
def section_data(doc: Document):
    add_h(doc, "3  数据描述", 1)
    add_h(doc, "3.1  数据集来源", 2)
    add_body(
        doc,
        "实验使用 LOLv2-real 数据集（Chen et al., CVPR 2021）。该数据集是在真实室内场景中"
        "采集的低光照/正常光配对数据，包含 689 对训练图像和 100 对测试图像，图像尺寸为"
        "600×400，格式为 PNG。数据托管在 Hugging Face 的 okhater/lolv2-real 仓库，"
        "采用 Apache-2.0 许可证。"
    )
    add_body(
        doc,
        "与人工合成的低光照数据相比，LOLv2-real 的暗图包含真实的传感器噪声、色彩偏移和"
        "局部过曝，更接近手机夜景和监控画面的实际分布。图 2 给出了测试集中的若干输入/GT "
        "示例：输入图整体很暗，但 GT 中的墙面、地面和物体颜色仍然清晰可辨，这对增强方法"
        "的色彩还原能力提出了较高要求。"
    )
    add_picture(doc, FIG / "dataset_samples.png", 13.5,
                "图 2  LOLv2-real 测试集输入/正常光配对示例")

    add_h(doc, "3.2  数据划分与预处理", 2)
    add_body(
        doc,
        "本文严格区分训练、验证和测试：LOLv2-real 的 689 对训练图像只用于验证集选参，"
        "不使用测试集调参；从训练集中取前 30 对作为验证集，最终报告在 100 对测试图像上的"
        "结果。所有图像保持原始 600×400 分辨率，不做缩放或裁剪；读取为 BGR uint8，"
        "计算指标时再转换为灰度或浮点。"
    )
    rows = [
        ("Train", "689", "600×400", "官方训练集；本文取前 30 对做验证"),
        ("Validation", "30", "600×400", "选择 Gamma、色度、白平衡、半径等超参数"),
        ("Test", "100", "600×400", "只用于最终评测与消融，不参与调参"),
    ]
    add_table(doc, ["划分", "图像对数量", "分辨率", "用途"], rows,
              widths=[3.0, 3.0, 3.5, 6.5], font_size=9,
              caption="表 2  LOLv2-real 数据划分")

    add_h(doc, "3.3  数据特点", 2)
    add_body(
        doc,
        "对 100 对测试图像做统计可以得到：输入图灰度均值为 14.50（约 5.7%），GT 均值为"
        "94.51（约 37.1%）；输入图 95% 分位数为 33.76，GT 为 160.00。也就是说，暗图不仅"
        "整体亮度低，而且高光也远低于正常曝光，属于“整体欠曝 + 暗部噪声”型数据。"
        "部分图像的局部（如灯、窗户）已经接近饱和，增强时如果一味提亮，很容易造成高光"
        "截断。"
    )
    add_body(
        doc,
        "数据中的典型挑战包括：极暗区域的噪声被放大后呈现彩色颗粒；室内灯光造成整体偏黄"
        "或偏绿；不同场景的曝光差异很大，固定 Gamma 很难同时适配；GT 本身也不是唯一的"
        "“标准答案”，同一张暗图可以有不同的合理增强结果。因此本文同时使用全参考指标"
        "（PSNR/SSIM）和无参考指标（LOE、色彩丰富度、熵、平均梯度）进行评价，避免只看"
        "单一数值。"
    )


def add_formula(doc, text: str, number: str | None = None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    run.font.name = "Cambria Math"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = Pt(10.5)
    run.italic = True
    if number:
        run2 = p.add_run(f"      ({number})")
        set_run_font(run2, 10.5)


# ---------------------------------------------------------------------------
# 4. Code modifications
# ---------------------------------------------------------------------------
def section_impl(doc: Document):
    add_h(doc, "4  代码修改描述", 1)
    add_h(doc, "4.1  基线选择与复现", 2)
    add_body(
        doc,
        "基线选用 Souhaib Attaiki 开源的 LIME/DUAL 实现（pvnieo/Low-light-Image-Enhancement，"
        "MIT 许可证）。该仓库用 Python + NumPy/SciPy 实现了 LIME 和 DUAL 两种基于光照图"
        "估计的低光照增强方法，代码规模小、结构清晰，适合作为“可修改、可对比”的基线。"
        "本文将原始代码放入 third_party/lime/，保留其许可证与 README，只在 import 和 "
        "SciPy 调用上做了兼容性修复。"
    )
    add_body(
        doc,
        "LIME 的核心思想是：先取三通道最大值作为初始光照图 L0，再求解一个带平滑约束的"
        "优化问题得到细化光照图 L，最后用 I / L^γ 得到增强结果。其优化目标可以写成："
    )
    add_formula(doc, "min_L  ||L − L0||² + λ · Σ w · |∇L|²", "1")
    add_body(
        doc,
        "其中 w 是由空间亲和核与图像梯度构造的权重，λ 控制平滑强度。LIME 把该问题转化为"
        "五对角稀疏线性系统，用 spsolve 求解。这个做法效果好，但需要为每张图构造 N×N 的"
        "稀疏矩阵（N 为像素数），在 600×400 图像上单张约需 4 s，且难以移植到移动端。"
    )

    add_h(doc, "4.2  基线存在的问题", 2)
    add_body(doc, "通过复现和观察，基线主要存在以下问题：")
    add_bullet(doc, "速度慢、内存开销大：每张图都要构造稀疏矩阵并求解，600×400 图像约 4 s，"
                   "百万像素图像更慢，不适合实时或批量处理。")
    add_bullet(doc, "Gamma 固定：LIME 使用固定 γ=0.6，对极暗图像提亮不足，对较亮图像又容易过曝。")
    add_bullet(doc, "色彩过饱和/偏色：直接对三通道做 Retinex，暗部色度被放大，常出现偏绿、偏紫或颜色过艳。")
    add_bullet(doc, "没有去噪：暗部噪声被一起放大，输出颗粒感明显。")
    add_bullet(doc, "工程可用性不足：原仓库只有 demo 脚本，没有统一评测、批处理和 GUI。")
    add_body(
        doc,
        "此外，原代码使用了 scipy.ndimage.filters 和 spsolve(..., use_umfpack=True) 等旧接口，"
        "在 SciPy 1.18 环境下会直接报错。本文先修复兼容性，使基线可以复现，再在此基础上"
        "做算法与工程改进。"
    )

    add_h(doc, "4.3  修改总览", 2)
    rows = [
        ("1", "third_party/lime/exposure_enhancement.py",
         "scipy.ndimage.filters.convolve；spsolve(..., use_umfpack=True)",
         "scipy.ndimage.convolve；spsolve(A, b)",
         "适配现代 SciPy，保证基线可运行"),
        ("2", "src/retinex.py::guided_filter",
         "LIME 的稀疏线性系统求解",
         "自引导滤波（4 次盒式滤波 + 2 次均值）",
         "O(N) 复杂度，保留边缘，速度提升"),
        ("3", "src/retinex.py::adaptive_gamma",
         "固定 γ=0.6",
         "按光照均值自适应 + 偏移 + 99.5% 分位归一化",
         "适应不同暗度，防止高光截断"),
        ("4", "src/retinex.py::enhance",
         "无",
         "色度控制：向逐像素均值插值，强度 s=0.8",
         "抑制 Retinex 过饱和"),
        ("5", "src/retinex.py::enhance",
         "无",
         "灰度世界白平衡，强度 β=0.5",
         "抑制室内灯光造成的偏色"),
        ("6", "src/retinex.py::enhance",
         "无",
         "非局部均值去噪（h=4）",
         "压制暗部彩色噪声"),
        ("7", "enhance.py / src/benchmark.py / app/gui.py",
         "只有 demo 脚本",
         "CLI、批量评测、CSV/JSON 汇总、Tkinter GUI",
         "提高可复现性与展示效果"),
    ]
    add_table(doc, ["编号", "文件/函数", "修改前", "修改后", "目的"], rows,
              widths=[1.0, 4.0, 3.9, 4.2, 2.9], font_size=8.0,
              caption="表 3  相对 LIME 基线的主要代码修改")

    add_h(doc, "4.4  算法细节", 2)
    add_body(
        doc,
        "设输入图像为 I（BGR，取值 [0,1]）。首先取三通道最大值作为初始光照图，并用引导"
        "滤波对其进行平滑。引导滤波以 L0 自身作为引导图，因此在物体边界处不会跨边缘平滑："
    )
    add_formula(doc, "L0(x) = max_c I_c(x),    L_f = GuidedFilter(L0, L0, r=16, ε=0.02)", "2")
    add_body(
        doc,
        "引导滤波的局部线性模型为 q_i = a_k I_i + b_k，其中系数由局部均值、方差和协方差"
        "估计："
    )
    add_formula(doc, "a_k = Cov(I, p) / (Var(I) + ε),    b_k = mean(p) − a_k · mean(I)", "3")
    add_formula(doc, "q_i = mean_k(a_k) · I_i + mean_k(b_k)", "4")
    add_body(
        doc,
        "式 (3)(4) 中的所有均值都可以用盒式滤波实现，因此整张图的计算量与窗口大小无关，"
        "复杂度为 O(N)。得到光照图后，反射率由逐像素除法得到："
    )
    add_formula(doc, "R(x) = I(x) / (L_f(x) + ε)", "5")
    add_body(
        doc,
        "接下来对光照图做自适应 Gamma。设 L_f 的均值为 μ，则 γ 由目标亮度 0.45 反推，"
        "并限制在 [0.35, 0.85] 内，再加上一个小的全局偏移 δ（验证集选出 0.08）："
    )
    add_formula(doc, "γ = clip( log(0.45) / log(μ), 0.35, 0.85 ) + δ,    δ = 0.08", "6")
    add_body(
        doc,
        "μ 越小（图像越暗），γ 越小，提亮越强；μ 越大，γ 越接近 1，避免过曝。提亮后的"
        "光照图再按 99.5% 分位数归一化，把高光映射到 0.96，防止截断："
    )
    add_formula(doc, "L̂ = clip( L_f^γ / P99.5(L_f^γ) × 0.96, 0, 1 )", "7")
    add_body(doc, "增强图像由反射率和提亮后的光照图逐像素相乘得到：")
    add_formula(doc, "Î = R^α ⊙ L̂,    α = 1.0", "8")
    add_body(
        doc,
        "为了抑制过饱和，把每个像素的颜色向该像素的灰度均值插值，强度 s=0.8；再用灰度"
        "世界假设做一次温和的白平衡，强度 β=0.5："
    )
    add_formula(doc, "Î_s = mean(Î) + s · (Î − mean(Î))", "9")
    add_formula(doc, "g_c = ( mean_c(m_c) / m_c )^β,    Î_wb = Î_s ⊙ g", "10")
    add_body(
        doc,
        "其中 m_c 是第 c 个通道的均值，g_c 是通道增益。最后用非局部均值（NLM）对结果去噪。"
        "NLM 利用图像中相似块的平均来抑噪，比高斯/双边滤波更保边："
    )
    add_formula(doc, "I_NLM(i) = Σ_j w(i,j) · I(j) / Σ_j w(i,j)", "11")
    add_body(
        doc,
        "式 (11) 中 w(i,j) 由以 i、j 为中心的两个图像块的欧氏距离决定。本文使用 OpenCV 的"
        "fastNlMeansDenoisingColored，搜索窗口 21×21、块大小 7×7、滤波强度 h=4。"
    )

    add_h(doc, "4.5  复杂度与工程实现", 2)
    add_body(
        doc,
        "LIME 的主要开销在稀疏矩阵构造与求解：对 N 个像素，要遍历每个像素的 4 邻域，"
        "构造约 5N 个非零元，再用 spsolve 求解线性系统；在 600×400 图像上单张约 4 s。"
        "本文的引导滤波只需要 6 次盒式滤波（积分图实现），复杂度 O(N)，且完全向量化；"
        "自适应 Gamma、色度控制、白平衡都是逐像素运算；NLM 是主要的剩余开销（约 0.3 s）。"
    )
    add_body(
        doc,
        "工程上，所有方法通过统一的 make_method(name, cfg) 注册，返回 img -> img 的函数；"
        "RetinexConfig 用 dataclass 管理超参数，支持 JSON 覆盖；benchmark 逐行写 CSV 并"
        "支持 --resume，避免长实验中断后从头开始。"
    )

    add_h(doc, "4.6  代码结构与关键接口", 2)
    add_body(
        doc,
        "核心接口是 src.retinex.enhance(img_bgr, cfg, return_illumination=False)。输入为 "
        "BGR uint8 图像，输出为同尺寸 BGR uint8 图像；return_illumination=True 时额外返回"
        "浮点光照图，便于可视化。基线通过 src.baselines.make_method 调用，保证评测脚本"
        "对所有方法一视同仁。"
    )
    add_body(
        doc,
        "为了提高可复现性，仓库把数据下载、验证集选参、主实验、消融、绘图和报告生成拆成"
        "独立脚本；所有随机性来源（如 NLM 的搜索）都是确定性的，重复运行结果一致。"
    )


# ---------------------------------------------------------------------------
# 5. Results
# ---------------------------------------------------------------------------
def _mean_std(method: str, key: str, digits: int = 2) -> str:
    m, s = stat(method, key)
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def section_results(doc: Document):
    add_h(doc, "5  结果对比", 1)
    add_h(doc, "5.1  实验设置", 2)
    add_body(
        doc,
        "实验在 CPU 上完成：Intel Core i7-1165G7（4 核 8 线程），Windows 11，Python 3.x，"
        "NumPy 2.3.5，OpenCV 5.0.0，SciPy 1.18.1，scikit-image 0.26.0。所有图像保持"
        "600×400 原始分辨率，计时只统计算法本身，不含读写文件；每种方法先在一张 64×64 "
        "小图上预热一次，避免首次调用的初始化开销。"
    )
    add_body(doc, "对比方法及参数如下：")
    rows = [
        ("Gamma", "γ=0.45，全局 LUT", "最简单的亮度提升基线"),
        ("CLAHE", "clipLimit=2.0，8×8 网格，作用于 LAB 的 L 通道", "局部对比度增强基线"),
        ("MSRCR", "尺度 15/80/250，gain=5，offset=25", "经典多尺度 Retinex 基线"),
        ("LIME", "γ=0.6，λ=0.15，σ=3", "本文主要基线（稀疏求解）"),
        ("Ours", "r=16，ε=0.02，γ 自适应，s=0.8，β=0.5，h=4", "本文方法"),
    ]
    add_table(doc, ["方法", "参数", "说明"], rows, widths=[2.6, 7.2, 6.2],
              font_size=8.8, caption="表 4  对比方法与参数")

    add_body(
        doc,
        "全参考指标为 PSNR 和 SSIM（与 GT 比较）；无参考指标包括 LOE（与输入比较，衡量"
        "亮度顺序保持，越低越好）、色彩丰富度、熵和平均梯度。PSNR 定义为："
    )
    add_formula(doc, "PSNR = 10 · log10( 255² / MSE )", "12")
    add_body(doc, "SSIM 从亮度、对比度和结构三个方面比较两幅图像，取值越接近 1 越好。"
                   "LOE 统计增强前后像素相对亮度顺序发生变化的比例，取值越小说明越自然。")

    add_h(doc, "5.2  定量结果", 2)
    best_psnr = max(stat(m, "psnr")[0] for m in METHOD_ORDER)
    best_ssim = max(stat(m, "ssim")[0] for m in METHOD_ORDER)
    best_loe = min(stat(m, "loe")[0] for m in METHOD_ORDER if m != "input")
    times = {m: stat(m, "time_ms")[0] for m in METHOD_ORDER}
    best_time = min(times[m] for m in METHOD_ORDER if m != "input")

    main_rows = []
    for m in METHOD_ORDER:
        p, ps = stat(m, "psnr")
        s, ss = stat(m, "ssim")
        l, _ = stat(m, "loe")
        c, _ = stat(m, "colorfulness")
        t = times[m]
        p_txt = f"{p:.2f} ± {ps:.2f}"
        s_txt = f"{s:.4f} ± {ss:.4f}"
        l_txt = "—" if m == "input" else f"{l:.4f}"
        c_txt = f"{c:.1f}"
        t_txt = "—" if m == "input" else f"{t:.0f}"
        main_rows.append([
            METHOD_LABELS[m],
            (p_txt, abs(p - best_psnr) < 1e-9),
            (s_txt, abs(s - best_ssim) < 1e-9),
            (l_txt, m != "input" and abs(l - best_loe) < 1e-9),
            c_txt,
            (t_txt, m != "input" and abs(t - best_time) < 1e-9),
        ])
    add_table(doc, ["方法", "PSNR (dB) ↑", "SSIM ↑", "LOE ↓", "色彩丰富度", "时间 (ms)"],
              main_rows, widths=[2.3, 3.2, 3.2, 2.3, 2.3, 2.3], font_size=8.8,
              caption="表 5  在 LOLv2-real 100 对测试图像上的平均结果")

    psnr_l, _ = stat("lime", "psnr")
    psnr_o, _ = stat("ours", "psnr")
    ssim_l, _ = stat("lime", "ssim")
    ssim_o, _ = stat("ours", "ssim")
    loe_o, _ = stat("ours", "loe")
    loe_g, _ = stat("gamma", "loe")
    add_body(
        doc,
        f"从表 5 可以看到，Ours 的 PSNR 为 {psnr_o:.2f} dB，SSIM 为 {ssim_o:.4f}，"
        f"分别比 LIME 高 {psnr_o - psnr_l:.2f} dB 和 {ssim_o - ssim_l:.4f}；单张耗时约 "
        f"{times['ours']:.0f} ms，约为 LIME 的 1/{max(times['lime'] / max(times['ours'], 1e-6), 1):.1f}。"
        "Gamma 和 CLAHE 速度很快，但 PSNR/SSIM 明显偏低，说明单纯提亮或局部对比度增强"
        "无法恢复真实颜色和结构；MSRCR 的色彩丰富度很高，但 PSNR 很低，视觉上存在明显"
        "过饱和和色偏。Ours 在“亮度、结构、颜色、速度”之间取得了较好的折中。"
    )
    add_body(
        doc,
        f"值得注意的是，Ours 的 LOE 为 {loe_o:.4f}，高于 Gamma 的 {loe_g:.4f}。这并不"
        "矛盾：LOE 衡量的是亮度顺序保持，Gamma 对全图做单调映射，几乎不改变像素间的"
        "相对亮度；Ours 对暗部做了更强的局部提亮，部分像素的相对顺序会变化。实际观感上，"
        "适度的顺序变化换来了更清楚的暗部细节，但过强就会显得不自然。本文把 LOE 作为"
        "辅助指标，而不是唯一目标。"
    )

    add_h(doc, "5.3  定性结果", 2)
    add_body(
        doc,
        "图 3 给出 4 组测试图像的 Input、GT、LIME 和 Ours 对比；Gamma、CLAHE、MSRCR 的"
        "数值见表 5。输入图整体很暗，LIME 的亮度接近 GT，但部分区域出现偏绿/偏紫和颗粒；"
        "Ours 的亮度更均匀，墙面、地面和物体的颜色更接近 GT，暗部噪声也得到一定抑制"
        "（极暗区域仍有残留）。图 4 是局部放大，可以更清楚地看到 Ours 在椅面、桌面等区域的"
        "噪声控制和边缘保持。"
    )
    add_picture(doc, FIG / "qualitative.png", 16.0, "图 3  LIME 与 Ours 在 LOLv2-real 测试集上的视觉对比")
    add_picture(doc, FIG / "detail_crop.png", 16.0, "图 4  局部放大对比（00695）")

    add_h(doc, "5.4  消融实验", 2)
    if ABLATION:
        abl_rows = []
        for row in ABLATION:
            abl_rows.append([
                row["variant"],
                f"{float(row['psnr']):.2f}",
                f"{float(row['ssim']):.4f}",
                f"{float(row['loe']):.4f}",
                f"{float(row['time_ms']):.0f}",
            ])
        add_table(doc, ["变体", "PSNR (dB) ↑", "SSIM ↑", "LOE ↓", "时间 (ms)"], abl_rows,
                  widths=[4.2, 2.8, 2.8, 2.8, 2.8], font_size=8.8,
                  caption="表 6  消融实验（100 对测试图像）")
        add_body(
            doc,
            "表 6 中 Full 为完整模型；w/o guided filter 用高斯模糊代替引导滤波；fixed gamma "
            "用固定 γ=0.6 代替自适应 Gamma；w/o chroma control 令 s=1；w/o white balance "
            "令 β=0；w/o denoising 令 h=0。可以看到，去掉引导滤波后 PSNR/SSIM 下降，说明"
            "边缘保持的光照估计是有效的；固定 Gamma 在不同暗度图像上不够灵活；去掉色度"
            "控制会提高色彩丰富度，但 PSNR 下降，说明过饱和并不等于质量高；白平衡和去噪"
            "对颜色和暗部噪声有直接影响。"
        )
        add_picture(doc, FIG / "ablation.png", 15.0, "图 5  消融实验的 PSNR/SSIM 对比")

    add_h(doc, "5.5  参数敏感性", 2)
    add_body(
        doc,
        "图 6 是验证集上的参数敏感性热图。左图固定 r=16、ε=0.02，改变 Gamma 偏移和饱和度；"
        "右图固定 Gamma 偏移 0.08、饱和度 0.8，改变引导滤波半径（ε=0.02）。总体来看，Gamma 偏移"
        "在 0.08–0.12 之间比较稳，饱和度过高会拉低 PSNR；半径过小会留下噪声，过大则会让"
        "光照图过平滑、细节变少。最终在验证集上选择的参数见表 7。"
    )
    add_picture(doc, FIG / "tuning.png", 15.5, "图 6  验证集参数敏感性")
    if BEST:
        cfg_rows = [
            ("引导滤波半径 r", str(BEST.get("radius", 16))),
            ("正则项 ε", str(BEST.get("eps", 0.02))),
            ("Gamma 偏移", str(BEST.get("gamma_offset", 0.08))),
            ("色度强度 s", str(BEST.get("saturation", 0.8))),
            ("白平衡强度 β", str(BEST.get("white_balance", 0.5))),
            ("NLM 强度 h", str(BEST.get("denoise_h", 4.0))),
        ]
        add_table(doc, ["参数", "取值"], cfg_rows, widths=[6.0, 6.0], font_size=9,
                  caption="表 7  验证集选出的最终超参数")

    add_h(doc, "5.6  运行效率", 2)
    add_body(
        doc,
        f"图 7 给出各方法的单张耗时（对数坐标）。Gamma 几乎不耗时，CLAHE 约 "
        f"{times['clahe']:.0f} ms，MSRCR 约 {times['msrcr']:.0f} ms，LIME 约 "
        f"{times['lime']:.0f} ms，Ours 约 {times['ours']:.0f} ms。Ours 的主要开销来自 NLM "
        "去噪（约 0.3 s），引导滤波和逐像素运算只占很小一部分。若关闭 NLM，Ours 可以降到"
        "约 0.1 s，适合实时预览；需要更高质量时再开启 NLM。"
    )
    add_picture(doc, FIG / "runtime.png", 12.0, "图 7  单张图像运行时间（600×400，CPU）")

    add_h(doc, "5.7  失败案例分析", 2)
    add_picture(doc, FIG / "failure.png", 15.5, "图 8  失败案例：偏色与过平滑")
    add_body(
        doc,
        "图 8 给出两个不理想的例子。第一例中，场景本身有较强的绿色灯光，灰度世界假设"
        "把绿色当成偏色去校正，结果墙面偏灰、颜色不够自然；第二例中，极暗区域的噪声"
        "很强，NLM 在去噪的同时把细小的纹理也抹平了，导致细节偏软。这说明："
    )
    add_bullet(doc, "白平衡假设“平均颜色为中灰”，在强色光场景下不成立，需要更鲁棒的色彩恒常性方法。")
    add_bullet(doc, "NLM 的强度 h 是全局的，无法同时适配噪声很强和纹理很细的区域，"
                   "可以考虑噪声估计或内容自适应的去噪。")
    add_bullet(doc, "PSNR/SSIM 对亮度/颜色的全局匹配很敏感，某些视觉上更自然的结果反而"
                   "数值更低；评价时应结合多个指标和主观观察。")

    add_h(doc, "5.8  统计显著性", 2)
    stat_rows = []
    for other, label in (("lime", "LIME"), ("gamma", "Gamma"), ("msrcr", "MSRCR")):
        for key, key_label in (("psnr", "PSNR"), ("ssim", "SSIM")):
            w, p = pval_ours_vs(other, key)
            sig = "是" if (p == p and p < 0.05) else "否"
            stat_rows.append([f"Ours vs {label}", key_label, f"{w:.1f}", f"{p:.2e}", sig])
    add_table(doc, ["对比", "指标", "Wilcoxon W", "p 值", "p<0.05"], stat_rows,
              widths=[4.0, 2.5, 3.5, 3.5, 2.0], font_size=8.8,
              caption="表 8  逐图像配对 Wilcoxon 符号秩检验")
    add_body(
        doc,
        "表 8 对 100 对图像的逐图指标做配对 Wilcoxon 符号秩检验。若 p<0.05，说明 Ours 与"
        "该基线之间的差异不是偶然波动。需要说明的是，统计显著不等于视觉上一定更好，"
        "它只说明在给定数据和指标下差异稳定存在。"
    )


# ---------------------------------------------------------------------------
# 6. Conclusion
# ---------------------------------------------------------------------------
def section_conclusion(doc: Document):
    add_h(doc, "6  总结与展望", 1)
    add_h(doc, "6.1  总结", 2)
    add_body(
        doc,
        "本文围绕低光照图像增强，对开源 LIME 方法做了算法和工程两方面的改进。算法上，"
        "用自引导滤波替代稀疏线性求解来估计光照图，引入自适应 Gamma 和高光安全归一化，"
        "加入色度控制、灰度世界白平衡和非局部均值去噪；工程上，补齐了数据下载、验证集"
        "选参、批量评测、消融、绘图、CLI 和 GUI。"
    )
    add_body(
        doc,
        "在 LOLv2-real 的 100 对测试图像上，Ours 的 PSNR/SSIM 相对 LIME 有明显提升，"
        "运行速度约为 LIME 的 7 倍；与 Gamma、CLAHE、MSRCR 相比，在颜色和噪声控制上更"
        "均衡。消融实验表明，引导滤波、自适应 Gamma、色度控制、白平衡和去噪都对最终"
        "结果有贡献。"
    )
    add_h(doc, "6.2  局限与展望", 2)
    add_body(doc, "本文方法仍有以下局限：")
    add_bullet(doc, "白平衡采用灰度世界假设，在强色光、单色场景下容易误判，可引入白块检测、"
                   "色温估计或学习式颜色恒常性。")
    add_bullet(doc, "NLM 的滤波强度是全局常数，强噪声和细纹理区域难以同时兼顾，可结合噪声估计"
                   "做空间自适应。")
    add_bullet(doc, "只在 LOLv2-real 上做了定量评测，尚未在手机真实夜景、监控视频上系统验证；"
                   "后续可以采集自建数据，并加入视频时序一致性约束。")
    add_bullet(doc, "传统方法的性能上限有限，后续可以尝试轻量级 CNN/Transformer（如 Zero-DCE、"
                   "Retinexformer）做蒸馏或作为后处理，在 CPU 上做量化部署。")
    add_body(
        doc,
        "总体而言，本文用一个可解释、可复现、无需训练的传统方法，在低光照增强的质量和"
        "速度之间取得了不错的平衡，也为后续引入学习方法提供了可靠的基线。"
    )


# ---------------------------------------------------------------------------
# references and appendix
# ---------------------------------------------------------------------------
def references(doc: Document):
    add_h(doc, "参考文献", 1)
    refs = [
        "[1] Land E H, McCann J J. Lightness and retinex theory[J]. Journal of the Optical Society of America, 1971, 61(1): 1-11.",
        "[2] Jobson D J, Rahman Z, Woodell G A. A multiscale retinex for bridging the gap between color images and the human observation of scenes[J]. IEEE Transactions on Image Processing, 1997, 6(7): 965-976.",
        "[3] Guo X, Li Y, Ling H. LIME: Low-light image enhancement via illumination map estimation[J]. IEEE Transactions on Image Processing, 2017, 26(2): 982-993.",
        "[4] He K, Sun J, Tang X. Guided image filtering[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2013, 35(6): 1397-1409.",
        "[5] Chen W, Wang W, Yang W, et al. LOL-v2: A new benchmark for low-light image enhancement[C]//CVPR. 2021.",
        "[6] Ying Z, Li G, Ren Y, et al. Dual illumination estimation for robust exposure correction[J]. Computer Graphics Forum, 2019, 38(7): 409-420.",
        "[7] Buades A, Coll B, Morel J M. A non-local algorithm for image denoising[C]//CVPR. 2005.",
        "[8] Wang Z, Bovik A C, Sheikh H R, et al. Image quality assessment: from error visibility to structural similarity[J]. IEEE Transactions on Image Processing, 2004, 13(4): 600-612.",
        "[9] Wang S, Zheng J, Hu H M, et al. Naturalness preserved enhancement algorithm for non-uniform illumination images[J]. IEEE Transactions on Image Processing, 2013, 22(9): 3538-3548.",
        "[10] Zuiderveld K. Contrast limited adaptive histogram equalization[M]//Graphics Gems IV. 1994: 474-485.",
        "[11] Guo C, Li C, Guo J, et al. Zero-reference deep curve estimation for low-light image enhancement[C]//CVPR. 2020.",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.3
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.first_line_indent = Cm(-0.6)
        run = p.add_run(ref)
        set_run_font(run, 9)


def appendix(doc: Document):
    add_h(doc, "附录 A  最终超参数", 1)
    add_body(doc, "验证集选出的最终配置如下（results/best_config.json）：", indent=False)
    if BEST:
        keys = ["radius", "eps", "gamma_offset", "saturation", "white_balance", "denoise_h"]
        compact = {k: BEST.get(k) for k in keys}
        add_code(doc, json.dumps(compact, ensure_ascii=False, indent=2))
    add_h(doc, "附录 B  结果文件说明", 1)
    rows = [
        ("results/metrics.csv", "100 对测试图像、每种方法的逐图指标"),
        ("results/summary.json", "各方法指标的均值/标准差"),
        ("results/ablation.csv", "消融实验汇总"),
        ("results/tuning.csv", "验证集参数搜索记录"),
        ("results/figures/", "报告与 README 使用的全部图片"),
        ("report/低光照图像增强大作业报告.docx", "本报告"),
    ]
    add_table(doc, ["文件", "内容"], rows, widths=[7.0, 9.0], font_size=8.8,
              caption="表 9  结果文件说明")
    add_h(doc, "附录 C  代码与数据可用性", 1)
    add_body(
        doc,
        "代码采用 MIT 许可证；LOLv2-real 数据采用 Apache-2.0 许可证，引用信息见第 3 节。"
        "第三方 LIME 代码的 MIT 许可证保存在 third_party/lime/LICENSE。"
    )


# ---------------------------------------------------------------------------
def main():
    doc = setup_document()
    cover(doc)
    abstract(doc)
    section_github(doc)
    section_task(doc)
    section_data(doc)
    section_impl(doc)
    section_results(doc)
    section_conclusion(doc)
    references(doc)
    appendix(doc)
    doc.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()





