# -*- coding: utf-8 -*-
"""A small Tkinter GUI for before/after comparison.

Run:
    python app/gui.py
"""
from __future__ import annotations

import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import make_method
from src.config import RetinexConfig
from src.io_utils import imread, imwrite

METHODS = [
    ("Ours (guided-filter Retinex)", "ours"),
    ("LIME (baseline)", "lime"),
    ("Gamma correction", "gamma"),
    ("CLAHE", "clahe"),
    ("MSRCR", "msrcr"),
]
CANVAS_W, CANVAS_H = 560, 420


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("低光照图像增强 · Guided-filter Retinex")
        self.geometry("1240x720")
        self.minsize(980, 620)
        self.configure(bg="#f4f5f7")

        self.input_bgr: np.ndarray | None = None
        self.output_bgr: np.ndarray | None = None
        self.input_path: Path | None = None
        self.cfg = RetinexConfig()
        self.queue: queue.Queue = queue.Queue()
        self._busy = False

        self._build_style()
        self._build_ui()
        self._bind_keys()
        self.after(100, self._poll_queue)

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", padding=(10, 5), font=("Microsoft YaHei UI", 10))
        style.configure("TLabel", background="#f4f5f7", font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 18, "bold"), foreground="#111827")
        style.configure("Sub.TLabel", font=("Microsoft YaHei UI", 9), foreground="#6b7280")
        style.configure("TCombobox", padding=4)
        style.configure("Card.TFrame", background="white")

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=(18, 14, 18, 6))
        header.pack(fill="x")
        ttk.Label(header, text="低光照图像增强", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="Guided-filter Retinex · LOLv2-real", style="Sub.TLabel").pack(
            side="left", padx=(12, 0), pady=(8, 0))

        toolbar = ttk.Frame(self, padding=(18, 6, 18, 6))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="打开图片", command=self.open_image).pack(side="left")
        self.method_var = tk.StringVar(value=METHODS[0][0])
        ttk.Combobox(toolbar, textvariable=self.method_var, values=[m[0] for m in METHODS],
                     state="readonly", width=30).pack(side="left", padx=(10, 6))
        self.run_btn = ttk.Button(toolbar, text="增强", command=self.run_enhance)
        self.run_btn.pack(side="left")
        ttk.Button(toolbar, text="保存结果", command=self.save_result).pack(side="left", padx=(6, 0))
        self.status_var = tk.StringVar(value="请打开一张低光照图片")
        ttk.Label(toolbar, textvariable=self.status_var, style="Sub.TLabel").pack(side="right")

        body = ttk.Frame(self, padding=(18, 6, 18, 14))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self.in_canvas = self._make_panel(body, 0, "输入")
        self.out_canvas = self._make_panel(body, 1, "输出")

        footer = ttk.Frame(self, padding=(18, 0, 18, 12))
        footer.pack(fill="x")
        ttk.Label(footer, text="提示：Ours 约 0.5 s / 600×400 CPU；LIME 约 4 s。",
                  style="Sub.TLabel").pack(side="left")

    def _make_panel(self, parent, col: int, title: str) -> tk.Canvas:
        card = ttk.Frame(parent, style="Card.TFrame", padding=8)
        card.grid(row=0, column=col, sticky="nsew", padx=(0, 8) if col == 0 else (8, 0))
        card.rowconfigure(1, weight=1)
        card.columnconfigure(0, weight=1)
        ttk.Label(card, text=title, background="white",
                  font=("Microsoft YaHei UI", 11, "bold"), foreground="#374151").grid(
            row=0, column=0, sticky="w", padx=4, pady=(0, 6))
        canvas = tk.Canvas(card, width=CANVAS_W, height=CANVAS_H, bg="#111827",
                           highlightthickness=0)
        canvas.grid(row=1, column=0, sticky="nsew")
        return canvas

    def _bind_keys(self) -> None:
        self.bind("<Control-o>", lambda e: self.open_image())
        self.bind("<Control-s>", lambda e: self.save_result())
        self.bind("<Return>", lambda e: self.run_enhance())

    def _show(self, canvas: tk.Canvas, bgr: np.ndarray | None) -> None:
        canvas.delete("all")
        if bgr is None:
            canvas.create_text(CANVAS_W // 2, CANVAS_H // 2, text="尚未加载",
                               fill="#6b7280", font=("Microsoft YaHei UI", 12))
            return
        h, w = bgr.shape[:2]
        cw = max(canvas.winfo_width(), CANVAS_W)
        ch = max(canvas.winfo_height(), CANVAS_H)
        scale = min(cw / w, ch / h, 1.0)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(pil)
        canvas.create_image(cw // 2, ch // 2, image=self._tk_img, anchor="center")

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择低光照图片",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            img = imread(path)
        except Exception as exc:
            messagebox.showerror("读取失败", str(exc))
            return
        self.input_path = Path(path)
        self.input_bgr = img
        self.output_bgr = None
        self._show(self.in_canvas, self.input_bgr)
        self._show(self.out_canvas, None)
        self.status_var.set(f"{self.input_path.name}  {img.shape[1]}×{img.shape[0]}")

    def run_enhance(self) -> None:
        if self._busy:
            return
        if self.input_bgr is None:
            messagebox.showinfo("提示", "请先打开一张图片")
            return
        self._busy = True
        self.run_btn.configure(state="disabled")
        self.status_var.set("处理中…")
        label = self.method_var.get()
        key = dict(METHODS)[label]
        img = self.input_bgr.copy()
        cfg = self.cfg

        def worker() -> None:
            try:
                t0 = time.perf_counter()
                out = make_method(key, cfg)(img)
                dt = (time.perf_counter() - t0) * 1000
                self.queue.put(("ok", out, dt))
            except Exception as exc:  # pragma: no cover - GUI safety net
                self.queue.put(("err", exc, 0))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self) -> None:
        try:
            kind, payload, dt = self.queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_queue)
            return
        self._busy = False
        self.run_btn.configure(state="normal")
        if kind == "ok":
            self.output_bgr = payload
            self._show(self.out_canvas, self.output_bgr)
            self.status_var.set(f"完成 · {dt:.0f} ms")
        else:
            messagebox.showerror("处理失败", str(payload))
            self.status_var.set("处理失败")
        self.after(100, self._poll_queue)

    def save_result(self) -> None:
        if self.output_bgr is None:
            messagebox.showinfo("提示", "还没有可保存的结果")
            return
        path = filedialog.asksaveasfilename(
            title="保存增强结果", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if path:
            imwrite(path, self.output_bgr)
            self.status_var.set(f"已保存：{Path(path).name}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
