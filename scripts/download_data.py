# -*- coding: utf-8 -*-
"""Download LOLv2-real test set and a small validation subset from Hugging Face.

Dataset: https://huggingface.co/datasets/okhater/lolv2-real (Apache-2.0)
Structure after download:
    data/lolv2/Test/{Input,GT}/00690.png ... 00789.png
    data/lolv2/Train/{Input,GT}/00001.png ... 00030.png
"""
from huggingface_hub import snapshot_download

REPO = "okhater/lolv2-real"
LOCAL = r"C:\dev\图像处理大作业\lowlight-retinex\data\lolv2"

val_ids = [f"{i:05d}.png" for i in range(1, 31)]
patterns = ["Test/Input/*.png", "Test/GT/*.png"]
for split in ("Input", "GT"):
    patterns += [f"Train/{split}/{n}" for n in val_ids]

path = snapshot_download(
    repo_id=REPO,
    repo_type="dataset",
    local_dir=LOCAL,
    allow_patterns=patterns,
    max_workers=8,
)
print("downloaded to", path)
