# -*- coding: utf-8 -*-
"""Download LOLv2-real from Hugging Face.

Dataset: https://huggingface.co/datasets/okhater/lolv2-real (Apache-2.0)
Structure after download:
    data/lolv2/Test/{Input,GT}/00690.png ... 00789.png
    data/lolv2/Train/{Input,GT}/00001.png ... 00689.png

The full Train split is needed by scripts/select_validation.py, which builds a
brightness-matched validation subset.
"""
from huggingface_hub import snapshot_download

REPO = "okhater/lolv2-real"
LOCAL = r"C:\dev\图像处理大作业\lowlight-retinex\data\lolv2"

patterns = [
    "Test/Input/*.png",
    "Test/GT/*.png",
    "Train/Input/*.png",
    "Train/GT/*.png",
]

path = snapshot_download(
    repo_id=REPO,
    repo_type="dataset",
    local_dir=LOCAL,
    allow_patterns=patterns,
    max_workers=8,
)
print("downloaded to", path)
