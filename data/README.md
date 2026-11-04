# 数据说明

本项目使用 **LOLv2-real** 低光照图像增强数据集。

- 来源：Hugging Face [okhater/lolv2-real](https://huggingface.co/datasets/okhater/lolv2-real)
- 许可证：Apache-2.0
- 规模：689 对训练图像 + 100 对测试图像
- 分辨率：600 × 400，PNG
- 目录结构：

```
data/lolv2/
  Test/
    Input/00690.png ... 00789.png
    GT/00690.png    ... 00789.png
  Train/
    Input/00001.png ... 00689.png
    GT/00001.png    ... 00689.png
```

运行 `python scripts/download_data.py` 会自动下载 100 对测试图像和 30 对验证图像
（Train 前 30 对）。数据体积约 94 MB，不纳入 git 版本管理；`assets/samples/` 中保留了
两对示例，便于快速试用。

引用：

> Chen W, Wang W, Yang W, et al. LOL-v2: A new benchmark for low-light image
> enhancement. CVPR 2021.
