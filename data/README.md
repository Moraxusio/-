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
  Val/                    # select_validation.py 生成
    Input/*.png (30)
    GT/*.png (30)
```

运行 `python scripts/download_data.py` 会自动下载 689 对训练图像和 100 对测试图像。
随后运行 `python scripts/select_validation.py`，按测试集 GT 亮度分布从训练集中分层抽取
30 对，复制到 `data/lolv2/Val/` 作为亮度匹配验证集。数据体积约 300 MB，不纳入 git
版本管理；`assets/samples/` 中保留了两对示例，便于快速试用。

引用：

> Chen W, Wang W, Yang W, et al. LOL-v2: A new benchmark for low-light image
> enhancement. CVPR 2021.
