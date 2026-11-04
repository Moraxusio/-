# results 目录

- `metrics.csv`：100 对测试图像、每种方法的逐图 PSNR/SSIM/LOE/色彩丰富度/熵/平均梯度/耗时。
- `summary.json`：各方法指标的均值和标准差，以及最终配置。
- `tuning.csv`：验证集参数搜索记录；`best_config.json`：选出的配置。
- `ablation.csv` / `ablation_summary.json`：消融实验。
- `figures/`：报告和 README 使用的图片（pipeline、qualitative、detail、metric、ablation、tuning、runtime、failure）。
- `enhanced/`：可选的增强结果输出目录（不纳入 git）。
