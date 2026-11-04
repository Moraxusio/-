# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import numpy as np, cv2
ROOT = Path(r"C:\dev\图像处理大作业\lowlight-retinex")
sys.path.insert(0, str(ROOT))
from src.io_utils import imread
inp_mean=[]; gt_mean=[]; inp_p95=[]; gt_p95=[]
for p in sorted((ROOT/"data"/"lolv2"/"Test"/"Input").glob("*.png")):
    g=p.parent.parent/"GT"/p.name
    a=imread(p); b=imread(g)
    ga=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY); gb=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
    inp_mean.append(ga.mean()); gt_mean.append(gb.mean())
    inp_p95.append(np.percentile(ga,95)); gt_p95.append(np.percentile(gb,95))
print("input mean", np.mean(inp_mean), "std", np.std(inp_mean), "range", min(inp_mean), max(inp_mean))
print("gt mean", np.mean(gt_mean), "std", np.std(gt_mean), "range", min(gt_mean), max(gt_mean))
print("input p95", np.mean(inp_p95), "gt p95", np.mean(gt_p95))
