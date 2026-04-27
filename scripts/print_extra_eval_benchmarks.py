#!/usr/bin/env python3
"""
说明：除 3DPW/H36M/COCO 等之外的「固定」评测集接入方式（已写入 hmr2/configs/datasets_eval.yaml）。

1) 3DPW-OCC-TEST（遮挡子集）
   - npz: {HUMANS_ROOT}/hmr2_evaluation_data/3dpw_occ_test.npz（官方 eval 包已带）
   - 图像: 与 3DPW-TEST 相同根目录 .../data/3DPW/
   - 运行: --dataset 3DPW-OCC-TEST 或 ALL（已包含在 ALL_EVAL_DATASETS）

2) MPI-INF-3DHP-TEST（户外固定协议，论文常作泛化/野外补充）
   - 需自备与 HMR2 ImageDataset 一致的 mpi_inf_3dhp_test.npz 及帧图像目录。
   - 官方数据与协议: https://vcai.mpi-inf.mpg.de/3dhp-dataset/
   - npz 通常由 HMR2/PARE/SPIN 等仓库的预处理脚本从原始序列生成；文件名约定为 mpi_inf_3dhp_test.npz
   - 放置:
       {HUMANS_ROOT}/hmr2_evaluation_data/mpi_inf_3dhp_test.npz
       {HUMANS_ROOT}/data/mpi_inf_3dhp/   # imgname 相对路径与此目录拼接
   - 可选环境变量覆盖图像根: HMR2_EVAL_IMG_DIR_MPIINF

3) 与「倒立/瑜伽/竞技」的关系
   - MPI-INF-3DHP 以**户外多活动**为主，是社区通行的**固定 benchmark**，便于与文献对比；
   - 若需语义上的「极端姿态」子集，需在 benchmark 之外另做 curated subset（本脚本不覆盖）。
"""
from __future__ import annotations

import textwrap

if __name__ == "__main__":
    print(textwrap.dedent(__doc__).strip())
