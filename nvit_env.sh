#!/bin/bash
# NViT Environment & Ultimate Memory Optimizations
# Auto-activate 4D-humans environment
conda activate 4D-humans 2>/dev/null || source activate 4D-humans 2>/dev/null

# # Memory and CPU optimizations
# export MALLOC_ARENA_MAX=2
# export NCCL_SHM_DISABLE=1
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# # Hydra Debugging
# export HYDRA_FULL_ERROR=1

# 防多卡通讯爆炸
export TORCH_NCCL_SHM_DISABLE=1
# 防 3090 显存颗粒碎片化报错
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
# 开三代硬件加速
export NVIDIA_TF32_OVERRIDE=1
# 打印完整报错日志
export HYDRA_FULL_ERROR=1