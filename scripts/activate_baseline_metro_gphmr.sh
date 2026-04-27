#!/usr/bin/env bash
# 在已安装 nvit_metro_cu101（或 BASELINE_LEGACY_ENV 指定名）后使用；避免宿主机 LD_LIBRARY_PATH 指向 Py3.12 的 torch。
# 用法: source scripts/activate_baseline_metro_gphmr.sh
# 仍使用旧环 baseline_metro_gphmr37:  BASELINE_LEGACY_ENV=baseline_metro_gphmr37 source scripts/activate_baseline_metro_gphmr.sh

CONDA="${CONDA:-/cpfs_infra/shared/yangz/opt/Miniconda3/bin/conda}"
ENV_NAME="${BASELINE_LEGACY_ENV:-nvit_metro_cu101}"
CONDA_BASE="$("$CONDA" info --base)"
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"
unset LD_LIBRARY_PATH
unset PYTHONPATH
conda activate "$ENV_NAME"
echo "Activated: $ENV_NAME (LD_LIBRARY_PATH cleared for correct torch)."
