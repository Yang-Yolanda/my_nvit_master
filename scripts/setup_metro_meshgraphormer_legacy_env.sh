#!/usr/bin/env bash
# 为 METRO（MeshTransformer）与 Mesh Graphormer（MeshGraphormer）创建「官方 README 同款」旧栈环境。
# 二者依赖一致（Python 3.7 / PyTorch 1.4 / cu10.1 / apex），默认共用一个 conda env，两个源码目录并排。
#
# 用法:
#   bash scripts/setup_metro_meshgraphormer_legacy_env.sh
#   默认: 代码克隆到 <NViT-master>/nvit/external_baselines/ ；conda 环境名默认 nvit_metro_cu101
#   新建/重建环境: bash scripts/setup_metro_meshgraphormer_legacy_env.sh --force
#   指定环境名: BASELINE_LEGACY_ENV=我的环名 bash ... --force
#   旧目录（可选）: INSTALL_ROOT=/path bash ...
#
# 说明:
# - 与 NViT / 4D-Humans 的 PyTorch 2.x 环境隔离，避免版本冲突。
# - NVIDIA 驱动需支持运行 cu10.1 的 PyTorch wheel（一般新驱动向后兼容用户态 runtime）。
# - apex 编译可能因 GCC 版本失败；失败时可设 SKIP_APEX=1 跳过（部分功能可能不可用）。
# - MeshGraphormer 官方 INSTALL 使用 umich opendr git；若拉取失败，脚本会尝试 pip opendr 兜底。
#
set -euo pipefail

# 集群/容器里常残留无效 http(s)_proxy，会导致 conda 报 ProxyError；安装阶段统一去掉。
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY ftp_proxy FTP_PROXY || true

# 容器里常把系统 Py3.12 的 torch/lib 塞进 LD_LIBRARY_PATH，导致本 env 的 Py3.7 误链到错误 libtorch_python.so
# 部分机器在 ~/.config/pip 等位置锁了与 Py3.7 不兼容的 PIP_CONSTRAINT（如 matplotlib 3.10），安装时必须去掉
clean_gpu_python() { env -u LD_LIBRARY_PATH -u PYTHONPATH -u PIP_CONSTRAINT "$@"; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA:-/cpfs_infra/shared/yangz/opt/Miniconda3/bin/conda}"
# 与 4D-humans 分离的独立 conda 环境；换名可 export BASELINE_LEGACY_ENV
ENV_NAME="${BASELINE_LEGACY_ENV:-nvit_metro_cu101}"
# 与 SMPLer 适配器同放 nvit/external_baselines（MeshTransformer / MeshGraphormer 目录并列）
INSTALL_ROOT="${INSTALL_ROOT:-${ROOT}/nvit/external_baselines}"
INSTALL_ROOT="$(mkdir -p "$INSTALL_ROOT" && cd "$INSTALL_ROOT" && pwd)"

FORCE=0
for a in "$@"; do
  if [[ "$a" == "--force" ]]; then FORCE=1; fi
done

if [[ ! -x "$CONDA" ]]; then
  echo "ERROR: conda 不可执行: $CONDA  （可 export CONDA=/你的/miniconda3/bin/conda）" >&2
  exit 1
fi

echo "INSTALL_ROOT=$INSTALL_ROOT"
echo "ENV_NAME=$ENV_NAME"
echo "CONDA=$CONDA"

if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  if [[ "$FORCE" != "1" ]]; then
    echo "环境已存在: $ENV_NAME （加 --force 可删后重建）"
  else
    echo "移除旧环境: $ENV_NAME"
    "$CONDA" remove -y -n "$ENV_NAME" --all || true
  fi
fi

if ! "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$CONDA" create -y -n "$ENV_NAME" python=3.7 pip
fi

PY="$(clean_gpu_python "$CONDA" run -n "$ENV_NAME" which python)"
echo "python=$PY"

# CUDA 10.1 wheel：官方索引已下架 torch==1.4.0+cu101；用仍带 +cu101 的 1.6 栈（MeshTransformer/MeshGraphormer 多数代码可兼容）
# 若必须复刻论文同款 1.4，可自备 wheel 后 export TORCH_WHEEL_FILE / TV_WHEEL_FILE 指向本地 .whl
TORCH_INDEX="${TORCH_WHEEL_INDEX:-https://download.pytorch.org/whl/cu101/torch_stable.html}"
if [[ -n "${TORCH_WHEEL_FILE:-}" && -n "${TV_WHEEL_FILE:-}" ]]; then
  clean_gpu_python "$CONDA" run -n "$ENV_NAME" env -u PIP_CONSTRAINT -u PIP_REQUIRE_VIRTUALENV -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL \
    python -m pip install --no-cache-dir "${TORCH_WHEEL_FILE}" "${TV_WHEEL_FILE}"
else
  clean_gpu_python "$CONDA" run -n "$ENV_NAME" env -u PIP_CONSTRAINT -u PIP_REQUIRE_VIRTUALENV -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL \
    python -m pip install --index-url https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    "torch==1.6.0+cu101" "torchvision==0.7.0+cu101" \
    -f "$TORCH_INDEX"
fi

# METRO / MeshGraphormer 官方 INSTALL：matplotlib + opendr（尽量在 clone 仓库前装好）
# 默认用 pip 固定 3.5.x（小、快）；若设 MATPLOTLIB_VIA=conda 则会拉带 Qt 的大型 matplotlib 栈（更慢）
if [[ "${MATPLOTLIB_VIA:-pip}" == "conda" ]]; then
  "$CONDA" install -y -n "$ENV_NAME" matplotlib || true
else
  clean_gpu_python "$CONDA" run -n "$ENV_NAME" python -m pip install "matplotlib==3.5.3"
fi
clean_gpu_python "$CONDA" run -n "$ENV_NAME" pip install "six" || true
# opendr 编译常依赖系统 libOSMesa/GL；无头环境可 SKIP_OPENDR=1
if [[ "${SKIP_OPENDR:-0}" != "1" ]]; then
  # 先 pip 装 opendr；umich git 仅当 OPENDR_TRY_UMICH=1
  if ! clean_gpu_python "$CONDA" run -n "$ENV_NAME" pip install "opendr==0.78" "chumpy" "Cython"; then
    echo "WARN: pip opendr 0.78 失败。可设 SKIP_OPENDR=1 重跑跳过（需无 OpenGL 渲染的评测路径）。" >&2
  fi
  if [[ "${OPENDR_TRY_UMICH:-0}" == "1" ]]; then
    clean_gpu_python "$CONDA" run -n "$ENV_NAME" pip install "git+https://gitlab.eecs.umich.edu/ngv-python-modules/opendr.git" || true
  fi
else
  echo "SKIP_OPENDR=1 — 不装 opendr"
fi

cd "$INSTALL_ROOT"

# --- apex（可选）---
if [[ "${SKIP_APEX:-0}" != "1" ]]; then
  if [[ ! -d apex ]]; then
    git clone https://github.com/NVIDIA/apex.git
  fi
  pushd apex >/dev/null
  clean_gpu_python "$CONDA" run -n "$ENV_NAME" python setup.py install --cuda_ext --cpp_ext || {
    echo "WARN: apex 安装失败。可 export SKIP_APEX=1 后重跑脚本跳过。" >&2
  }
  popd >/dev/null
else
  echo "SKIP_APEX=1 — 跳过 apex"
fi

# --- METRO / MeshTransformer ---
if [[ ! -d MeshTransformer ]]; then
  if ! git clone --recursive https://github.com/microsoft/MeshTransformer.git; then
    echo "ERROR: 无法从 github.com 克隆 MeshTransformer（DNS/代理/网络）。有网后请手动:" >&2
    echo "  cd \"$INSTALL_ROOT\" && git clone --recursive https://github.com/microsoft/MeshTransformer.git" >&2
  fi
fi
if [[ -f MeshTransformer/setup.py ]]; then
  pushd MeshTransformer >/dev/null
  clean_gpu_python "$CONDA" run -n "$ENV_NAME" python setup.py build develop
  clean_gpu_python "$CONDA" run -n "$ENV_NAME" pip install -r requirements.txt
  clean_gpu_python "$CONDA" run -n "$ENV_NAME" pip install ./manopth/.
  popd >/dev/null
fi

# --- Mesh Graphormer ---
if [[ ! -d MeshGraphormer ]]; then
  if ! git clone --recursive https://github.com/microsoft/MeshGraphormer.git; then
    echo "ERROR: 无法从 github.com 克隆 MeshGraphormer。请联网后:" >&2
    echo "  cd \"$INSTALL_ROOT\" && git clone --recursive https://github.com/microsoft/MeshGraphormer.git" >&2
  fi
fi
if [[ -f MeshGraphormer/setup.py ]]; then
  pushd MeshGraphormer >/dev/null
  clean_gpu_python "$CONDA" run -n "$ENV_NAME" python setup.py build develop
  clean_gpu_python "$CONDA" run -n "$ENV_NAME" pip install -r requirements.txt
  clean_gpu_python "$CONDA" run -n "$ENV_NAME" pip install ./manopth/.
  popd >/dev/null
fi

echo
echo "=== 完成 ==="
CONDA_BASE="$("$CONDA" info --base)"
echo "激活环境:"
echo "  source \"$CONDA_BASE/etc/profile.d/conda.sh\" && conda activate $ENV_NAME"
echo "源码路径:"
echo "  METRO(MeshTransformer): $INSTALL_ROOT/MeshTransformer"
echo "  MeshGraphormer:         $INSTALL_ROOT/MeshGraphormer"
echo "预训练权重请按各仓库 docs/DOWNLOAD.md（MeshGraphormer）与 MeshTransformer README 自行下载。"
echo
echo "若 import torch 报错链到 /usr/local/.../python3.12/.../torch：shell 里执行 unset LD_LIBRARY_PATH 再 conda activate，"
echo "或: source $ROOT/scripts/activate_baseline_metro_gphmr.sh"
