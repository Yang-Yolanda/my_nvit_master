#!/bin/bash

# [Robust Root Detection]
# This script dynamically sets the project environment regardless of its location.
# Source this file before running any NViT experiments: source nvit_env.sh

# 1. Detect NViT-master root (the location of this script)
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 2. Detect 4D-Humans sibling root
export HUMANS_ROOT="$( cd "$PROJECT_ROOT/.." && pwd )/4D-Humans"

if [ ! -d "$HUMANS_ROOT" ]; then
    echo "⚠️ Warning: 4D-Humans sibling directory not found at $HUMANS_ROOT"
    echo "Current PROJECT_ROOT: $PROJECT_ROOT"
fi

# 3. Setup PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$HUMANS_ROOT:$PROJECT_ROOT/nvit:$PYTHONPATH"

# 4. Standardize Conda Environment (Optional)
# export CONDA_ENV_PATH="/path/to/conda/4D-humans"

# 5. Output for Verification
echo "✅ NViT Environment Initialized"
echo "   PROJECT_ROOT: $PROJECT_ROOT"
echo "   HUMANS_ROOT:  $HUMANS_ROOT"
echo "   PYTHONPATH:   ... (updated)"

# FIRST=0
# if [ "$FIRST" -eq 1 ]; then
#     curl -o ossutil-2.1.2-linux-amd64.zip https://gosspublic.alicdn.com/ossutil/v2/2.1.2/ossutil-2.1.2-linux-amd64.zip
#     unzip ossutil-2.1.2-linux-amd64.zip
#     cd ossutil-2.1.2-linux-amd64
#     chmod 755 ossutil
#     sudo mv ossutil /usr/local/bin/ && sudo ln -s /usr/local/bin/ossutil /usr/bin/ossutil
# fi

export PATH=$PATH:/usr/local/go/bin

# --- 7. 自动启动环境 (关键) ---
# 每次打开终端都会自动进入 4D-humans 环境
if conda info --envs | grep -q '4D-humans'; then
    conda activate 4D-humans
fi
# --- 7. Go 安装及环境变量向导 ---
export PATH=$PATH:/usr/local/go/bin
if ! command -v go &> /dev/null; then
    echo "⚠️ Go not found. Downloading and installing..."
    wget -qO go.tar.gz https://go.dev/dl/go1.22.2.linux-amd64.tar.gz
    rm -rf /usr/local/go && tar -C /usr/local -xzf go.tar.gz
    rm go.tar.gz
    echo "✅ Go installed to /usr/local/go"
fi

# --- 8. 自动启动 Mihomo 代理引擎 ---
# 如果进程没在跑，就去对应目录后台拉起
if ! pgrep -x "mihomo" > /dev/null; then
    echo "⚙️ Starting Mihomo proxy in background..."
    nohup /cpfs_infra/shared/yangz/mihomo/mihomo -d /cpfs_infra/shared/yangz/mihomo > /cpfs_infra/shared/yangz/mihomo/mihomo.log 2>&1 &
    sleep 2 # 给进程启动时间
    echo "✅ Mihomo proxy started."
else
    echo "✅ Mihomo proxy is already running."
fi

# --- 9. 设定全局代理环境变量 ---
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export all_proxy="socks5://127.0.0.1:7890"
export ALL_PROXY="socks5://127.0.0.1:7890"
export no_proxy=localhost,127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,100.64.0.0/10,.cn

echo "✅ Network Proxy Environment Variables Set"
echo "   http_proxy  = $http_proxy"
echo "   https_proxy = $https_proxy"