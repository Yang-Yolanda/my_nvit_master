#!/usr/bin/env bash
# 误拼 mnt_nvim 时仍可用；实际执行 mnt_nvit_compare.sh
# 日常出表见 run_ch6_external_mnt_nvit_compare.sh 文件头与 artifacts/ch6_best_vs_baselines.py
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/run_ch6_external_mnt_nvit_compare.sh" "$@"
