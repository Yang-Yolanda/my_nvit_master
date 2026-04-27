#!/usr/bin/env bash
# Deprecated: last.ckpt is large (optimizer state) and can OOM on shared GPUs.
# Use: bash scripts/run_best_max_step_eval.sh
echo "This script is deprecated. Use scripts/run_best_max_step_eval.sh (max step_step=*.ckpt per run)." >&2
exec "$(dirname "$0")/run_best_max_step_eval.sh"
