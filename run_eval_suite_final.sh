#!/usr/bin/env bash
# Back-compat wrapper: SMPLer eval on GPU 0 (see scripts/run_best_max_step_eval.sh smpler-only).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${ROOT}/scripts/run_best_max_step_eval.sh" smpler-only
