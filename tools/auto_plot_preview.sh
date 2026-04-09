#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
PY=/home/yangz/.conda/envs/4D-humans/bin/python

ROOT=/home/yangz/NViT-master
ART=$ROOT/artifacts
OUTDIR=$ART/preview_latest
mkdir -p $OUTDIR

PARETO_PY=$ROOT/tools/plot_pareto.py
LAYER_PY=$ROOT/tools/plot_layerwise_metrics.py

echo "Starting auto-plot preview service..."
echo "Monitoring artifacts directory: $ART"
echo "Outputs will be saved to: $OUTDIR"

while true; do
  # Find latest CSV
  CSV=$(find $ART -name "lightweight_controls.csv" | xargs ls -t 2>/dev/null | head -n 1 || true)
  if [[ -n "${CSV}" ]]; then
    echo "[$(date +%T)] Updating Pareto plots from $CSV"
    $PY $PARETO_PY --csv_glob "$CSV" --dataset_key "3DPW" --out "$OUTDIR/pareto_3dpw.png" || true
    $PY $PARETO_PY --csv_glob "$CSV" --dataset_key "H36M" --out "$OUTDIR/pareto_h36m.png" || true
  fi

  # Find latest JSONS
  JSONS=$(find $ART -name "layerwise_metrics*.json" | xargs ls -t 2>/dev/null | head -n 10 || true)
  if [[ -n "${JSONS}" ]]; then
    echo "[$(date +%T)] Updating Layer-wise plots"
    # Note: LAYER_PY takes a glob, we pass the directory pattern
    $PY $LAYER_PY --json_glob "$ART/**/layerwise_metrics*.json" --metric kmi    --out "$OUTDIR/kti_layerwise.png" || true
    $PY $LAYER_PY --json_glob "$ART/**/layerwise_metrics*.json" --metric entropy --out "$OUTDIR/entropy_layerwise.png" || true
    $PY $LAYER_PY --json_glob "$ART/**/layerwise_metrics*.json" --metric rank    --out "$OUTDIR/erank_layerwise.png" || true
  fi

  sleep 60
done
