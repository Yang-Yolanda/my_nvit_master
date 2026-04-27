#!/usr/bin/env bash
# NViT unified eval entrypoints (single-GPU or multi-GPU cluster layout).
# 常用预设（full8 / ch6 八卡分片 / 画图）：scripts/workflow_unified_eval.sh
#
# Modes (first argument):
#   (empty) | maxstep — one process: max-step checkpoint per run folder (default: physical GPU 6).
#   prepare-data — create eval image dirs + write eval_data_status.txt;
#       FETCH_HR_LSPET=1 — stream-download HR-LSPET (~2.7GiB, no proxy; long — use tmux).
#   download-hr-lspet — same as prepare-data with FETCH_HR_LSPET=1.
#   finish-hr-lspet — unzip $HUMANS_ROOT/_hr-lspet.zip after manual wget -c completes.
#   cluster — parallel layout (defaults = 8 张物理卡常见排布):
#       SMPLer → CLUSTER_SMPLER_GPU (default 0)
#       Ch6 all-step eval → CH6_GPU_LIST 的第一张卡（默认 "1"）；多张卡则自动分片（见下）
#       Ch5 ablation M0..M5 → CH5_GPU_LIST 的 6 张卡（默认 2,3,4,5,6,7）
#   smpler-only — only SMPLer path on GPU 0 (same as former run_eval_suite_final.sh).
#
# Environment (examples):
#   CLUSTER_OUT_DIR — 结果根目录（默认 $ROOT/artifacts/eval_unified）
#   CLUSTER_RUN_ID — 日志子目录名（默认时间戳）；日志在 $CLUSTER_OUT_DIR/cluster_logs/$CLUSTER_RUN_ID/
#   CLUSTER_ENABLE_CH5 / CLUSTER_ENABLE_CH6 / CLUSTER_ENABLE_SMPLER — 设为 0 可跳过对应块（只跑你想跑的部分）
#   CH5_GPU_LIST="2,3,4,5,6,7" — ch5 六路，每张卡一个消融组（需 6 个 GPU id）
#   CH6_GPU_LIST — ch6 使用的物理 GPU：
#       单个 id（默认 1）→ 一张卡跑全部 ch6 checkpoint；
#       多个 id（如 "0,1,2,3,4,5,6,7"）→ 每张卡一个进程，按 checkpoint 轮转分片（--ch6-shard-*）
#       注意：子 shell 已 export CUDA_VISIBLE_DEVICES=物理卡号；不要再传 --cuda-visible-devices 0，
#       否则 unified_eval_batch 会把环境覆盖成只用卡 0（见 scripts/unified_eval_batch.py 里对 env 的赋值）。
#   CLUSTER_CH6_GPU — 兼容旧名；未设置 CH6_GPU_LIST 时等价于 CH6_GPU_LIST=$CLUSTER_CH6_GPU
#   CLUSTER_SMPLER_GPU (default 0)
#   CLUSTER_LIMIT_BATCHES — smoke 时传给 NViT ch5/ch6
#   CLUSTER_DATASETS — cluster 默认 ALL
#   CLUSTER_SKIP_EXISTING_JSON — 默认 1：ch5/ch6 若对应 json 已存在则跳过该 checkpoint（只补未测）；设为 0 强制重算
#   CH6_EXPERIMENT_DIR — 可选，cluster 的 ch6 段传给 unified_eval_batch --ch6-experiment-dir
#       须指向**具体实验根目录**（其下为 train/runs/<日期>/checkpoints/step_step=*.ckpt）；
#       不能指向只含多实验子目录的父级（如 .../nvit_output 本身，下面应是 ch6_xxx/ 再 train/runs）。
#       未设置时默认 <本脚本 NViT>/output/ch6。绝对路径、或 $ROOT/output/ch6_xxx 均可。
#       若已设置且 CLUSTER_ENABLE_CH6=1：cluster 全部 wait 结束后，会按 metrics_master 中
#       与 artifacts/ch6_best_vs_baselines.py 相同的 composite 规则，把该实验目录下最优一行
#       追加到 CH6_APPEND_COMPOSITE_BEST_FILE（默认 $CLUSTER_OUT_DIR/ch6_experiment_composite_best.log）。
#       设 CH6_APPEND_COMPOSITE_BEST_LINE=0 可关闭此追加。
#   CH5_ABLATION_MAX_STEP_ONLY — 设为 1：ch5 消融每组只评 step 最大的 ckpt（仍写 ablation/ch5/）；需与 --ch5-ablation-all-steps 同用
#   HUMANS_ROOT, HMR2_CFG_REFERENCE_CKPT (required for smpler-only / cluster SMPLer)
#   SMPLER_ROOT, SMPLER_CKPT_3DPW, SMPLER_CKPT_H36M
#
set -euo pipefail
set -o pipefail

ROOT="/cpfs_infra/shared/yangz/NViT-master"
EVAL_OUT_DIR="${CLUSTER_OUT_DIR:-$ROOT/artifacts/eval_unified}"
export PYTHON="${PYTHON:-/cpfs_infra/shared/yangz/opt/Miniconda3/envs/4D-humans/bin/python}"
export PYTHONUNBUFFERED=1
cd "$ROOT"
export PYTHONPATH="$ROOT:/cpfs_infra/shared/yangz/4D-Humans:${PYTHONPATH:-}"
export HUMANS_ROOT="${HUMANS_ROOT:-/cpfs_infra/shared/yangz/4D-Humans}"

MODE="${1:-maxstep}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATASETS="${DATASETS:-3DPW-TEST}"

# Optional 3DPW image root (same logic as former run_eval_suite_final.sh)
if [[ -z "${HMR2_EVAL_IMG_DIR_3DPW:-}" ]]; then
  if [[ -d "${HUMANS_ROOT}/data/3DPW/imageFiles" ]]; then
    export HMR2_EVAL_IMG_DIR_3DPW="${HUMANS_ROOT}/data/3DPW"
  elif [[ -d /cpfs_infra/shared/yangz/data/3DPW.backup ]]; then
    export HMR2_EVAL_IMG_DIR_3DPW="/cpfs_infra/shared/yangz/data/3DPW.backup"
  fi
fi
# H36M: set explicit image root so CH5 rebase is unambiguous (cpfs / multi-root setups).
if [[ -z "${HMR2_EVAL_IMG_DIR_H36M:-}" && -d "${HUMANS_ROOT}/data/h36m" ]]; then
  export HMR2_EVAL_IMG_DIR_H36M="${HUMANS_ROOT}/data/h36m"
fi

run_smpler_suite() {
  # Runs on caller's CUDA_VISIBLE_DEVICES (use physical GPU 0 in cluster mode).
  local OUT="${ROOT}/artifacts/external_baselines/SMPLer"
  mkdir -p "${OUT}/logs"
  if [[ -z "${HMR2_CFG_REFERENCE_CKPT:-}" ]]; then
    echo "ERROR: Set HMR2_CFG_REFERENCE_CKPT to a loadable NViT/HMR2 checkpoint." >&2
    return 1
  fi
  local SMPLER_ROOT="${SMPLER_ROOT:-/home/yangz/external_baselines/SMPLer}"
  local SMPLER_CKPT_3DPW="${SMPLER_CKPT_3DPW:-${SMPLER_ROOT}/pretrained/SMPLer_3dpw.pt}"
  local SMPLER_CKPT_H36M="${SMPLER_CKPT_H36M:-${SMPLER_ROOT}/pretrained/SMPLer_h36m.pt}"
  local DATA_DIR="${HMR2_EVAL_DATA_DIR:-${HUMANS_ROOT}/hmr2_evaluation_data}"

  # SMPLer: default num_workers=0 (cpfs/DSW: multi-worker imread can race or fail; single loader is stable).
  local _SMPLER_BS="${SMPLER_BATCH_SIZE:-${BATCH_SIZE:-16}}"
  local _SMPLER_NW="${SMPLER_NUM_WORKERS:-0}"
  run_one() {
    local name="$1"
    local ds="$2"
    local mode="$3"
    local ckpt="$4"
    local lim=()
    if [[ -n "${CLUSTER_LIMIT_BATCHES:-}" ]]; then
      lim=(--limit_batches "${CLUSTER_LIMIT_BATCHES}")
    fi
    "${PYTHON}" "${ROOT}/nvit/eval_smpler_ch5.py" \
      --ckpt "${HMR2_CFG_REFERENCE_CKPT}" \
      --smpler_ckpt "${ckpt}" \
      --smpler_root "${SMPLER_ROOT}" \
      --dataset "${ds}" \
      --data_mode "${mode}" \
      --data_dir "${DATA_DIR}" \
      --batch_size "${_SMPLER_BS}" \
      --num_workers "${_SMPLER_NW}" \
      --gpu 0 \
      "${lim[@]}" \
      --output "${OUT}/smpler_${name}.json" \
      2>&1 | tee "${OUT}/logs/${name}.log"
  }

  echo "=== SMPLer 3DPW-TEST ==="
  run_one "3dpw" "3DPW-TEST" "3dpw" "${SMPLER_CKPT_3DPW}"
  echo "=== SMPLer H36M-VAL-P2 ==="
  run_one "h36m" "H36M-VAL-P2" "h36m" "${SMPLER_CKPT_H36M}"

  "${PYTHON}" "${ROOT}/nvit/external_baselines/aggregate_smpler_results.py" \
    --project_root "${ROOT}" \
    --out_csv "${OUT}/results.csv"

  "${PYTHON}" "${ROOT}/scripts/unified_eval_batch.py" \
    --python "${PYTHON}" \
    --out-dir "${EVAL_OUT_DIR}" \
    --ingest-smpler-json \
    --smpler-chapter ch6 \
    --smpler-3dpw-json "${OUT}/smpler_3dpw.json" \
    --smpler-h36m-json "${OUT}/smpler_h36m.json" \
    --skip-nvit
  echo "SMPLer artifacts: ${OUT}/results.csv + metrics_master.csv rows"
}

case "$MODE" in
  maxstep)
    LOG="$EVAL_OUT_DIR/tmux_best_ch5_ch6_maxstep_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "$LOG")"
    echo "Logging to $LOG"
    stdbuf -oL -eL env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}" \
      "$PYTHON" "$ROOT/scripts/unified_eval_batch.py" \
      --python "$PYTHON" \
      --gpu 0 \
      --out-dir "$EVAL_OUT_DIR" \
      --chapters ch5,ch6 \
      --checkpoint-mode max-step-per-run \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --datasets "$DATASETS" \
      2>&1 | tee "$LOG"
    ;;

  smpler-only)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    run_smpler_suite
    ;;

  prepare-data)
    EXTRA=()
    if [[ "${FETCH_HR_LSPET:-0}" == "1" ]]; then
      EXTRA+=(--fetch-hr-lspet)
    fi
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
      "$PYTHON" "$ROOT/scripts/unified_eval_batch.py" --prepare-eval-layout "${EXTRA[@]}"
    echo "See artifacts/eval_unified/eval_data_status.txt"
    ;;

  download-hr-lspet)
    export FETCH_HR_LSPET=1
    exec bash "$0" prepare-data
    ;;

  finish-hr-lspet)
    env -u http_proxy -u https_proxy -u ALL_PROXY \
      "$PYTHON" "$ROOT/scripts/unified_eval_batch.py" --finish-hr-lspet
    ;;

  cluster)
    CLUSTER_ENABLE_CH5="${CLUSTER_ENABLE_CH5:-1}"
    CLUSTER_ENABLE_CH6="${CLUSTER_ENABLE_CH6:-1}"
    CLUSTER_ENABLE_SMPLER="${CLUSTER_ENABLE_SMPLER:-1}"
    CLUSTER_SMPLER_GPU="${CLUSTER_SMPLER_GPU:-0}"
    CH6_GPU_LIST="${CH6_GPU_LIST:-${CLUSTER_CH6_GPU:-1}}"
    IFS=',' read -r -a CH6_GPUS <<< "${CH6_GPU_LIST// /}"
    CH6_N="${#CH6_GPUS[@]}"
    CH6_EXP_ARGS=()
    if [[ -n "${CH6_EXPERIMENT_DIR:-}" ]]; then
      if [[ ! -d "$CH6_EXPERIMENT_DIR" && "$CH6_EXPERIMENT_DIR" == /output/* ]]; then
        CH6_EXPERIMENT_DIR="${ROOT}/output/${CH6_EXPERIMENT_DIR#/output/}"
        echo "已修正 CH6_EXPERIMENT_DIR 为 $CH6_EXPERIMENT_DIR（原 /output/... 不存在，多半因未在 shell 中 export ROOT）" >&2
      fi
      CH6_EXP_ARGS=(--ch6-experiment-dir "${CH6_EXPERIMENT_DIR}")
      echo "CH6_EXPERIMENT_DIR=${CH6_EXPERIMENT_DIR} — ch6 checkpoints 来自该目录（非默认 output/ch6）"
    fi

    IFS=',' read -r -a GPUS <<< "${CH5_GPU_LIST:-2,3,4,5,6,7}"
    # Not named GROUPS — that is a readonly bash built-in (user's group ids).
    CH5_ABLATION_GROUPS=(M0_NoMask M1_Pos16 M2_Pos24 M3_8PlusSoft M4_AdaptiveKTI M5_8PlusHard)
    if [[ "$CLUSTER_ENABLE_CH5" == "1" ]]; then
      if [[ "${#GPUS[@]}" -ne "${#CH5_ABLATION_GROUPS[@]}" ]]; then
        echo "ERROR: need 6 GPUs in CH5_GPU_LIST (got ${#GPUS[@]}). Set CLUSTER_ENABLE_CH5=0 to skip ch5." >&2
        exit 1
      fi
    fi

    CLUSTER_DATASETS="${CLUSTER_DATASETS:-ALL}"
    RUN_TAG="${CLUSTER_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
    LOGDIR="$EVAL_OUT_DIR/cluster_logs/$RUN_TAG"
    mkdir -p "$LOGDIR"
    echo "Cluster logs: $LOGDIR"
    echo "EVAL_OUT_DIR=$EVAL_OUT_DIR"
    LIMIT_ARGS=()
    if [[ -n "${CLUSTER_LIMIT_BATCHES:-}" ]]; then
      LIMIT_ARGS=(--limit-batches "${CLUSTER_LIMIT_BATCHES}")
      echo "CLUSTER_LIMIT_BATCHES=${CLUSTER_LIMIT_BATCHES} (smoke mode)"
    fi

    CLUSTER_SKIP_EXISTING_JSON="${CLUSTER_SKIP_EXISTING_JSON:-1}"
    SKIP_EXISTING_ARGS=()
    if [[ "${CLUSTER_SKIP_EXISTING_JSON}" == "1" ]]; then
      SKIP_EXISTING_ARGS=(--skip-existing-json)
      echo "CLUSTER_SKIP_EXISTING_JSON=1 — skip checkpoints that already have json under out-dir (resume)."
    fi

    CH5_MAXSTEP_ONLY_ARGS=()
    if [[ "${CH5_ABLATION_MAX_STEP_ONLY:-0}" == "1" ]]; then
      CH5_MAXSTEP_ONLY_ARGS=(--ch5-ablation-max-step-only)
      echo "CH5_ABLATION_MAX_STEP_ONLY=1 — ch5 ablation: latest step only per group."
    fi

    if [[ "$CLUSTER_ENABLE_CH5" == "1" ]]; then
      rm -f "$EVAL_OUT_DIR/ablation/ch5/summary_best_composite.csv"
      for i in "${!CH5_ABLATION_GROUPS[@]}"; do
        G="${CH5_ABLATION_GROUPS[$i]}"
        (
          export CUDA_VISIBLE_DEVICES="${GPUS[$i]}"
          stdbuf -oL -eL "$PYTHON" "$ROOT/scripts/unified_eval_batch.py" \
            --python "$PYTHON" \
            --gpu 0 \
            --out-dir "$EVAL_OUT_DIR" \
            --datasets "$CLUSTER_DATASETS" \
            --use-mean-alignment \
            --batch-size "$BATCH_SIZE" \
            --num-workers "$NUM_WORKERS" \
            "${LIMIT_ARGS[@]}" \
            --ch5-ablation-all-steps \
            --ablation-groups "$G" \
            --ch5-summary-append \
            "${CH5_MAXSTEP_ONLY_ARGS[@]}" \
            "${SKIP_EXISTING_ARGS[@]}" \
            2>&1 | tee "$LOGDIR/ch5_${G}.log"
        ) &
        echo "Started ch5 $G on GPU ${GPUS[$i]} (pid $!)"
      done
    else
      echo "CLUSTER_ENABLE_CH5=0 — skip ch5 ablation workers."
    fi

    if [[ "$CLUSTER_ENABLE_CH6" == "1" ]]; then
      if [[ "$CH6_N" -lt 1 ]]; then
        echo "ERROR: CH6_GPU_LIST empty." >&2
        exit 1
      fi
      if [[ "$CH6_N" -eq 1 ]]; then
        (
          export CUDA_VISIBLE_DEVICES="${CH6_GPUS[0]}"
          stdbuf -oL -eL "$PYTHON" "$ROOT/scripts/unified_eval_batch.py" \
            --python "$PYTHON" \
            --gpu 0 \
            --out-dir "$EVAL_OUT_DIR" \
            --datasets "$CLUSTER_DATASETS" \
            --use-mean-alignment \
            --batch-size "$BATCH_SIZE" \
            --num-workers "$NUM_WORKERS" \
            "${LIMIT_ARGS[@]}" \
            --ch6-all-steps \
            "${CH6_EXP_ARGS[@]}" \
            --ch6-shard-index 0 \
            --ch6-shard-total 1 \
            "${SKIP_EXISTING_ARGS[@]}" \
            2>&1 | tee "$LOGDIR/ch6.log"
        ) &
        echo "Started ch6 (single GPU) on ${CH6_GPUS[0]} (pid $!)"
      else
        for ((idx = 0; idx < CH6_N; idx++)); do
          (
            export CUDA_VISIBLE_DEVICES="${CH6_GPUS[$idx]}"
            stdbuf -oL -eL "$PYTHON" "$ROOT/scripts/unified_eval_batch.py" \
              --python "$PYTHON" \
              --gpu 0 \
              --out-dir "$EVAL_OUT_DIR" \
              --datasets "$CLUSTER_DATASETS" \
              --use-mean-alignment \
              --batch-size "$BATCH_SIZE" \
              --num-workers "$NUM_WORKERS" \
              "${LIMIT_ARGS[@]}" \
              --ch6-all-steps \
              "${CH6_EXP_ARGS[@]}" \
              --ch6-shard-index "$idx" \
              --ch6-shard-total "$CH6_N" \
              "${SKIP_EXISTING_ARGS[@]}" \
              2>&1 | tee "$LOGDIR/ch6_shard$((idx + 1))of${CH6_N}.log"
          ) &
          echo "Started ch6 shard $((idx + 1))/${CH6_N} on GPU ${CH6_GPUS[$idx]} (pid $!)"
        done
      fi
    else
      echo "CLUSTER_ENABLE_CH6=0 — skip ch6."
    fi

    if [[ "$CLUSTER_ENABLE_SMPLER" == "1" ]]; then
      (
        export CUDA_VISIBLE_DEVICES="$CLUSTER_SMPLER_GPU"
        run_smpler_suite
      ) >"$LOGDIR/smpler.log" 2>&1 &
      echo "Started SMPLer on GPU $CLUSTER_SMPLER_GPU (pid $!)"
    else
      echo "CLUSTER_ENABLE_SMPLER=0 — skip SMPLer."
    fi

    wait
    echo "Cluster eval finished. Logs: $LOGDIR"

    if [[ -n "${CH6_EXPERIMENT_DIR:-}" && "$CLUSTER_ENABLE_CH6" == "1" ]] \
      && [[ "${CH6_APPEND_COMPOSITE_BEST_LINE:-1}" != "0" ]]; then
      CH6_SUB="$(basename "${CH6_EXPERIMENT_DIR%/}")"
      CH6_BEST_LOG="${CH6_APPEND_COMPOSITE_BEST_FILE:-$EVAL_OUT_DIR/ch6_experiment_composite_best.log}"
      echo "=== composite best 追加一行 -> $CH6_BEST_LOG (checkpoint 子串: $CH6_SUB) ==="
      "$PYTHON" "$ROOT/artifacts/append_ch6_experiment_composite_best_line.py" \
        --metrics-csv "$EVAL_OUT_DIR/metrics_master.csv" \
        --checkpoint-path-contains "$CH6_SUB" \
        --append-to "$CH6_BEST_LOG" \
        || echo "[warn] append_ch6_experiment_composite_best_line.py 失败" >&2
    fi
    ;;

  *)
    echo "Unknown mode: $MODE (use: maxstep | cluster | smpler-only | prepare-data | download-hr-lspet | finish-hr-lspet)" >&2
    exit 1
    ;;
esac
