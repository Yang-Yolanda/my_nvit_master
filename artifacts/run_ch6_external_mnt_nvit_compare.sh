#!/usr/bin/env bash
#
# 【何时才跑本脚本】仅当需要：同步 mnt 镜像、对外部 *.ckpt/*.pth 全量跑 standard_eval、
# 更新 manifest、重画 ch6_external_vs_ch6best.*、跑推理 bench 时。
# 【日常不跑】与 METRO / Mesh Graphormer 对齐的 NViT 论文式表格，应用：
#     python3 artifacts/ch6_best_vs_baselines.py
#   由 artifacts/eval_unified/metrics_master.csv（及可选 REFERENCE_* 常数）出表，无需每次走本套外部测评。
#
# 1) 同步 /mnt/shared/yangz/nvit 或 /mnt/yangz/nvit → outputs/.../mnt_nvit_mirror
# 2) 对每个 *.ckpt 跑 standard_eval ALL
# 3) 文件名含 mid_heavy 的剪枝 .pth：--pth-ref-ckpt 重载 MLP 后 standard_eval；其它 .pth 记 skip
# 4) 写 manifest + 图/表 ch6_external_vs_ch6best.*
# 5) 多模型 bench（含 Ch6 best + 各 external ckpt）
# 环境变量: GPU, HUMANS_ROOT, HMR2_EVAL_DATA_DIR, CH6_EVAL_JSON, CH6_CKPT, NVIT_CH6_PARAMS_M,
#          EXTERNAL_ROOT=…（优先）, METRICS_MASTER=…（ingest 目标 CSV）
# 跳过 GPU 评测、直接读已有 standard_eval JSON（如 hmr2_mid_heavy.pth 已测过）:
#   CH6_EXTERNAL_SKIP_STANDARD_EVAL=1 \
#   CH6_EXTERNAL_REUSE_JSON="/abs/a.json;/abs/b.json" \
#   bash artifacts/run_ch6_external_mnt_nvim_compare.sh
#   此时默认同时跳过 mnt 同步（无需 NFS）；若要仍同步镜像: CH6_EXTERNAL_SKIP_RSYNC=0 且配置 EXTERNAL_ROOT。
# 续行请保持 Unix LF；若曾用 CRLF 编辑本文件，行尾 \ 会失效并出现「--dataset: command not found」类错误。
set -euo pipefail
ROOT="/cpfs_infra/shared/yangz/NViT-master"
cd "$ROOT"
export PYTHONPATH="${ROOT}:/cpfs_infra/shared/yangz/4D-Humans${PYTHONPATH:+:$PYTHONPATH}"
export HUMANS_ROOT="${HUMANS_ROOT:-/cpfs_infra/shared/yangz/4D-Humans}"
PY="${PYTHON:-/cpfs_infra/shared/yangz/opt/Miniconda3/envs/4D-humans/bin/python}"
DATA_DIR="${HMR2_EVAL_DATA_DIR:-${HUMANS_ROOT}/hmr2_evaluation_data}"
GPU="${GPU:-0}"

CH6_CKPT="${CH6_CKPT:-/mnt/yangz/nvit_output/ch6/train/runs/2026-04-17_13-28-24/checkpoints/step_step=492000.ckpt}"
CH6_EVAL_JSON="${CH6_EVAL_JSON:-${ROOT}/artifacts/eval_unified/json/nvit/ch6_2026-04-17_13-28-24_step_492000.json}"
NVIT_CH6_PARAMS_M="${NVIT_CH6_PARAMS_M:-208.128}"
METRICS_MASTER="${METRICS_MASTER:-${ROOT}/artifacts/eval_unified/metrics_master.csv}"
OUT_STEM="ch6_external_vs_ch6best"
TAG="$(date +%Y%m%d_%H%M%S)"
JSON_DIR="${ROOT}/artifacts/eval_unified/logs"
LOG="${JSON_DIR}/ch6_external_compare_${TAG}.log"
export LOG
mkdir -p "$JSON_DIR" "${ROOT}/outputs/eval_global/Ch6A" "${ROOT}/artifacts/eval_unified/json"
exec > >(tee -a "$LOG") 2>&1
echo "=== 日志: $LOG ==="

SKIP_EVAL="${CH6_EXTERNAL_SKIP_STANDARD_EVAL:-0}"
SKIP_RSYNC="${CH6_EXTERNAL_SKIP_RSYNC:-0}"
REUSE_JSON="${CH6_EXTERNAL_REUSE_JSON:-}"

if [[ "$SKIP_EVAL" == "1" ]]; then
  SKIP_RSYNC="${CH6_EXTERNAL_SKIP_RSYNC:-1}"
  if [[ -z "$REUSE_JSON" ]]; then
    echo "[err] CH6_EXTERNAL_SKIP_STANDARD_EVAL=1 时必须设置 CH6_EXTERNAL_REUSE_JSON（分号分隔多个 JSON 绝对路径）"
    exit 1
  fi
fi

MIRROR="${ROOT}/outputs/eval_global/Ch6A/mnt_nvit_mirror"
mkdir -p "$MIRROR"

if [[ "$SKIP_RSYNC" != "1" ]]; then
  if [[ -n "${EXTERNAL_ROOT:-}" && -d "${EXTERNAL_ROOT}" ]]; then
    SRC="$EXTERNAL_ROOT"
  else
    SRC=""
    for d in "/mnt/shared/yangz/nvit" "/mnt/yangz/nvit"; do
      if [[ -d "$d" ]]; then
        SRC="$d"
        break
      fi
    done
  fi
  if [[ -z "$SRC" || ! -d "$SRC" ]]; then
    echo "[err] 未找到外部目录: 可设 EXTERNAL_ROOT=…（还试过 /mnt/shared/yangz/nvit, /mnt/yangz/nvit）"
    exit 1
  fi
  echo "=== 同步: $SRC/ -> $MIRROR/ ==="
  cp -a "$SRC/." "$MIRROR/"
else
  SRC=""
  echo "=== 跳过 mnt 同步（CH6_EXTERNAL_SKIP_RSYNC=1）==="
fi

# 剪枝 hmr2_mid_heavy：需与 OSS yaml 中 base 相同的 HMR2 Lightning 参考断点
HMR2_PTH_REF="${HMR2_PTH_REF_CKPT:-/cpfs_infra/shared/yangz/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt}"
export HMR2_PTH_REF_CKPT="${HMR2_PTH_REF}"

echo "=== 拉取 OSS 元数据 (可选) ==="
YAML_OSS="oss://kai-ego/nvit/ablation/hmr2_mid_heavy_model_config.yaml"
YAML_DST="${MIRROR}/ablation/hmr2_mid_heavy_model_config.yaml"
mkdir -p "${MIRROR}/ablation"
if command -v ossutil >/dev/null 2>&1; then
  if [[ ! -f "$YAML_DST" ]]; then
    ossutil cp "$YAML_OSS" "$YAML_DST" 2>&1 || echo "[warn] ossutil 拉 yaml 失败，可忽略"
  else
    echo "已有: $YAML_DST"
  fi
else
  echo "无 ossutil，跳过 yaml 下载"
fi

# 行记录 → python 再写正式 manifest
LIST_F="$(mktemp)"
BENCH_SNIP="$(mktemp)"
export LIST_F
trap 'rm -f "$LIST_F" "$BENCH_SNIP"' EXIT

BENCH_ENTRY_ARGS=()
# Ch6
BENCH_ENTRY_ARGS+=(--entry "NViT_ch6_best=$CH6_CKPT")

if [[ "$SKIP_EVAL" == "1" ]]; then
  echo "=== 跳过 standard_eval，复用已有 JSON（CH6_EXTERNAL_SKIP_STANDARD_EVAL=1）==="
  "$PY" "$ROOT/artifacts/ch6_external_register_reuse_eval_json.py" \
    --list-file "$LIST_F" \
    --bench-snippet "$BENCH_SNIP" \
    --also-ingest-metrics "$METRICS_MASTER" \
    --paths-list "$REUSE_JSON"
  # shellcheck disable=SC1090
  source "$BENCH_SNIP"
fi

while IFS= read -r -d '' c; do
  [[ "$SKIP_EVAL" == "1" ]] && break
  [[ -z "$c" ]] && continue
  name="$(basename "$c")"
  name="${name%.*}"
  outj="${ROOT}/artifacts/eval_unified/json/external_mnt_nvit_${name}_${TAG}.json"
  echo "=== standard_eval ALL: $c ==="
  se_ckpt=(
    "$PY" -m nvit.skills.evaluate_model.standard_eval
    --ckpt "$c"
    --dataset ALL
    --gpu "$GPU"
    --use_mean_alignment
    --data_dir "$DATA_DIR"
    --output "$outj"
  )
  if "${se_ckpt[@]}"; then
    echo "OK_JSON::$outj::$c::$name" >> "$LIST_F"
    BENCH_ENTRY_ARGS+=(--entry "${name}=$c")
    "$PY" "$ROOT/artifacts/ingest_eval_json_to_metrics_master.py" \
      --json "$outj" --metrics-csv "$METRICS_MASTER" \
      --family external_mnt --chapter ch6_external --experiment "$name" \
      || echo "[warn] ingest metrics_master 失败（可忽略后手动跑 ingest 脚本）"
  else
    echo "FAIL::$c" >> "$LIST_F"
  fi
done < <(find "$MIRROR" -type f \( -name '*.ckpt' -o -name '*.CKPT' \) -print0 2>/dev/null)

while IFS= read -r -d '' p; do
  [[ "$SKIP_EVAL" == "1" ]] && break
  [[ -z "$p" ]] && continue
  bn="${p##*/}"; bn="${bn%.*}"
  pl="${p,,}"
  if [[ "$pl" == *"mid_heavy"* ]] && [[ -f "$HMR2_PTH_REF" ]]; then
    outj="${ROOT}/artifacts/eval_unified/json/external_mnt_nvit_${bn}_pth_${TAG}.json"
    echo "=== standard_eval ALL (pruned pth + HMR2 ref): $p ==="
    se_pth=(
      "$PY" -m nvit.skills.evaluate_model.standard_eval
      --ckpt "$p"
      --pth-ref-ckpt "$HMR2_PTH_REF"
      --dataset ALL
      --gpu "$GPU"
      --use_mean_alignment
      --data_dir "$DATA_DIR"
      --output "$outj"
    )
    if "${se_pth[@]}"; then
      echo "OK_JSON::$outj::$p::$bn" >> "$LIST_F"
      BENCH_ENTRY_ARGS+=(--entry "${bn}=${p}")
      "$PY" "$ROOT/artifacts/ingest_eval_json_to_metrics_master.py" \
        --json "$outj" --metrics-csv "$METRICS_MASTER" \
        --family external_mnt --chapter ch6_external --experiment "${bn}_pth" \
        || echo "[warn] ingest metrics_master 失败（可忽略后手动跑 ingest 脚本）"
    else
      echo "FAIL::$p" >> "$LIST_F"
    fi
  else
    echo "PTH::$p::$bn" >> "$LIST_F"
  fi
done < <(find "$MIRROR" -type f \( -name '*.pth' -o -name '*.PTH' -o -name '*.pt' -o -name '*.PT' \) -print0 2>/dev/null)

MANIFEST="${ROOT}/artifacts/eval_unified/json/ch6_external_compare_${TAG}.json"
export OUT_STEM
export CH6_CKPT
export CH6_EVAL_JSON
export NVIT_CH6_PARAMS_M
export TAG
export MIRROR SRC MANIFEST
"$PY" - <<'EOPY'
import json
import os
import pathlib

ch6 = {
    "row_label": "NViT (Ch6 best, step=492k)",
    "label": "composite",
    "eval_json": os.environ["CH6_EVAL_JSON"],
    "checkpoint": os.environ["CH6_CKPT"],
    "params_m": float(os.environ.get("NVIT_CH6_PARAMS_M", "208.128")),
}
rows: list[dict] = []
list_f = os.environ.get("LIST_F", "")
if list_f and pathlib.Path(list_f).is_file():
    for line in pathlib.Path(list_f).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("OK_JSON::"):
            s = line[len("OK_JSON::") :]
            p1, s2 = s.split("::", 1)
            p2, nm = s2.rsplit("::", 1) if "::" in s2 else (s2, "ext")
            jpath, ck = p1, p2
            rows.append(
                {
                    "kind": "external",
                    "label": nm,
                    "status": "ok",
                    "eval_json": jpath,
                    "params_m": 0.0,
                    "checkpoint": ck,
                    "source": "mnt_nvit_mirror",
                }
            )
        elif line.startswith("FAIL::"):
            ckf = line[6:]
            rows.append(
                {
                    "kind": "external",
                    "label": pathlib.Path(ckf).stem or "ckpt",
                    "status": "eval_failed",
                    "checkpoint": ckf,
                    "note": "standard_eval 失败，见本运行日志",
                }
            )
        elif line.startswith("PTH::"):
            rest = line[len("PTH::") :]
            pth, bn = rest.rsplit("::", 1) if "::" in rest else (rest, "")
            if not bn or bn == pth:
                import os

                bn = os.path.splitext(os.path.basename(pth))[0]
            rows.append(
                {
                    "kind": "external",
                    "label": bn,
                    "status": "skipped_pth",
                    "params_m": 0.0,
                    "checkpoint": pth,
                    "note": "仅 state_dict/剪枝 backbone；与默认 HMR2 全结构不一致。需同实验 config + LayerAblation 建网后再标准评测。",
                }
            )
mp = {
    "ch6": ch6,
    "output_stem": os.environ.get("OUT_STEM", "ch6_external_vs_ch6best"),
    "rows": rows,
    "source_mirror": os.environ.get("MIRROR", ""),
    "source_nfs": os.environ.get("SRC", ""),
    "log": os.environ.get("LOG", ""),
}
out = os.environ["MANIFEST"]
pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(out).write_text(json.dumps(mp, ensure_ascii=False, indent=2), encoding="utf-8")
print("manifest written:", out)
EOPY
rm -f "$LIST_F"
trap - EXIT
unset LIST_F

echo "=== 对比表/图 ==="
"$PY" "$ROOT/artifacts/plot_ch6_external_manifest.py" --manifest "$MANIFEST" \
  --output-dir "$ROOT/outputs/eval_global/Ch6A"

echo "=== 推理测速 ==="
BENCH_CSV="${ROOT}/outputs/eval_global/Ch6A/ch6_external_inference_${TAG}.csv"
"$PY" "$ROOT/artifacts/bench_list_inference.py" \
  --gpu "$GPU" --batch 1 --warmup 20 --iters 100 --out-csv "$BENCH_CSV" \
  "${BENCH_ENTRY_ARGS[@]}"
echo "[ok] $BENCH_CSV"

echo "======== DONE (ch6 external) ========"
echo "  out_csv:  ${ROOT}/outputs/eval_global/Ch6A/${OUT_STEM}.csv"
echo "  out_md:   ${ROOT}/outputs/eval_global/Ch6A/${OUT_STEM}.md"
echo "  out_png:  ${ROOT}/outputs/eval_global/Ch6A/${OUT_STEM}.png"
echo "  bench:    $BENCH_CSV"
echo "  mirror:   $MIRROR"
echo "  manifest: $MANIFEST"
echo "  log:      $LOG"
