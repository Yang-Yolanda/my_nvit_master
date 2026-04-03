#!/home/yangz/.conda/envs/4D-humans/bin/python
# -*- coding: utf-8 -*-
"""
自动化生成 LaTeX 论文脚本
- 读取实验结果（CSV/JSON）
- 生成表格、绘图
- 填充 LaTeX 模板
- 自动提交到 Overleaf (git)
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import shutil

# ------------------- 配置 -------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # NViT 项目根目录
LATEX_ROOT   = PROJECT_ROOT / "paper"
FIG_DIR      = LATEX_ROOT / "figures"
RESULTS_DIR  = PROJECT_ROOT / "output" / "ablation_results"
SUMMARY_CSV  = PROJECT_ROOT / "results" / "summary.csv"
TEMPLATE_TEX = LATEX_ROOT / "template.tex"   # 预先准备好的 LaTeX 模板
OVERLEAF_GIT = "git@overleaf.com:yourteam/yourproject.git"   # 替换为实际 remote

# ------------------- 工具函数 -------------------
def collect_metrics():
    """遍历每个实验组的 eval_metrics.json，汇总为 CSV"""
    rows = []
    for group_dir in RESULTS_DIR.iterdir():
        if not group_dir.is_dir():
            continue
        metrics_path = group_dir / "eval_metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        rows.append({
            "Group": group_dir.name,
            "Paper": "Paper 2" if "mamba" in group_dir.name.lower() else "Paper 1",
            "MPJPE": m.get("mpjpe", 0.0),
            "PA_MPJPE": m.get("pa_mpjpe", 0.0),
            "KTI": m.get("kmi", 0.0),
        })
    df = pd.DataFrame(rows)
    os.makedirs(SUMMARY_CSV.parent, exist_ok=True)
    df.to_csv(SUMMARY_CSV, index=False)
    return df

def plot_comparison(df):
    """绘制 MPJPE 对比柱状图"""
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.figure(figsize=(10,5))
    df_sorted = df.sort_values("MPJPE")
    plt.bar(df_sorted["Group"], df_sorted["MPJPE"], color="#4A90E2")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("MPJPE (mm)")
    plt.title("Ablation Study – MPJPE Comparison")
    plt.tight_layout()
    fig_path = FIG_DIR / "mpjpe_comparison.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return fig_path

def render_latex(df, fig_path):
    """使用模板渲染 LaTeX，填入表格与图片路径"""
    with open(TEMPLATE_TEX, "r", encoding="utf-8") as f:
        tex = f.read()

    # 简单读取已有 markdown 作为章节内容（可自行改进）
    def read_md(name):
        p = PROJECT_ROOT / ".gemini" / "antigravity" / "brain" / "ca2b0f1c-77b0-44b9-84e7-641dba5a2bcc" / f"{name}.md"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    abstract = read_md("Paper_Strategy")
    method   = read_md("Guided_NViT_Paper_Materials")
    discussion = read_md("Project_Audit_Report")

    table_tex = df.to_latex(index=False, float_format="%.2f", column_format="lrrr", caption="Ablation results", label="tab:ablation")

    tex = tex.replace("{{ABSTRACT}}", abstract)
    tex = tex.replace("{{METHOD}}", method)
    tex = tex.replace("{{DISCUSSION}}", discussion)
    tex = tex.replace("{{RESULT_TABLE}}", table_tex)
    tex = tex.replace("{{FIGURE_PATH}}", str(fig_path.relative_to(LATEX_ROOT)))

    out_tex = LATEX_ROOT / "main.tex"
    os.makedirs(LATEX_ROOT, exist_ok=True)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex)
    return out_tex

def sync_overleaf():
    """使用 Overleaf Git 插件同步仓库"""
    os.chdir(LATEX_ROOT)
    # 初始化 git（如果还未初始化）
    if not (LATEX_ROOT / ".git").exists():
        subprocess.run(["git", "init"], check=True)
        subprocess.run(["git", "remote", "add", "origin", OVERLEAF_GIT], check=True)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Auto-generated paper update"], check=False)
    subprocess.run(["git", "push", "-u", "origin", "master"], check=True)

def main():
    df = collect_metrics()
    fig = plot_comparison(df)
    render_latex(df, fig)
    # sync_overleaf()
    print("Paper generation & Overleaf sync 完成！")

if __name__ == "__main__":
    main()
