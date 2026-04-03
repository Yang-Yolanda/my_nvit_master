import os
import sys
import pandas as pd
import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def generate_chapter_report(chapter):
    chapter_dir = BASE_DIR / "outputs" / "eval_global" / chapter
    if not chapter_dir.exists():
        print(f"❌ No results found for {chapter}")
        return

    all_data = []
    
    # 1. Collect all summary.csv from subfolders
    for run_dir in chapter_dir.iterdir():
        if run_dir.is_dir():
            summary_csv = run_dir / "metrics_suite.json" # Or aggregate from JSON if primary
            # For simplicity, we use our aggregated summary.csv created by global_evaluator
            run_summary = chapter_dir / "summary.csv" 
            if run_summary.exists():
                df = pd.read_csv(run_summary)
                # Ensure we only take data for THIS chapter
                df = df[df['Chapter'] == chapter]
                all_data = df.to_dict('records')
            break # Summary.csv is chapter-wide in the chapter root now based on my glob_eval logic

    if not all_data:
        print(f"❌ No valid data records for {chapter}")
        return

    df = pd.DataFrame(all_data)
    
    # 2. Generate Markdown Table
    md_file = chapter_dir / f"{chapter}_Global_Report.md"
    
    md_content = [f"# {chapter} 实验全局汇总报告\n"]
    md_content.append(f"生成时间: {pd.Timestamp.now()}\n")
    
    # Select key columns for the summary table
    # Standard Metrics: 3DPW_MPJPE, H36M_VAL_P2_MPJPE, COCO_VAL_KPL2, etc.
    # Standard Diags: KTI, MAD, EffectiveRank, Entropy
    
    cols = ["Run"]
    # Task Metrics
    for c in df.columns:
        if "MPJPE" in c or "KPL2" in c: cols.append(c)
    # Diag Metrics
    for c in ["KTI", "MAD", "EffectiveRank", "Entropy"]:
        if c in df.columns: cols.append(c)
        
    md_table = df[cols].to_markdown(index=False)
    md_content.append(md_table)
    
    with open(md_file, "w") as f:
        f.write("\n".join(md_content))
    
    print(f"✅ Markdown report saved to {md_file}")

    # 3. Generate LaTeX Table (Draft)
    tex_file = chapter_dir / f"{chapter}_Global_Table.tex"
    try:
        # Simplified LaTeX conversion
        latex_table = df[cols].to_latex(index=False, caption=f"{chapter} Results Summary", label=f"tab:{chapter.lower()}")
        with open(tex_file, "w") as f:
            f.write(latex_table)
        print(f"✅ LaTeX table template saved to {tex_file}")
    except:
        print("⚠️ Failed to generate LaTeX table (missing dependency?).")

def main():
    parser = argparse.ArgumentParser(description="NViT Global Reporter")
    parser.add_argument("--chapter", type=str, required=True, choices=['Ch4', 'Ch5', 'Ch6A', 'Ch6B'])
    args = parser.parse_args()
    
    generate_chapter_report(args.chapter)

if __name__ == "__main__":
    main()
