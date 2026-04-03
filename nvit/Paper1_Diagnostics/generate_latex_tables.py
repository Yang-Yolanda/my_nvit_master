import pandas as pd
from pathlib import Path

def csv_to_latex_ablation():
    csv_path = "/home/yangz/NViT-master/nvit/Paper1_Diagnostics/finetune_eval_results.csv"
    if not Path(csv_path).exists():
        return "% Ablation data not found"
    
    df = pd.read_csv(csv_path)
    
    # Pre-processing
    df['MPJPE'] = df['MPJPE'].round(2)
    df['PA-MPJPE'] = df['PA-MPJPE'].round(2)
    
    # Sorting
    order = ["Control", "T2-Static-Late", "T2-Static-Mid", "T2-A-S-Baseline", "T2-KTI-Adaptive", "T2-A-H-Baseline"]
    df['Group'] = pd.Categorical(df['Group'], categories=order, ordered=True)
    df = df.sort_values('Group')

    latex = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Ablation Study of Masking Strategies on 3DPW Test Set. $\\downarrow$ indicates lower is better.}",
        "\\label{tab:ablation_results}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Strategy & MPJPE (mm) $\\downarrow$ & PA-MPJPE (mm) $\\downarrow$ \\\\",
        "\\midrule"
    ]
    
    for _, row in df.iterrows():
        name = row['Group'].replace('_', '\\_')
        if row['Group'] == "Control":
            name = "\\textbf{" + name + "}"
        elif row['MPJPE'] == df['MPJPE'].min():
            name = "\\textit{" + name + "} (Best)"
            
        line = f"{name} & {row['MPJPE']} & {row['PA-MPJPE']} \\\\"
        latex.append(line)
        
    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(latex)

def csv_to_latex_robustness():
    csv_path = "/home/yangz/NViT-master/nvit/Paper1_Diagnostics/robustness_results.csv"
    if not Path(csv_path).exists():
        return "% Robustness data not found"
        
    df = pd.read_csv(csv_path)
    # Pivot for table view
    pivot = df.pivot(index='Group', columns='Occlusion', values='MPJPE').round(2)
    
    latex = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Robustness Analysis: MPJPE (mm) under Systematic Patch Occlusion.}",
        "\\label{tab:robustness_results}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & Occ=0.1 & Occ=0.2 & Occ=0.3 \\\\",
        "\\midrule"
    ]
    
    for group in ["Control", "T2-KTI-Adaptive"]:
        if group in pivot.index:
            row = pivot.loc[group]
            name = "\\textbf{Ours}" if group == "T2-KTI-Adaptive" else "Control"
            line = f"{name} & {row[0.1]} & {row[0.2]} & {row[0.3]} \\\\"
            latex.append(line)
            
    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(latex)

if __name__ == "__main__":
    out_path = Path("/home/yangz/NViT-master/nvit/Paper1_Diagnostics/paper1_figures_refined/Paper1_Tables.tex")
    with open(out_path, "w") as f:
        f.write("% --- Ablation Results ---\n")
        f.write(csv_to_latex_ablation())
        f.write("\n\n% --- Robustness Results ---\n")
        f.write(csv_to_latex_robustness())
    print(f"LaTeX tables saved to {out_path}")
