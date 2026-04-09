import argparse, glob
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_glob", type=str, required=True)
    ap.add_argument("--dataset_key", type=str, default="3DPW")
    ap.add_argument("--xcol", type=str, default=None)  # Latency_ms / FLOPs_G / Params_M
    ap.add_argument("--ycol", type=str, default="PA-MPJPE(mm)")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.csv_glob))
    if not paths:
        print(f"No csv matches: {args.csv_glob}")
        return

    df_list = []
    for p in paths:
        try:
            df_list.append(pd.read_csv(p))
        except Exception as e:
            print(f"Error reading {p}: {e}")
            
    if not df_list: return
    df = pd.concat(df_list, ignore_index=True).drop_duplicates()
    
    # Try to find the dataset key in the 'Dataset' column
    sub = df[df["Dataset"].str.contains(args.dataset_key, case=False, na=False)].copy()
    if sub.empty:
        print(f"No rows for dataset_key={args.dataset_key}")
        return

    # Auto-detect xcol if not provided
    if args.xcol is None:
        for c in ["Latency_ms", "FLOPs_G", "Params_M"]:
            if c in sub.columns:
                args.xcol = c
                break
        if args.xcol is None:
            # Fallback to the first numeric column that isn't the target
            numeric_cols = sub.select_dtypes(include=['number']).columns
            for c in numeric_cols:
                if c != args.ycol:
                    args.xcol = c
                    break
        if args.xcol is None:
            print(f"Need xcol. Available columns: {sub.columns.tolist()}")
            return

    plt.figure(figsize=(8, 6))
    for method, g in sub.groupby("Method"):
        # Sort by xcol to draw lines if multiple points exist
        g = g.sort_values(args.xcol)
        plt.plot(g[args.xcol], g[args.ycol], marker='o', label=method)
        plt.scatter(g[args.xcol], g[args.ycol], s=50)

    plt.xlabel(args.xcol)
    plt.ylabel(args.ycol)
    plt.title(f"Pareto Front ({args.dataset_key})")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print("Saved Pareto plot to:", args.out)

if __name__ == "__main__":
    main()
