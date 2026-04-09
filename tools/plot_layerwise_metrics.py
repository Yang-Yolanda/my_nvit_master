import argparse, json, glob
import matplotlib.pyplot as plt
import os
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_glob", required=True, type=str)
    ap.add_argument("--metric", default="kmi", type=str)  # kmi/entropy/rank
    ap.add_argument("--out", required=True, type=str)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.json_glob))
    if not paths:
        print(f"No files matched: {args.json_glob}")
        return

    plt.figure(figsize=(10, 6))
    for p in paths:
        try:
            with open(p, "r") as f:
                data = json.load(f)
            
            # Handle possible nested structure if many points per layer
            # data is usually mapping {layer_idx: {metric: [values]}}
            layers = sorted([int(k) for k in data.keys()])
            y_vals = []
            for l in layers:
                v = data[str(l)].get(args.metric, [])
                if v:
                    y_vals.append(np.mean(v))
                else:
                    y_vals.append(0)
            
            name = os.path.basename(p).replace("layer_metrics_", "").replace(".json", "")
            plt.plot(layers, y_vals, marker='x', linewidth=2, label=name)
        except Exception as e:
            print(f"Error processing {p}: {e}")

    plt.xlabel("ViT block index")
    plt.ylabel(args.metric.upper())
    plt.title(f"Layerwise {args.metric.upper()}")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print("Saved layerwise plot to:", args.out)

if __name__ == "__main__":
    main()
