import sys
import re
import csv
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    mpjpe = "N/A"
    pa_mpjpe = "N/A"
    mpjpe_abs = "N/A"

    if os.path.exists(args.log):
        with open(args.log, 'r', encoding='utf-8') as f:
            content = f.read()
            # Try to find MPJPE (which is the Hip-Mean root-aligned MPJPE now)
            m1 = re.search(r'MPJPE\s*[:=]?\s*([0-9\.]+)', content, re.IGNORECASE)
            # Try to find PA-MPJPE
            m2 = re.search(r'PA[-_]MPJPE\s*[:=]?\s*([0-9\.]+)', content, re.IGNORECASE)
            # Try to find [DEBUG_MPJPE] MPJPE_abs
            m3 = re.search(r'\[DEBUG_MPJPE\].*?MPJPE_abs\s*[:=]?\s*([0-9\.]+)', content, re.IGNORECASE)
            
            if m1: mpjpe = m1.group(1)
            if m2: pa_mpjpe = m2.group(1)
            if m3: mpjpe_abs = m3.group(1)

    # Output to CSV
    file_exists = os.path.exists(args.out)
    mode = 'a' if args.append else 'w'
    with open(args.out, mode, newline='') as f:
        writer = csv.writer(f)
        if not file_exists or not args.append:
            writer.writerow(["Method", "Dataset", "MPJPE_root_hipmean(mm)", "PA-MPJPE(mm)", "MPJPE_abs(mm)"])
        writer.writerow([args.method, args.dataset, mpjpe, pa_mpjpe, mpjpe_abs])
        
if __name__ == "__main__":
    main()
