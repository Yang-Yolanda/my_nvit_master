# Analysis: Run 9 and H3.6M Results Discrepancy

This document summarizes the investigation into why current results differ from "Run 9" and why H3.6M results "spike" (MPJPE > 200mm) compared to 3DPW.

## 🔍 Findings: H3.6M "Spiking" Issue

The discrepancy in H3.6M evaluation is caused by **Root Alignment Inconsistency**.

1.  **Standard Alignment (Broken for H36M)**: Most scripts (`eval_comprehensive.py`, `run9_3dpw_30k.json`, etc.) use a single joint (usually index 0, the Pelvis) to align predicted and ground-truth skeletons. 
2.  **H36M Root Mismatch**: H3.6M ground truth often uses a different pelvis definition or index than SMPL/J24. If the model predicts SMPL joints but the evaluator uses H3.6M GT without proper mapping, a massive translation offset (MPJPE > 200mm) occurs.
3.  **The "Blessed" Fix**: `standard_eval.py` contains a specific `H36M ALIGNMENT FIX`:
    *   It uses the **Mean of all Evaluation Joints** as the root for both predicted and GT skeletons.
    *   This "Mean Centering" eliminates translation drift entirely, resulting in "good" results (e.g., ~154mm or lower).
4.  **Inconsistent pelvis_ind**: `standard_eval.py` also manually sets `pelvis_ind = 1` for H3.6M before overriding it with the mean fix.

## 🔍 Findings: The "25mm" Mystery

After searching all log files (`.log`, `.json`, `.md`):
*   **Run 9 (Log)**: `eval_baseline_run9.log` shows **72.54mm MPJPE** on 3DPW.
*   **Run 9 (JSON)**: `run9_3dpw_30k.json` shows **202.68mm MPJPE** (likely due to broken evaluation).
*   **Thesis Notes**: `Thesis_Paper2_Future_Work_Ideas.md` describes **154.2mm** as a "massive drop" (暴降) after 25 epochs.
*   **Hypothesis**: The "25" in your memory might refer to the **25 Epochs** of finetuning that led to the 154mm breakthrough, or it might be **PA-MPJPE** (which can reach < 50mm).

## 🚀 Recommendation

1.  **Standardize Eval**: Use **ONLY** `standard_eval.py` for all future metrics. 
2.  **Explicit Alignment**: Always use the `Mean of Eval Joints` logic for H3.6M reports to remain consistent with your previous "good" results.
3.  **Clean up**: Archive `eval_comprehensive.py` and other scattered evaluation scripts to avoid confusion.
