#!/bin/bash
set -e

echo "Starting centralized result organization according to Clean Workflow..."
BASE_DIR="/home/yangz/NViT-master/nvit/Paper1_Diagnostics"
cd "${BASE_DIR}"

# 1. Create centralized figure directories
mkdir -p Phase1_Figures/overall_trends
mkdir -p Phase1_Figures/per_model
mkdir -p Phase1_Figures/ablations
mkdir -p Phase1_Figures/supplementary

# 2. Move High-Level Bimodal / Architecture Diagrams (General)
mv -f kti_bimodal_chart.png Phase1_Figures/overall_trends/ 2>/dev/null || true
mv -f diagnosis_*.png Phase1_Figures/overall_trends/ 2>/dev/null || true
mv -f initial_baseline_layer_trends.png Phase1_Figures/overall_trends/ 2>/dev/null || true
mv -f initial_baseline_comparison.png Phase1_Figures/overall_trends/ 2>/dev/null || true
mv -f Phase3_Bimodal_Analysis_Preview.png Phase1_Figures/overall_trends/ 2>/dev/null || true
mv -f kti_mapping_sanity_visual.png Phase1_Figures/supplementary/ 2>/dev/null || true
mv -f kti_math_comparison.png Phase1_Figures/supplementary/ 2>/dev/null || true
mv -f titration_*.png Phase1_Figures/supplementary/ 2>/dev/null || true

# 3. Clean up the paper1_figures_xxx folders into the centralized Phase1_Figures
# Move refined figures
if [ -d "paper1_figures_refined" ]; then
    mv -f paper1_figures_refined/Paper1_Fig1_Robustness.png Phase1_Figures/overall_trends/ 2>/dev/null || true
    mv -f paper1_figures_refined/HMR2_Phase_Transition.png Phase1_Figures/overall_trends/ 2>/dev/null || true
    mv -f paper1_figures_refined/Paper1_Fig2a_Alignment.png Phase1_Figures/overall_trends/ 2>/dev/null || true
    mv -f paper1_figures_refined/Paper1_Fig2b_Scatter.png Phase1_Figures/overall_trends/ 2>/dev/null || true
    mv -f paper1_figures_refined/Ablation_MPJPE_Comparison.png Phase1_Figures/ablations/ 2>/dev/null || true
    mv -f paper1_figures_refined/cross_*.png Phase1_Figures/overall_trends/ 2>/dev/null || true
    mv -f paper1_figures_refined/HMR2_KTI_Curve.png Phase1_Figures/per_model/ 2>/dev/null || true
    gio trash paper1_figures_refined 2>/dev/null || rm -rf paper1_figures_refined
fi

# Move final figures
if [ -d "paper1_figures_final" ]; then
    mv -f paper1_figures_final/* Phase1_Figures/overall_trends/ 2>/dev/null || true
    gio trash paper1_figures_final 2>/dev/null || rm -rf paper1_figures_final
fi

if [ -d "paper1_figures_gt_kti" ]; then
    mv -f paper1_figures_gt_kti/* Phase1_Figures/overall_trends/ 2>/dev/null || true
    gio trash paper1_figures_gt_kti 2>/dev/null || rm -rf paper1_figures_gt_kti
fi

if [ -d "Experiment3_Masking/paper1_figures" ]; then
    mv -f Experiment3_Masking/paper1_figures/* Phase1_Figures/ablations/ 2>/dev/null || true
    gio trash Experiment3_Masking/paper1_figures 2>/dev/null || rm -rf Experiment3_Masking/paper1_figures
fi


# 4. Standardize Experiment1_Entropy (Core Baseline Run)
if [ -f "Experiment1_Entropy/results/entropy_curve_comparison.png" ]; then
    mv Experiment1_Entropy/results/entropy_curve_comparison.png Phase1_Figures/overall_trends/ 2>/dev/null || true
fi

# 5. Standardize Experiment2_KTI
if [ -f "Experiment2_KTI/results/kti_bar_comparison.png" ]; then
    mv Experiment2_KTI/results/kti_bar_comparison.png Phase1_Figures/overall_trends/ 2>/dev/null || true
fi
if [ -f "Experiment2_KTI/results/kti_curve_others.png" ]; then
    mv Experiment2_KTI/results/kti_curve_others.png Phase1_Figures/overall_trends/ 2>/dev/null || true
fi
if [ -f "Experiment2_KTI/results/kti_curve_hmr_hsmr.png" ]; then
    mv Experiment2_KTI/results/kti_curve_hmr_hsmr.png Phase1_Figures/overall_trends/ 2>/dev/null || true
fi

# Move old individual KTI results into the standard structure (Control) if they aren't there
for MODEL in HMR2 HSMR PromptHMR CameraHMR Robot SigLIP AniMeR HaMeR TaskB_Robot; do
    if [ -d "Experiment2_KTI/results/${MODEL}" ] && [ ! -d "Experiment2_KTI/results/${MODEL}/Control" ]; then
        mkdir -p "Experiment2_KTI/results/${MODEL}/Control"
        mv Experiment2_KTI/results/${MODEL}/*.png "Experiment2_KTI/results/${MODEL}/Control/" 2>/dev/null || true
        mv Experiment2_KTI/results/${MODEL}/*.csv "Experiment2_KTI/results/${MODEL}/Control/" 2>/dev/null || true
        mv Experiment2_KTI/results/${MODEL}/*.json "Experiment2_KTI/results/${MODEL}/Control/" 2>/dev/null || true
    fi
done

# 6. Create workflow definition for future runs
cat << 'WORKFLOW_EOF' > /home/yangz/NViT-master/.agent/workflows/clean_workflow_guide.md
---
description: NViT Workspace Storage Workflow
---

# Clean Workflow Guide

When running diagnostic or training experiments, all outputs MUST follow this structure:

## Individual Model Data:
All raw `.csv`, `.json`, and model-specific `diagnostic_plot.png` files must be saved under:
`/home/yangz/NViT-master/nvit/Paper1_Diagnostics/Experiment[X]_[Name]/results/[ModelName]/[GroupName]/`

*Example:* `Experiment1_Entropy/results/HMR2/Control/diagnostic_plot.png`
*Example:* `Experiment3_Masking/results/PromptHMR/Hybrid-Mamba-GCN/results.csv`

## Cross-Model Summary Figures (For Paper):
Any plot that aggregates data across multiple models or serves as a final thesis illustration must be saved directly to:
`/home/yangz/NViT-master/nvit/Paper1_Diagnostics/Phase1_Figures/` (under subfolders like `overall_trends`, `ablations`, etc.)

**DO NOT** leave `.png` or `.csv` files floating in the root `Paper1_Diagnostics` directory.
WORKFLOW_EOF

# 7. Delete empty results folders if left behind
find Phase1_Figures -type d -empty -delete 2>/dev/null || true

echo "Organization complete! All summary figures are in Phase1_Figures/."
