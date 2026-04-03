---
name: Autonomous Training Monitor
description: Standardized protocol for monitoring deep learning experiments, detecting failures (NaN, OOM, Hangs), and executing self-healing actions.
---

# Autonomous Training Monitor Skill

This skill defines the rigorous procedure for monitoring the Run 9 training session.

## 0. Core Development Principles

*   **No "Reinventing the Wheel" (不要反复造轮子)**: Prioritize reusing existing modules (`diagnostic_engine.py`, `GuidedHMR2Module`, etc.) over writing new standalone logic.
*   **Process Optimization (优化流程)**: Focus on improving the efficiency, stability, and diagnostic depth of existing code paths rather than creating redundant variants.
*   **Modular Extensions**: If new functionality is needed, implement it as a modular extension to the existing codebase to maintain consistency across the project.

## 1. Routine Health Check
**Frequency**: Every 10 minutes.

### 1.1 Check Process Status
Run the following to verify the process is alive:
```bash
tmux capture-pane -pt run9_training | tail -n 20
```
*   **Criteria**: `tmux` output should show progressing metrics (loss decreasing, epoch counting).

### 1.2 Check GPU Utilization
```bash
nvidia-smi
```
*   **Criteria**: GPU 2 Utilization > 80%, Memory Usage stable (approx 24-32GB for ViT-H).

### 1.3 Scan & Analyze Logs
Check the latest logs for specific danger keywords:
```bash
### 1.4 Check Gradient Health (CRITICAL)
Check for skipped steps due to gradient explosion:
```bash
grep "Grad norm" pilot_train_v9.log | tail -n 10
```
*   **Criteria**: If "Skipping optimizer step" appears continuously (>10 times), the model is **STALLED**.

## 2. Failure Diagnosis & Recovery (Self-Healing)
If any check fails, immediately enter **Recovery Mode**:

1.  **Stop**: Kill the hanging/Zombie process (`pkill -f train_guided.py`).
2.  **Diagnose**: 
    *   **NaN Loss**: Read log tail for "NaN".
    *   **Grad Explosion**: Check `full_scan_kmi.json` or loss components to see which head is exploding.
    *   **OOM**: Check NVIDIA-SMI.
3.  **Fix SOP**:
    *   **Case: Grad Explosion**: 
        1. Temporarily disabling Heatmap Loss (set weight=0.0) -> Re-launch to see if it stabilizes.
        2. If stable, re-introduce Heatmap at 0.1 (Warmup).
        3. If still exploding, reduce Global LR by 10x.
    *   **Case: NaN**: Tighten numerical clamp in `soft_argmax`.
4.  **Restart**:
    ```bash
    tmux new-session -d -s run9_training "bash scripts/run9_pose_fix.sh"
    ```
5.  **Log**: Record the incident in `walkthrough.md` under "Stability Log".
