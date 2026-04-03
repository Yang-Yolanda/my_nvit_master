# CLEANUP_LIST.md - Proposed Archives and Deletions

The following files are identified as legacy, redundant, or messy. They will be moved to `archive_legacy/` upon approval.

## 📦 To Be Archived (Move to `archive_legacy/`)

### High-Priority Redundancy
- [ ] `nvit/train_ablation.py` (Redundant to `train_guided.py`)
- [ ] `nvit/finetune_dense.py` (Legacy finetuning)
- [ ] `nvit/finetune_hybrid.py` (Legacy finetuning)
- [ ] `nvit/diagnose_paper2_v1.py` (Old diagnostic)
- [ ] `nvit/diagnose_paper2_v2.py` (Old diagnostic)

### Secondary Scripts
- [ ] `nvit/experiment_layer_ablation.py` (Absorbed into controlled plan)
- [ ] `nvit/experiment_kti_sensitivity.py` (Absorbed into controlled plan)
- [ ] `nvit/run_ablation_campaign.sh` (Old automation)
- [ ] `nvit/run_nvit.sh` (Old launcher)
- [ ] `nvit/train_nvit.sh` (Duplicate of above)

### Log Cleanup (Move to `archive_logs/`)
- [ ] `nvit/eval_gpu7_*.log` (Main directory logs)
- [ ] `nvit/monitor_healing.log` 
- [ ] `nvit/eval_res_*.json` (Scattered results)

## 🗑️ To Be Deleted (After Archiving)
- *None for now. Safety first.*

## 🛠️ Archiving Command (Draft)
```bash
mkdir -p archive_legacy archive_logs
mv nvit/train_ablation.py nvit/finetune_*.py nvit/diagnose_paper2_*.py archive_legacy/
mv nvit/eval_gpu7_*.log nvit/*.json archive_logs/
```
