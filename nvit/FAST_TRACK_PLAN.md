# NViT Scientific Accelerator (Fast-Track Training)

## Goal
Accelerate the Paper 1 Titration ablation study by 3x (from 28 hours to ~10 hours) while maintaining scientific accuracy for relative comparisons. This plan uses a "Scientific Seed" approach: training on a representative, deterministic subset with more epochs.

## Design Rationale
- **Relative Comparison > Absolute Accuracy**: In ablation studies, the *gap* between Group A (Hero) and Group B (Baseline) is more important than the absolute MPJPE.
- **Dataset Scale**: The current dataset (900k+ samples) is too large for 4 epochs if we need fast feedback.
- **Consistency**: All groups MUST see the exact same subset of data to ensure the comparison is fair.

## User Review Required
> [!IMPORTANT]
> This plan will use **15% of the full dataset (approx 135,000 samples)** and run for **15 epochs**.
> - **Expected Result**: Clear trends in **2-3 hours**.
> - **Total Completion**: **10-12 hours**.
> - **Risk**: The final MPJPE might be slightly higher than training on the full 1M set, but the *ranking* of experiments will remain valid.

## Proposed Changes

### [Component] `finetune_dense.py`
Add a `SubsetIterableDataset` wrapper to ensure all groups train on the same leading part of the data stream.

#### [MODIFY] [finetune_dense.py](file:///home/yangz/NViT-master/nvit/finetune_dense.py)
```python
# Create this class before the main() function
class SubsetIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, max_samples):
        self.dataset = dataset
        self.max_samples = max_samples
        
    def __iter__(self):
        count = 0
        for item in self.dataset:
            if count >= self.max_samples:
                break
            yield item
            count += 1
            
# Inside main after dataset creation:
if args.fast_track:
    print(f"🚀 FAST TRACK: Subsetting dataset to {args.subset_size} samples")
    dataset_train = SubsetIterableDataset(dataset_train, args.subset_size)
```

### [Component] Launcher Scripts
Create a new launcher that uses the `--fast-track` flag.

#### [NEW] [run_fast_track_blitz.sh](file:///home/yangz/NViT-master/nvit/run_fast_track_blitz.sh)
- Similar to `run_optimized_blitz.sh`
- `batch_size`: 12
- `epochs`: 15
- `--fast-track`: ON
- `--subset-size`: 135000

## Verification Plan

### Automated Tests
1. **Verification of Subset Consistency**: Run two identical small groups for 50 steps and check if their loss values are EXACTLY identical.
2. **GPU Monitoring**: Ensure utilization remains high despite subsetting.

### Manual Verification
1. **Dashboard Check**: Use `status_dashboard.sh` to see if epoch progress is 10x faster.
