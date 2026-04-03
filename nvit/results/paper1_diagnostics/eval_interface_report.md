# 评测接口核对报告 (Evaluation Interface Report)

## 1. 评测脚本来源
- **核心评测脚本**: `nvit/skills/evaluate_model/standard_eval.py`
- **评价器引擎**: `hmr2.utils.Evaluator`
- 该脚本被 `run_batch_hmr_diagnostics.py` 及 `global_evaluator.py` 共享用于全局评测工作流。

## 2. 接口定义核对
- **输入格式**: 脚本期望 PyTorch DataLoader 直接推理，或者包含 `pred_keypoints_3d` 键的字典。
- **坐标系**: Camera Space。预测和GT的 3D keypoints 必须在相机坐标系下对齐。
- **单位缩放**: **Meter 毫米坑点发现**。HMR2 原始输出和数据集 `keypoints_3d` 在内存中均以 **Meter (米)** 为单位。然而，`hmr2.utils.pose_utils.compute_error` 会默认在返回值时 `return 1000 * mpjpe`。因此，调用 `Evaluator.get_metrics_dict()` 后，数值**已经被转换为毫米 (mm)**。之前如果脚本额外执行 `val * 1000` 将导致数值出现 700,000 以上的极大误差（放大了1,000倍）。
- **Root对齐 (Pelvis Alignment)**: 对于 3DPW，标准 HMR2 配置要求使用 `pelvis_ind=0` 或者不配置而由 Evaluator 内部利用其自身的 `keypoint_list` 进行0点偏移。如果外部评测脚本未在送入前后执行减去 Root 并手动将 Root 置 0（如 `H36M` 对齐修正代码段），就会导致全量特征未对齐而在空间漂移，即使单位转换后误差也极高（可达 700+ mm）。
- **Joint Mapping**: 3DPW-TEST 默认使用 14 关节评估 (H36M Subset, `HMR_14`)。
