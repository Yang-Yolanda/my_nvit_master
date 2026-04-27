# 统一评测工作流（详见 scripts/workflow_unified_eval.sh）
.PHONY: help eval-full8 eval-ch6-8gpu eval-env plot-ch5

ROOT := $(abspath .)

help:
	@echo "Targets:"
	@echo "  eval-full8      8卡: SMPLer + ch6 + ch5（需 HMR2_CFG_REFERENCE_CKPT）"
	@echo "  eval-ch6-8gpu   仅 ch6，8卡分片"
	@echo "  eval-env        打印 CLUSTER_* / CH*_GPU_LIST 等说明"
	@echo "  plot-ch5        CH5 最大 step 柱状图（读 metrics_master.csv）"
	@echo "示例: make eval-full8 HMR2_CFG_REFERENCE_CKPT=/path/to/ref.ckpt"

eval-full8:
	bash $(ROOT)/scripts/workflow_unified_eval.sh full8

eval-ch6-8gpu:
	bash $(ROOT)/scripts/workflow_unified_eval.sh ch6-only-8gpu

eval-env:
	bash $(ROOT)/scripts/workflow_unified_eval.sh env-print

plot-ch5:
	bash $(ROOT)/scripts/workflow_unified_eval.sh plot-ch5 --eval-mode full
