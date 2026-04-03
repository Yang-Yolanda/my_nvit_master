import os
import sys
import time
import torch
import pandas as pd
from pathlib import Path
from thop import profile
import logging
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / 'nvit'))
sys.path.append('/home/yangz/4D-Humans')

from yacs.config import CfgNode
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from nvit2_models.guided_hmr2 import GuidedHMR2Module
from nvit.bio_dataset import BioMambaDataset
from hmr2.utils.geometry import aa_to_rotmat

# Standard evaluator
from hmr2.configs import dataset_eval_config
from hmr2.datasets.image_dataset import ImageDataset
from hmr2.utils import Evaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Profiler")

# --- Model Configurations ---
def get_config(g_name):
    # Base config from HMR2
    _, cfg = load_hmr2(DEFAULT_CHECKPOINT)
    # Ensure nested configs exist
    cfg.defrost()
    if 'BACKBONE' not in cfg.MODEL: cfg.MODEL.BACKBONE = CfgNode()
    if 'SMPL_HEAD' not in cfg.MODEL: cfg.MODEL.SMPL_HEAD = CfgNode()
    if 'TRANSFORMER_DECODER' not in cfg.MODEL.SMPL_HEAD: cfg.MODEL.SMPL_HEAD.TRANSFORMER_DECODER = CfgNode()
    if 'LOSS_WEIGHTS' not in cfg: cfg.LOSS_WEIGHTS = CfgNode(new_allowed=True)
    cfg.LOSS_WEIGHTS.HEATMAP = 2.0
    
    # We will let guided_hmr2 override the backbone.
    if g_name == "G0":
        # Full 32-layer ViT
        cfg.MODEL.BACKBONE.depth = 32
        cfg.MODEL.BACKBONE.switch_layer_1 = 32
        cfg.MODEL.BACKBONE.switch_layer_2 = 32
        cfg.MODEL.BACKBONE.mamba_variant = 'spiral'
        cfg.MODEL.BACKBONE.gcn_variant = 'guided'
        
    elif g_name == "G1" or g_name == "G2":
        # Naive Truncated 12-layer ViT / KTI Keep 12
        cfg.MODEL.BACKBONE.depth = 12
        cfg.MODEL.BACKBONE.switch_layer_1 = 12
        cfg.MODEL.BACKBONE.switch_layer_2 = 12
        cfg.MODEL.BACKBONE.mamba_variant = 'spiral'
        cfg.MODEL.BACKBONE.gcn_variant = 'guided'
        
    elif g_name == "G3":
        # Hybrid 12L (0-7 ViT, 8-10 Mamba, 11 GCN)
        cfg.MODEL.BACKBONE.depth = 12
        cfg.MODEL.BACKBONE.switch_layer_1 = 8
        cfg.MODEL.BACKBONE.switch_layer_2 = 11
        cfg.MODEL.BACKBONE.mamba_variant = 'spiral' # or 'kinetic_out' based on paper 2? Let's use spiral for default.
        cfg.MODEL.BACKBONE.gcn_variant = 'skeleton' # GCN uses skeleton topology
        
    return cfg

def load_checkpoint_if_exists(model, g_name):
    # Dictionary of known checkpoint paths from ablation script
    ckpts = {
        "G0": "/home/yangz/NViT-master/nvit/outputs/baseline_32L/checkpoints/best.ckpt", 
        "G1": "/home/yangz/NViT-master/nvit/outputs/trunc_11L/checkpoints/best.ckpt",
        "G3": "/home/yangz/NViT-master/nvit/outputs/ch6_safe_ckpts/best.ckpt", 
    }
    
    ckpt_path = ckpts.get(g_name, None)
    if ckpt_path and os.path.exists(ckpt_path):
        logger.info(f"Loading weights for {g_name} from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        model.load_state_dict(state_dict, strict=False)
        return True
    return False

def get_dataloader(batch_size=16, is_train=True):
    dataset_file = '/home/yangz/4D-Humans/hmr2_evaluation_data/3dpw_test.npz'
    img_dir = '/home/yangz/4D-Humans/data/3DPW'
    _, m_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    
    if is_train:
        ds = BioMambaDataset(m_cfg, dataset_file=dataset_file, img_dir=img_dir, train=True)
    else:
        cfg_eval = dataset_eval_config()['3DPW-TEST']
        cfg_eval.defrost()
        cfg_eval.DATASET_FILE = dataset_file
        cfg_eval.freeze()
        ds = ImageDataset(cfg=m_cfg, dataset_file=cfg_eval.DATASET_FILE, img_dir=img_dir, train=False)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2), cfg_eval

    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)

def profile_efficiency(model, device):
    model.eval()
    model.to(device)
    
    # 1. Params & Model Size
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 ** 2)
    
    # 2. FLOPs via thop
    dummy_img = torch.randn(1, 3, 256, 192).to(device)
    dummy_input = {'img': dummy_img}
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops_g = macs * 2 / 1e9 # THOP computes MACs. FLOPs is roughly 2*MACs. Output in GFLOPs.
    
    # 3. Latency (bs=1)
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy_input)
            
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        with torch.no_grad():
            model(dummy_input)
    torch.cuda.synchronize()
    latency_ms = ((time.time() - t0) / 50) * 1000
    
    # 4. Throughput (bs=16)
    dummy_img_16 = torch.randn(16, 3, 256, 192).to(device)
    dummy_input_16 = {'img': dummy_img_16}
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            model(dummy_input_16)
            
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        with torch.no_grad():
            model(dummy_input_16)
    torch.cuda.synchronize()
    total_time = time.time() - t0
    throughput_fps = (16 * 20) / total_time
    
    return {
        "Params (M)": f"{total_params / 1e6:.2f}",
        "FLOPs (G)": f"{flops_g:.2f}",
        "ModelSize (MB)": f"{model_size_mb:.1f}",
        "Latency bs1 (ms)": f"{latency_ms:.1f}",
        "Throughput bs16 (FPS)": f"{throughput_fps:.1f}",
    }
    
def profile_training_cost(model, train_loader, device, freeze_depth=0):
    model.train()
    model.to(device)
    
    # Apply freeze policy
    if hasattr(model, 'nvit_backbone'):
        model.nvit_backbone.surgical_freeze(freeze_depth=freeze_depth)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    step_times = []
    
    def to_device(obj, dev):
        if isinstance(obj, torch.Tensor):
            return obj.to(dev)
        elif isinstance(obj, dict):
            return {k: to_device(v, dev) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_device(v, dev) for v in obj]
        else:
            return obj
            
    for i, batch in enumerate(train_loader):
        if i >= 15: break # Run 15 steps
        
        batch = to_device(batch, device)
        
        torch.cuda.synchronize()
        t0 = time.time()
        
        output = model.forward_step(batch, train=True)
        loss = model.compute_loss(batch, output, train=True)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        step_times.append(time.time() - t0)
        
    avg_step_time_ms = (sum(step_times[5:]) / len(step_times[5:])) * 1000 # skip first 5
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    
    # Estimate total hours for 50k steps
    total_steps = 50000 
    gpu_hours = (avg_step_time_ms / 1000) * total_steps / 3600
    
    return {
        "TrainableParams (M)": f"{trainable_params / 1e6:.2f}",
        "PeakVRAM (GB)": f"{peak_vram_gb:.2f}",
        "StepTime (ms)": f"{avg_step_time_ms:.1f}",
        "Est.GPU_Hours (50k steps)": f"{gpu_hours:.1f}"
    }

def evaluate_accuracy(model, eval_loader, cfg_eval, device):
    model.eval()
    model.to(device)
    
    evaluator = Evaluator(
        dataset_length=len(eval_loader.dataset),
        keypoint_list=cfg_eval.KEYPOINT_LIST,
        pelvis_ind=39,
        metrics=['mode_mpjpe', 'mode_re']
    )
    
    limit_batches = 50 # to save time for this profiler
    def to_device(obj, dev):
        if isinstance(obj, torch.Tensor):
            return obj.to(dev)
        elif isinstance(obj, dict):
            return {k: to_device(v, dev) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_device(v, dev) for v in obj]
        else:
            return obj

    for i, batch in enumerate(eval_loader):
        if i >= limit_batches: break
        
        batch = to_device(batch, device)
        with torch.no_grad():
            output = model(batch)
            pred_keypoints_3d = output['pred_keypoints_3d']
            
            evaluator(output, batch)
            
    metrics = evaluator.get_metrics_dict()
    return {
        "MPJPE": f"{metrics.get('mode_mpjpe', 0.0)*1000:.1f}",
        "PA-MPJPE": f"{metrics.get('mode_re', 0.0)*1000:.1f}"
    }

def main():
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu') # Use GPU 4 to avoid conflict
    
    groups = ["G0", "G1", "G2", "G3"]
    policies = [("T0", "FullFinetune", 0), ("T1", "Freeze0-7", 8)]
    
    train_loader = get_dataloader(batch_size=8, is_train=True)
    eval_loader, cfg_eval = get_dataloader(batch_size=8, is_train=False)
    
    results = []
    
    for g in groups:
        logger.info(f"--- Processing {g} ---")
        cfg = get_config(g)
        model = GuidedHMR2Module(cfg, init_renderer=False)
        
        # Load weights if available
        has_weights = load_checkpoint_if_exists(model, g)
        
        # 1. Profile Efficiency (Constant per Model Arch)
        eff_stats = profile_efficiency(model, device)
        
        # 2. Evaluate Accuracy (Constant per Model Weights)
        # Note: If G1/G2 has no weights, accuracy is low.
        acc_stats = evaluate_accuracy(model, eval_loader, cfg_eval, device)
        
        # 3. Training Policy Cost
        # If G0/G1 don't need T1, we can skip or run just to get data
        for p_id, p_name, freeze_depth in policies:
            if g == "G0" and p_id == "T1": continue # G0 is baseline, full 32 layer, freezing 8 layers doesn't match paper logic
            
            logger.info(f"Benchmarking {g} with {p_name} ({p_id})")
            
            try:
                train_stats = profile_training_cost(model, train_loader, device, freeze_depth=freeze_depth)
                status = "Finetuned" if p_name != "EvalOnly" else "eval-only"
                if not has_weights: status += " (Untrained Weights)"
                
                row = {"Model": g, "TrainPolicy": p_name, "Status": status}
                row.update(acc_stats)
                row.update(eff_stats)
                row.update(train_stats)
                
                results.append(row)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Failed on {g} {p_name}: {e}")
                
        # Free memory before next model
        del model
        torch.cuda.empty_cache()
                
    # Format and Output Data
    df = pd.DataFrame(results)
    
    # Fill in T1 for models that skipped to keep table complete, just copy T0 efficiency / mark eval-only
    
    out_dir = BASE_DIR / "nvit" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Correct order
    cols_order = ["Model", "TrainPolicy", "MPJPE", "PA-MPJPE", "Params (M)", "FLOPs (G)", "ModelSize (MB)", 
                  "Latency bs1 (ms)", "Throughput bs16 (FPS)", "TrainableParams (M)", "PeakVRAM (GB)", "Est.GPU_Hours (50k steps)"]
    
    df = df[cols_order]
    
    csv_path = out_dir / "efficiency_table.csv"
    tex_path = out_dir / "efficiency_table.tex"
    
    df.to_csv(csv_path, index=False)
    
    # Save LaTeX table
    latex_str = df.to_latex(index=False, float_format="%.2f", na_rep="-", 
                            column_format="l|l|cc|cccp{1cm}p{1cm}|ccc")
    with open(tex_path, "w") as f:
        f.write(latex_str)
        
    logger.info(f"✅ Success! Results saved to {csv_path} and {tex_path}")
    print(df.to_string())

if __name__ == "__main__":
    main()
