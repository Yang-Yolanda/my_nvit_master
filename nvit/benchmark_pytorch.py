#!/home/yangz/.conda/envs/4D-humans/bin/python
import torch
import time
import argparse
from pathlib import Path
from experiment_layer_ablation import LayerAblationPruner

def benchmark_pytorch(mode="original", runs=100, warmups=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Map mode to the correct pruner config
    # We use the pruner logic to transform the model structure
    if mode == "original":
        from model_manager import ModelManager
        mm = ModelManager({'device': device, 'enable_detector': False})
        mm.load_model()
        model = mm.model
    else:
        # mode will be shallow_heavy, mid_heavy, deep_heavy
        config = {
            'device': device,
            'enable_detector': False,
            'ablation_mode': mode,
            'heavy_sparsity': 0.7,
            'base_sparsity': 0.05
        }
        pruner = LayerAblationPruner(config)
        model = pruner.prune() # This applies the structural transformation
        
    model = model.to(device).float()
    model.eval()
    
    dummy_input = torch.randn(1, 3, 256, 256).to(device).float()
    
    print(f"🔥 Warming up {mode} ({warmups} runs)...")
    with torch.no_grad():
        for _ in range(warmups):
            _ = model({'img': dummy_input})
    
    if device == 'cuda':
        torch.cuda.synchronize()
        
    print(f"🚀 Benchmarking {mode} ({runs} runs)...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model({'img': dummy_input})
            if device == 'cuda':
                torch.cuda.synchronize()
    end_time = time.time()
    
    avg_lat = (end_time - start_time) / runs * 1000
    fps = 1000 / avg_lat
    print(f"✅ [RESULT][{mode}] Avg Latency: {avg_lat:.2f} ms | FPS: {fps:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="original")
    args = parser.parse_args()
    benchmark_pytorch(mode=args.mode)
