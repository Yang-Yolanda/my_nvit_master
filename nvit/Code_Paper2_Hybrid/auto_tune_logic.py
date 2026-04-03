def auto_tune_pruning_params(target_neurons, current_neurons):
    import torch
    
    if not torch.cuda.is_available():
        return 2048, 40 # Default CPU/Fallback

    try:
        # Get Device Info
        gpu_id = torch.cuda.current_device()
        properies = torch.cuda.get_device_properties(gpu_id)
        total_mem_gb = properies.total_memory / (1024**3)
        device_name = properies.name
        
        # Heuristic Logic
        # Standard: Step 2304, Interval 30
        
        step = 2304
        interval = 30
        
        print(f"🛠️  [Auto-Tune] Detected {device_name} ({total_mem_gb:.1f} GB)")
        
        if total_mem_gb > 40: # A6000, A40, A100 80G
            print("🚀 High-End GPU Detected: Enabling Aggressive Pruning")
            step = 5120  # Fast removal
            interval = 20 # Frequent updates (computational overhead is fine)
        elif total_mem_gb > 20: # 3090, 4090, A10g
            print("⚡ Mid-High GPU: Standard+ Mode")
            step = 3072
            interval = 25
        elif total_mem_gb < 12: # 3080, 2080Ti
            print("⚠️ Consumer GPU: Conservative Mode")
            step = 1024
            interval = 50
            
        # Refine based on remaining neurons
        # If we are close to target, slow down
        remaining_to_prune = current_neurons - target_neurons
        if remaining_to_prune < 20000:
             step = max(512, step // 4)
             interval = max(10, interval // 2)
             print(f"📉 Approaching Target: Slowing down (Step={step})")
             
        return step, interval

    except Exception as e:
        print(f"⚠️ Auto-Tune Failed: {e}")
        return 2304, 30
