#!/home/yangz/.conda/envs/4D-humans/bin/python
        # C. Save Explicitly with Sparsity Name
        save_filename = f'hmr2_pruned_sparsity_{sparsity_percent}.pth'

        if utils.is_main_process():
            print(f"✅ Pruning Done for {sparsity_percent}%. Saving to {save_filename}...")

        if args.output_dir:
            from engine_4d import generate_pruned_structure
            try:
                 p_struct = generate_pruned_structure(model_without_ddp)
            except:
                 p_struct = {"error": "generator_failed"}

            # Temp Save Masrked Model
            temp_ckpt_path = Path(args.output_dir) / save_filename
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
                'sparsity': ratio,
                'pruned_structure': p_struct
            }, temp_ckpt_path)
            
            # --- 🔥 PHYSICAL PRUNING INJECTION 🔥 ---
            if utils.is_main_process():
                print(f"🔪 Starting Physical Pruning Conversion for {sparsity_percent}%...")
                try:
                    # 1. Initialize Physical Pruner
                    from apply_structural_pruning_v2 import HessianChannelPruner 
                    sys.path.append(os.getcwd())
                    
                    pruner_config = {
                        'sparsity': ratio,
                        'prune_output_channels': True,
                        'coupled_pruning': True
                    }
                    
                    physical_ckpt_name = f'hmr2_physical_{sparsity_percent}.pth'
                    physical_ckpt_path = Path(args.output_dir) / physical_ckpt_name
                    
                    print(f"    Reading from: {temp_ckpt_path}")
                    
                    pruner = HessianChannelPruner(args, pruning_config=pruner_config)
                    
                    state_dict = torch.load(temp_ckpt_path, map_location='cpu')['model']
                    pruner.model.load_state_dict(state_dict, strict=False)
                    
                    calib_data = []
                    for i, batch in enumerate(data_loader_train):
                        if i >= 10: break
                        calib_data.append(batch)
                        
                    pruner.prune(calibration_data=calib_data)
                    
                    torch.save(pruner.model.state_dict(), physical_ckpt_path)
                    print(f"✅ Physical Pruning Complete! Saved to: {physical_ckpt_name} (Size reduced)")
                    
                except Exception as e:
                    print(f"❌ Physical Pruning Failed: {e}")
                    import traceback
                    traceback.print_exc()

            if (Path(args.output_dir) / save_filename).exists():
                 print(f"✅ Verified: {save_filename}")

        if args.distributed:
            dist.barrier()
