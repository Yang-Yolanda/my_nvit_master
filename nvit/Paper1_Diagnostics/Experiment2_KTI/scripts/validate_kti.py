#!/home/yangz/.conda/envs/4D-humans/bin/python
"""
KTI Validation Diagnostic Tool

Performs comprehensive validation of KTI calculation pipeline:
1. Input Validation (Dataset & Topology)
2. Process Validation (Algorithm)
3. Output Validation (Results)
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.append('nvit/Paper1_Diagnostics/diagnostic_core')

class KTIValidator:
    """Comprehensive KTI validation suite"""
    
    def __init__(self):
        self.results = defaultdict(list)
        self.errors = []
        self.warnings = []
        
    def validate_topology(self, parents_list, name="Unknown"):
        """Validate topology is a valid tree structure"""
        print(f"\n{'='*60}")
        print(f"VALIDATING TOPOLOGY: {name}")
        print(f"{'='*60}")
        
        num_joints = len(parents_list)
        print(f"Number of joints: {num_joints}")
        print(f"Parents list: {parents_list}")
        
        # Check 1: Root has parent -1
        if parents_list[0] != -1:
            self.errors.append(f"{name}: Root must have parent -1, got {parents_list[0]}")
            return False
        print("✓ Root node valid (parent = -1)")
        
        # Check 2: No forward references (parent index < child index)
        for i, p in enumerate(parents_list):
            if p != -1 and p >= i:
                self.errors.append(f"{name}: Parent {p} >= child {i} (forward reference)")
                return False
        print("✓ No forward references")
        
        # Check 3: All joints reachable from root
        reachable = {0}
        changed = True
        iterations = 0
        while changed and iterations < num_joints:
            changed = False
            for i, p in enumerate(parents_list):
                if p in reachable and i not in reachable:
                    reachable.add(i)
                    changed = True
            iterations += 1
        
        if len(reachable) != num_joints:
            self.errors.append(f"{name}: Only {len(reachable)}/{num_joints} joints reachable from root")
            return False
        print(f"✓ All {num_joints} joints reachable from root")
        
        # Check 4: Compute number of edges
        num_edges = sum(1 for p in parents_list if p != -1)
        print(f"✓ Topology has {num_edges} edges (tree property: n-1 edges)")
        
        self.results[f'{name}_topology'] = 'PASS'
        return True
    
    def validate_kti_json(self, json_path, model_type):
        """Validate KTI output JSON"""
        print(f"\n{'='*60}")
        print(f"VALIDATING KTI OUTPUT: {json_path}")
        print(f"{'='*60}")
        
        if not os.path.exists(json_path):
            self.errors.append(f"File not found: {json_path}")
            return False
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print(f"Number of layers: {len(data)}")
        
        all_kti = []
        for layer_idx, metrics in data.items():
            kti_values = metrics.get('kti', [])
            if not kti_values:
                self.warnings.append(f"{json_path}: Layer {layer_idx} has no KTI values")
                continue
            
            kti_array = np.array(kti_values)
            all_kti.extend(kti_values)
            
            # Check for NaN/Inf
            if np.isnan(kti_array).any():
                self.errors.append(f"{json_path}: Layer {layer_idx} contains NaN")
            if np.isinf(kti_array).any():
                self.errors.append(f"{json_path}: Layer {layer_idx} contains Inf")
            
            # Check non-negative
            if (kti_array < 0).any():
                self.errors.append(f"{json_path}: Layer {layer_idx} has negative KTI")
            
            print(f"Layer {layer_idx}: mean={kti_array.mean():.4f}, std={kti_array.std():.4f}, "
                  f"min={kti_array.min():.4f}, max={kti_array.max():.4f}")
        
        # Overall statistics
        all_kti = np.array(all_kti)
        
        if len(all_kti) == 0:
            self.warnings.append(f"{json_path}: No KTI values found in any layer")
            print(f"\n⚠ WARNING: No KTI data available")
            return False
        
        print(f"\n{'='*60}")
        print(f"OVERALL STATISTICS")
        print(f"{'='*60}")
        print(f"Total samples: {len(all_kti)}")
        print(f"Mean KTI: {all_kti.mean():.4f}")
        print(f"Std KTI: {all_kti.std():.4f}")
        print(f"Min KTI: {all_kti.min():.4f}")
        print(f"Max KTI: {all_kti.max():.4f}")
        
        # Model-specific validation
        if model_type == 'HMR2':
            if all_kti.max() < 0.2:
                self.warnings.append(f"{json_path}: HMR2 should have KTI > 0.2 (got {all_kti.max():.4f})")
            else:
                print("✓ HMR2 shows structural signal (max KTI > 0.2)")
        
        elif model_type in ['SigLIP', 'CLIP']:
            if all_kti.max() > 0.3:
                self.warnings.append(f"{json_path}: General model should have low KTI (got {all_kti.max():.4f})")
            else:
                print("✓ General model shows expected low KTI")
        
        elif 'Robot_Scratch' in model_type:
            if all_kti.max() > 0.15:
                self.warnings.append(f"{json_path}: Scratch model should have minimal KTI (got {all_kti.max():.4f})")
            else:
                print("✓ Scratch model shows expected minimal structure")
        
        elif 'Robot_Pretrained' in model_type:
            if all_kti.max() < 0.1 or all_kti.max() > 0.4:
                self.warnings.append(f"{json_path}: Pretrained robot KTI out of expected range (got {all_kti.max():.4f})")
            else:
                print("✓ Pretrained model shows weak structural prior")
        
        # Check variance
        if all_kti.std() < 0.01:
            self.warnings.append(f"{json_path}: Very low variance (std={all_kti.std():.4f})")
        
        self.results[f'{model_type}_output'] = 'PASS'
        return True
    
    def validate_attention_properties(self, attn_map):
        """Validate attention map properties (for runtime checks)"""
        # Check shape
        if len(attn_map.shape) != 4:
            self.errors.append(f"Attention map should be 4D (B,H,N,N), got {attn_map.shape}")
            return False
        
        # Check probability (should sum to 1)
        attn_sum = attn_map.sum(dim=-1)
        if not torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-4):
            self.warnings.append(f"Attention doesn't sum to 1 (max diff: {(attn_sum - 1).abs().max():.6f})")
        
        # Check range [0, 1]
        if (attn_map < 0).any() or (attn_map > 1).any():
            self.errors.append(f"Attention values out of [0,1] range")
            return False
        
        return True
    
    def generate_report(self):
        """Generate validation report"""
        print(f"\n{'='*60}")
        print(f"VALIDATION REPORT")
        print(f"{'='*60}")
        
        print(f"\n✓ PASSED: {len(self.results)}")
        for test, status in self.results.items():
            print(f"  - {test}: {status}")
        
        if self.warnings:
            print(f"\n⚠ WARNINGS: {len(self.warnings)}")
            for w in self.warnings:
                print(f"  - {w}")
        
        if self.errors:
            print(f"\n✗ ERRORS: {len(self.errors)}")
            for e in self.errors:
                print(f"  - {e}")
            return False
        
        print(f"\n{'='*60}")
        print(f"✓ ALL VALIDATIONS PASSED")
        print(f"{'='*60}")
        return True

def main():
    validator = KTIValidator()
    
    # 1. Validate Topologies
    print("\n" + "="*60)
    print("PHASE 1: TOPOLOGY VALIDATION")
    print("="*60)
    
    # SMPL (HMR2)
    smpl_parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    validator.validate_topology(smpl_parents, "SMPL (HMR2)")
    
    # Robot (from URDF)
    robot_parents = [-1, 0, 1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 14, 0, 16, 17, 18, 19, 20, 21, 22]
    validator.validate_topology(robot_parents, "Robot (URDF)")
    
    # MANO (HaMeR)
    mano_parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
    validator.validate_topology(mano_parents, "MANO (HaMeR)")
    
    # 2. Validate KTI Outputs
    print("\n" + "="*60)
    print("PHASE 2: OUTPUT VALIDATION")
    print("="*60)
    
    results_dir = Path('nvit/Paper1_Diagnostics/Experiment2_KTI/results')
    
    # Task A: HMR2
    hmr2_json = results_dir / 'HMR2' / 'layer_metrics_Control.json'
    if hmr2_json.exists():
        validator.validate_kti_json(str(hmr2_json), 'HMR2')
    
    # Task B: Robot
    robot_scratch_json = results_dir / 'Toy_Robot_Scratch' / 'layer_metrics_Control.json'
    if robot_scratch_json.exists():
        validator.validate_kti_json(str(robot_scratch_json), 'Robot_Scratch')
    
    robot_pretrained_json = results_dir / 'Toy_Robot_Pretrained' / 'layer_metrics_Control.json'
    if robot_pretrained_json.exists():
        validator.validate_kti_json(str(robot_pretrained_json), 'Robot_Pretrained')
    
    # Task C: SigLIP
    siglip_json = results_dir / 'SigLIP' / 'layer_metrics_Control.json'
    if siglip_json.exists():
        validator.validate_kti_json(str(siglip_json), 'SigLIP')
    
    # Task D: HaMeR
    hamer_json = results_dir / 'HaMeR_Random' / 'layer_metrics_Control.json'
    if hamer_json.exists():
        validator.validate_kti_json(str(hamer_json), 'HaMeR_Random')
    
    # Generate final report
    success = validator.generate_report()
    
    # Save report
    report_path = results_dir / 'validation_report.txt'
    with open(report_path, 'w') as f:
        f.write("KTI Validation Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Passed: {len(validator.results)}\n")
        f.write(f"Warnings: {len(validator.warnings)}\n")
        f.write(f"Errors: {len(validator.errors)}\n\n")
        
        if validator.warnings:
            f.write("Warnings:\n")
            for w in validator.warnings:
                f.write(f"  - {w}\n")
        
        if validator.errors:
            f.write("\nErrors:\n")
            for e in validator.errors:
                f.write(f"  - {e}\n")
    
    print(f"\n✓ Report saved to {report_path}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
