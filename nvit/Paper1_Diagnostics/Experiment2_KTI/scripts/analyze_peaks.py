#!/home/yangz/.conda/envs/4D-humans/bin/python

import json
import numpy as np
import argparse
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter
import yaml

def analyze_kti_peaks(metrics_file, output_config_path=None):
    """
    Analyze KTI metrics to find peaks and generate adaptive config.
    
    Args:
        metrics_file: Path to layer_metrics_Control.json
        output_config_path: Where to save the adaptive config YAML
    
    Returns:
        dict: Configuration with switch_layer_1 and switch_layer_2
    """
    print(f"Loading KTI metrics from: {metrics_file}")
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    # Extract layer indices and mean KTI
    layers = sorted([int(l) for l in data.keys()])
    kti_means = [np.mean(data[str(l)]['kti']) for l in layers]
    
    print(f"Layers: {layers}")
    print(f"KTI Means: {kti_means}")
    
    # Smooth the curve (Savitzky-Golay filter)
    if len(kti_means) > 5:
        kti_smooth = savgol_filter(kti_means, window_length=min(5, len(kti_means)//2*2+1), polyorder=2)
    else:
        kti_smooth = kti_means
    
    # Find peaks
    peaks, properties = find_peaks(kti_smooth, prominence=0.01)
    
    print(f"Detected {len(peaks)} peaks at layers: {[layers[p] for p in peaks]}")
    
    if len(peaks) == 0:
        print("WARNING: No peaks detected. Using default configuration.")
        switch_layer_1 = 8
        switch_layer_2 = 10
    elif len(peaks) == 1:
        print("WARNING: Only one peak detected. Using heuristic split.")
        peak_layer = layers[peaks[0]]
        switch_layer_1 = max(1, peak_layer - 2)
        switch_layer_2 = peak_layer
    else:
        # Use first two peaks
        peak_1_idx = peaks[0]
        peak_2_idx = peaks[1]
        
        switch_layer_1 = layers[peak_1_idx]
        switch_layer_2 = layers[peak_2_idx]
        
        print(f"Peak 1 at Layer {switch_layer_1} (KTI: {kti_smooth[peak_1_idx]:.4f})")
        print(f"Peak 2 at Layer {switch_layer_2} (KTI: {kti_smooth[peak_2_idx]:.4f})")
    
    # Generate config
    config = {
        'MODEL': {
            'BACKBONE': {
                'switch_layer_1': int(switch_layer_1),
                'switch_layer_2': int(switch_layer_2),
                'mamba_variant': 'spiral',
                'gcn_variant': 'grid'
            }
        },
        'metadata': {
            'source': 'KTI Peak Detection',
            'peaks_detected': len(peaks),
            'peak_layers': [int(layers[p]) for p in peaks],
            'kti_values': [float(kti_smooth[p]) for p in peaks]
        }
    }
    
    # Save config
    if output_config_path:
        output_path = Path(output_config_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Saved adaptive config to: {output_path}")
    
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze KTI peaks and generate adaptive architecture config')
    parser.add_argument('--metrics', type=str, required=True, help='Path to layer_metrics_Control.json')
    parser.add_argument('--output', type=str, default='nvit/Paper1_Diagnostics/Experiment2_KTI/results/adaptive_config.yaml', help='Output config path')
    args = parser.parse_args()
    
    analyze_kti_peaks(args.metrics, args.output)
