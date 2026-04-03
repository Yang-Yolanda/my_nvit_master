#!/home/yangz/.conda/envs/4D-humans/bin/python

import onnxruntime as ort
import numpy as np
import time
import argparse
import os

def benchmark(model_path, name="Model", runs=100):
    if not os.path.exists(model_path):
        print(f"Skipping {name}: {model_path} not found.")
        return

    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    # Handle dynamic batch
    if isinstance(input_shape[0], str): input_shape[0] = 1
    
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    print(f"Warming up {name}...")
    for _ in range(10):
        session.run(None, {input_name: dummy_input})
        
    # Benchmark
    print(f"Benchmarking {name} ({runs} runs)...")
    start = time.time()
    for _ in range(runs):
        session.run(None, {input_name: dummy_input})
    end = time.time()
    
    avg_lat = (end - start) / runs * 1000 # ms
    fps = 1000 / avg_lat
    print(f"[{name}] Avg Latency: {avg_lat:.2f} ms | FPS: {fps:.2f}")

if __name__ == "__main__":
    benchmark('/home/yangz/NViT-master/nvit/hmr2_pruned.onnx', "FP32 Original")
    benchmark('/home/yangz/NViT-master/nvit/hmr2_pruned_int8.onnx', "INT8 Quantized")
