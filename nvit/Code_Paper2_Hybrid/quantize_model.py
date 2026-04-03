#!/home/yangz/.conda/envs/4D-humans/bin/python

import os
import argparse
import sys

def quantize(args):
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("Error: onnxruntime or onnx not installed.")
        print("Please run: pip install onnx onnxruntime")
        sys.exit(1)

    input_model = args.input
    output_model = args.output
    
    if not os.path.exists(input_model):
        print(f"Error: Input model {input_model} not found.")
        sys.exit(1)

    print(f"Quantizing {input_model} (Dynamic INT8)...")
    
    # Quantize
    quantize_dynamic(
        model_input=input_model,
        model_output=output_model,
        weight_type=QuantType.QUInt8, # Quantize weights to Int8
        # optimize_model=True # Basic optimizations
    )
    
    # Verify sizes
    orig_size = os.path.getsize(input_model) / (1024*1024)
    quant_size = os.path.getsize(output_model) / (1024*1024)
    
    print(f"Success!")
    print(f"Original Size: {orig_size:.2f} MB")
    print(f"Quantized Size: {quant_size:.2f} MB")
    print(f"Reduction: {orig_size/quant_size:.1f}x")
    print(f"Saved to: {output_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/home/yangz/NViT-master/nvit/hmr2_pruned.onnx', help='Path to input ONNX')
    parser.add_argument('--output', type=str, default='/home/yangz/NViT-master/nvit/hmr2_pruned_int8.onnx', help='Path to output INT8 ONNX')
    args = parser.parse_args()
    
    quantize(args)
