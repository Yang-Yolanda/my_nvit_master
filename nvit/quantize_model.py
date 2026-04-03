#!/home/yangz/.conda/envs/4D-humans/bin/python
import os
import argparse
import sys
try:
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    print("Error: Please run 'pip install onnx onnxruntime'")
    sys.exit(1)

def quantize(args):
    input_model = args.input
    output_model = args.output
    if not os.path.exists(input_model):
        print(f"Error: {input_model} not found.")
        sys.exit(1)
        
    print(f"Quantizing {input_model} (INT8)...")
    quantize_dynamic(input_model, output_model, weight_type=QuantType.QUInt8)
    
    orig_s = os.path.getsize(input_model)/(1024*1024)
    quant_s = os.path.getsize(output_model)/(1024*1024)
    print(f"Success! {orig_s:.2f}MB -> {quant_s:.2f}MB (Reduction: {orig_s/quant_s:.1f}x)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/home/yangz/NViT-master/nvit/hmr2_pruned.onnx')
    parser.add_argument('--output', default='/home/yangz/NViT-master/nvit/hmr2_pruned_int8.onnx')
    args = parser.parse_args()
    quantize(args)
