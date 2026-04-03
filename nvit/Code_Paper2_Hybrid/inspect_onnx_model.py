#!/home/yangz/.conda/envs/4D-humans/bin/python
import onnxruntime as ort
import sys
import os

def inspect_model(model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    try:
        session = ort.InferenceSession(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"--- Model Inspection: {os.path.basename(model_path)} ---")
    
    print("\nInputs:")
    for i, meta in enumerate(session.get_inputs()):
        print(f"  [{i}] Name: {meta.name}")
        print(f"      Shape: {meta.shape}")
        print(f"      Type:  {meta.type}")

    print("\nOutputs:")
    for i, meta in enumerate(session.get_outputs()):
        print(f"  [{i}] Name: {meta.name}")
        print(f"      Shape: {meta.shape}")
        print(f"      Type:  {meta.type}")
        
    # Metadata (optional)
    meta = session.get_modelmeta()
    print(f"\nMetadata:")
    print(f"  Description: {meta.description}")
    print(f"  Domain: {meta.domain}")
    print(f"  Graph Name: {meta.graph_name}")
    print(f"  Producer Name: {meta.producer_name}")
    print(f"  Version: {meta.version}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_onnx_model.py <path_to_onnx_file>")
    else:
        inspect_model(sys.argv[1])
