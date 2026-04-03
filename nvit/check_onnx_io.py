#!/home/yangz/.conda/envs/4D-humans/bin/python
import onnxruntime as ort
import sys

try:
    path = '/home/yangz/NViT-master/nvit/hmr2_pruned_int8.onnx'
    session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    print("Inputs:")
    for i in session.get_inputs():
        print(f"  Name: {i.name}, Shape: {i.shape}, Type: {i.type}")
    print("Outputs:")
    for o in session.get_outputs():
        print(f"  Name: {o.name}, Shape: {o.shape}, Type: {o.type}")
except Exception as e:
    print(e)
