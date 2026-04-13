#!/usr/bin/env python
import onnxruntime as ort
import sys
from nvit.utils.path_utils import get_humans_root, get_project_root, resolve_data_path


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
