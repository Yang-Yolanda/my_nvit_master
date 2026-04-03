import os
import sys
import subprocess

MODELS = [
    {"name": "HMR2", "gpu": 0, "type": "Body"},
    {"name": "PromptHMR", "gpu": 1, "type": "Body"},
    {"name": "HSMR", "gpu": 2, "type": "Body"},
    {"name": "HaMeR", "gpu": 3, "type": "Hand"},
    {"name": "AniMer", "gpu": 4, "type": "Animal"},
    {"name": "Robot", "gpu": 5, "type": "Robot"},
    {"name": "SigLIP", "gpu": 6, "type": "Vision"},
]

def run_diag(model_info):
    name = model_info['name']
    gpu = model_info['gpu']
    print(f"🚀 Launching Full Diagnostics for {name} on GPU {gpu}...")
    # This script will call the respective model runners with flags for all metrics
    # For now, let's target the unified evaluate scripts in Experiment2_KTI/scripts
    # and ensure they save to all 3 experiment dirs.
    pass

if __name__ == "__main__":
    print("Master Diagnostic Runner Initialized.")
