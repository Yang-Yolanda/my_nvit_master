#!/bin/bash

# [Robust Root Detection]
# This script dynamically sets the project environment regardless of its location.
# Source this file before running any NViT experiments: source nvit_env.sh

# 1. Detect NViT-master root (the location of this script)
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 2. Detect 4D-Humans sibling root
export HUMANS_ROOT="$( cd "$PROJECT_ROOT/.." && pwd )/4D-Humans"

if [ ! -d "$HUMANS_ROOT" ]; then
    echo "⚠️ Warning: 4D-Humans sibling directory not found at $HUMANS_ROOT"
    echo "Current PROJECT_ROOT: $PROJECT_ROOT"
fi

# 3. Setup PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$HUMANS_ROOT:$PROJECT_ROOT/nvit:$PYTHONPATH"

# 4. Standardize Conda Environment (Optional)
# export CONDA_ENV_PATH="/path/to/conda/4D-humans"

# 5. Output for Verification
echo "✅ NViT Environment Initialized"
echo "   PROJECT_ROOT: $PROJECT_ROOT"
echo "   HUMANS_ROOT:  $HUMANS_ROOT"
echo "   PYTHONPATH:   ... (updated)"
