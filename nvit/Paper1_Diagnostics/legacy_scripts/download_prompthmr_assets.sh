#!/bin/bash
mkdir -p ./external_models/PromptHMR/data/pretrain
cd ./external_models/PromptHMR

# PromptHMR checkpoints
gdown --folder -O ./data/pretrain/ https://drive.google.com/drive/folders/1EQ7arZz135T-WpxkS_K1R_hjZp3prh-y?usp=share_link
gdown --folder -O ./data/pretrain/ https://drive.google.com/drive/folders/18SywG7Fc_iTfVNaikjHAZmy-A9I85eKv?usp=sharing

# Thirdparty checkpoints
gdown --folder -O ./data/pretrain/ https://drive.google.com/drive/folders/1OKhTdL1QVFH3f4hbIEa7jLANx4azuPi1?usp=sharing

# Examples (Checking for body models)
gdown --folder -O ./data/ https://drive.google.com/drive/folders/1uhy_8rCjOELqR9G5BXBKu0-cnQOSHFkD?usp=sharing

# Specifically fetch smplx2smpl.pkl from the supplementary folder mentioned in fetch_smplx.sh
# 1v9Qy7ZXWcTM8_a9K2nSLyyVrJMFYcUOk
mkdir -p ./data/body_models
gdown -O ./data/body_models/smplx2smpl.pkl 1v9Qy7ZXWcTM8_a9K2nSLyyVrJMFYcUOk # Attempt

