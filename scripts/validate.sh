#!/bin/bash
export TERM=xterm-256color

dataset="multidex"  # ("dexgrab" "realdex" "dexgraspnet" "unidexgrasp" "multidex")
robots=("barrett" "allegro" "shadowhand")
radius=0.01

cd ~/Documents/goag
conda deactivate

# Activate the 'model' conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate model

# Run the inference script for each robot
for robot in "${robots[@]}"; do
    python utils_validation/validate_models.py --robot_name "$robot" --radius "$radius" --dataset "$dataset"
done

# Activate the 'isaac_env' conda environment
conda deactivate
conda activate isaac_env

# Run the validation script
for robot in "${robots[@]}"; do
    python utils_validation/validate_isaac.py --robot_name "$robot" --radius "$radius" --dataset "$dataset"
done

# Deactivate the 'isaac_env' environment
conda deactivate
conda activate model
