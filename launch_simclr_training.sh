#!/bin/bash
# launch_simclr_training.sh
# Launch SimCLR foundation model training on GPU 7

source activate nnFormer

# Set GPU
export SIMCLR_GPU_ID=4

# Create experiment directory
mkdir -p /home/pahm409/stroke_foundation_experiments

echo "=========================================="
echo "Training SimCLR Foundation Model (T1-only)"
echo "GPU: $SIMCLR_GPU_ID"
echo "=========================================="

python train_simclr.py --config config_simclr_foundation.yaml

echo ""
echo "Training complete!"
