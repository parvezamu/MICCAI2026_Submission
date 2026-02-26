#!/bin/bash
# launch_patch_based_training.sh

source activate nnFormer
export SIMCLR_GPU_ID=4

echo "========================================================================"
echo "PATCH-BASED TRAINING (Faster + Better DSC)"
echo "========================================================================"

python train_patch_based.py \
  --pretrained-checkpoint /home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260108_152856/checkpoints/checkpoint_epoch_70.pth \
  --epochs 100 \
  --batch-size 8 \
  --patch-size 96 96 96 \
  --patches-per-volume 4 \
  --lr 0.0001 \
  --save-every 10

echo ""
echo "Training complete!"

