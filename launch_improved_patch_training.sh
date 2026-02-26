#!/bin/bash
# launch_improved_patch_training.sh

source activate nnFormer
export SIMCLR_GPU_ID=4

echo "========================================================================"
echo "IMPROVED PATCH-BASED TRAINING"
echo "========================================================================"
echo "Key improvements:"
echo "  - 10 patches per volume (vs 4)"
echo "  - 70% lesion-focused sampling"
echo "  - Dense sliding window validation"
echo "  - No train-val gap!"
echo "========================================================================"

python train_patch_improved.py \
  --pretrained-checkpoint /home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260108_152856/checkpoints/checkpoint_epoch_70.pth \
  --epochs 100 \
  --batch-size 8 \
  --patch-size 96 96 96 \
  --patches-per-volume 10 \
  --lesion-focus-ratio 0.7 \
  --validation-overlap 0.5 \
  --lr 0.0001 \
  --save-every 10

echo ""
echo "Training complete!"
echo "Check results in: /home/pahm409/patch_improved_experiments/"
