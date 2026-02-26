#!/bin/bash

CHECKPOINT="/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260108_152856/checkpoints/checkpoint_epoch_70.pth"

for FOLD in 0 1 2 3 4
do
  echo "========================================="
  echo "Starting Fold ${FOLD}"
  echo "========================================="
  
  python train_patch_with_reconstruction_five_fold.py \
    --pretrained-checkpoint $CHECKPOINT \
    --fold $FOLD \
    --epochs 100 \
    --batch-size 8 \
    --patch-size 96 96 96 \
    --patches-per-volume 10 \
    --lesion-focus-ratio 0.7 \
    --lr 0.0001 \
    --validate-recon-every 5 \
    --save-nifti-every 25 
  
  echo ""
  echo "Fold ${FOLD} complete!"
  echo ""
done

echo "========================================="
echo "All 5 folds complete!"
echo "========================================="
