#!/bin/bash
# launch_final_finetuning.sh

source activate nnFormer
export SIMCLR_GPU_ID=4

echo "========================================================================"
echo "Starting Final Segmentation Fine-tuning"
echo "Using SimCLR Epoch 70 Checkpoint"
echo "========================================================================"

python finetune_segmentation_final.py \
  --pretrained-checkpoint /home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260108_152856/checkpoints/checkpoint_epoch_70.pth \
  --epochs 150 \
  --lr 0.0001 \
  --save-every 10

echo ""
echo "Fine-tuning complete!"
echo "Check results in: /home/pahm409/segmentation_final_experiments/"
