#!/bin/bash
#
# run_all_folds_simclr.sh
# Run all 5 folds with SimCLR pretrained encoder
#
# Author: Parvez
# Date: January 2026
import os
export CUDA_VISIBLE_DEVICES=0
echo "========================================================================"
echo "  EXPERIMENT: SimCLR Pretrained Encoder (5-Fold Cross-Validation)"
echo "========================================================================"
echo ""

# Configuration
CHECKPOINT="/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth"
OUTPUT_DIR="/home/pahm409/experiments_comparison/SimCLR_Pretrained1"
PYTHON_SCRIPT="train_segmentation_clean.py"


# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: SimCLR checkpoint not found at:"
    echo "  $CHECKPOINT"
    echo ""
    echo "Please update the CHECKPOINT path in this script."
    exit 1
fi

echo "SimCLR Checkpoint: $CHECKPOINT"
echo "Output Directory:  $OUTPUT_DIR"
echo ""
echo "========================================================================"
echo ""

# Run all 5 folds
for FOLD in 0 1 2 3 4
do
  echo ""
  echo "========================================================================"
  echo "  Starting Fold ${FOLD} - SimCLR Pretrained"
  echo "========================================================================"
  echo ""
  
  python $PYTHON_SCRIPT \
    --pretrained-checkpoint $CHECKPOINT \
    --output-dir $OUTPUT_DIR \
    --fold $FOLD \
    --epochs 50 \
    --batch-size 16 \
    --patch-size 96 96 96 \
    --patches-per-volume 10 \
    --lesion-focus-ratio 0.7 \
    --lr 0.0001 \
    --validate-recon-every 5 \
    --save-nifti-every 25
  
  EXIT_CODE=$?
  
  if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Fold ${FOLD} failed with exit code ${EXIT_CODE}"
    echo "Stopping execution."
    exit $EXIT_CODE
  fi
  
  echo ""
  echo "========================================================================"
  echo "  Fold ${FOLD} Complete!"
  echo "========================================================================"
  echo ""
done

echo ""
echo "========================================================================"
echo "  ALL 5 FOLDS COMPLETE - SimCLR Pretrained"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next step: Run analysis script to compare with baseline"
echo "  python analyze_all_results.py"
echo ""
echo "========================================================================"
