#!/bin/bash
#
# run_all_folds_random.sh
# Run all 5 folds with Random Initialization (BASELINE)
#
# Author: Parvez
# Date: January 2026


import os
export CUDA_VISIBLE_DEVICES=1

echo "========================================================================"
echo "  BASELINE: Random Initialization (5-Fold Cross-Validation)"
echo "========================================================================"
echo ""

# Configuration
OUTPUT_DIR="/home/pahm409/experiments_comparison/Random_Init1"
PYTHON_SCRIPT="train_clean.py"

echo "Initialization: Random (No Pretraining)"
echo "Output Directory: $OUTPUT_DIR"
echo ""
echo "NOTE: This is the BASELINE to compare against SimCLR pretraining"
echo ""
echo "========================================================================"
echo ""

# Run all 5 folds
for FOLD in 0 1 2 3 4
do
  echo ""
  echo "========================================================================"
  echo "  Starting Fold ${FOLD} - Random Initialization (BASELINE)"
  echo "========================================================================"
  echo ""
  
  # NOTE: No --pretrained-checkpoint argument! This triggers random init
  echo "Random Init Fold ${FOLD} on GPU 2"
  python $PYTHON_SCRIPT \
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
echo "  ALL 5 FOLDS COMPLETE - Random Initialization BASELINE"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next step: Run analysis script to compare with SimCLR"
echo "  python analyze_all_results.py"
echo ""
echo "========================================================================"
