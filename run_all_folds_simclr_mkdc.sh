#!/bin/bash
#
# run_all_folds_simclr_mkdc.sh
# Run all 5 folds with SimCLR pretrained encoder + MKDC
#
# Author: Parvez
# Date: January 2026

export CUDA_VISIBLE_DEVICES=1

echo "========================================================================"
echo "  EXPERIMENT: SimCLR + MKDC (Full Method)"
echo "  Purpose: Evaluate complete proposed method"
echo "========================================================================"
echo ""

# Configuration
CHECKPOINT="/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth"
OUTPUT_DIR="/home/pahm409/experiments_comparison/SimCLR_MKDC"
PYTHON_SCRIPT="train_segmentation_clean_mkr.py"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: SimCLR checkpoint not found at:"
    echo "  $CHECKPOINT"
    echo ""
    echo "Please update the CHECKPOINT path in this script."
    exit 1
fi

echo "Configuration:"
echo "  SimCLR Checkpoint: $CHECKPOINT"
echo "  MKDC: ENABLED (kernels [1, 3, 5])"
echo "  Output Directory: $OUTPUT_DIR"
echo ""
echo "Key Features:"
echo "  ✓ SimCLR pretrained encoder"
echo "  ✓ MKDC multi-scale skip connections"
echo "  ✓ Multi-kernel depthwise convolutions [1×1, 3×3, 5×5]"
echo "  ✓ Expected: +2-5% improvement over SimCLR alone"
echo ""
echo "========================================================================"
echo ""

# Run all 5 folds
for FOLD in 0 1 2 3 4
do
  echo ""
  echo "========================================================================"
  echo "  Starting Fold ${FOLD} - SimCLR + MKDC (Full Method)"
  echo "========================================================================"
  echo ""
  
  python $PYTHON_SCRIPT \
    --pretrained-checkpoint $CHECKPOINT \
    --use-mkdc \
    --output-dir $OUTPUT_DIR \
    --fold $FOLD \
    --epochs 100 \
    --batch-size 8 \
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
echo "  ALL 5 FOLDS COMPLETE - SimCLR + MKDC"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Expected performance: DSC ~0.58-0.63"
echo "  SimCLR contribution: +8-10%"
echo "  MKDC contribution:   +2-5%"
echo "  Total improvement:   +10-15% over baseline"
echo ""
echo "========================================================================"
