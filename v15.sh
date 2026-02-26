#!/bin/bash
# run_minimal_balanced_test.sh
#
# MINIMAL TEST: Just 2 configs to prove balanced sampling works
# Total: 2 configs × 5 folds × 1 run = 10 experiments (~2-3 days)

CHECKPOINT="/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth"
OUTPUT_DIR="/home/pahm409/segmentation_balanced_test"

echo "=============================================="
echo "MINIMAL BALANCED SAMPLING TEST"
echo "=============================================="
echo "Total experiments: 10 (2 configs × 5 folds × 1 run)"
echo ""
echo "Config 1: Baseline (Random sampling)"
echo "Config 2: Baseline (Balanced sampling)"
echo ""
echo "Goal: Prove balanced sampling helps"
echo "If successful, expand to full experiment set"
echo "=============================================="
echo ""

for FOLD in 0 1 2 3 4; do
    echo ""
    echo "======================================"
    echo "FOLD $FOLD"
    echo "======================================"
    
    # Config 1: Random sampling (baseline)
    echo ""
    echo ">>> Running: Baseline + Random sampling"
    python v15.py \
        --fold $FOLD \
        --run-id 0 \
        --attention mkdc \
        --deep-supervision \
        --pretrained-checkpoint $CHECKPOINT \
        --epochs 100 \
        --batch-size 16 \
        --optimizer adamw \
        --output-dir $OUTPUT_DIR

done

echo ""
echo "======================================"
echo "MINIMAL TEST COMPLETE!"
echo "======================================"
echo "Results: $OUTPUT_DIR"
echo ""
echo "Analysis:"
echo "1. Compare mean DSC: Random vs Balanced"
echo "2. Check ATLAS vs UOA performance gap"
echo "3. Look at small/medium/large lesion stratification"
echo ""
echo "If balanced sampling helps by >1%, expand to full experiments"
echo "======================================"
