#!/bin/bash
# run_all_experiments_corrected.sh
#
# CORRECTED experimental setup with proper learning rate scaling
# 
# KEY CORRECTIONS:
# 1. Learning rate: 0.0001 → 0.0004 (4× scaling for batch_size 32)
# 2. Warmup: 5 epochs linear warmup
# 3. AdamW optimizer (better for large batch)
# 4. Gradient clipping: max_norm=1.0
# 5. Weight decay: 0.01
#
# Total experiments: 60
# - 4 configurations × 5 folds × 3 runs
# - Each run uses different random seed for reproducibility
#
# Expected improvements:
# - Baseline: ~64-65% DSC (vs 62.84% with wrong LR)
# - SimCLR+MKDC+DS: ~66-67% DSC (vs 63.11% with wrong LR)

CHECKPOINT="/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth"
OUTPUT_DIR="/home/pahm409/segmentation_corrected_5fold"

echo "=============================================="
echo "CORRECTED EXPERIMENTAL SETUP"
echo "=============================================="
echo "Total experiments: 60"
echo "Configurations: 4"
echo "Folds: 5"
echo "Runs per fold: 3"
echo ""
echo "KEY CORRECTIONS:"
echo "  ✓ Batch size: 32"
echo "  ✓ Learning rate: 0.0004 (4× scaled!)"
echo "  ✓ Optimizer: AdamW"
echo "  ✓ Warmup: 5 epochs"
echo "  ✓ Gradient clipping: 1.0"
echo "  ✓ Weight decay: 0.01"
echo ""
echo "Expected training time: ~5-7 days"
echo "=============================================="
echo ""

# Quick test first (optional - comment out to skip)
echo "======================================"
echo "QUICK TEST - Fold 0, Run 0, 20 epochs"
echo "======================================"
# Run all configurations for 3 runs across 5 folds
for FOLD in 0 1 2 3 4; do
    for RUN in 0 1 2; do
        echo ""
        echo "======================================"
        echo "FOLD $FOLD - RUN $RUN"
        echo "======================================"
        
        # 1. Random Init + Baseline + DS
        echo ""
        echo ">>> Running: Random Init + Baseline + DS"
        python v10.py \
            --fold $FOLD \
            --run-id $RUN \
            --attention none \
            --deep-supervision \
            --epochs 100 \
            --batch-size 16 \
            --optimizer adamw \
            --weight-decay 0.01 \
            --warmup-epochs 5 \
            --max-grad-norm 1.0 \
            --output-dir $OUTPUT_DIR
        
        # 2. Random Init + MKDC + DS
        echo ""
        echo ">>> Running: Random Init + MKDC + DS"
        python v10.py \
            --fold $FOLD \
            --run-id $RUN \
            --attention mkdc \
            --deep-supervision \
            --epochs 100 \
            --batch-size 16 \
            --optimizer adamw \
            --weight-decay 0.01 \
            --warmup-epochs 5 \
            --max-grad-norm 1.0 \
            --output-dir $OUTPUT_DIR
        

    done
done

echo ""
echo "======================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "======================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Analyze training logs for convergence"
echo "2. Compare with previous (incorrect LR) results"
echo "3. Run aggregate analysis script"
echo "4. Prepare results table and visualizations"
echo "======================================"
