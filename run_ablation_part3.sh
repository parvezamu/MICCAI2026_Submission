#!/bin/bash
# run_ablation_part3.sh
# Experiment 3: SimCLR + Baseline (no DS)

CHECKPOINT="/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth"
OUTPUT_DIR="/home/pahm409/ablation_ds_main_only"
FOLD=0
BATCH_SIZE=32
LR=0.0002
EPOCHS=100

echo "========================================================================"
echo "EXPERIMENT 3: SimCLR + Baseline (no DS)"
echo "========================================================================"

python DS_MAIN_ONLY2.py \
    --fold $FOLD \
    --attention none \
    --pretrained-checkpoint $CHECKPOINT \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --epochs $EPOCHS \
    --optimizer adamw \
    --output-dir $OUTPUT_DIR

echo ""
echo "✓ EXPERIMENT 3 COMPLETE"
echo "========================================================================"
