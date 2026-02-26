#!/bin/bash
# run_ablation_part2.sh
# Experiment 2: Random Init + MKDC + DS (main-only loss)

CHECKPOINT="/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth"
OUTPUT_DIR="/home/pahm409/ablation_ds_main_only"
FOLD=0
BATCH_SIZE=32
LR=0.0002
EPOCHS=100

echo "========================================================================"
echo "EXPERIMENT 2: Random Init + MKDC + DS (main-only loss)"
echo "========================================================================"

python train_segmentation_corrected_DS_MAIN_ONLY.py \
    --fold $FOLD \
    --attention mkdc \
    --deep-supervision \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --epochs $EPOCHS \
    --optimizer adamw \
    --output-dir $OUTPUT_DIR

echo ""
echo "✓ EXPERIMENT 2 COMPLETE"
echo "========================================================================"
