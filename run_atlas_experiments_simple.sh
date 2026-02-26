#!/bin/bash

echo "========================================================================"
echo "ATLAS → ISLES EXPERIMENTS (Simplified)"
echo "========================================================================"
echo ""

ATLAS_SIMCLR="/home/pahm409/simclr_output/atlas_simclr_t1_strokes_20260206_191211/checkpoints/best_model.pth"

if [ ! -f "$ATLAS_SIMCLR" ]; then
    echo "❌ ATLAS SimCLR checkpoint not found!"
    echo "   Expected: $ATLAS_SIMCLR"
    exit 1
fi

echo "✓ Found ATLAS SimCLR checkpoint"
echo ""

# Experiment: ATLAS SimCLR → ISLES
echo "========================================================================"
echo "EXPERIMENT: ATLAS SimCLR (655 stroke cases) → ISLES"
echo "========================================================================"
echo ""

python finetune_on_isles_FIXED.py \
    --pretrained-checkpoint "$ATLAS_SIMCLR" \
    --fold 0 \
    --epochs 50 \
    --batch-size 8 \
    --decoder-lr 0.0001 \
    --encoder-lr-ratio 0.1 \
    --freeze-encoder-epochs 3 \
    --isles-only \
    --output-dir /home/pahm409/finetuned_atlas_simclr_to_isles

if [ $? -ne 0 ]; then
    echo "❌ Fine-tuning failed!"
    exit 1
fi

echo ""
echo "✓ Fine-tuning complete"
echo ""

# Find checkpoint
CHECKPOINT=$(find /home/pahm409/finetuned_atlas_simclr_to_isles/fold_0 -name "best_finetuned_model.pth" | head -1)

if [ -z "$CHECKPOINT" ]; then
    echo "❌ Could not find finetuned checkpoint!"
    exit 1
fi

echo "Found checkpoint: $CHECKPOINT"
echo ""

# Evaluate
echo "========================================================================"
echo "EVALUATING ON TEST SET"
echo "========================================================================"
echo ""

python evaluate_isles_test.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir /home/pahm409/test_results/atlas_simclr_to_isles

echo ""
echo "========================================================================"
echo "COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved to: /home/pahm409/test_results/atlas_simclr_to_isles"
echo ""
echo "Generate summary:"
echo "  python summarize_final_results.py"
