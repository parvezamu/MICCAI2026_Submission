#!/bin/bash

################################################################################
# run_low_data_experiments.sh
#
# Run all low-data experiments to show when transfer learning helps
#
# Tests 3 data regimes:
#   - C1: 25 training cases (very low data)
#   - C2: 50 training cases (low data)  
#   - C3: 100 training cases (medium data)
#
# Each regime tests:
#   - WITH transfer learning (BraTS pretrained, discriminative LR)
#   - WITHOUT transfer learning (random init, uniform LR)
################################################################################

set -e  # Exit on error

# Configuration
BRATS_CKPT="/home/pahm409/ISLES2029/simclr_output/brats/checkpoints/best_model.pth"
FOLD=0
EPOCHS=50
BATCH_SIZE=8
DECODER_LR=0.0001

# Directories
OUTPUT_DIR="/home/pahm409/finetuned_low_data"
ISLES_DIR="/home/pahm409/preprocessed_stroke_foundation"

# Check if checkpoint exists
if [ ! -f "$BRATS_CKPT" ]; then
    echo "ERROR: BraTS checkpoint not found: $BRATS_CKPT"
    exit 1
fi

echo "================================================================================"
echo "LOW-DATA TRANSFER LEARNING EXPERIMENTS"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Pretrained checkpoint: $BRATS_CKPT"
echo "  Fold: $FOLD"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Decoder LR: $DECODER_LR"
echo ""
echo "Testing 3 data regimes:"
echo "  C1: 25 training cases  (very low data - transfer should HELP a lot)"
echo "  C2: 50 training cases  (low data - transfer should HELP moderately)"
echo "  C3: 100 training cases (medium data - transfer should HELP minimally)"
echo ""
echo "Each regime runs 2 experiments:"
echo "  1. WITH transfer learning (encoder LR = 0.1 × decoder LR)"
echo "  2. WITHOUT transfer learning (encoder LR = decoder LR, no freeze)"
echo ""
echo "Total experiments: 6 (3 regimes × 2 conditions)"
echo "Estimated time: 12-16 hours"
echo "================================================================================"
echo ""

# Function to run experiment
run_experiment() {
    local NUM_CASES=$1
    local EXPERIMENT_NAME=$2
    local ENCODER_LR_RATIO=$3
    local FREEZE_EPOCHS=$4
    local OUTPUT_SUBDIR=$5
    
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "EXPERIMENT: $EXPERIMENT_NAME"
    echo "--------------------------------------------------------------------------------"
    echo "  Training cases: $NUM_CASES"
    echo "  Encoder LR ratio: $ENCODER_LR_RATIO"
    echo "  Freeze epochs: $FREEZE_EPOCHS"
    echo "  Output: $OUTPUT_DIR/$OUTPUT_SUBDIR"
    echo "--------------------------------------------------------------------------------"
    echo ""
    
    python finetune_on_isles_LOW_DATA.py \
        --pretrained-checkpoint "$BRATS_CKPT" \
        --isles-dir "$ISLES_DIR" \
        --fold $FOLD \
        --max-train-cases $NUM_CASES \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --decoder-lr $DECODER_LR \
        --encoder-lr-ratio $ENCODER_LR_RATIO \
        --freeze-encoder-epochs $FREEZE_EPOCHS \
        --isles-only \
        --output-dir "$OUTPUT_DIR/$OUTPUT_SUBDIR"
    
    echo ""
    echo "✓ COMPLETED: $EXPERIMENT_NAME"
    echo ""
}

################################################################################
# EXPERIMENT C1: 25 Training Cases (Very Low Data)
################################################################################

echo ""
echo "================================================================================"
echo "C1: 25 TRAINING CASES (VERY LOW DATA)"
echo "================================================================================"
echo ""

# C1a: WITH transfer learning
run_experiment \
    25 \
    "C1a: 25 cases + Transfer Learning" \
    0.1 \
    3 \
    "25cases_transfer"

# C1b: WITHOUT transfer learning (random init)
run_experiment \
    25 \
    "C1b: 25 cases + Random Init" \
    1.0 \
    0 \
    "25cases_scratch"

################################################################################
# EXPERIMENT C2: 50 Training Cases (Low Data)
################################################################################

echo ""
echo "================================================================================"
echo "C2: 50 TRAINING CASES (LOW DATA)"
echo "================================================================================"
echo ""

# C2a: WITH transfer learning
run_experiment \
    50 \
    "C2a: 50 cases + Transfer Learning" \
    0.1 \
    3 \
    "50cases_transfer"

# C2b: WITHOUT transfer learning
run_experiment \
    50 \
    "C2b: 50 cases + Random Init" \
    1.0 \
    0 \
    "50cases_scratch"

################################################################################
# EXPERIMENT C3: 100 Training Cases (Medium Data)
################################################################################

echo ""
echo "================================================================================"
echo "C3: 100 TRAINING CASES (MEDIUM DATA)"
echo "================================================================================"
echo ""

# C3a: WITH transfer learning
run_experiment \
    100 \
    "C3a: 100 cases + Transfer Learning" \
    0.1 \
    3 \
    "100cases_transfer"

# C3b: WITHOUT transfer learning
run_experiment \
    100 \
    "C3b: 100 cases + Random Init" \
    1.0 \
    0 \
    "100cases_scratch"

################################################################################
# Summary
################################################################################

echo ""
echo "================================================================================"
echo "ALL LOW-DATA EXPERIMENTS COMPLETED!"
echo "================================================================================"
echo ""
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Directory structure:"
echo "  $OUTPUT_DIR/"
echo "    ├── 25cases_transfer/   (C1a: 25 cases + transfer)"
echo "    ├── 25cases_scratch/    (C1b: 25 cases + random)"
echo "    ├── 50cases_transfer/   (C2a: 50 cases + transfer)"
echo "    ├── 50cases_scratch/    (C2b: 50 cases + random)"
echo "    ├── 100cases_transfer/  (C3a: 100 cases + transfer)"
echo "    └── 100cases_scratch/   (C3b: 100 cases + random)"
echo ""
echo "Next steps:"
echo "  1. Evaluate all checkpoints: ./evaluate_all_low_data.sh"
echo "  2. Summarize results: python summarize_low_data_results.py"
echo ""
echo "================================================================================"
