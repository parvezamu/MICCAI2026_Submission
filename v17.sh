#!/bin/bash
# run_balanced_sampling_experiments.sh
#
# Experimental setup comparing BALANCED vs RANDOM sampling
# 
# Goal: Test if balanced dataset×lesion-size sampling prevents shortcuts
#
# Experiment design:
# 1. Baseline (Random sampling) - 5 folds × 3 runs = 15 experiments
# 2. Baseline (Balanced sampling) - 5 folds × 3 runs = 15 experiments
# Total: 30 experiments (quick test of sampling effect)
#
# If balanced sampling helps, then test with prompts:
# 3. Full enhancement (Random) - 5 folds × 3 runs = 15 experiments  
# 4. Full enhancement (Balanced) - 5 folds × 3 runs = 15 experiments
# Total with prompts: 60 experiments
#
# Author: Parvez
# Date: January 2026

CHECKPOINT="/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth"
#CHECKPOINT=""
OUTPUT_DIR="/home/pahm409/segmentation_with_prompts"

echo "=============================================="
echo "BALANCED SAMPLING EXPERIMENTAL SETUP"
echo "=============================================="
echo "Phase 1: Test balanced sampling (30 experiments)"
echo "  - Baseline + Random sampling (15)"
echo "  - Baseline + Balanced sampling (15)"
echo ""
echo "If Phase 1 shows improvement:"
echo "Phase 2: Full enhancement comparison (30 more)"
echo "  - Full + Random sampling (15)"
echo "  - Full + Balanced sampling (15)"
echo ""
echo "Key settings:"
echo "  ✓ Batch size: 16"
echo "  ✓ Learning rate: 0.0002 (auto-scaled)"
echo "  ✓ Optimizer: AdamW"
echo "  ✓ Warmup: 5 epochs"
echo "=============================================="
echo ""

# ============================================================================
# QUICK TEST FIRST (20 epochs, fold 0, run 0)
# ============================================================================
echo "======================================"
echo "QUICK TEST - Comparing sampling methods"
echo "======================================"
echo ""

# Test 1: Baseline without balanced sampling
echo "Test 1: Baseline (Random sampling)..."
python v17.py \
    --fold 0 \
    --run-id 0 \
    --attention mkdc \
    --deep-supervision \
    --pretrained-checkpoint $CHECKPOINT \
    --epochs 20 \
    --batch-size 16 \
    --optimizer adamw \
    --output-dir "${OUTPUT_DIR}_test"

echo ""

# Test 2: Baseline with balanced sampling
echo "Test 2: Baseline (Balanced sampling)..."
python v17.py \
    --fold 0 \
    --run-id 0 \
    --attention mkdc \
    --deep-supervision \
    --balanced-sampling \
    --pretrained-checkpoint $CHECKPOINT \
    --epochs 20 \
    --batch-size 16 \
    --optimizer adamw \
    --output-dir "${OUTPUT_DIR}_test"

echo ""
echo "======================================"
echo "Quick test complete!"
echo "======================================"
echo ""
echo "Compare the two test runs:"
echo "1. Check if balanced sampling improves val DSC"
echo "2. Look at training curves for stability"
echo "3. Check if ATLAS/UOA performance is more balanced"
echo ""
read -p "Continue with Phase 1 (30 experiments)? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Exiting. Review test results first."
    exit 1
fi

# ============================================================================
# PHASE 1: BASELINE COMPARISON (30 experiments)
# ============================================================================
echo ""
echo "=============================================="
echo "PHASE 1: BASELINE SAMPLING COMPARISON"
echo "=============================================="
echo "Running 30 experiments (5 folds × 3 runs × 2 sampling methods)"
echo ""

for FOLD in 0 1 2 3 4; do
    for RUN in 0 1 2; do
        echo ""
        echo "======================================"
        echo "FOLD $FOLD - RUN $RUN"
        echo "======================================"
        
        # Config 1: Baseline + Random sampling
        echo ""
        echo ">>> Config 1: Baseline (SimCLR + MKDC + DS) - RANDOM sampling"
        python v17.py \
            --fold $FOLD \
            --run-id $RUN \
            --attention mkdc \
            --deep-supervision \
            --pretrained-checkpoint $CHECKPOINT \
            --epochs 100 \
            --batch-size 16 \
            --optimizer adamw \
            --output-dir $OUTPUT_DIR
        
        # Config 2: Baseline + Balanced sampling
        echo ""
        echo ">>> Config 2: Baseline (SimCLR + MKDC + DS) - BALANCED sampling"
        python v17.py \
            --fold $FOLD \
            --run-id $RUN \
            --attention mkdc \
            --deep-supervision \
            --balanced-sampling \
            --pretrained-checkpoint $CHECKPOINT \
            --epochs 100 \
            --batch-size 16 \
            --optimizer adamw \
            --output-dir $OUTPUT_DIR
        
        echo ""
        echo "✓ Completed FOLD $FOLD - RUN $RUN (Phase 1)"
        echo ""
    done
done

echo ""
echo "======================================"
echo "PHASE 1 COMPLETE!"
echo "======================================"
echo "Analyze results to decide on Phase 2"
echo ""
read -p "Did balanced sampling improve results? Continue with Phase 2? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Stopping after Phase 1. Analyze results."
    exit 0
fi

# ============================================================================
# PHASE 2: FULL ENHANCEMENT COMPARISON (30 more experiments)
# ============================================================================
echo ""
echo "=============================================="
echo "PHASE 2: FULL ENHANCEMENT COMPARISON"
echo "=============================================="
echo "Running 30 more experiments (prompts + balanced sampling)"
echo ""

for FOLD in 0 1 2 3 4; do
    for RUN in 0 1 2; do
        echo ""
        echo "======================================"
        echo "FOLD $FOLD - RUN $RUN (Phase 2)"
        echo "======================================"
        
        # Config 3: Full enhancement + Random sampling
        echo ""
        echo ">>> Config 3: Full Enhancement - RANDOM sampling"
        python v17.py \
            --fold $FOLD \
            --run-id $RUN \
            --attention mkdc \
            --deep-supervision \
            --use-prompts \
            --num-prompts 10 \
            --use-domain-prompts \
            --multi-scale-supervision \
            --pretrained-checkpoint $CHECKPOINT \
            --epochs 100 \
            --batch-size 16 \
            --optimizer adamw \
            --output-dir $OUTPUT_DIR
        
        # Config 4: Full enhancement + Balanced sampling
        echo ""
        echo ">>> Config 4: Full Enhancement - BALANCED sampling"
        python v17.py \
            --fold $FOLD \
            --run-id $RUN \
            --attention mkdc \
            --deep-supervision \
            --use-prompts \
            --num-prompts 10 \
            --use-domain-prompts \
            --multi-scale-supervision \
            --balanced-sampling \
            --pretrained-checkpoint $CHECKPOINT \
            --epochs 100 \
            --batch-size 16 \
            --optimizer adamw \
            --output-dir $OUTPUT_DIR
        
        echo ""
        echo "✓ Completed FOLD $FOLD - RUN $RUN (Phase 2)"
        echo ""
    done
done

echo ""
echo "======================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "======================================"
echo "Total experiments run: 60"
echo "  Phase 1 (Baseline comparison): 30"
echo "  Phase 2 (Full enhancement): 30"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Expected improvements:"
echo "  Phase 1: Balanced sampling should improve DSC by 1-2%"
echo "           and equalize ATLAS/UOA performance"
echo "  Phase 2: Prompts + balanced should give best results"
echo ""
echo "Next steps:"
echo "1. Run aggregate analysis script"
echo "2. Compare sampling methods (Random vs Balanced)"
echo "3. Analyze per-dataset performance (ATLAS vs UOA)"
echo "4. Check lesion-size stratified performance"
echo "5. Prepare results for paper"
echo "======================================"
