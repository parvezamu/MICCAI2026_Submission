#!/bin/bash
# launch_monitored_training.sh

source activate nnFormer
export SIMCLR_GPU_ID=7

echo "=========================================="
echo "Starting SimCLR Training with Monitoring"
echo "=========================================="

# Start training
python train_simclr_with_monitoring.py --config config_simclr_foundation.yaml &

# Get PID
TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"

# Wait a bit for experiment directory to be created
sleep 10

# Find the latest experiment directory
EXP_DIR=$(ls -dt /home/pahm409/stroke_foundation_experiments1/stroke_foundation_simclr_t1only_* | head -1)

echo "Experiment directory: $EXP_DIR"
echo ""
echo "Monitoring commands:"
echo "  python monitor_all.py --exp-dir $EXP_DIR"
echo "  tail -f $EXP_DIR/logs/training.log"
echo ""
echo "Training in progress..."
