
#!/bin/bash
# simpleclr_mkdc_ds.sh

CHECKPOINT="/home/pahm409/stroke_foundation_experiments/stroke_foundation_simclr_t1only_20260113_111634/checkpoints/best_model.pth"
OUTPUT_DIR="/home/pahm409/segmentation_3runs"

# Run all configurations for 3 runs across 5 folds
for FOLD in 0 1 2 3 4; do
    for RUN in 0 1 2; do
        echo "======================================"
        echo "FOLD $FOLD - RUN $RUN"
        echo "======================================"


        echo "Running: SimCLR + Baseline + DS"
        python v8.py \
            --fold $FOLD \
            --run-id $RUN \
            --attention none \
            --deep-supervision \
            --pretrained-checkpoint $CHECKPOINT \
            --epochs 100 \
            --output-dir $OUTPUT_DIR



    done
done

echo "======================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "======================================"


