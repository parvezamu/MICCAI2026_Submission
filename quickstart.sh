#!/bin/bash
# quickstart.sh - Updated for GPU support

echo "=========================================="
echo "Stroke Foundation Model - Preprocessing"
echo "GPU-Accelerated Version"
echo "=========================================="

# Make sure you're in the right environment
source activate nnFormer

# Create logs directory
mkdir -p logs

# Check GPU availability
echo ""
echo "Checking GPU availability..."
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB' if torch.cuda.is_available() else 'N/A')"

# 1. Verify datasets
echo ""
echo "Step 1: Verifying datasets..."
python verify_datasets.py

# 2. Create configuration (ATLAS and UOA only - T1 data)
echo ""
echo "Step 2: Creating configuration..."
python create_stroke_config.py \
    --output stroke_config.json \
    --output-dir /home/pahm409/preprocessed_stroke_foundation \
    --datasets ATLAS UOA \
    --verify

# 3. Show configuration
echo ""
echo "Configuration created:"
cat stroke_config.json

# 4. Run preprocessing
echo ""
echo "Step 3: Starting preprocessing with GPU acceleration..."
echo "This will take some time. Monitor GPU usage with: watch -n 1 nvidia-smi"
python preprocess_stroke_foundation.py --config stroke_config.json

echo ""
echo "=========================================="
echo "Done! Check output in:"
echo "/home/pahm409/preprocessed_stroke_foundation"
echo "=========================================="
