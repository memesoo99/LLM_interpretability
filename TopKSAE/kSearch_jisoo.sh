#!/bin/bash

# Base configuration
DATASET_PATH="/workspace/mistral_pile"
INPUT_DIM=4096
HIDDEN_DIM_MULT=100
BATCH_SIZE=256
NUM_EPOCHS=30
LEARNING_RATE=1e-4

# Array of k values to test
# k_values=(24 48 64 100 128 256)
k_values=(24 48)

# Create a directory for this batch of experiments
BATCH_DIR="${HIDDEN_DIM_MULT}x_k_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "experiments/${BATCH_DIR}"

# Log file for overall batch results
echo "Starting batch experiments for different k values" > "experiments/${BATCH_DIR}/batch_summary.log"
echo "Started at: $(date)" >> "experiments/${BATCH_DIR}/batch_summary.log"
echo "----------------------------------------" >> "experiments/${BATCH_DIR}/batch_summary.log"

# Run experiments for each k value
for k in "${k_values[@]}"; do
    echo "Starting experiment with k=${k}"
    echo "Running experiment with k=${k}" >> "experiments/${BATCH_DIR}/batch_summary.log"

    python trainSAE2.py \
        --dataset_path "$DATASET_PATH" \
        --exp_name "${BATCH_DIR}/mistral_pile_k${k}_experiment" \
        --input_dim $INPUT_DIM \
        --hidden_dim_multiplier $HIDDEN_DIM_MULT \
        --k $k \
        --batch_size $BATCH_SIZE \
        --num_epochs $NUM_EPOCHS \
        --learning_rate $LEARNING_RATE \
        2>&1 | tee "experiments/${BATCH_DIR}/k${k}_experiment.log"
    
    echo "Completed experiment with k=${k}"
    echo "----------------------------------------" >> "experiments/${BATCH_DIR}/batch_summary.log"
done