#!/bin/bash

# Base configuration
DATASET_PATH="./dataset"
INPUT_DIM=3072
HIDDEN_DIM_MULT=4
BATCH_SIZE=400
NUM_EPOCHS=15
LEARNING_RATE=1e-4

# Array of k values to test
# k_values=(100 128)

# Create a directory for this batch of experiments
BATCH_DIR="k_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "experiments/${BATCH_DIR}"

# Log file for overall batch results
echo "Starting batch experiments for different k values" > "experiments/${BATCH_DIR}/batch_summary.log"
echo "Started at: $(date)" >> "experiments/${BATCH_DIR}/batch_summary.log"
echo "----------------------------------------" >> "experiments/${BATCH_DIR}/batch_summary.log"


python trainSAE2.py --dataset_path "$DATASET_PATH" --exp_name "K_32" --input_dim $INPUT_DIM --hidden_dim_multiplier 4 --k 32 --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --learning_rate $LEARNING_RATE
python trainSAE2.py --dataset_path "$DATASET_PATH" --exp_name "K_64" --input_dim $INPUT_DIM --hidden_dim_multiplier 4 --k 64 --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --learning_rate $LEARNING_RATE
python trainSAE2.py --dataset_path "$DATASET_PATH" --exp_name "K_128" --input_dim $INPUT_DIM --hidden_dim_multiplier 4 --k 128 --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --learning_rate $LEARNING_RATE
python trainSAE2.py --dataset_path "$DATASET_PATH" --exp_name "K_256" --input_dim $INPUT_DIM --hidden_dim_multiplier 4 --k 256 --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --learning_rate $LEARNING_RATE


# --tied_weights


# python trainSAE2.py --dataset_path "$DATASET_PATH" --exp_name "K_100" --input_dim $INPUT_DIM --hidden_dim_multiplier $HIDDEN_DIM_MULT --k 100 --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --learning_rate $LEARNING_RATE 

# python trainSAE2.py --dataset_path "$DATASET_PATH" --exp_name "K_128" --input_dim $INPUT_DIM --hidden_dim_multiplier $HIDDEN_DIM_MULT --k 128 --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --learning_rate $LEARNING_RATE 

# python trainSAE2.py --dataset_path "$DATASET_PATH" --exp_name "K_100_tied" --input_dim $INPUT_DIM --hidden_dim_multiplier $HIDDEN_DIM_MULT --k 100 --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --learning_rate $LEARNING_RATE --tied_weights

# python trainSAE2.py --dataset_path "$DATASET_PATH" --exp_name "K_128_tied" --input_dim $INPUT_DIM --hidden_dim_multiplier $HIDDEN_DIM_MULT --k 128 --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS --learning_rate $LEARNING_RATE --tied_weights


# python trainSAE2.py --dataset_path "./dataset" --exp_name "K_100" --input_dim 3072 --hidden_dim_multiplier 4 --k 100 --batch_size 1024 --num_epochs $NUM_EPOCHS --learning_rate $LEARNING_RATE

# # Run experiments for each k value
# for k in "${k_values[@]}"; do
#     echo "Starting experiment with k=${k}"
#     echo "Running experiment with k=${k}" >> "experiments/${BATCH_DIR}/batch_summary.log"

#     python trainSAE2.py \
#         --dataset_path "$DATASET_PATH" \
#         --exp_name "k${k}_experiment" \
#         --input_dim $INPUT_DIM \
#         --hidden_dim_multiplier $HIDDEN_DIM_MULT \
#         --k $k \
#         --batch_size $BATCH_SIZE \
#         --num_epochs $NUM_EPOCHS \
#         --learning_rate $LEARNING_RATE \
#         2>&1 | tee "experiments/${BATCH_DIR}/k${k}_experiment.log"

#     echo "Completed experiment with k=${k}"
#     echo "----------------------------------------" >> "experiments/${BATCH_DIR}/batch_summary.log"
# done
