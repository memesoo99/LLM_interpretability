import argparse
import os
import time
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Train Sparse Autoencoder')
    
    # Dataset parameters
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory')
    
    # Model parameters
    parser.add_argument('--input_dim', type=int, default=3072, help='Input dimension')
    parser.add_argument('--hidden_dim_multiplier', type=int, default=4, help='Multiplier for hidden dimension (hidden_dim = input_dim * multiplier)')
    parser.add_argument('--k', type=int, default=48, help='Number of active neurons (suggested: input_dim/64)')
    parser.add_argument('--dead_steps_threshold', type=int, default=1000000, help='Steps threshold for dead neuron detection')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2048, help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=35, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--l2_lambda', type=float, default=1e-5, help='L2 regularization strength')
    
    # Output parameters
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name for output directory')
    parser.add_argument('--tied_weights', action='store_true', help='Tied Decoder')
    
    return parser.parse_args()

def setup_experiment_dir(exp_name):
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    exp_dir = f"experiments/{exp_name}_{timestamp}"
    
    # Create directories
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/models", exist_ok=True)
    os.makedirs(f"{exp_dir}/logs", exist_ok=True)
    os.makedirs(f"{exp_dir}/plots", exist_ok=True)
    
    return exp_dir

def save_training_config(args, exp_dir):
    config_path = f"{exp_dir}/config.txt"
    with open(config_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
