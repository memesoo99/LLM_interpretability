import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import matplotlib.pyplot as plt
import time
from datetime import datetime
from SAEModel import SparseAutoencoder
from torch.nn import functional as F
import argparse



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

class EmbeddingDataset(Dataset):
    def __init__(self, path, file_pattern):
        # Load all `.pt` files based on the pattern
        self.file_path = path
        self.files = sorted(glob.glob(path+"/"+file_pattern))
        print(f"Num of files found : {len(self.files)}")
        self.data = []

        # Read and store all embeddings from all files
        for file in self.files:
            batch_data = torch.load(file)
            # Extract embeddings and flatten them into a list
            self.data.extend([item["embedding"] for item in batch_data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    
def main():
    args = parse_args()
    print(args)
    exp_dir = setup_experiment_dir(args.exp_name)
    save_training_config(args, exp_dir)
    
    # Hyperparameters
    # 1,048,576 (~1M), 4,194,304 (~4M), and 33,554,432 (~34M)
    input_dim = args.input_dim  # Input and output dimensions
    hidden_dim = input_dim * args.hidden_dim_multiplier  # Hidden layer dimension
    learning_rate = 5e-5

    model = SparseAutoencoder(input_dim, hidden_dim, k = args.k, dead_steps_threshold=args.dead_steps_threshold) # initial lambda is 0
    model = model.cuda()
    # Adam optimizer beta1=0.9, beta2=0.999 and no weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))


    ###########Dataset############
    # Define your dataset
    file_pattern = "residual_data_batch_*.pt"  # Adjust the path if needed
    path = args.dataset_path
    embedding_dataset = EmbeddingDataset(path, file_pattern)

    from torch.utils.data import Dataset, DataLoader, random_split

    # Split dataset: 80% for training, 20% for validation
    train_size = int(0.85 * len(embedding_dataset))
    val_size = len(embedding_dataset) - train_size
    train_dataset, val_dataset = random_split(embedding_dataset, [train_size, val_size])
    print(f"Train size: {train_size}, Validation size: {val_size}")

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = args.num_epochs
    total_steps = len(train_loader)*num_epochs

    # Dataset scaling
    # X is data tensor (embeddings?)
    # X = X * (input_dim ** 0.5) / torch.norm(X, dim=1, keepdim=True).mean()  # Scaling dataset


    train_loss_values = []
    val_loss_values = []


    def loss_fn(x, recons, auxk, l2_lambda=1e-5):
        mse_scale = 1.0 #/ 20  # Adjust based on your data
        auxk_coeff = 1.0 / 32.0

        mse_loss = mse_scale * F.mse_loss(recons, x)
        if auxk is not None:
            auxk_loss = auxk_coeff * F.mse_loss(auxk, x - recons).nan_to_num(0)
        else:
            auxk_loss = torch.tensor(0.0)

        l2_loss = l2_lambda * model.w_enc.pow(2).sum()
        return mse_loss + l2_loss, auxk_loss


    train_steps = 0
    # Training loop
    for epoch in range(num_epochs):
        start = time.time()
        print(f"Starting Epoch [{epoch}/{num_epochs}]")
        model.train()
        avg_train_loss = 0
        done_vals = 0


        for batch_num, batch in enumerate(train_loader):

            batch = batch.cuda().to(torch.float32)
            recons, auxk, num_dead, latents = model(batch)
            mse_loss, auxk_loss = loss_fn(batch, recons, auxk)
            loss = mse_loss + auxk_loss

            optimizer.zero_grad()

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            avg_train_loss += loss.item()
            done_vals += 1
            if batch_num % 200 == 0:
                print(f"\t Batch {batch_num} Loss: {avg_train_loss / done_vals:.4f} AuXKLoss: {auxk_loss} Num Dead: {num_dead}")

            train_steps += len(batch)

        avg_train_loss /= done_vals
        train_loss_values.append(avg_train_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Train Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        avg_val_loss = 0
        done_vals = 0
        with torch.no_grad():
            for batch_num, batch in enumerate(val_loader):
                batch = batch.cuda().to(torch.float32)

                recons, auxk, num_dead, latents = model(batch)

                mse_loss, auxk_loss = loss_fn(batch, recons, auxk)

                loss = mse_loss + auxk_loss
                avg_val_loss += loss.item()
                done_vals += 1
            avg_val_loss /= done_vals
            val_loss_values.append(avg_val_loss)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Validation Loss: {avg_val_loss:.4f}")

        if (epoch + 1) % 2 == 0:
            checkpoint_path = f"{exp_dir}/models/model_epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")
        print(f"Done in {time.time() - start}")


    print(train_loss_values)
    print(val_loss_values)
    # Plotting the loss curves
    plt.figure(figsize=(10, 6))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    plt.plot(range(1, num_epochs + 1), train_loss_values, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_loss_values, label='Validation Loss', linestyle='--')
    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f"{exp_dir}/plots/curve_train_exp_{timestamp}.png")
    plt.show()


    # Conceptually a feature’s activation is now f i ∣ ∣ W d , i ∣ ∣ 2 f i ​ ∣∣W d,i ​ ∣∣ 2 ​ instead of f i f i ​ .
    # Normalize W_d and adjust encoder and bias after training
    # with torch.no_grad():
    #     W_d_norm = model.decoder.weight.norm(dim=0, keepdim=True)
    #     model.decoder.weight /= W_d_norm
    #     model.encoder.weight *= W_d_norm
    #     model.encoder.bias /= W_d_norm

if __name__ == "__main__":
    main()