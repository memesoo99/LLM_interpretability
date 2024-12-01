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
from SAEModel import *
from torch.nn import functional as F
import argparse
from dataset_embedding import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split
from sae_utils import *
    
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
    
    initial_k = 256  # Starting K value
    final_k = args.k    # Final K value
    
    model = SparseAutoencoder(input_dim, hidden_dim, k = initial_k, dead_steps_threshold=args.dead_steps_threshold, tied_weights = args.tied_weights) # initial lambda is 0
    model = model.cuda()
    # Adam optimizer beta1=0.9, beta2=0.999 and no weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    
    
    ###########Dataset############
    # Define your dataset
    file_pattern = "residual_data_batch_*.pt"  # Adjust the path if needed
    path = args.dataset_path
    # embedding_dataset = EmbeddingDataset(path, file_pattern)
    embedding_dataset = EmbeddingDataset(path=path, file_pattern=file_pattern)

    # Split dataset: 80% for training, 20% for validation
    train_size = int(0.85 * len(embedding_dataset))
    val_size = len(embedding_dataset) - train_size
    train_dataset, val_dataset = random_split(embedding_dataset, [train_size, val_size])
    print(f"Train size: {train_size}, Validation size: {val_size}")

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    total_steps = len(train_loader) * args.num_epochs
    k_scheduler = KScheduler(initial_k, final_k, total_steps)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=learning_rate * 0.1)
    
    
    num_epochs = args.num_epochs
    total_steps = len(train_loader)*num_epochs

    # Dataset scaling
    # X is data tensor (embeddings?)
    # X = X * (input_dim ** 0.5) / torch.norm(X, dim=1, keepdim=True).mean()  # Scaling dataset


    train_loss_values = []
    val_loss_values = []


    def loss_fn(x, recons, auxk, l2_lambda=1e-5):
        mse_scale = 1.0 #/ 20  # Adjust based on your data
        auxk_coeff = 1.0 / 10.0

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
            
            current_k = k_scheduler.get_k(global_step)
            model.k = current_k
        
            batch = batch.cuda().to(torch.float32)
            recons, auxk, num_dead, latents = model(batch)
            mse_loss, auxk_loss = loss_fn(batch, recons, auxk)
            loss = mse_loss + auxk_loss

            optimizer.zero_grad()

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            lr_scheduler.step()
            avg_train_loss += loss.item()
            done_vals += 1
            if batch_num % 1000 == 0:
                print(f"\t Batch {batch_num} Loss: {avg_train_loss / done_vals:.4f} "
                      f"AuxKLoss: {auxk_loss} Num Dead: {num_dead} "
                      f"Current K: {current_k} Current LR: {current_lr:.6f}")
                
            train_steps += 1

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