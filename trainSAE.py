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
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lambda_reg):
        super(SparseAutoencoder, self).__init__()
        self.relu = nn.ReLU()
        # Encoder and Decoder weights with initializations as described
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

        self.encoder.bias.data.fill_(0)
        self.decoder.bias.data.fill_(0)

        # Initialize encoder and decoder weights
        with torch.no_grad():
            for i in range(hidden_dim):
                # Create a random vector for each column
                random_vector = torch.randn(input_dim)
                # Set L2 norm to a random value between 0.05 and 1
                norm = torch.FloatTensor(1).uniform_(0.05, 1.0)
                # Normalize the vector and scale it by the chosen norm
                self.decoder.weight[:, i] = (random_vector / random_vector.norm()) * norm

        # Initialize W_e as W_d^T
        self.encoder.weight.data = self.decoder.weight.data.T.clone()

        self.lambda_reg = lambda_reg

    def forward(self, x):
        # Forward pass
        hidden = self.relu(self.encoder(x)) #f(x) Shape :
        reconstructed = self.decoder(hidden) #
        return reconstructed, hidden

    def compute_loss(self, x, reconstructed, hidden):
        # Reconstruction loss
        reconstruction_loss = torch.mean((x - reconstructed) ** 2)

        # Sparsity penalty
        sparsity_loss = self.lambda_reg * torch.sum(
            torch.abs(hidden) @ torch.norm(self.decoder.weight, dim=0)
        )
        
        zero_count = (hidden == 0).sum(dim=1).float()  # Count zeros per batch element
        avg_zero_count = zero_count.mean().item()  # Average across the batch
        max_zero_count = zero_count.max().item()  # Average across the batch

        return reconstruction_loss + sparsity_loss, avg_zero_count, max_zero_count




class EmbeddingDataset(Dataset):
    def __init__(self, path, file_pattern):
        # Load all `.pt` files based on the pattern
        self.file_path = path
        self.files = sorted(glob.glob(path+file_pattern))
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

# Hyperparameters
# 1,048,576 (~1M), 4,194,304 (~4M), and 33,554,432 (~34M)
input_dim = 3072  # Input and output dimensions
hidden_dim = 3072*10  # Hidden layer dimension
# hidden_dims = [128, 512, 4096]
final_lambda = 5  # Final regularization strength after 5% of training steps
learning_rate = 5e-5

# Model, optimizer
model = SparseAutoencoder(input_dim, hidden_dim, 0) # initial lambda is 0
model = model.cuda()
# Adam optimizer beta1=0.9, beta2=0.999 and no weight decay
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))


###########Dataset############
# Define your dataset
file_pattern = "*residual_data_batch_*.pt"  # Adjust the path if needed
path = "./dataset/"
embedding_dataset = EmbeddingDataset(path, file_pattern)

from torch.utils.data import Dataset, DataLoader, random_split

# Split dataset: 80% for training, 20% for validation
train_size = int(0.85 * len(embedding_dataset))
val_size = len(embedding_dataset) - train_size
train_dataset, val_dataset = random_split(embedding_dataset, [train_size, val_size])
print(f"Train size: {train_size}, Validation size: {val_size}")

batch_size = 2048
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)





num_epochs = 35 #200000  # as per scaling laws
total_steps = len(train_dataset)*num_epochs


lambda_increase_steps = int(total_steps * 0.05) # Increase the lambda linearly over the first 5% of training steps


# Dataset scaling
# X is data tensor (embeddings?)
# X = X * (input_dim ** 0.5) / torch.norm(X, dim=1, keepdim=True).mean()  # Scaling dataset


train_loss_values = []
val_loss_values = []

train_steps = 0
# Training loop
for epoch in range(num_epochs):
    start = time.time()
    print(f"Starting Epoch [{epoch}/{num_epochs}]")
    model.train()
    avg_train_loss = 0
    done_vals = 0
    
    # Linearly increase λ over the first 5% of steps
    
    if train_steps < lambda_increase_steps:
        model.lambda_reg = final_lambda * (train_steps / lambda_increase_steps)
    else:
        model.lambda_reg = final_lambda
    
    # Decay learning rate linearly over the last 20% of training
    if epoch > num_epochs * 0.8:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 1 - (epoch - num_epochs * 0.8) / (num_epochs * 0.2)

    # Change the Batch sampling 
    # indices = torch.randperm(len(X))[:batch_size]
    # batch = X[indices] 
    # Training loop
    for batch_num, batch in enumerate(train_loader):
        batch = batch.cuda().to(torch.float32)
        reconstructed, hidden = model(batch)
        loss, avg_zeros, max_zeros = model.compute_loss(batch, reconstructed, hidden)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        avg_train_loss += loss.item()
        done_vals += 1
        if batch_num % 200 == 0:
            print(f"\t Batch {batch_num} Loss: {avg_train_loss / done_vals:.4f} Lambda: {model.lambda_reg} Average Zeros: {avg_zeros} Max Zeros: {max_zeros}")
        
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
            reconstructed, hidden = model(batch)
            loss = model.compute_loss(batch, reconstructed, hidden)
            avg_val_loss += loss.item()
            done_vals += 1
        avg_val_loss /= done_vals
        val_loss_values.append(avg_val_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Validation Loss: {avg_val_loss:.4f}")

    if (epoch + 1) % 2 == 0:
        checkpoint_path = f"./models/model_epoch_{epoch + 1}.pt"
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
plt.savefig(f"./curve_train_exp_{timestamp}.png")
plt.show()


# Conceptually a feature’s activation is now f i ∣ ∣ W d , i ∣ ∣ 2 f i ​ ∣∣W d,i ​ ∣∣ 2 ​ instead of f i f i ​ .
# Normalize W_d and adjust encoder and bias after training
with torch.no_grad():
    W_d_norm = model.decoder.weight.norm(dim=0, keepdim=True)
    model.decoder.weight /= W_d_norm
    model.encoder.weight *= W_d_norm
    model.encoder.bias /= W_d_norm
