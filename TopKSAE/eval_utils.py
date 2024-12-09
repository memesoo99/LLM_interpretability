import torch
from tqdm import tqdm
from collections import Counter


def most_common_neurons(neuron_dict, K):
    # Flatten all indices into a single list
    all_indices = [index for indices in neuron_dict.values() for index in indices]
    
    # Count the frequency of each index
    index_counts = Counter(all_indices)
    
    # Sort by frequency (descending) and then by index (ascending)
    sorted_indices = sorted(index_counts.items(), key=lambda x: (-x[1], x[0]))
    
    # Return the top K indices
    return [index for index, _ in sorted_indices[:K]]

def get_word_list(index, word_active_neurons):
    word_list = []
    for key, val in word_active_neurons.items():
        if index in val:
            word_list.append(key)
    return word_list


def getDict(loader, sae_model, K=10):
    word_active_neurons = {}
    with torch.no_grad():
        for batch_num, batch in tqdm(enumerate(loader), total = len(loader), desc="Processing Batches"):
            # Batch to GPU
            batch, text = batch
            batch = batch.cuda().to(torch.float32)

            # Perform forward pass to get latent representations
            recons, auxk, num_dead, latents = sae_model(batch)  # Assuming `model` returns latent as the second output

            # Find top K active neuron indices for each input in the batch
            active_neuron_indices = torch.topk(latents, K, dim=1).indices
            # print(active_neuron_indices.shape) # (Batch_size, K)
            
            # Save results
            for i, indices in enumerate(active_neuron_indices):
                word = text[i]  # Assuming dataset has words or identifiers
                word_active_neurons[word] = indices.cpu().tolist()

            if batch_num % 100 == 0:
                print(f"Processed Batch {batch_num}...")
    return word_active_neurons
