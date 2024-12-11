from flask import Flask, request, jsonify
import torch  # Assuming you are using PyTorch for GPU-based tasks
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib import cm
from IPython.display import HTML, display
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import matplotlib.pyplot as plt
import time
from datetime import datetime
from SAEModel import SparseAutoencoder
from torch.nn import functional as F
import json
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from collections import Counter


def most_common_neurons(neuron_dict, K):
    # Flatten all indices into a single list
    all_indices = [inde for indices in neuron_dict.values() for index in indices]
    
    # Count the frequency of each index
    index_counts = Counter(all_indices)
    
    # Sort by frequency (descending) and then by index (ascending)
    sorted_indices = sorted(index_counts.items(), key=lambda x: (-x[1], x[0]))
    
    # Return the top K indices
    return [index for index, _ in sorted_indices[:K]]

## Residual Hook Function
def gather_residual_activations(model, target_layer, inputs):
  target_act = None
  def gather_target_act_hook(mod, inputs, outputs):
    nonlocal target_act # make sure we can modify the target_act from the outer scope
    target_act = outputs[0]
    return outputs
  handle = model.model.layers[target_layer].register_forward_hook(gather_target_act_hook)
  _ = model.forward(inputs)
  handle.remove()
  return target_act

def get_word_vector_pairs(tokenizer_output, model_output, tokenizer):
    wordVectorPairs = []
    for i, word in enumerate(tokenizer_output):
        #print(i, word, tokenizer.decode(word), model_output[i, :].shape)
        wordVectorPairs.append({"text":tokenizer.decode(word), "embedding": model_output[i, :]})
    # print(wordVectorPairs)
    return wordVectorPairs

# def sentence_heatmap_visualization_with_activations(batch_text, activations, feature_index, activation_threshold=0, as_html=True):
#     """
#     Visualize activations as a heatmap over text, with optional HTML string return.

#     Args:
#         batch_text (list of str): List of tokens in the sentence.
#         activations (ndarray): Normalized activations for each token.
#         feature_index (int): The feature index being visualized.
#         activation_threshold (float): Minimum activation threshold for highlighting.
#         as_html (bool): If True, return the HTML string instead of displaying it.

#     Returns:
#         str: HTML string if as_html=True.
#     """
#     # Extract normalized activations for the given feature index
#     feature_activations = activations[:, feature_index]

#     # Create HTML with tokens highlighted
#     sentence_html = ""
#     cmap = cm.get_cmap("Reds")  # Color map for activation intensity

#     # Define transparency scaling factor (e.g., 0.7 for 70% transparency)
#     alpha_scaling = 0.7

#     for token, act in zip(batch_text, feature_activations):
#         # Skip tokens with activations below the threshold
#         if act < activation_threshold or np.isnan(act):  # Handle NaN values gracefully
#             color_hex = "rgba(255, 255, 255, 0)"  # Fully transparent for below-threshold activation
#         else:
#             color = cmap(act)
#             color_hex = f"rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {color[3] * alpha_scaling})"
#         token = token.replace("<", "&lt;").replace(">", "&gt;")  # Sanitize tokens for HTML
#         sentence_html += f'<span style="background-color: {color_hex}; padding: 2px 4px; margin: 2px; text-decoration: none;">{token}</span> '

#     # Render HTML visualizations as a single paragraph
#     html_visualization = f"<p style='font-family: monospace; text-decoration: none;'>{sentence_html}</p>"

#     if as_html:
#         return html_visualization  # Return the HTML string for external use
#     else:
#         display(HTML(html_visualization))  # Display in notebook or inline context

def sentence_heatmap_visualization_with_activations(
    batch_text, activations, feature_index, activation_threshold=0, as_html=True, tokenizer=None
):
    """
    Visualize activations as a heatmap over text, with tokens concatenated into words.

    Args:
        batch_text (list of str): List of tokens in the sentence.
        activations (ndarray): Normalized activations for each token.
        feature_index (int): The feature index being visualized.
        activation_threshold (float): Minimum activation threshold for highlighting.
        as_html (bool): If True, return the HTML string instead of displaying it.
        tokenizer: Tokenizer used for splitting tokens. Needed to group tokens into words.

    Returns:
        str: HTML string if as_html=True.
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided to handle token-word mapping.")

    # Extract normalized activations for the given feature index
    feature_activations = activations[:, feature_index]

    # Create HTML with tokens grouped by words
    sentence_html = ""
    cmap = cm.get_cmap("Blues")  # Color map for activation intensity
    alpha_scaling = 1

    for i, (token, act) in enumerate(zip(batch_text, feature_activations)):
        # Skip tokens with activations below the threshold
        if act < activation_threshold or np.isnan(act):  # Handle NaN values gracefully
            color_hex = "rgba(255, 255, 255, 0)"  # Fully transparent for below-threshold activation
            text_color = "#000"  # Default text color (black) for non-highlighted tokens
        else:
            color = cmap(act)
            color_hex = f"rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {color[3] * alpha_scaling})"
            text_color = "#FFF"  # White text for highlighted tokens

        # Add spacing between tokens if the token starts a new word
        if i > 0 and not token.startswith("Ġ"):  # "Ġ" is common in tokenizers to mark a new word
            sentence_html += f'<span style="background-color: {color_hex}; color: {text_color}; padding: 2px 3px; margin: 0px 1px; text-decoration: none;">{token}</span>'
        else:
            # Add a space before the token if it's a new word
            sentence_html += f' <span style="background-color: {color_hex}; color: {text_color}; padding: 2px 3px; margin: 0px 1px; text-decoration: none;">{token}</span>'

    # Render HTML visualizations as a single paragraph
    html_visualization = f"<p style='font-family: monospace; text-decoration: none; margin-bottom: 5px;'>{sentence_html.strip()}</p>"

    if as_html:
        return html_visualization  # Return the HTML string for external use
    else:
        display(HTML(html_visualization))  # Display in notebook or inline context

def analyze_feature_responses_grouped_and_visualize(theme_texts, tokenizer, model, sae_model, gather_residual_activations, layer_index=16, activation_threshold=0.5, num_activations=0.8, display_n=5):
    feature_to_text_map = defaultdict(list)  # Map each feature to sentences that activate it
    dummy_index = len(theme_texts) - 1  # Index of the dummy sentence (last entry in theme_texts)
    dummy_text = theme_texts[dummy_index]
    theme_texts = theme_texts[:-1]

    # Process each text and track feature activations
    for text_id, text in enumerate(theme_texts):
        # Prepare batch embeddings and text
        batch_embeddings, batch_text = prepare_batch_embeddings(text, tokenizer, model, gather_residual_activations, layer_index)
        batch_embeddings = batch_embeddings.to(torch.float32)

        # Forward pass through SAE model
        sae_model.eval()
        with torch.no_grad():
            _, _, _, latents = sae_model(batch_embeddings.cuda())
        latents = latents.cpu().numpy()

        # Min-max normalization per token
        latents_normalized = (latents - latents.min(axis=-1, keepdims=True)) / (
            latents.max(axis=-1, keepdims=True) - latents.min(axis=-1, keepdims=True) + 1e-8
        )

        # Identify features with activations above threshold
        activated_features = (latents_normalized > activation_threshold).nonzero()[1].tolist()

        # Avoid duplicates for the same feature
        unique_features = set(activated_features)  # Deduplicate feature indices

        # Track sentences and activations for each feature
        for feature_index in unique_features:
            if (text, batch_text, latents_normalized) not in feature_to_text_map[feature_index]:
                feature_to_text_map[feature_index].append((text, batch_text, latents_normalized))

    # Process the dummy sentence
    dummy_embeddings, dummy_batch_text = prepare_batch_embeddings(dummy_text, tokenizer, model, gather_residual_activations, layer_index)
    dummy_embeddings = dummy_embeddings.to(torch.float32)

    sae_model.eval()
    with torch.no_grad():
        _, _, _, dummy_latents = sae_model(dummy_embeddings.cuda())
    dummy_latents = dummy_latents.cpu().numpy()

    # Min-max normalization for dummy latents
    dummy_latents_normalized = (dummy_latents - dummy_latents.min(axis=-1, keepdims=True)) / (
        dummy_latents.max(axis=-1, keepdims=True) - dummy_latents.min(axis=-1, keepdims=True) + 1e-8
    )

    # Always add the dummy sentence for visualization
    for feature_index in range(dummy_latents_normalized.shape[1]):
        feature_to_text_map[feature_index].append((dummy_text, dummy_batch_text, dummy_latents_normalized))

    # Filter features activated by more than n texts
    active_features = [feature for feature, texts in feature_to_text_map.items() if len(texts) / len(theme_texts) > num_activations]
    # print(f"Features activated by more than {num_activations} texts: {len(active_features)}, {active_features}")

    run = 0
    html_visualizations = []
    # Visualize for each feature
    if display_n == -1:
        display_n = len(active_features)
    for feature_index in active_features:
        if run >= display_n:
            break

        # print(f"\nVisualizing Feature {feature_index}")
        feature_html = f'<h3 style="padding-top: 10px;">Feature {feature_index}</h3>'
        for text_id, (text, batch_text, activations) in enumerate(feature_to_text_map[feature_index]):
            # Modify visualization to return HTML instead of directly rendering
            visualization_html = sentence_heatmap_visualization_with_activations(
                batch_text, activations, feature_index, activation_threshold=activation_threshold, as_html=True, tokenizer=tokenizer
            )
            feature_html += visualization_html
        html_visualizations.append(feature_html)
        run += 1

    # Return the HTML for all visualizations
    return html_visualizations





def prepare_batch_embeddings(text, tokenizer, model, gather_residual_activations, layer_index=16):

    inputs = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    
    with torch.no_grad():
        residuals = gather_residual_activations(model, layer_index, inputs)
    
    out = get_word_vector_pairs(inputs[0].cpu().tolist(), residuals[0].cpu(), tokenizer)

    batch_text = [item["text"] for item in out]
    batch_embeddings = torch.stack([item["embedding"] for item in out])

    return batch_embeddings, batch_text


class EmbeddingDataset(Dataset):
    def __init__(self, path, file_pattern, use_files = 40):
        # Load all `.pt` files based on the pattern
        self.file_path = path
        self.files = sorted(glob.glob(path+"/"+file_pattern))
        print(f"Num of files found : {len(self.files)}")
        print(f"Num of files used : {len(self.files[:use_files])}")
        self.data = []

        # Read and store all embeddings from all files
        for file in self.files[:use_files]:
            batch_data = torch.load(file)
            # Extract embeddings and flatten them into a list
            self.data.extend([(item["embedding"], item["text"]) for item in batch_data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

        
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
