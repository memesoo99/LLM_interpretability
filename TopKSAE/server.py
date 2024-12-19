from flask import Flask, request, jsonify, render_template

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import glob
import os
import matplotlib.pyplot as plt
import time
import json

from datetime import datetime
from torch.nn import functional as F
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.messages import UserMessage
from datasets import load_dataset
import numpy as np

from SAEModel import SparseAutoencoder
from eval_utils import sentence_heatmap_visualization_with_activations, analyze_feature_responses_grouped_and_visualize, prepare_batch_embeddings, gather_residual_activations, EmbeddingDataset, getDict, most_common_neurons, get_word_list


# Flask app initialization
app = Flask(__name__)

from dotenv import load_dotenv
load_dotenv()

# Retrieve the Hugging Face token
access_token = os.getenv("HF_TOKEN")
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token)

# Load SAE model
if 'sae_model' in globals() and sae_model is not None:
    del sae_model
    torch.cuda.empty_cache()  # Clear the GPU memory

input_dim = 4096  # Input and output dimensions
hidden_dim = input_dim * 100  # Hidden layer dimension
K = 12
state_dict = torch.load("/workspace/LLM_interpretability/TopKSAE/experiments/100x_k_comparison_20241209_032005/mistral_pile_k12_experiment_20241209/models/model_epoch_20.pt")
# state_dict = torch.load("/workspace/LLM_interpretability/TopKSAE/experiments/100x_k_comparison_20241209_005036/mistral_pile_k24_experiment_20241209/models/model_epoch_25.pt")
# state_dict = torch.load("/workspace/LLM_interpretability/TopKSAE/experiments/100x_k_comparison_20241210_043339/mistral_pile_k24_experiment_20241210/models/model_epoch_30.pt")

# Load the weights into the model
sae_model = SparseAutoencoder(input_dim, hidden_dim, k = K, dead_steps_threshold=1000000) # initial lambda is 0
sae_model.load_state_dict(state_dict)
sae_model.to('cuda')

@app.route('/')
def home():
    # Render the HTML homepage
    return render_template('index.html')

@app.route('/test_tokenizer')
def test_tokenizer():
    try:
        test_sentence = "This is a test."
        tokens = tokenizer.encode(test_sentence)
        return jsonify({"tokens": tokens})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        sentences = data.get('sentences', [])
        activation_threshold = data.get('activation_threshold', 0.7)
        num_activations = data.get('num_activations', 0.8)
        display_n = data.get('display_n', 3)

        if not sentences:
            return jsonify({"error": "No sentences provided"}), 400

        # Call your analysis function
        visualizations = analyze_feature_responses_grouped_and_visualize(
            theme_texts=sentences,
            model=model,
            sae_model=sae_model,
            tokenizer=tokenizer,
            gather_residual_activations=gather_residual_activations,
            layer_index=16,
            activation_threshold=activation_threshold,
            num_activations=num_activations,
            display_n=display_n
        )

        # Return HTML results
        html_response = "".join(visualizations)
        return html_response
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
