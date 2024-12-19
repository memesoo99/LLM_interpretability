import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from datasets import load_dataset
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.messages import UserMessage
from datasets import load_dataset

# Dataset create configuration
batch_size = 1
samples_per_file = 1000
max_samples = 20000
model_type = "phi"
model_type= "mistral"

data_full = []
file_count = 1
sample_count = 0

access_token = os.environ.get('ACCESS_TOKEN') 

## 1. Model and Tokenizer Selection
if model_type == "phi":
    model_name = "microsoft/Phi-3-mini-4k-instruct" #3.8
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype="auto", trust_remote_code=True, attn_implementation='eager')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
elif model_type == "mistral":
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token, device_map="cuda")
    tokenizer = MistralTokenizer.v1()

## 2. Dataset to extract residual
# ds = load_dataset("tatsu-lab/alpaca")
ds = load_dataset("monology/pile-uncopyrighted", data_files="train/00.jsonl.zst", split="train")

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

def get_word_vector_pairs(tokenizer_output, model_output):
    wordVectorPairs = []
    for i, word in enumerate(tokenizer_output):
        wordVectorPairs.append({"text":tokenizer.decode(word), "embedding": model_output[i, :]})
    return wordVectorPairs


for i in range(0, len(ds['train']), 1):
    if sample_count >= 25000:
        break
    
    # Select a batch of samples
    text = ds['train'][i]['output']

    # Tokenize the batch of text
    inputs = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to("cuda")

    # Get the activations for the middle layer
    try:
        with torch.no_grad():
            residuals = gather_residual_activations(model, 16, inputs)
    
        out = get_word_vector_pairs(inputs[0].cpu().tolist(), residuals[0].cpu())
        data_full.extend(out)
        sample_count += len(out)
        print(i)
    except:
        continue
    # Save batch to disk every 5,000 samples
    if sample_count % samples_per_file == 0: # and data_batch:
        torch.save(data_full, f"./dataset/residual_data_batch_{file_count}.pt")
        print(f"Done with {file_count} files")
        data_full = []
        file_count += 1