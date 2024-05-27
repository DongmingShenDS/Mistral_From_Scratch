import logging
import os
from pathlib import Path
from typing import List, Optional

import fire  # type: ignore
import torch
from mistral_model import *
from mistral_inference import *

# Define necessary model arguments
args = ModelArgs(
    dim=128,
    n_layers=1,
    hidden_dim=256,
    head_dim=16,
    n_heads=8,
    n_kv_heads=2,
    vocab_size=1000,
    norm_eps=1e-5,
    max_batch_size=8,
    max_seq_len=64,
    attn_window=4,
    rope_theta=10000.0,
    moe=MoeArgs(n_experts=4, n_experts_per_tok=2),
    debug=False
)

device = "cuda" if torch.cuda.is_available() else "cpu"
mistral = Mistral(args)
print(mistral.model)

import torch
from torchviz import make_dot

# Define your model
model = Transformer(args)  # Replace your_model with your actual model
x = torch.ones(3, 1, dtype=torch.long)  # Adjust the size according to your input
y = model(x, 0)

# Create a dictionary to rename or remove nodes
node_attr = dict(style='filled', shape='box', align='left', fontsize='12', ranksep='0.1', height='0.2')
params = {name: param for name, param in model.named_parameters() if 'bias' not in name}

# Make the dot graph
dot = make_dot(y, params=params)

# Optional: Simplify names in the graph for readability
simple_names = {}
for node in dot.body:
    if 'Conv' in node or 'BatchNorm' in node or 'ReLU' in node:
        name = node.split('[')[0]
        simple_names[name] = name.replace('\"', '').split('/')[-1]

# Update node names
for i in range(len(dot.body)):
    for key, value in simple_names.items():
        if key in dot.body[i]:
            dot.body[i] = dot.body[i].replace(key, value)

# Render the pruned graph to a file
dot.render('model_visualization', format='png')
exit(0)
















encoded_prompts = [[10, 2, 4, 4, 3, 7, 8], [4, 5, 6, 2, 3, 8, 9], [4, 5, 6, 2, 3, 4, 5, 6, 2, 3]]

tokens, text = mistral.generate(
    prompts=encoded_prompts,
    temperature=0.6,
    top_p=0.9,
    max_gen_len=10
)

print(tokens)


exit(0)














# Instantiate the transformer directly
print(args)
transformer_block = TransformerBlock(args=args).to(device)
transformer = Transformer(args=args).to(device)
tok = transformer.tok_embeddings
# batch size = 3
encoded_prompts = [[100, 2, 4, 4, 3, 7, 8], [4, 5, 6, 2, 3, 8, 9], [4, 5, 6, 2, 3, 4, 5, 6, 2, 3]]
print(encoded_prompts)
# first token from each batch: (batch_size, 1)
prompt_chunks = [p[0:0 + 1] for p in encoded_prompts]
print(prompt_chunks)
# input tensor shape: (batch_size)
input_tensor = torch.tensor(sum(prompt_chunks, []), device=transformer.device, dtype=torch.long)
print(input_tensor.shape)
# input tensor shape: (batch_size * dim)
print(tok(input_tensor).shape)



exit(0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


total_params = count_parameters(transformer)
print(f"Total trainable parameters in the model: {total_params}")

## inference without cache
# input = torch.tensor([args.vocab_size - 1] * (args.dim * args.max_batch_size)).to(device)
# seqlens = [args.dim] * args.max_batch_size  # Assuming all sequences are of maximum length for simplicity
# output = transformer(input, seqlens)
# print(output)

## inference with cache

encoded_prompts = [[1, 2, 4, 4, 3, 7, 8], [4, 5, 6, 2, 3, 8, 9], [4, 5, 6, 2, 3, 4, 5, 6, 2, 3]]
generated_sequences = generate(
    encoded_prompts=encoded_prompts,
    model=transformer,
    max_tokens=5,
    temperature=0.8,
    eos_id=None  # Replace with an appropriate eos_id if available
)
print(generated_sequences)
