import logging
import os
from pathlib import Path
from typing import List, Optional

import fire  # type: ignore
import torch
from mistral_model import *

# Define necessary model arguments
args = ModelArgs()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate the transformer directly
print(args)
transformer_block = TransformerBlock(args=args).to(device)
transformer = Transformer(args=args).to(device)

print(transformer.tok_embeddings)

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
