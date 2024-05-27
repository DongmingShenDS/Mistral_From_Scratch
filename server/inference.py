from model import *
from moe import *
from lora import *
from generate import *

device = "cuda" if torch.cuda.is_available() else "cpu"
# Define necessary model arguments
args = ModelArgs(
    n_layers=4,
    head_dim=4,
    hidden_dim=20,
    n_heads=2,
    dim=16,            # embedding dimension for each input token
    n_kv_heads=2,
    norm_eps=1e-6,
    vocab_size=100,     # vocab size (number of possible tokens)
    max_batch_size=2,  # maximum batch size
    rope_theta=10000.0, #rotation angle
    moe=MoeArgs(
        num_experts=4,
        num_experts_per_tok=2,
    ),
    lora=LoraArgs(
        rank=4,
        scaling=2
    )
)
    
# Instantiate the transformer directly
transformer = Transformer(
    args=args,
    pipeline_rank=0,       # Assuming single machine, non-distributed
    num_pipeline_ranks=1   # Not using pipeline parallelism
).to(device)

# print(transformer)

# def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
# total_params = count_parameters(transformer)
# print(f"Total trainable parameters in the model: {total_params}")

## inference without cache
# input = torch.tensor([args.vocab_size - 1] * (args.dim * args.max_batch_size)).to(device)
# seqlens = [args.dim] * args.max_batch_size  # Assuming all sequences are of maximum length for simplicity
# output = transformer(input, seqlens)
# print(output)

## inference with cache

encoded_prompts = [[1, 2, 4], [4, 5, 6, 2, 3]]
generated_sequences = generate(
    encoded_prompts=encoded_prompts,
    model=transformer,
    max_tokens=2,
    temperature=0.8,
    eos_id=None  # Replace with an appropriate eos_id if available
)
print(generated_sequences)
