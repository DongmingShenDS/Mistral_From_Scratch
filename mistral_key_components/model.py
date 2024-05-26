import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional
# This Class ignores some MOE components in the original model for testing purposes


@dataclass
class ModelArgs:
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 8  # number of heads for the Q
    n_kv_heads: Optional[int] = None  # number of heads for the K and V (can be different from Q)
    vocab_size: int = 1000
    multiple_of: int = 256  # feedforward dimension must be multiple of this
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    device: str = None
    # for KV Cache
    max_batch_size: int = 32
    max_seq_len: int = 512


def precompute_freqs_pos_frequencis(head_dim: int, seq_len: int, device: str, theta: float = 10000.0) -> torch.Tensor:
    assert head_dim % 2 == 0, "head_dim must be even as proposed in https://arxiv.org/pdf/2104.09864"
    # theta parameter = a sequence according to the paper
    # theta_i = 10000^(-2(i-1)/dim) for i in range(1, dim / 2 + 1)
    # shape (both theta_denominator and theta): (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # build the "m" in the paper (aka the positions)
    # shape: (seq_len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product (for all possible combinations of the two)
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    # Compute the complex number polar form: c = R * exp(m * theta) w/ R = 1
    # (seq_len, head_dim / 2) -> (seq_len, head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str) -> torch.Tensor:
    # Separate the last dimension pairs of 2 values (aka real and imaginary parts of the complex number) => make complex
    # Each pair of 2 consecutive values in head_dim is transformed into a single complex number (thus head_dim / 2
    # (B, seq_len, H, head_dim) => (B, seq_len, H, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape feqs_complex tensor to match the shape of the x_complex tensor (use unsqueeze to add extra dimension of 1)
    # (seq_len, head_dim / 2) => (1, seq_len, 1, head_dim / 2)
    feqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)  # recall .unsqueeze(i) means add extra dimension at i
    # Element-wise multiplication with broadcasting
    # This results in the rotation of the complex number as shown in the Figure 1 of the paper
    # (B, seq_len, H, head_dim / 2) * (1, seq_len, 1, head_dim / 2) => (B, seq_len, H, head_dim / 2)
    x_rotated = x_complex * feqs_complex
    # Convert the complex number back to the real number: the additional 2 in the final dim is for real from imag
    # (B, seq_len, H, head_dim / 2) => (B, seq_len, H * head_dim / 2, 2)
    x_real = torch.view_as_real(x_rotated)
    # Flatten the last two dimensions back into 2nd last dimension, using reshape to have same size with x
    # (B, seq_len, H * head_dim / 2, 2) => (B, seq_len, H * head_dim)
    return x_real.reshape(*x.shape)


class RMSNorm(nn.Module):
    # https://arxiv.org/pdf/1910.07467
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma (g) parameter that is trainable to perform the rescaling on the norm
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # RMSNorm statistics, (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        rms_reciprocal = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)    # rsqrt = 1 / sqrt
        return x * rms_reciprocal

    def forward(self, x: torch.Tensor):
        # This completes the equation 4 in the paper
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        # auto-broadcasting expands (Dim) to (1, 1, Dim) to multiplied to the last dimension of (B, Seq_Len, Dim)
        # recall: Automatic broadcasting in PyTorch occurs when dimensions match or are broadcastable starting from the trailing dimensions (i.e., from right to left)
        return self.weight * self._norm(x.float()).type_as(x)


class SelfAttention(nn.Module):
    # Decoder only with causal attention:
    # only care about current token and its corresponding attention (with support from the KV Cache)
    # Extended support for GQA (grouped query attention)
    def __init__(self, args: ModelArgs):
        super().__init__()
        # set the number of KV heads for GQA (see the paper), default to Q heads (then just MHA)
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads











class TransformerBlock(nn.Module):
    # a single transformer block (different for Llama & Mistral)
    # here for simplicity, we did not include MoE
    def __int__(self, args: ModelArgs):
        super().__int__()
        self.n_head = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads  # this is conventional!!! (concat of heads should give dim)
        # Model Components
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        # RMS Normalization before Attention
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # RMS Normalization before Feed Forward
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        # start_pos: the position of the token in the sequence we are dealing with
        # because dealing with only 1 token at a time
        # (B, seq_Len, dim) + (B, seq_Len, dim) => (B, seq_Len, dim) with residual connection
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex
        )
        # (B, seq_Len, dim) + (B, seq_Len, dim) => (B, seq_Len, dim)
        out = h + self.feed_forward.forward(
            self.ffn_norm(h)
        )
        return out







class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size > 0, "vocab size should be set"
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        # input embedding
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        # each transformer layers in (Nx part) of the model, total self.n_layers blocks
        # NOTE: this is the most important part of the model and is where different LLMs architectures are implemented
        self.layers = nn.ModuleList([TransformerBlock(args=args) for _ in range(args.n_layers)])
        # RMS Normalization (better than LayerNorm) - used in Llama and Mistral
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)    # eps for numerical stability never divided by 0
        # output layer
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        # precomputed frequencies for ROPE positional encoding (https://arxiv.org/pdf/2104.09864)
        self.freqs_complex = precompute_freqs_pos_frequencis(   # to precompute the sin and cos in the paper
            self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device
        )

    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        """
        note that with the KV Cache, only need the latest tokens, no need all tokens: info about previous tokens are saved in the cache
        NOTE: this is only for inference, not training (in training there's no KV cache)
        """
        # (B, Seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "One token at a time at inference time"
        # (B, Seq_len) -> (B, Seq_len, dim)
        h = self.tok_embeddings(tokens)
        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        # Apply precomputed frequencies to the encoding layers for positional encoding
        for layer in self.layers:
            h = layer(h, freqs_complex, start_pos, None)    # these are the Nx TransformerBlock layers
        # Apply RMS Normalization after all layers
        h = self.norm(h)
        # Output layer
        output = self.output(h)
        return output

















def main():
    print("Hello, Llama!")


if __name__ == "__main__":
    main()
