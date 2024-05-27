import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional
# This Class ignores some MOE components in the original model for testing purposes (same as Llama)


@dataclass
class ModelArgs:
    dim: int = 128
    hidden_dim: int = 256   # this is default to dim*8/3 in Llama but in general should be free to set
    n_layers: int = 4
    n_heads: int = 8  # number of heads for the Q
    n_kv_heads: Optional[int] = None  # number of heads for the K and V (can be different from Q)
    vocab_size: int = 1000
    multiple_of: int = 256  # feedforward dimension must be multiple of this
    ffn_dim_multiplier: Optional[float] = None  # this is not used in Mistral b.c. hidden_dim is set directly
    norm_eps: float = 1e-5
    device: torch.device = None
    # for KV Cache
    max_batch_size: int = 32
    max_seq_len: int = 512


def precompute_freqs_pos_frequencis(
        head_dim: int, seq_len: int, device: torch.device, theta: float = 10000.0
) -> torch.Tensor:
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


def apply_rotary_embeddings(
        x: torch.Tensor, freqs_complex: torch.Tensor, device: torch.device
) -> torch.Tensor:
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
    x_out = x_real.reshape(*x.shape)
    return x_out.type_as(x).to(device)


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
    # Decoder only with causal attention (only work for inference)
    # only care about current token and its corresponding attention (with support from the KV Cache)
    # Extended support for GQA (grouped query attention)
    def __init__(self, args: ModelArgs):
        super().__init__()
        # set the number of KV heads for GQA (see the paper), default to Q heads (then just MHA)
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # set the number of Q heads, should always be args.n_heads
        self.n_heads_q = args.n_heads
        # get num times the KV should be repeated in GQA
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # dim of each head = dim / n_heads (the part of the embedding that each head will be responsible for)
        self.head_dim = args.dim // args.n_heads

        # q k v o weights in transformer attention
        self.wq = nn.Linear(args.dim, self.n_heads_q * self.head_dim)   # for Q
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim)  # for K
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim)  # for V
        self.wo = nn.Linear(self.n_heads_q * self.head_dim, args.dim)   # for O (here n_heads_q * head_dim == dim)

        # Cache for K and V
        self.cache_k = torch.zeros((args.max_batch_size, self.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, self.max_seq_len, self.n_kv_heads, self.head_dim))

    def repeat_kv(self, kv: torch.Tensor) -> torch.Tensor:  # just copy, but can be optimized...
        # in GQA, each Q group shares the same KV heads, thus just repeat KV heads for the Q in the same group
        # goal shape: (B, prefix_seq_len, n_kv_heads, Head_Dim) => (B, prefix_seq_len, n_heads_q, Head_Dim)
        batch_size, seq_len, n_kv_heads, head_dim = kv.shape
        if self.n_rep == 1:  # Q and KV are 1-to-1 (just a normal MHA)
            return kv
        else:  # GQA
            return (
                # (B, prefix_seq_len, n_kv_heads, 1, Head_Dim)
                kv[:, :, :, None, :]
                # (B, prefix_seq_len, n_kv_heads, n_rep, Head_Dim) just copy n_rep times
                .expand(batch_size, seq_len, n_kv_heads, self.n_rep, head_dim)
                # (B, prefix_seq_len, n_kv_heads * n_rep, Head_Dim) = (B, prefix_seq_len, n_heads_q, Head_Dim)
                .reshape(batch_size, seq_len, n_kv_heads * self.n_rep, head_dim)
            )

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        # recall start_pos = the position of the token in the sequence we are dealing with
        # this is the standard self-attention mechanism computation with slight modifications (Llama / Mistral)
        # goal shape: (B, 1, Dim) => (B, 1, Dim)

        batch_size, seq_len, _ = x.shape    # (B, 1, Dim)
        # compute Q K V from the weights wq wk wv
        # (B, 1, Dim) => (B, 1, n_heads_q * Head_Dim)
        xq = self.wq(x)
        # (B, 1, Dim) => (B, 1, n_kv_heads * Head_Dim)
        xk = self.wk(x)
        # (B, 1, Dim) => (B, 1, n_kv_heads * Head_Dim)
        xv = self.wv(x)

        # reshape Q K V to get individual single heads (Qi, Ki, Vi) from the tensors
        # (B, 1, n_heads_q * Head_Dim) => (B, 1, n_heads_q, Head_Dim)
        xq = xq.reshape(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, n_heads_q * Head_Dim) => (B, 1, n_heads_q, Head_Dim)
        xk = xk.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, n_heads_q * Head_Dim) => (B, 1, n_heads_q, Head_Dim)
        xv = xv.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # apply RoPE on Q and K, both should have the same shape before and after RoPE
        xq = apply_rotary_embeddings(xq, freqs_complex, x.device)    # (B, 1, n_heads_q, Head_Dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, x.device)    # (B, 1, n_kv_heads, Head_Dim)

        # replace the entry in the KV cache's respective position (aka update KV Cache)
        # fill (:B, idx) part of the (max_B, max_seq_len, n_kv_heads, Head_Dim) cache with (B, 1, n_kv_heads, Head_Dim)
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

        # retrieve complete K and V from KV Cache for Attention Computation
        # (B, prefix_seq_len, n_kv_heads, Head_Dim)
        keys = self.cache_k[:batch_size, :start_pos + seq_len, :, :]
        values = self.cache_v[:batch_size, :start_pos + seq_len, :, :]

        # in GQA, each Q group shares the same KV heads, thus just repeat KV heads for the Q in the same group
        # (B, prefix_seq_len, n_kv_heads, Head_Dim) => (B, prefix_seq_len, n_heads_q, Head_Dim)
        keys = self.repeat_kv(keys)
        values = self.repeat_kv(values)

        # reshape: equivalent to X.reshape(B, n_heads_q, 1 or prefix_seq_len, Head_Dim)
        # (B, 1, n_heads_q, Head_Dim) => (B, n_heads_q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, prefix_seq_len, n_heads_q, Head_Dim) => (B, n_heads_q, prefix_seq_len, Head_Dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # attention weight and score computation
        # NOTE about MATMUL: for tensors with more than 2 dimensions, torch.matmul treats the last two dimensions as matrices and performs batch matrix multiplication on the other dimensions. The result is a tensor where each batch element is the result of matrix multiplication on the corresponding batch elements of the input tensors
        # (B, n_heads_q, 1, Head_Dim) @ (B, n_heads_q, Head_Dim, prefix_seq_len) => (B, n_heads_q, 1, prefix_seq_len)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # softmax(QK/sqrt(dk)): (B, n_heads_q, 1, prefix_seq_len) => (B, n_heads_q, 1, prefix_seq_len)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # dim=-1 means softmax along last dimension (sum=1)

        # attention computation with the values
        # (B, n_heads_q, 1, prefix_seq_len) @ (B, n_heads_q, prefix_seq_len, Head_Dim) => (B, n_heads_q, 1, Head_Dim)
        output = torch.matmul(scores, values)
        # (B, n_heads_q, 1, Head_Dim) => (B, 1, n_heads_q, Head_Dim) and make sure contiguous in memory
        output = output.transpose(1, 2).contiguous()
        # (B, 1, n_heads_q, Head_Dim) => (B, 1, n_heads_q * Head_Dim) = (B, 1, dim)
        output = output.view(batch_size, seq_len, -1)   # -1 means infer the last dimension's shape

        # apply the attention's output layer
        # (B, 1, dim) => (B, 1, dim)
        output = self.wo(output)
        return output


class FeedForward(nn.Module):
    # FFN with SwiGLU activation
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = args.hidden_dim
        # recall, this is not used in Mistral b.c. hidden_dim is set directly in args
        if args.ffn_dim_multiplier is not None: hidden_dim = int(hidden_dim * args.ffn_dim_multiplier)
        # round hidden_dim to the nearest multiple of the args.multiple_of parameter (bigger or equal)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        # ffn weights initialize
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (S(XW1) * XV)XW2 = (ss)XW2; goal shape: (B, seq_len, Dim) => (B, seq_len, Dim)
        # (B, seq_len, Dim) w1=> (B, seq_len, Hidden_Dim)
        xw1 = self.w1(x)
        # (B, seq_len, Hidden_Dim) => (B, seq_len, Hidden_Dim)
        sxw1 = F.silu(xw1)
        # (B, seq_len, Dim) w3=> (B, seq_len, Hidden_Dim)
        xv = self.w3(x)
        # (B, seq_len, Hidden_Dim) * (B, seq_len, Hidden_Dim) = (B, seq_len, Hidden_Dim) = element wise multiplication
        sxw1xv = sxw1 * xv
        # (B, seq_len, Hidden_Dim) w2=> (B, seq_len, Dim)
        return self.w2(sxw1xv)


class TransformerBlock(nn.Module):
    # a single transformer block (different for Llama & Mistral)
    # here for simplicity, we did not include MoE
    def __int__(self, args: ModelArgs):
        super().__int__()
        self.n_head = args.n_heads
        self.dim = args.dim
        self.hidden_dim = args.hidden_dim
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
