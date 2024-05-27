import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional
# This file migrated from llama_model for testing only currently
# TODO: add MoE support in Mixtral
# TODO: add LoRA support in Mistral
# TODO: add RollingBuffer support in Mistral


@dataclass
class ModelArgs:
    dim: int = 128      # embedding dimension for each input token and general dim in model layers
    n_layers: int = 4   # number of transformer layers in the model
    hidden_dim: int = 256   # hidden dimension used in ffn
    head_dim: int = 32  # head dimension used in attention (conventionally set to hidden_dim / n_heads)
    n_heads: int = 8  # number of heads for the Q
    n_kv_heads: Optional[int] = None  # number of heads for the K and V (can be different from Q)
    vocab_size: int = 1000  # vocab size (number of possible tokens) usually from tokenizer.vocab_size
    norm_eps: float = 1e-5   # for numerical stability
    max_batch_size: int = 8     # maximum batch size
    max_seq_len: int = 64   # maximum sequence length (not directly used in Mistral)
    attn_window: Optional[int] = None  # attention window and rolling buffer size, if None, it is set to max_seq_len
    rope_theta: float = 10000.0  # theta for rotary embeddings


def precompute_theta_pos_frequencies(
        head_dim: int, seq_len: int, device: torch.device, theta: float
) -> torch.Tensor:
    # note: here seq_len is actual max_seq_len * 2
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
    # (B, seq_len, H=n_heads, head_dim) => (B, seq_len, H, head_dim / 2)
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


class RollingBufferKVCache:
    def __init__(self, max_batch_size, attn_window, n_kv_heads, head_dim):
        # implemented based on idea from original Mistral paper https://arxiv.org/abs/2310.06825
        self.max_batch_size = max_batch_size
        self.attn_window = attn_window
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        # initialize the KV cache with zeros with shape (B, attn_window, n_kv_heads, Head_Dim)
        self.cache_k = torch.zeros((self.max_batch_size, self.attn_window, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((self.max_batch_size, self.attn_window, self.n_kv_heads, self.head_dim))

    def update_cache(self, xk, xv, batch_size, start_pos):
        # get the position of rolling window cache using the modulo operation
        # ensures that the position wraps around within the attn_window size
        cache_position = start_pos % self.attn_window
        # update the entry in the KV cache's respective calculated position with the new KV values
        # fill (:B, idx) part of the (max_B, max_seq_len, n_kv_heads, Head_Dim) cache with (B, 1, n_kv_heads, Head_Dim)
        # shape of xk and xv: (batch_size, 1, n_kv_heads, head_dim)
        self.cache_k[:batch_size, cache_position:cache_position + 1] = xk
        self.cache_v[:batch_size, cache_position:cache_position + 1] = xv

    def update_cache_multiple(self, xk, xv, batch_size, start_pos, seq_len):
        # used when seq_len > 1, yet in inference we only care about the seq_len = 1 case
        # can be optimized in the future to support Mistral's pre-fill and chunking (to handle prompts)
        for i in range(seq_len):
            self.update_cache(xk[:, i:i+1, :, :], xv[:, i:i+1, :, :], batch_size, start_pos + i)

    def retrieve_cache(self, batch_size, start_pos):
        # calculate the effective start position considering the rolling buffer's nature
        # NOTE: start_pos should be updated to be start_pos + seq_len when called after update_cache
        effective_start_pos = start_pos % self.attn_window
        # retrieve KV from the cache, split into 2 parts to handle the wrap-around
        keys = torch.cat([
            self.cache_k[:batch_size, effective_start_pos:, :, :],
            self.cache_k[:batch_size, :effective_start_pos, :, :]
        ], dim=1)
        values = torch.cat([
            self.cache_v[:batch_size, effective_start_pos:, :, :],
            self.cache_v[:batch_size, :effective_start_pos, :, :]
        ], dim=1)
        # select the last seq_len tokens from the concatenated keys and values (to handle when < attn_window)
        keys = keys[:, -start_pos:, :, :]
        values = values[:, -start_pos:, :, :]
        return keys, values


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
        self.head_dim = args.head_dim
        # cache size or attention window size, if not specified, default to full attention
        self.attn_window = args.attn_window if args.attn_window is not None else args.max_seq_len

        # q k v o weights in transformer attention
        self.wq = nn.Linear(args.dim, self.n_heads_q * self.head_dim)   # for Q
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim)  # for K
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim)  # for V
        self.wo = nn.Linear(self.n_heads_q * self.head_dim, args.dim)   # for O (here n_heads_q * head_dim == dim)

        # KV Cache with support of Sliding Window Attention & Rolling Buffer Cache
        # this is modified from the Llama implementation which does not support rolling buffer
        self.kv_cache = RollingBufferKVCache(args.max_batch_size, self.attn_window, self.n_kv_heads, self.head_dim)

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
        assert seq_len == 1, "only support 1D input for now for debugging"  # TODO test support when seq_len > 1
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
        self.kv_cache.update_cache(xk, xv, batch_size, start_pos)

        # retrieve complete K and V from KV Cache for Attention Computation
        # (B, prefix_seq_len, n_kv_heads, Head_Dim)
        keys, values = self.kv_cache.retrieve_cache(batch_size, start_pos + seq_len)

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
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_head = args.n_heads
        self.dim = args.dim
        self.hidden_dim = args.hidden_dim
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
        # input tokens embedding (note: vocab_size is handled internally by nn.Embedding)
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        # each transformer layers in (Nx part) of the model, total self.n_layers blocks
        # NOTE: this is the most important part of the model and is where different LLMs architectures are implemented
        self.layers = nn.ModuleList([TransformerBlock(args=args) for _ in range(args.n_layers)])
        # RMS Normalization (better than LayerNorm) - used in Llama and Mistral
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)    # eps for numerical stability never divided by 0
        # output layer
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        # precomputed frequencies for ROPE positional encoding (https://arxiv.org/pdf/2104.09864)
        self.freqs_complex = precompute_theta_pos_frequencies(   # to precompute the sin and cos in the paper
            self.args.head_dim, self.args.max_seq_len * 2,
            device=self.device, theta=args.rope_theta
        )

    @property
    def dtype(self) -> torch.dtype:
        # Returns the data type of the parameters of the model
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        # Returns the device on which the model parameters are stored
        return next(self.parameters()).device

    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        """
        note that with the KV Cache, only need the latest tokens, no need all tokens: info about previous tokens are saved in the cache
        NOTE: this is only for inference, not training (in training there's no KV cache)
        """
        # (B, Seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "One token at a time at inference time"
        # (B, Seq_len) -> (B, Seq_len, dim)
        # the input should be tokens within the vocabulary range, and the output will be the embedding
        h = self.tok_embeddings(tokens)
        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        # Apply precomputed frequencies to the encoding layers for positional encoding
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)    # these are the Nx TransformerBlock layers
        # Apply RMS Normalization after all layers
        h = self.norm(h)
        # Output layer
        output = self.output(h)
        return output


def main():
    print("Hello, Llama!")


if __name__ == "__main__":
    main()
