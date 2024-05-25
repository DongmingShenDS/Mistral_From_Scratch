"""
The rope.py file contains the implementation of the ROPE (Rope-based Positional Encoding) module.

ROPE is a positional encoding technique used in Transformer models to provide contextual information about the position of tokens in a sequence.
It consists of a series of sinusoidal functions that are added to the input embeddings to encode the positional information.

The file starts with import statements for necessary modules and packages.
It then defines the `Rope` class, which represents the ROPE module.
The class contains several methods and attributes, including initialization, forward pass computation, and utility functions for generating positional encodings.

The file also includes the `precompute_freqs_cis` function, which precomputes the frequencies used in the ROPE encoding.
"""

from typing import Tuple

import torch


def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    """
    Precomputes frequencies and complex numbers for the Rotary Embedding.
    Args:
        dim (int): Dimension of the embeddings.
        end (int): End of the range of frequencies.
        theta (float): Base for the frequency exponentiation.
    Returns:
        torch.Tensor: The precomputed frequencies and complex numbers.
    """
    # Compute the frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # Create a range of indices from 0 to end
    t = torch.arange(end, device=freqs.device)
    # Compute the outer product of t and freqs
    freqs = torch.outer(t, freqs).float()
    # Compute the polar form of complex numbers
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    xq: torch.Tensor,   # Query tensor
    xk: torch.Tensor,   # Key tensor
    freqs_cis: torch.Tensor,    # Frequencies and complex numbers tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings (https://arxiv.org/abs/2104.09864) to query and key tensors.
    Args:
        xq (torch.Tensor): Query tensor.
        xk (torch.Tensor): Key tensor.
        freqs_cis (torch.Tensor): Frequencies and complex numbers tensor.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of query and key tensors with rotary embeddings applied.
    """
    # Reshape xq, xk tensor to complex number format
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # Reshape freqs_cis tensor to broadcastable format: (batch_size, freqs_dim) => (batch_size, 1, freqs_dim)
    freqs_cis = freqs_cis[:, None, :]
    # Multiply xq_, xk_ tensors with freqs_cis tensor and flatten 3 dimensions
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    # Return query and key tensors with rotary embeddings applied back to complex number format
    return xq_out.type_as(xq), xk_out.type_as(xk)
