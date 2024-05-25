"""
The cache.py file contains the implementation of the cache functionality for the Mistral model.

The cache is used to store key-value pairs and metadata related to the input sequences.
It is used to improve the efficiency of the model by reducing the number of computations required for repeated sequences.

The file starts with import statements for necessary modules and packages.
It then defines the `CacheInputMetadata` class, which represents the metadata related to the input sequences.
The class has attributes such as `positions`, `cache_positions`, `prefill`, `mask`, and `seqlens`.

The file also defines the `BufferCache` class, which represents the cache.
The class has methods for initializing the cache, updating the cache with new key-value pairs, and retrieving views of the cache.

The `CacheView` class is defined within the `BufferCache` class and represents a view of the cache for a specific layer and metadata.
It has methods for updating the cache with new key-value pairs and interleaving the key-value pairs.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from xformers.ops.fmha.attn_bias import (  # type: ignore
    AttentionBias,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalMask,
)


@dataclass
class CacheInputMetadata:
    """
    Dataclass for storing metadata related to cache input.
    Attributes:
        positions (torch.Tensor): The absolute positions of tokens in the rope.
        cache_positions (torch.Tensor): The positions where tokens should be placed in the cache.
        prefill (bool): If True, use block diagonal causal mask. Otherwise, use causal mask with padded key mask.
        mask (AttentionBias): The attention bias mask.
        seqlens (List[int]): The lengths of the input sequences.
    """
    positions: torch.Tensor         # rope absolute positions
    cache_positions: torch.Tensor   # where tokens should go in the cache
    prefill: bool   # if True, use block diagonal causal mask; else use causal with padded key mask
    mask: AttentionBias
    seqlens: List[int]


def interleave_list(
    l1: List[torch.Tensor], l2: List[torch.Tensor]
) -> List[torch.Tensor]:
    """
    Interleave two lists of tensors.
    Args:
        l1 (List[torch.Tensor]): The first list of tensors.
        l2 (List[torch.Tensor]): The second list of tensors.
    Returns:
        List[torch.Tensor]: The interleaved list of tensors.
    Raises:
        AssertionError: If the lengths of l1 and l2 are not equal.
    """
    assert len(l1) == len(l2), "Lengths of l1 and l2 must be equal"
    # Interleave l1 and l2 using list comprehension and zip
    interleaved_list = [v for pair in zip(l1, l2) for v in pair]
    return interleaved_list


class CacheView:
    """
    A class representing a view into a cache.

    The cache is represented as a tensor of keys and a tensor of values, along with
    metadata about the cache input and the sequence lengths of the keys and values.
    This class provides methods for accessing and manipulating the cache.
    """
    def __init__(
        self,
        cache_k: torch.Tensor,  # Tensor containing the keys of the cache
        cache_v: torch.Tensor,  # Tensor containing the values of the cache
        metadata: CacheInputMetadata,  # Metadata about the cache input
        kv_seqlens: torch.Tensor,  # Tensor containing the sequence lengths of the keys and values
    ):
        """
        Initialize a CacheView object.
        Args:
            cache_k (torch.Tensor): Tensor containing the keys of the cache.
            cache_v (torch.Tensor): Tensor containing the values of the cache.
            metadata (CacheInputMetadata): Metadata about the cache input.
            kv_seqlens (torch.Tensor): Tensor containing the sequence lengths of the keys and values.
        """
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.metadata = metadata
        self.kv_seqlens = kv_seqlens

    def update(self, xk: torch.Tensor, xv: torch.Tensor) -> None:
        """
        Update the cache with new keys and values.
        Args:
            xk (torch.Tensor): The new keys to be added to the cache.
            xv (torch.Tensor): The new values to be added to the cache.
        Updates the cache by replacing the last [max_seq_len] tokens in each sequence with the new keys and values.
        """
        # Get the number of key-value heads and the head dimension
        n_kv_heads, head_dim = self.cache_k.shape[-2:]
        # Flatten the cache tensors
        flat_cache_k = self.cache_k.view(-1, n_kv_heads, head_dim)
        flat_cache_v = self.cache_v.view(-1, n_kv_heads, head_dim)
        # Update the cache with the new keys and values
        flat_cache_k.index_copy_(0, self.metadata.cache_positions, xk)
        flat_cache_v.index_copy_(0, self.metadata.cache_positions, xv)

    def interleave_kv(
        self, xk: torch.Tensor, xv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interleave the keys and values with the cache.
        Args:
            xk (torch.Tensor): The keys to be interleaved.
            xv (torch.Tensor): The values to be interleaved.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The interleaved keys and values.
        This is a naive implementation and not optimized for speed.
        """
        # Check the shape and dimension of the keys and values
        assert xk.ndim == xv.ndim == 3, "The keys and values should have shape (B * T, H, D)"
        assert xk.shape == xv.shape, "The keys and values should have the same shape"
        # If all sequence lengths are zero, there is no cache to interleave
        if all([s == 0 for s in self.metadata.seqlens]):
            return xk, xv   # No cache to interleave
        # Split the keys and values into a list of tensors [(T, H, D)] for each sequence
        xk: Tuple[torch.Tensor] = torch.split(xk, self.metadata.seqlens)  # type: ignore
        xv: Tuple[torch.Tensor] = torch.split(xv, self.metadata.seqlens)  # type: ignore
        assert len(xk) == len(
            self.kv_seqlens
        ), f"Batch size is {len(self.kv_seqlens)}, got {len(xk)}"
        # Retrieve the cache keys and values
        cache_k = [
            cache_k[:seq_len] for cache_k, seq_len in zip(self.cache_k, self.kv_seqlens)
        ]
        cache_v = [
            cache_v[:seq_len] for cache_v, seq_len in zip(self.cache_v, self.kv_seqlens)
        ]
        # Interleave the cache keys and values with the new keys and values
        interleaved_k = interleave_list(cache_k, list(xk))
        interleaved_v = interleave_list(cache_v, list(xv))
        # Concatenate the interleaved keys and values along the batch dimension
        return torch.cat(interleaved_k, dim=0), torch.cat(interleaved_v, dim=0)

    @property
    def max_seq_len(self) -> int:
        """
        Returns the maximum sequence length of the cache.
        """
        # Get the second dimension of the cache_k tensor, which represents the sequence length
        return self.cache_k.shape[1]

    @property
    def key(self) -> torch.Tensor:
        """
        Returns the cache key tensor up to the maximum sequence length of the cache.
        """
        # Get the cache key tensor up to the maximum sequence length of the cache
        return self.cache_k[: len(self.kv_seqlens)]

    @property
    def value(self) -> torch.Tensor:
        """
        Returns the value tensor of the cache up to the maximum sequence length of the cache.
        """
        # Get the value tensor of the cache up to the maximum sequence length of the cache
        return self.cache_v[: len(self.kv_seqlens)]

    @property
    def prefill(self) -> bool:
        """
        Returns a boolean indicating whether the cache is being pre-filled.
        """
        return self.metadata.prefill

    @property
    def mask(self) -> AttentionBias:
        """
        Returns the mask associated with the cache, obtained from the metadata of the cache.
        """
        # Get the mask from the metadata of the cache
        return self.metadata.mask


class BufferCache:
    """
    This is an example that implements a buffer cache, allowing for variable length sequences.
    Allocated cache is rectangular which is wasteful
    (see PagedAttention https://arxiv.org/pdf/2309.06180 for better mechanisms)

    This class represents a buffer cache for variable length sequences.
    The cache stores key-value pairs for each layer, batch element, and sequence position.
    The cache is used to store the key and value tensors for each layer in a transformer model.
    The cache is designed to handle sequences of variable length, where some elements may be padded.
    """

    def __init__(
        self,
        n_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
    ):
        """
        Initialize a BufferCache object.
        Args:
            n_layers (int): The number of layers in the cache.
            max_batch_size (int): The maximum batch size of the cache.
            max_seq_len (int): The maximum sequence length of the cache.
            n_kv_heads (int): The number of key-value heads in the cache.
            head_dim (int): The dimension of each head in the cache.
        """
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        # Allocate memory for the cache
        self.cache_k = torch.empty(
            (n_layers, max_batch_size, max_seq_len, n_kv_heads, head_dim)
        )
        self.cache_v = torch.empty(
            (n_layers, max_batch_size, max_seq_len, n_kv_heads, head_dim)
        )
        # Initialize & hold the valid length for each batch element in the cache
        self.kv_seqlens: Optional[torch.Tensor] = None

    def get_view(self, layer_id: int, metadata: CacheInputMetadata) -> CacheView:
        """
        Get a view of the cache for a specific layer and input metadata.
        Args:
            layer_id (int): The ID of the layer in the cache.
            metadata (CacheInputMetadata): The input metadata for the cache.
        Returns:
            CacheView: A view of the cache for the specified layer and input metadata.
        Raises:
            AssertionError: If the `kv_seqlens` attribute is None.
        """
        assert self.kv_seqlens is not None  # Ensure the `kv_seqlens` attribute is not None
        # Create and return a CacheView object with the cache_k, cache_v, metadata, and kv_seqlens
        return CacheView(
            self.cache_k[layer_id], self.cache_v[layer_id], metadata, self.kv_seqlens
        )

    def reset(self) -> None:
        """
        Reset the cache by setting the `kv_seqlens` attribute to None.
        This method is used to clear the cache and start with a fresh state.
        """
        self.kv_seqlens = None

    def init_kvseqlens(self, batch_size: int) -> None:
        """
        Initialize the `kv_seqlens` attribute with zeros for each batch element.
        Args:
            batch_size (int): The number of batch elements.
        """
        # The tensor is stored on the same device as the cache and has long integer dtype
        self.kv_seqlens = torch.zeros(
            (batch_size,), device=self.device, dtype=torch.long
        )

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the cache tensors are stored.
        Returns:
            torch.device: The device on which the cache tensors are stored.
        """
        # The device is the same for both cache_k and cache_v tensors
        return self.cache_k.device

    def to(self, device: torch.device, dtype: torch.dtype) -> "BufferCache":
        """
        Move the cache tensors to the specified device and data type.
        Args:
            device (torch.device): The device to move the tensors to.
            dtype (torch.dtype): The data type to convert the tensors to.
        Returns:
            BufferCache: The updated BufferCache object with the moved tensors.
        """
        # Move the cache_k and cache_v tensors to the specified device and data type
        self.cache_k = self.cache_k.to(device=device, dtype=dtype)
        self.cache_v = self.cache_v.to(device=device, dtype=dtype)
        # Return the updated BufferCache object
        return self

    def update_seqlens(self, seqlens: List[int]) -> None:
        """
        Update the valid length for each batch element in the cache.
        Args:
            seqlens (List[int]): A list of the valid lengths for each batch element.
        """
        assert self.kv_seqlens is not None  # Ensure kv_seqlens attribute is not None
        # Convert the list of valid lengths to a tensor and add it to the existing kv_seqlens tensor
        self.kv_seqlens += torch.tensor(seqlens, device=self.device, dtype=torch.long)

    def get_input_metadata(self, seqlens: List[int]) -> CacheInputMetadata:
        """
        Get metadata about cache positions?
        Args:
            seqlens (List[int]): A list of the valid lengths for each batch element.
        Returns:
            CacheInputMetadata: An object containing metadata about cache positions.
        """
        # Initialize kv_seqlens if it is None
        if self.kv_seqlens is None:
            self.init_kvseqlens(len(seqlens))
        # Check kv_seqlens is an instance of torch.Tensor andif the length of seqlens matches the length of kv_seqlens
        assert isinstance(self.kv_seqlens, torch.Tensor)
        assert len(seqlens) == len(
            self.kv_seqlens
        ), f"Batch size is {len(self.kv_seqlens)}, got {len(seqlens)}, did you forget to reset cache?"
        # Convert kv_seqlens to a list
        seqpos = self.kv_seqlens.tolist()
        assert len(seqlens) > 0, seqlens    # Check if the length of seqlens is greater than 0
        # Create a tensor of cached elements
        cached_elements = torch.tensor(seqlens, device=self.device, dtype=torch.long)
        # Create a tensor of positions using seqlens and seqpos
        positions = torch.cat(
            [torch.arange(pos, pos + seqlen) for pos, seqlen in zip(seqpos, seqlens)]
        ).to(device=self.device, dtype=torch.long)
        # Create a tensor of batch indices
        batch_idx = torch.tensor(
            sum([[i] * seqlen for i, seqlen in enumerate(seqlens)], []),
            device=self.device,
            dtype=torch.long,
        )
        # Create a tensor of cache positions
        cache_positions = positions + batch_idx * self.max_seq_len
        # Check if the first element of seqlens is 0 and setup first_prefill
        first_prefill = seqpos[0] == 0
        # Check if any element of seqlens is greater than 1 and setup subsequent_prefill
        subsequent_prefill = any(seqlen > 1 for seqlen in seqlens)
        # Create a mask based on the values of first_prefill and subsequent_prefill
        if first_prefill:
            assert all([pos == 0 for pos in seqpos]), seqpos    # Check if all elements in seqpos are 0
            # Create a BlockDiagonalCausalMask with seqlens and make local attention
            mask = BlockDiagonalCausalMask.from_seqlens(seqlens).make_local_attention(
                self.max_seq_len
            )
        elif subsequent_prefill:
            # Create a BlockDiagonalMask with seqlens and kv_seqlen
            mask = BlockDiagonalMask.from_seqlens(
                q_seqlen=seqlens,
                kv_seqlen=[
                    s + cached_s.clamp(max=self.max_seq_len).item()
                    for (s, cached_s) in zip(seqlens, self.kv_seqlens)
                ],
            ).make_local_attention_from_bottomright(self.max_seq_len)
        else:
            # Create a BlockDiagonalCausalWithOffsetPaddedKeysMask with seqlens, kv_padding, and kv_seqlen
            mask = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=seqlens,
                kv_padding=self.max_seq_len,
                kv_seqlen=(self.kv_seqlens + cached_elements)
                .clamp(max=self.max_seq_len)
                .tolist(),
            )
        # Return an object containing metadata about cache positions
        return CacheInputMetadata(
            positions=positions,
            cache_positions=cache_positions,
            prefill=first_prefill or subsequent_prefill,
            mask=mask,
            seqlens=seqlens,
        )
