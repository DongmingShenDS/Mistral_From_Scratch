import json
import logging
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple, Union

import safetensors.torch
import torch
from simple_parsing.helpers import Serializable
from torch import nn
# from xformers.ops.fmha import memory_efficient_attention  # type: ignore => not working on MAC

from mistral_inference.helpers import simple_scaled_dot_product_attention
from mistral_inference.cache import (
    BufferCache,
    CacheInputMetadata,
    CacheView,
)
from mistral_inference.lora import LoraArgs, LoRALinear, LoRALoaderMixin
from mistral_inference.moe import MoeArgs, MoeLayer
from mistral_inference.rope import apply_rotary_emb, precompute_freqs_cis


@dataclass
class ModelArgs(Serializable):
    """
    Arguments for the main model.
    Attributes:
        dim (int): The dimensionality of the model.
        n_layers (int): The number of layers in the model.
        head_dim (int): The dimensionality of the attention heads.
        hidden_dim (int): The dimensionality of the hidden layers.
        n_heads (int): The number of attention heads.
        n_kv_heads (int): The number of key-value attention heads.
        norm_eps (float): The epsilon value for the layer normalization.
        vocab_size (int): The size of the vocabulary.
        max_batch_size (int, optional): The maximum batch size for the model. Defaults to 0.
        rope_theta (float, optional): The theta value for rotary embeddings. If not set, it will be inferred.
        moe (MoeArgs, optional): The arguments for using MoE layers instead of dense layers.
        lora (LoraArgs, optional): The arguments for loading LoRA linear layers instead of linear layers.
    """
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    max_batch_size: int = 0
    # For rotary embeddings. If not set, will be inferred.
    rope_theta: Optional[float] = None
    # If this is set, will use MoE layers instead of dense layers.
    moe: Optional[MoeArgs] = None
    # If this is set, will load LoRA linear layers instead of linear layers. (for PEFT?)
    lora: Optional[LoraArgs] = None


@dataclass
class SimpleInputMetadata:
    """
    Class to store metadata for simple input.
    Attributes:
        positions (torch.Tensor): Tensor of positions for each input sequence.
    """
    # rope absolute positions
    positions: torch.Tensor

    @staticmethod
    def from_seqlens(seqlens: List[int], device: torch.device) -> "SimpleInputMetadata":
        """
        Create SimpleInputMetadata object from a list of sequence lengths and a device.
        Args:
            seqlens (List[int]): List of sequence lengths.
            device (torch.device): Device to place the tensor on.
        Returns:
            SimpleInputMetadata: Object with positions tensor.
        """
        # Create a tensor of positions for each sequence length
        positions = torch.cat([torch.arange(0, seqlen) for seqlen in seqlens])
        # Move the tensor to the specified device and convert it to long dtype
        positions = positions.to(device=device, dtype=torch.long)
        # Create and return the SimpleInputMetadata object
        return SimpleInputMetadata(positions=positions)


def repeat_kv(
    keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Repeat the keys and values tensors along a specified dimension.
    Args:
        keys (torch.Tensor): Tensor of keys to be repeated.
        values (torch.Tensor): Tensor of values to be repeated.
        repeats (int): Number of times to repeat each key-value pair.
        dim (int): Dimension along which to repeat.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Repeated keys and values tensors.
    """
    # Repeat the keys tensor along the specified dimension
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    # Repeat the values tensor along the specified dimension
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    # Return the repeated keys and values tensors
    return keys, values


def maybe_lora(args: ModelArgs) -> Union[nn.Linear, LoRALinear]:
    """
    Returns an instance of `nn.Linear` if `args.lora` is None, otherwise returns a partial function
    that creates an instance of `LoRALinear` with the specified `rank` and `scaling` parameters.
    Args:
        args (ModelArgs): The arguments containing the `lora` parameter.
    Returns:
        Union[nn.Linear, LoRALinear]: An instance of `nn.Linear` or a partial function that creates a `LoRALinear`.
    """
    # If `args.lora` is None, return an instance of `nn.Linear`
    if args.lora is None:
        return nn.Linear
    # Otherwise, return a partial function that creates an instance of `LoRALinear` with the specified `rank` and `scaling` parameters
    else:
        return partial(LoRALinear, rank=args.lora.rank, scaling=args.lora.scaling)


class Attention(nn.Module):
    """
    Implements a multi-head attention mechanism with optional caching and rotary embeddings for positional encoding.
    The class is designed to allow different numbers of heads for queries and key-value pairs, which can be useful for
    handling complex interactions within sequences. It includes scaling and optionally utilizes LoRA for weights.
    Attributes:
        args (ModelArgs): Configuration parameters for the model.
        n_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        n_kv_heads (int): Number of key/value heads.
        repeats (int): Number of times key/value pairs are repeated to match the number of query heads.
        scale (float): Scaling factor for the query vectors.
        wq, wk, wv, wo (nn.Module): Weight matrices for queries, keys, values, and outputs, potentially using LoRA.
    Methods:
        forward(x, freqs_cis, cache): Defines the computation performed at every call.
    """
    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.
        Args:
            args (ModelArgs): The arguments containing the model parameters.
        """
        super().__init__()
        self.args = args
        # Extract the number of heads, head dimension, and number of key-value heads from the arguments
        self.n_heads: int = args.n_heads
        self.head_dim: int = args.head_dim
        self.n_kv_heads: int = args.n_kv_heads
        # Calculate the number of repeats for the key-value heads
        self.repeats = self.n_heads // self.n_kv_heads
        # Calculate the scaling factor for the attention scores
        self.scale = self.args.head_dim**-0.5
        # Determine the type of linear layer based on the `args.lora` parameter
        MaybeLora = maybe_lora(args)
        # Initialize the linear layers for query, key, and value weights
        self.wq = MaybeLora(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = MaybeLora(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = MaybeLora(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        # Initialize the linear layer for the output weights
        self.wo = MaybeLora(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[CacheView],
    ) -> torch.Tensor:
        """
        Forward pass of the Attention module.
        Args:
            x (torch.Tensor): Input tensor of shape (B, S, D), where B is the batch size,
                S is the sequence length, and D is the input dimension.
            freqs_cis (torch.Tensor): Frequency-phase tensor for rotary embedding.
            cache (Optional[CacheView]): Cache view for efficient computation.
        Returns:
            torch.Tensor: Output tensor of shape (B, S, H * D), where B is the batch size,
                S is the sequence length, H is the number of attention heads, and D is the head dimension.
        """
        # Apply linear transformations and reshape for multi-head attention
        # x shape = (B, dim) = (B, n_heads * head_dim)
        print("Attention input shape", x.shape)
        # xq, xk, xv shape = (B, n_heads * head_dim), (B, n_kv_heads * head_dim), (B, n_kv_heads * head_dim)
        seqlen_sum, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        # xq, xk, xv shape = (B, n_heads, head_dim), (B, n_kv_heads, head_dim), (B, n_kv_heads, head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)  # Apply rotary embeddings

        # Handle caching for keys and values
        if cache is None:
            key, val = xk, xv
        elif cache.prefill:
            key, val = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
        else:
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            key = key.view(seqlen_sum * cache.max_seq_len, self.n_kv_heads, self.head_dim)
            val = val.view(seqlen_sum * cache.max_seq_len, self.n_kv_heads, self.head_dim)

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)
        # xformers memory_efficient_attention requires (B=1, S, H, D) => output = memory_efficient_attention(xq, key, val, None if cache is None else cache.mask)
        # Reshape for attention function (assumed format for specific attention implementation)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        # xq, key, val shape = (1, B, n_heads, head_dim), (1, B, n_heads, head_dim), (1, B, n_heads, head_dim) due to None
        output = simple_scaled_dot_product_attention(xq, key, val, None if cache is None else cache.mask)
        output = output.view(seqlen_sum, self.n_heads * self.head_dim)

        assert isinstance(output, torch.Tensor)   # Ensure output is tensor
        return self.wo(output)  # Transform output to final dimension and return


class FeedForward(nn.Module):
    """
    Implements a two-layer feed-forward network with an optional LoRA for the linear layers.
    This module performs a linear transformation, applies a SiLU activation, and follows up with another linear transformation, optionally enhancing each layer with LoRA adjustments.
    Attributes:
        args (ModelArgs): Configuration parameters for the model including input, hidden, and output dimensions.
        w1, w2, w3 (nn.Module): Linear transformation modules where w1 and w2 are the main transformation layers, and w3 is used for an element-wise multiplication after the first activation layer, potentially adapted with LoRA.
    Methods:
        forward(x): Conducts the forward pass through the feed-forward network using the input tensor `x`, applying linear transformations and activation to produce the output tensor.
    This module is crucial for adding depth and non-linearity to the Transformer's processing capabilities, making it suitable for complex pattern recognition and data transformation tasks in neural networks.
    """
    def __init__(self, args: ModelArgs):
        """
        Initialize the FeedForward module.
        """
        super().__init__()
        # Create the linear layers with optional LORA weights if applicable
        MaybeLora = maybe_lora(args)
        self.w1 = MaybeLora(args.dim, args.hidden_dim, bias=False)
        self.w2 = MaybeLora(args.hidden_dim, args.dim, bias=False)
        self.w3 = MaybeLora(args.dim, args.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        print("FeedForward input shape", x.shape)
        # Apply linear transformation with weights w1 and element-wise multiplication with weights w3
        # Apply linear transformation with weights w2 and activation function silu
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(torch.nn.Module):
    """
    Implements Root Mean Square Normalization (RMSNorm) for tensors, typically used in Transformer models.
    RMSNorm is similar to layer normalization but uses the root mean square of tensor elements to normalize the data.
    This can help stabilize the learning process by maintaining normalized activations across the network.
    Attributes:
        dim (int): Dimension of the input features, typically the last dimension of the tensor.
        eps (float): A small constant (epsilon) added to the denominator for numerical stability.
        weight (torch.nn.Parameter): A learnable scaling parameter applied after normalization.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initializes the RMSNorm module with the given dimension and epsilon value.
        Args:
            dim (int): Dimension of the input features.
            eps (float, optional): A small constant added to the denominator for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Private method to compute the root mean square normalization.
        Args:
            x (torch.Tensor): The input tensor to normalize.
        Returns:
            torch.Tensor: The normalized tensor.
        RMS normalization is a technique used in Transformer models to stabilize the learning process by
        maintaining normalized activations across the network. It computes the mean square of the input tensor,
        adds a small constant (epsilon) to the denominator for numerical stability, and then computes the
        reciprocal square root.
        """
        # Compute the mean square of the input tensor
        mean_square = x.pow(2).mean(-1, keepdim=True)
        # Add epsilon to the denominator for numerical stability
        denominator = mean_square + self.eps
        # Compute the reciprocal square root
        normalized = x * torch.rsqrt(denominator)
        return normalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RMSNorm layer.
        Applies root mean square normalization and then scales the result by learnable weights.
        Args:
            x (torch.Tensor): The input tensor to be normalized.
        Returns:
            torch.Tensor: The normalized and scaled tensor.
        """
        # Convert the input tensor to float to avoid potential precision issues
        # Convert the output tensor back to the same data type as the input tensor
        # Normalize the input tensor using root mean square normalization
        output = self._norm(x.float()).type_as(x)
        # Multiply the normalized tensor by the learnable weight parameter
        return output * self.weight


class TransformerBlock(nn.Module):
    """
    Represents a single block of a Transformer model, integrating multi-head attention,
    normalization (RMSNorm), and a feed-forward network. This block can optionally use a Mixture of Experts (MoE)
    as its feed-forward component to increase model capacity and tailor responses to different types of input.
    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimensionality of the input features.
        attention (Attention): The attention mechanism used in the block.
        attention_norm (RMSNorm): Normalization layer for the output of the attention.
        ffn_norm (RMSNorm): Normalization layer for the output of the feed-forward network.
        feed_forward (nn.Module): The feed-forward network, possibly a Mixture of Experts.
    """

    def __init__(self, args: ModelArgs):
        """
        Initializes the Transformer block with specified settings.
        Args:
            args (ModelArgs): The arguments for initializing the Transformer block.
        Attributes:
            n_heads (int): The number of attention heads.
            dim (int): The dimensionality of the input features.
            attention (Attention): The attention mechanism used in the block.
            attention_norm (RMSNorm): The normalization layer for the output of the attention.
            ffn_norm (RMSNorm): The normalization layer for the output of the feed-forward network.
            feed_forward (nn.Module): The feed-forward network, possibly a Mixture of Experts.
            args (ModelArgs): The arguments used for initializing the Transformer block.
        """
        super().__init__()
        # Set the number of attention heads, dimensionality, and arguments
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.args = args
        # Initialize the attention mechanism
        self.attention = Attention(args)
        # Initialize the normalization layers for attention and feed-forward networks
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Conditional initialize the feed-forward layer on Mixture of Experts if specified
        self.feed_forward: nn.Module
        if args.moe is not None:
            # If specified, use a Mixture of Experts as the feed-forward component
            self.feed_forward = MoeLayer(
                experts=[FeedForward(args=args) for _ in range(args.moe.num_experts)],
                gate=nn.Linear(args.dim, args.moe.num_experts, bias=False),
                moe_args=args.moe,
            )
        else:
            # Otherwise, use a standard feed-forward network
            self.feed_forward = FeedForward(args=args)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, cache: Optional[CacheView]
    ) -> torch.Tensor:
        """
        Process input through one Transformer block, applying attention, normalization, and feed-forward network.
        Args:
            x (torch.Tensor): The input tensor to the Transformer block.
            freqs_cis (torch.Tensor): Tensor containing frequency and cumulative indices for rotary position embeddings.
            cache (Optional[CacheView]): Optional cache object for storing state across multiple forward passes.
        Returns:
            torch.Tensor: The output tensor from the Transformer block.
        """
        print("TransformerBlock input shape", x.shape)
        # Apply attention mechanism followed by normalization
        r = self.attention.forward(self.attention_norm(x), freqs_cis, cache)
        # Residual connection around the attention layer
        h = x + r
        # Apply feed-forward network followed by normalization
        r = self.feed_forward.forward(self.ffn_norm(h))
        # Residual connection around the feed-forward layer
        out = h + r
        return out


class Transformer(nn.Module, LoRALoaderMixin):
    """
    Implements a Transformer model with optional pipeline parallelism. This class handles the distribution
    of transformer blocks across different devices or pipeline ranks and supports partial computation on sub-segments
    of the model for efficiency in distributed training environments.
    Attributes:
        args (ModelArgs): Configuration parameters for the model, such as number of layers, dimensions, and vocabulary size.
        vocab_size (int): Size of the vocabulary.
        n_layers (int): Number of transformer blocks in the model.
        pipeline_rank (int): The rank of the pipeline this instance is part of.
        num_pipeline_ranks (int): Total number of pipeline ranks in the distributed setup.
        layers (nn.ModuleDict): Dictionary of transformer blocks assigned to this pipeline rank.
    """

    def __init__(
        self,
        args: ModelArgs,
        pipeline_rank: int = 0,
        num_pipeline_ranks: int = 1,
    ):
        """
        Initializes the Transformer model with the specified pipeline configuration.
        Args:
            args (ModelArgs): Configuration parameters for the model, such as number of layers, dimensions, and vocabulary size.
            pipeline_rank (int): The rank of the pipeline this instance is part of. Defaults to 0.
            num_pipeline_ranks (int): Total number of pipeline ranks in the distributed setup. Defaults to 1.
        Raises:
            AssertionError: If the vocabulary size is not positive or if the pipeline rank is out of range.
        """
        super().__init__()
        # Store configuration parameters
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        # Initialize precomputed frequencies tensor
        self._precomputed_freqs_cis: Optional[torch.Tensor] = None
        # Validate vocabulary size and pipeline rank
        assert self.vocab_size > 0, "Vocabulary size must be positive"
        assert pipeline_rank < num_pipeline_ranks, f"Pipeline rank {pipeline_rank} is out of range for {num_pipeline_ranks} total ranks"
        # Store pipeline rank and number of pipeline ranks
        self.pipeline_rank = pipeline_rank
        self.num_pipeline_ranks = num_pipeline_ranks

        # Initialize modules specific to some ranks
        self.tok_embeddings: Optional[nn.Embedding] = None
        self.norm: Optional[RMSNorm] = None
        self.output: Optional[nn.Linear] = None
        # Initialize token embeddings and output layers depending on the pipeline rank
        if pipeline_rank == 0:
            # Initialize token embeddings for the first pipeline rank
            self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        if pipeline_rank == num_pipeline_ranks - 1:
            # Initialize normalization and output layers for the last pipeline rank
            self.norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Initialize transformer blocks and allocate them to appropriate ranks according to the pipeline configuration
        # Create transformer blocks for all layers initially
        layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        # Determine the subset of layers that this rank is responsible for
        num_layers_per_rank = math.ceil(self.n_layers / self.num_pipeline_ranks)
        offset = self.pipeline_rank * num_layers_per_rank
        end = min(self.n_layers, offset + num_layers_per_rank)
        # Assign the responsible subset of layers to this pipeline rank
        self.layers = nn.ModuleDict({str(i): layers[i] for i in range(offset, end)})
        # Count the number of layers assigned to this particular pipeline rank
        self.n_local_layers = len(self.layers)

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the data type of the parameters of the model.
        """
        # Get the first parameter of the model and return its data type
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the model parameters are stored.
        """
        # Get the first parameter of the model and return its device
        return next(self.parameters()).device

    @property
    def freqs_cis(self) -> torch.Tensor:
        # We cache freqs_cis but need to take care that it is on the right device
        # and has the right dtype (complex64). The fact that the dtype is different
        # from the module's dtype means we cannot register it as a buffer
        """
        Returns the precomputed frequencies and complex numbers for the Rotary Embedding.
        This property method ensures efficient management of precomputed tensors for rotary embeddings,
        caching them to avoid repetitive computation. It checks the device and data type to ensure compatibility
        with the module's current configuration.
        Returns:
            torch.Tensor: The precomputed frequencies and complex indices, adjusted for the head dimension and specified theta.
        """
        # Check if the frequencies and complex indices have been computed and cached
        if self._precomputed_freqs_cis is None:
            # If not cached, compute and store them. The value of theta can be customized
            theta = self.args.rope_theta or 1000000.0   # default to 10**6
            # Precompute the frequencies and complex indices based on head dimensions and theta
            self._precomputed_freqs_cis = precompute_freqs_cis(
                self.args.head_dim, 128_000, theta
            )
        # Ensure the tensor is on the correct device as the module's other parameters.
        # This step is necessary because the tensor's device may not automatically update with the module's device.
        if self._precomputed_freqs_cis.device != self.device:
            # Move the tensor to the same device as the module if it's not already there.
            self._precomputed_freqs_cis = self._precomputed_freqs_cis.to(
                device=self.device
            )
        return self._precomputed_freqs_cis

    def forward_partial(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        cache: Optional[BufferCache] = None,
    ) -> torch.Tensor:
        """ Local forward pass.
        Performs a local forward pass for the segment of the model corresponding to the current pipeline rank.
        This function handles input transformation, sequential processing through transformer blocks,
        and the sending or receiving of data to adjacent pipeline ranks, optimizing for efficient computation
        through caching and distributed processing.
        Args:
            input_ids (torch.Tensor): The input IDs to process, typically integers representing tokens.
            seqlens (List[int]): List of sequence lengths for batched input processing, ensuring variable-length input can be processed in fixed-size batches.
            cache (Optional[BufferCache]): Cache for storing intermediate states for efficient re-computation, especially beneficial in recurrent passes over data.
        Returns:
            torch.Tensor: The output tensor for this pipeline segment, or the final model outputs if this is the last segment.
        NOTE: If doing pipeline parallelism, this will return the activations of the last layer of this stage.
        For the last stage, this will return the normalized final embeddings.
        """
        # Ensure the batch size is within the pre-defined maximum to prevent resource over-allocation
        assert (
            len(seqlens) <= self.args.max_batch_size
        ), f"Max batch size is {self.args.max_batch_size}, got batch size of {len(seqlens)}"
        (num_toks,) = input_ids.shape
        # Validate that the sum of all sequence lengths matches the total number of input tokens
        assert sum(seqlens) == num_toks, (sum(seqlens), num_toks)

        # Determine the appropriate metadata format for caching or direct processing
        input_metadata: Union[CacheInputMetadata, SimpleInputMetadata]
        if cache is not None:
            input_metadata = cache.get_input_metadata(seqlens)
        else:
            input_metadata = SimpleInputMetadata.from_seqlens(seqlens, self.device)

        # Initialize the hidden state, either by embedding the input IDs or by receiving tensors from a previous rank
        # Pipeline-specific processing: initial embedding or receiving tensors from previous rank
        if self.pipeline_rank == 0:
            assert self.tok_embeddings is not None, "Token embeddings should be initialized for the first pipeline rank"
            h = self.tok_embeddings(input_ids)
        else:
            # Allocate an empty tensor to receive data from the previous pipeline rank
            h = torch.empty(num_toks, self.args.dim, device=self.device, dtype=self.dtype)
            torch.distributed.recv(h, src=self.pipeline_rank - 1)   # Receive input from the previous rank

        # Fetch the precomputed frequencies and complex indices for rotary embeddings
        freqs_cis = self.freqs_cis[input_metadata.positions]

        # Process input through each transformer block assigned to this pipeline rank
        for local_layer_id, layer in enumerate(self.layers.values()):
            if cache is not None:
                assert input_metadata is not None
                assert isinstance(input_metadata, CacheInputMetadata)
                cache_view = cache.get_view(local_layer_id, input_metadata)
            else:
                cache_view = None
            h = layer(h, freqs_cis, cache_view)

        # Update cache with new sequence lengths if cache is used
        if cache is not None:
            cache.update_seqlens(seqlens)

        # If not the last rank, send the processed tensor to the next rank in the pipeline
        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            torch.distributed.send(h, dst=self.pipeline_rank + 1)
            return h  # Return early as further processing occurs in subsequent ranks
        else:
            # For the last rank in the pipeline, apply a normalization step to finalize the outputs
            assert self.norm is not None
            return self.norm(h)  # Normalize and return the final embeddings as output

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        cache: Optional[BufferCache] = None,
    ) -> torch.Tensor:
        """
        Executes the complete forward pass for the Transformer model across all pipeline stages.
        This method orchestrates the sequential processing of input IDs through designated pipeline ranks,
        manages data transfer between stages, and ensures that the final output is computed and distributed correctly.
        Args:
            input_ids (torch.Tensor): The input IDs for the model, typically integer tokens that map to an embedding.
            seqlens (List[int]): List of sequence lengths for each input batch. This is critical for processing variable-length sequences within fixed-size batches.
            cache (Optional[BufferCache]): A cache object for storing intermediate computation states, used to enhance performance by reducing redundant computations across training iterations.
        Returns:
            torch.Tensor: The final output tensor after processing through all transformations and computational stages. The tensor contains logits or scores that are typically passed to a loss function or an activation layer.
        """
        # Perform the segment-specific forward pass to compute intermediate activations for this pipeline segment
        h = self.forward_partial(input_ids, seqlens, cache=cache)
        # Handle the output based on the rank of this pipeline segment
        # If this is not the last pipeline rank, send the intermediate activations to the next rank
        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            # For intermediate ranks, prepare a placeholder tensor to receive the final output
            # The actual content of 'outs' is not used (are ignored) because it will be overwritten by the broadcast from the last rank
            outs = torch.empty(h.shape[0], self.vocab_size, device=h.device, dtype=h.dtype)
        else:
            # The last rank in the pipeline computes the final output using a linear transformation layer
            assert self.output is not None, "Output layer should be initialized for the last pipeline rank"
            outs = self.output(h)
        # If the model is split across multiple ranks, ensure that the final output is available to all ranks by broadcasting it
        if self.num_pipeline_ranks > 1:
            # Broadcast the final output from the last rank to all other ranks
            torch.distributed.broadcast(outs, src=self.num_pipeline_ranks - 1)
        # Return the final output tensor
        return outs.float()

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> None:
        """
        Load the state dictionary into the model selectively based on the pipeline rank.
        This method adapts the standard load_state_dict behavior to accommodate pipeline parallelism by selectively
        loading parameters relevant to specific pipeline ranks. It ensures that each segment of the model in a distributed
        setup only handles its corresponding parameters.
        Args:
            state_dict (Mapping[str, Any]): The state dictionary to be loaded.
            strict (bool, optional): Whether to strictly enforce that the keys in the state dictionary match the keys in the model. Defaults to True.
            assign (bool, optional): Whether to assign the state dictionary to the model. Defaults to False.
        Raises:
            ValueError: If an unexpected key is found in the state dictionary.
        """
        # Initialize a dictionary to store the parameters to be loaded and a set to track skipped parameters
        state_to_load = {}
        skipped = set([])

        # Iterate through each key-value pair in the provided state dictionary
        for k, v in state_dict.items():
            # Check if the key starts with "tok_embeddings" - Token embeddings are only relevant for the first pipeline rank
            if k.startswith("tok_embeddings"):
                # If the current pipeline rank is 0, add the key-value pair to the state to be loaded
                if self.pipeline_rank == 0:
                    state_to_load[k] = v
                # Otherwise, skip the parameter and add the key to the set of skipped keys
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            # Check if the key starts with "norm" or "output" - Normalization and output layers are only relevant for the last pipeline rank
            elif k.startswith("norm") or k.startswith("output"):
                # If the current pipeline rank is the last pipeline rank, add the key-value pair to the state to be loaded
                if self.pipeline_rank == self.num_pipeline_ranks - 1:
                    state_to_load[k] = v
                # Otherwise, skip the parameter and add the key to the set of skipped keys
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            # Check if the key starts with "layers" - Only load layers that are within this rank's responsibility
            elif k.startswith("layers"):
                # Get the layer ID from the key
                layer_id = k.split(".")[1]
                # If the layer ID is in the model's layers, add the key-value pair to the state to be loaded
                if layer_id in self.layers:
                    state_to_load[k] = v
                # Otherwise, skip the parameter and add the key to the set of skipped keys
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            # Raise an error for any keys that do not correspond to expected model parameters
            else:
                raise ValueError(f"Unexpected key {k}")
        # Verify that all keys from the state dictionary are either loaded or intentionally skipped
        assert set(state_dict.keys()) == skipped.union(set(state_to_load.keys())), "Mismatch between state dictionary keys and the handled keys."
        # Load the state dictionary into the model using the superclass method, applying strict or assignment conditions
        super().load_state_dict(state_to_load, strict=strict, assign=assign)

    @staticmethod
    def from_folder(
        folder: Union[Path, str],
        max_batch_size: int = 1,
        num_pipeline_ranks: int = 1,
        device: Union[torch.device, str] = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> "Transformer":
        """
        Loads a Transformer model from a specified folder, setting it up for distributed processing if required.
        Args:
            folder (Union[Path, str]): The folder containing the model files.
            max_batch_size (int, optional): The maximum batch size. Defaults to 1.
            num_pipeline_ranks (int, optional): The number of pipeline ranks. Defaults to 1.
            device (Union[torch.device, str], optional): The device to load the model on. Defaults to "cuda".
            dtype (Optional[torch.dtype], optional): The data type of the model. Defaults to None.
        Returns:
            Transformer: The loaded Transformer model.
        Raises:
            FileNotFoundError: If neither the pt_model_file nor the safetensors_model_file exists in the folder.
            FileExistsError: If both the pt_model_file and the safetensors_model_file exist in the folder.
        """
        # Load the model configuration parameters from the params.json file
        with open(Path(folder) / "params.json", "r") as f:
            model_args = ModelArgs.from_dict(json.load(f))
        # Set the max_batch_size in the model arguments
        model_args.max_batch_size = max_batch_size
        # Set up pipeline rank based on the number of ranks specified
        pipeline_rank = torch.distributed.get_rank() if num_pipeline_ranks > 1 else 0
        # Initialize the Transformer model using loaded parameters
        with torch.device("meta"):
            model = Transformer(
                model_args,
                pipeline_rank=pipeline_rank,
                num_pipeline_ranks=num_pipeline_ranks,
            )

        # Define paths for possible model state files
        pt_model_file = Path(folder) / "consolidated.00.pth"
        safetensors_model_file = Path(folder) / "consolidated.safetensors"

        # Ensure that exactly one of the model state files exists
        assert (
            pt_model_file.exists() or safetensors_model_file.exists()
        ), f"Make sure either {pt_model_file} or {safetensors_model_file} exists"
        assert not (
            pt_model_file.exists() and safetensors_model_file.exists()
        ), f"Both {pt_model_file} and {safetensors_model_file} cannot exist"

        # Load the model state from the appropriate file type
        if pt_model_file.exists():
            loaded = torch.load(str(pt_model_file), mmap=True)
        else:
            loaded = safetensors.torch.load_file(str(safetensors_model_file))

        # Load the state dictionary into the model
        model.load_state_dict(loaded, assign=True, strict=True)
        # Configure the model to the appropriate device and data type
        return model.to(device=device, dtype=dtype)
