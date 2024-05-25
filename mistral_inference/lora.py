"""
The lora.py file contains the implementation of the LORA (Localized and Randomized) model, which is a variant of the Transformer model.
The LORA model is designed to improve the efficiency and scalability of the Transformer by localizing attention patterns and randomizing attention weights.

The file starts with import statements for necessary modules and packages.
It then defines the LoraConfig dataclass, which stores the configuration parameters for the LORA model.

The file continues with the definition of the LoraLayer class, which represents a single layer in the LORA model.
The class contains several methods and attributes, including initialization, forward pass computation, and utility functions for generating attention masks and computing attention scores.
"""


import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, NamedTuple, Union

import safetensors.torch
import torch
import torch.nn as nn
from simple_parsing.helpers import Serializable


@dataclass
class LoraArgs(Serializable):
    """
    Data class for LoraArgs.
    Attributes:
        rank (int): The rank of the LoraArgs.
        scaling (float): The scaling factor of the LoraArgs.
    """
    rank: int
    scaling: float

    def __post_init__(self):
        """
        Post-initialization of LoraArgs: check that the rank and scaling factor are greater than zero.
        """
        assert self.rank > 0, "Rank must be greater than zero"
        assert self.scaling > 0.0, "Scaling factor must be greater than zero"


class LoRALinear(nn.Module):
    """
    Implementation of:
        - LoRA: https://arxiv.org/abs/2106.09685
    Notes:
        - Freezing is handled at network level, not layer level.
        - Scaling factor controls relative importance of LoRA skip
          connection versus original frozen weight. General guidance is
          to keep it to 2.0 and sweep over learning rate when changing the rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        scaling: float,
        bias: bool = False,
    ):
        """
        Initialize the LoRALinear module.
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            rank (int): Rank of the LoraArgs.
            scaling (float): Scaling factor of the LoraArgs.
            bias (bool, optional): Whether to use bias. Defaults to False.
        """
        super().__init__()  # Call parent constructor
        self.in_features = in_features
        self.out_features = out_features
        assert not bias, "Bias is not supported"
        self.bias = bias
        self.rank = rank
        self.scaling = scaling
        # Initialize the linear layers A & B to implement the low-rank LoRA techniques
        # takes (batch_size, self.in_features) and produces an output tensor (batch_size, self.rank)
        self.lora_A = nn.Linear(    # size (self.in_features, self.rank)
            self.in_features,
            self.rank,
            bias=self.bias,
        )
        # takes (batch_size, self.rank) and produces an output tensor (batch_size, self.out_features)
        self.lora_B = nn.Linear(    # size (self.rank, self.out_features)
            self.rank,
            self.out_features,
            bias=self.bias,
        )
        # Initialize the linear layer for the original weights
        self.linear = nn.Linear(self.in_features, self.out_features, bias=self.bias)
        # Register a post-load state dict hook to ignore missing keys
        # make sure no LoRA weights are marked as "missing" in load_state_dict

        def ignore_missing_keys(m: nn.Module, incompatible_keys: NamedTuple):
            """
            This function is used as a post-load state dict hook to ignore missing keys
            in the state dictionary of a PyTorch module. It is typically registered
            using the `register_load_state_dict_post_hook` method of a module.
            Args:
                m (nn.Module): The module.
                incompatible_keys (NamedTuple): The incompatible keys.
            """
            incompatible_keys.missing_keys[:] = []  # type: ignore
        self.register_load_state_dict_post_hook(ignore_missing_keys)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the model with LoRA transformation from LoRA weights.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Apply linear transformation with LoRA weights (low-rank matrix multiplication)
        lora = self.lora_B(self.lora_A(x))
        # Apply linear transformation with original weights
        linear_output = self.linear(x)
        # Scale the LoRA output and add it to the linear output
        output = linear_output + lora * self.scaling
        return output

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """
        This function loads the state dict into the model.
        It first checks if the key name for the weight exists in the state dict.
        If it does, it loads the full checkpoint. Otherwise, it loads the frozen weights.
        Args:
            state_dict (dict): The state dict to load.
            prefix (str): The prefix for the key name.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        # Construct the key name for the weight
        key_name = prefix + "weight"
        # Check if the key name for the weight exists in the state dict (full checkpoint)
        if key_name in state_dict:
            # Get the reference weight from the state dict
            w_ref = state_dict[key_name]
            # Load the frozen weights
            state_dict = {
                "linear.weight": w_ref,             # Load the linear weight
                "lora_A.weight": torch.zeros_like(  # Load the LoRA A weight
                    self.lora_A.weight, device=w_ref.device, dtype=w_ref.dtype
                ),
                "lora_B.weight": torch.zeros_like(  # Load the LoRA B weight
                    self.lora_B.weight, device=w_ref.device, dtype=w_ref.dtype
                ),
            }
            # Load the state dict into the model using the load_state_dict method in nn.Module
            self.load_state_dict(state_dict, assign=True, strict=True)


class LoRALoaderMixin:
    """
    Mixin class for loading LoRA (Low-Rank Adaptation) state dictionaries into a model.
    This mixin class provides functionality for loading LoRA state dictionaries into a model. It is typically mixed into a model class that inherits from `nn.Module`.
    Attributes: ?
        args (argparse.Namespace): Arguments passed to the model.
        layers (List[str]): List of layer names to include in the model.
        pipeline_rank (int): Pipeline rank of the model.
   """

    def load_lora(self, lora_path: Union[Path, str], scaling: float = 2.0):
        """
        Load the LoRA checkpoint from the given path.
        Args:
            lora_path (Union[Path, str]): The path to the LoRA checkpoint file.
            scaling (float, optional): The scaling factor. Defaults to 2.0.
        Raises:
            AssertionError: If the `lora_path` does not exist or is not a file.
        """
        # Convert the lora_path to a pathlib Path object if it's a string & check exist and is file
        lora_path = Path(lora_path)
        assert lora_path.is_file(), f"{lora_path} does not exist or is not a file"
        # Load the LoRA state_dict from the file
        state_dict = safetensors.torch.load_file(lora_path)
        # Load the LoRA state_dict into the model using the _load_lora_state_dict method
        self._load_lora_state_dict(state_dict, scaling=scaling)

    def _load_lora_state_dict(
        self, lora_state_dict: Dict[str, torch.Tensor], scaling: float = 2.0
    ):
        """
        Load the LoRA state_dict into the model.
        Args:
            lora_state_dict (Dict[str, torch.Tensor]): The LoRA state_dict to be loaded.
            scaling (float, optional): The scaling factor. Defaults to 2.0.
        Raises:
            AssertionError: If the LoRA weights have multiple different dtypes or if the dtype differs from the model's dtype.
        """
        # Check if all weights have the same dtype
        lora_dtypes = set([p.dtype for p in lora_state_dict.values()])
        assert (
            len(lora_dtypes) == 1
        ), f"LoRA weights have multipe different dtypes {lora_dtypes}. All weights need to have the same dtype"
        lora_dtype = lora_dtypes.pop()
        assert (
            lora_dtype == self.dtype
        ), f"LoRA weights dtype differs from model's dtype {lora_dtype} != {self.dtype}"
        assert all("lora" in key for key in lora_state_dict.keys())
        # Move tensors to device
        lora_state_dict = {k: v.to(self.device) for k, v in lora_state_dict.items()}
        # Get the current state dict of the model
        state_dict = self.state_dict()

        if self.args.lora is None:
            logging.info("Loading and merging LoRA weights...")
            # Replace every nn.Linear module with a LoRALinear module except the output layer
            named_modules = dict(self.named_modules())
            for name, module in named_modules.items():
                if isinstance(module, nn.Linear) and name != "output":
                    layer_id = name.split(".")[1]
                    if layer_id not in self.layers:
                        logging.debug(
                            "Skipping parameter %s at pipeline rank %d",
                            name,
                            self.pipeline_rank,
                        )
                    else:  # Merge the weights of the LoRA state_dict with the current state_dict of the model
                        # Retrieve the LoRA weights for the current layer
                        lora_B_weight = lora_state_dict[name + ".lora_B.weight"]
                        lora_A_weight = lora_state_dict[name + ".lora_A.weight"]
                        # Calculate the merged weight by adding original weight with the scaled product of LoRA weights
                        # NOTE: here matrix multiplication is used to get the actual weights
                        weight = (
                            module.weight + (lora_B_weight @ lora_A_weight) * scaling
                        )
                        state_dict[name + ".weight"] = weight   # Update the state dict with the merged weight
        else:
            logging.info("Loading LoRA weights...")
            # Update the state dict with the LoRA state_dict
            for k, v in lora_state_dict.items():
                state_dict.update(lora_state_dict)
                layer_id = k.split(".")[1]
                if layer_id in self.layers:
                    # Update the state_dict with the LoRA weight if the layer is included in self.layers
                    state_dict[k] = v
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )
        # Load the state dict into the model using the load_state_dict method in nn.Module
        self.load_state_dict(state_dict, strict=True)
