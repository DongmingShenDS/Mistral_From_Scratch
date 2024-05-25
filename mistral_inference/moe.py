"""
The moe.py file contains the implementation of the MoE (Mixture of Experts) layer.

The MoE layer is a variant of the Transformer architecture that utilizes expert modules to process different parts of the input sequence.
It consists of a list of expert modules, a gate module, and MoeArgs object.
The expert modules are responsible for processing different parts of the input sequence, while the gate module determines the importance of each expert module.
The MoeArgs object stores the configuration parameters for the MoE layer.

The file starts with import statements for necessary modules and packages. It then defines the `MoeLayer` class, which represents the MoE layer.
The class contains several methods and attributes, including initialization, forward pass computation, and utility functions for expert routing and expert gating.

The file also includes the `MoeArgs` class, which stores the configuration parameters for the MoE layer.

Overall, the `moe.py` file implements the MoE layer and provides a way to utilize expert modules for efficient processing of input sequences.
"""

import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn


@dataclasses.dataclass
class MoeArgs(Serializable):
    """
    Arguments for the MoE (Mixture of Experts) layer.
    Attributes:
        num_experts (int): The total number of experts.
        num_experts_per_tok (int): The number of experts per token. (for sparsity)
    """
    num_experts: int
    num_experts_per_tok: int


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        """
        Initialize the MoeLayer with the given experts (a list of linear layers), gate (a linear layer), and MoeArgs.
        https://en.wikipedia.org/wiki/Mixture_of_experts, https://arxiv.org/abs/2401.04088
        Args:
            experts (List[nn.Module]): A list of expert modules.
            gate (nn.Module): The gate module.
            moe_args (MoeArgs): The MoeArgs object.
        """
        super().__init__()          # Initialize the super class
        assert len(experts) > 0, "Experts list cannot be empty"
        # Create a ModuleList (subclass of nn.Module) to store the expert modules
        self.experts = nn.ModuleList(experts)
        # Initialize the gate module and MoeArgs object
        self.gate = gate            # MoE gating mechanism
        self.args = moe_args

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MoeLayer (a part of the larger MoE architecture).
        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, output_size)
        """
        # Get the gate logits (transformation of inputs) for each input token using a gate module
        gate_logits = self.gate(inputs)
        # Get the top k experts for each input token, using torch.topk
        weights, selected_experts = torch.topk(
            gate_logits, self.args.num_experts_per_tok
        )
        # Normalize the weights with softmax, to get the selected top k experts' weights
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        # Initialize the results tensor (same shape as inputs, why?)
        results = torch.zeros_like(inputs)
        # Iterate over each expert to compute the weighted sum of the outputs from each selected top k experts,
        # Then accumulates the results in the results tensor
        for i, expert in enumerate(self.experts):
            # For each expert, retrieves batch_idx and expert for each token, based on the selected top k experts
            batch_idx, nth_expert = torch.where(selected_experts == i)
            # Compute the weighted sum of the expert's output for each token in the batch
            # Add the weighted sum to the corresponding position in the results tensor
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs[batch_idx]
            )
        return results
