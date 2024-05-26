from typing import List, Optional, Tuple

import torch

from mistral_inference.cache import BufferCache
from mistral_inference.model import Transformer


@torch.inference_mode()
def generate(
    encoded_prompts: List[List[int]],
    model: Transformer,
    *,
    max_tokens: int,
    temperature: float,
    chunk_size: Optional[int] = None,
    eos_id: Optional[int] = None
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Generate text using a given model and encoded prompts.
    Args:
        encoded_prompts (List[List[int]]): The encoded prompts to generate text from.
        model (Transformer): The model to use for generation.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The temperature to use for sampling.
        chunk_size (Optional[int], optional): The size of chunks to process the prompts in. Defaults to None.
        eos_id (Optional[int], optional): The ID of the end of sequence token. Defaults to None.
    Returns:
        Tuple[List[List[int]], List[List[float]]]: A tuple containing the generated tokens and their log probabilities.

    The function first sets the model to evaluation mode. It then gets the batch size and vocabulary size from the input. It initializes a cache for storing intermediate results. It also initializes a list to store the log probabilities for each generated token.
    The function then encodes the prompts by splitting them into chunks of size chunk_size. It passes the concatenated prompt chunks through the model to get prelogits. If it's not the first pass, it calculates logits for the last token of each chunk in the previous pass and updates the log probabilities accordingly. It then calculates logits for each token in the chunk (excluding the first token) and updates the last_token_prelogits to the prelogits of the last token in the last chunk.
    After encoding the prompts, the function decodes the generated tokens. It samples the next token from the last_token_prelogits using the specified temperature and top_p value. It updates the is_finished tensor based on whether the next token is the EOS token. If all batch elements are finished, it exits the loop. It calculates logits for the last token in the sequence with softmax and appends the log probability of the next token to the corresponding logprob list. It appends the next token to the generated_tensors list, passes the next token through the model to get the prelogits for the next token, and updates the last_token_prelogits accordingly.
    Finally, the function converts the generated tensors into a list of lists of integers and returns the generated tokens and logprobs.
    """
    # Note: The function assumes that the model is in evaluation mode.
    model = model.eval()

    # Get batch size and vocabulary size
    B, V = len(encoded_prompts), model.args.vocab_size

    # Get lengths of encoded prompts
    seqlens = [len(x) for x in encoded_prompts]

    # Initialize Cache
    cache_window = max(seqlens) + max_tokens
    cache = BufferCache(
        model.n_local_layers,
        model.args.max_batch_size,
        cache_window,
        model.args.n_kv_heads,
        model.args.head_dim,
    )
    cache.to(device=model.device, dtype=model.dtype)
    cache.reset()

    # Bookkeeping: Initialize log probabilities
    logprobs: List[List[float]] = [[] for _ in range(B)]
    last_token_prelogits = None

    # Set one chunk size to maximum prompt length if not specified
    max_prompt_len = max(seqlens)
    if chunk_size is None:
        chunk_size = max_prompt_len

    # Encode prompt by chunks
    # It encodes the prompt by splitting it into chunks of size chunk_size
    # It then passes the concatenated prompt chunks through the model to get prelogits
    # If not the first pass, it calculates logits for the last token of each chunk in the previous pass.
    # It then calculates logits for each token in the chunk (excluding the first token) and updates last_token_prelogits to the prelogits of the last token in the last chunk.
    for s in range(0, max_prompt_len, chunk_size):
        # Split encoded prompts into chunks of size chunk_size
        prompt_chunks = [p[s : s + chunk_size] for p in encoded_prompts]
        assert all(len(p) > 0 for p in prompt_chunks)
        # Pass the concatenated prompt chunks through the model to get prelogits
        prelogits = model.forward(
            torch.tensor(sum(prompt_chunks, []), device=model.device, dtype=torch.long),
            seqlens=[len(p) for p in prompt_chunks],
            cache=cache,
        )
        logits = torch.log_softmax(prelogits, dim=-1)
        if last_token_prelogits is not None:
            # (Pass > 1) If this is not the first pass, calculate logits for the last token of each chunk in the previous pass
            last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
            for i_seq in range(B):
                logprobs[i_seq].append(
                    last_token_logits[i_seq, prompt_chunks[i_seq][0]].item()
                )
        offset = 0
        for i_seq, sequence in enumerate(prompt_chunks):
            # Calculate logits for each token in the chunk (excluding the first token)
            logprobs[i_seq].extend(
                [
                    logits[offset + i, sequence[i + 1]].item()
                    for i in range(len(sequence) - 1)
                ]
            )
            offset += len(sequence)
        # Update last_token_prelogits to the prelogits of the last token in the last chunk
        last_token_prelogits = prelogits.index_select(
            0,
            torch.tensor(
                [len(p) for p in prompt_chunks], device=prelogits.device
            ).cumsum(dim=0)
            - 1,
        )
        assert last_token_prelogits.shape == (B, V)    # Ensure shape is correct

    # Decode with autoregressive sampling
    generated_tensors = []  # Stores the generated tokens as tensors
    is_finished = torch.tensor([False for _ in range(B)])   # A boolean tensor indicating if each batch element is finished
    assert last_token_prelogits is not None     # Ensure last_token_prelogits is not None
    # Decode for a maximum of max_tokens tokens in a autoregressive manner
    # NOTE: Autoregressive token generation with state caching (every step only compute attention on last token)
    for _ in range(max_tokens):
        # Sample the next token from the last_token_prelogits using the specified temperature and top_p for randomness control
        next_token = sample(last_token_prelogits, temperature=temperature, top_p=0.8)
        if eos_id is not None:  # If an EOS token is specified
            # Update is_finished tensor based on whether the next token is the EOS token
            is_finished = is_finished ^ (next_token == eos_id).cpu()
        if is_finished.all():  # Exit the generation loop if all sequences are complete
            break
        # Calculate logits for the last token in the sequence with softmax
        last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
        # Record the log probability of the newly generated token (next_token) for each sequence in the batch
        for i in range(B):
            logprobs[i].append(last_token_logits[i, next_token[i]].item())
        # Append (store) the next token in a generated_tensors tensor list for further processing
        generated_tensors.append(next_token[:, None])
        # Pass the next token through the model to compute prelogits for the next generation step
        # This utilizes cached keys and values from previous tokens to efficiently compute attention only for the new token
        last_token_prelogits = model.forward(next_token, seqlens=[1] * B, cache=cache)
        # Ensure the shape of the prelogits is correct, confirming correct autoregressive behavior and caching usage
        assert last_token_prelogits.shape == (B, V)

    # Convert the generated tensors into a list of lists of integers
    generated_tokens: List[List[int]]
    if generated_tensors:
        generated_tokens = torch.cat(generated_tensors, 1).tolist()
    else:
        generated_tokens = []
    # Return the generated tokens and logprobs
    return generated_tokens, logprobs


def sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """
    Samples a token from the given logits based on the given temperature and top_p.
    Args:
        logits (torch.Tensor): The logits from which to sample a token.
        temperature (float): The temperature to apply to the logits. If temperature is 0, argmax is used.
        top_p (float): The cumulative probability threshold for the top_p sampling.
    Returns:
        torch.Tensor: The sampled token.
    """
    # If temperature is greater than 0, apply temperature scaling to the logits and sample from the probabilities
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    # If temperature is 0, directly sample the token with the highest logit (argmax sample)
    else:
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
    # Reshape the next token to a 1D tensor
    return next_token.reshape(-1)


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Samples a token from the given probabilities tensor based on the given top_p threshold.
    Args:
        probs (torch.Tensor): The probabilities tensor from which to sample a token.
        p (float): The cumulative probability threshold for the top_p sampling.
    Returns:
        torch.Tensor: The sampled token.
    """
    assert 0 <= p <= 1
    # Sort the probabilities in descending order
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # Calculate the cumulative sum of the probabilities
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # Create a mask for probabilities at tail (that exceed the top_p threshold), mask tail to be 0
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    # Normalize the leftover probabilities
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # Sample a token from the probabilities & Gather the index of the sampled token
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)
