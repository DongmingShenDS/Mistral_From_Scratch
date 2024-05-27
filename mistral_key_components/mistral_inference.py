from typing import List, Optional, Tuple
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from mistral_model import ModelArgs, Transformer


class Mistral:
    def __init__(self, model_args: ModelArgs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = model_args
        self.model = Transformer(args=model_args).to(device)

    def generate(
            self, prompts: List[List[int]], temperature: float = 0.6,
            top_p: float = 0.9, max_gen_len: Optional[int] = None
    ):

        # set max_gen_len to max_seq_len - 1 if not specified (max possible sequence length)
        # this is bottleneck by KV Cache in the current implementation, but should not be a problem in Mistral?
        if max_gen_len is None: max_gen_len = self.args.max_seq_len - 1

        # NOTE: prompts are not str but are assumed to be encoded: List[List[int]]
        # in reality, tokenize the prompts using tokenizer.encode method and add bos and eos (not used for inference)
        prompt_tokens = prompts

        # Make sure the batch size is not too large (cannot be larger than max_batch_size)
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, \
            f"batch size must be less than or equal to {self.args.max_batch_size}"

        # Make sure the prompt length is not larger than the maximum sequence length
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.args.max_seq_len, \
            f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # Create the list to hold the generated tokens, along with the initial prompt tokens
        pad_id = 0  # self.tokenizer.pad_id()
        # create a tensor of size (batch_size, total_len) filled with padding token
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.model.device)
        for k, t in enumerate(prompt_tokens):
            # fill the initial tokens with the prompt tokens to start with
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.model.device)
        # to keep track of if eos token as been reached in each batch (aka any of the prompts)
        eos_reached = torch.tensor([False] * batch_size, device=self.model.device)
        prompt_tokens_mask = tokens != pad_id  # True if the current token is a prompt token, False otherwise

        # Generate the tokens one at a time
        for cur_pos in tqdm(range(1, total_len), desc="Generating tokens"):
            with torch.no_grad():
                # pass in 1 token at a time (only current token) for all batches, at current position
                # b.c. using KV Cache, the model will output logits for this 1 token only at a time => for inference
                logits = self.model.forward(tokens[:, cur_pos - 1:cur_pos], cur_pos)
            # sample the next token using temperature and top_p
            next_token = self.sample_next_token(logits, temperature, top_p)
            next_token = next_token.reshape(-1)
            # Only replace token if it is a padding token (b.c. already have tokens that are from the prompt)
            # note: running the tokens in the prompt is solely for building the initial cache
            # torch.where(cond, a, b): if cond then a else b => if pad then next_token else curr_token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found an EOS token for a padding position
            # since we don't have tokenizer for testing, this part used -1
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == -1)  # self.tokenizer.eos_id)
            # if all prompts have reached EOS, we are done, break from the loop
            if all(eos_reached): break

        # prepare the output
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token if present, the prev part represents the generated output
            if -1 in current_prompt_tokens:  # self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)  # output tokens
            # out_text.append(self.tokenizer.decode(current_prompt_tokens))  # output texts need tokenizer
        return (out_tokens, out_text)

    def sample_next_token(self, probs, temperature, top_p):
        if temperature == 0:
            next_token = torch.argmax(probs[:, -1], dim=-1)
            return next_token
        # Temperature Sampling for Inference
        probs = torch.softmax(probs[:, -1] / temperature, dim=-1)
        # Top-P Sampling for Inference (Mistral also used Temperature, but here no such support)
        # sort in descending order, (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # get cumulative sum of probs, (B, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # binary mask for which tokens are selected, (B, vocab_size)
        # (substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
        mask = probs_sum - probs_sort > top_p
        # zeros out all the probabilities of tokens that are not selected by the Top-P (tail probs to ignore)
        probs_sort[mask] = 0.0
        # re-distribute the probabilities so that they sum up to 1.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # sample a token (its index) from these top-p distribution based on their distribution
        next_token = torch.multinomial(probs_sort, num_samples=1)  # torch.multinomial: (B, 1)
        # get the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token)  # torch.gather: (B, 1)
        return next_token
