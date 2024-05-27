# inference strategies:
# 1. Greedy decoding (cannot recover from mistake, no randomness no creativity)
# 2. Random Sampling (sampling from distribution => likely to pick rare tokens)
# 3. Beam Search with K>1 (better recover from mistake)
# 4. Nucleus Sampling Top-p (portion of distribution until the cumulative probability exceeds a threshold p)
# 5. Top-k Sampling (select from top k most likely tokens)
# 6. Temperature Sampling (low temperature => more deterministic, high temperature => more creative)

from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from llama_model import ModelArgs, Transformer


class LLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool,
              max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()
        # load model from checkpoints path
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, "No checkpoints found in {}".format(checkpoints_dir)
            check_path = checkpoints[-1]
            print(f"Loading from {check_path}")
            checkpoint = torch.load(check_path, map_location="cpu")
            print(f"Loaded model in {time.time() - prev_time} seconds")
            prev_time = time.time()
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        # load model args from params
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )
        # get vocab size from tokenizer used
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()
        # set type based on device
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)
        model = Transformer(model_args).to(device)
        # load model state dict
        if load_model:
            del checkpoint['rope.freqs']  # The only unmatched key in the checkpoint is rope.freqs. Remove it
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time: .2f}s")
        return LLaMA(model, tokenizer, model_args)

    def generate(self, prompts: list[str], temperature: float = 0.6,
                 top_p: float = 0.9, max_gen_len: Optional[int] = None):
        # set max_gen_len to max_seq_len - 1 if not specified (max possible sequence length)
        # this is bottleneck by KV Cache in the current implementation, but should not be a problem in Mistral?
        if max_gen_len is None: max_gen_len = self.args.max_seq_len - 1

        # tokenize the prompts using tokenizer.encode method and add bos and eos (not used for inference)
        prompt_tokens = [self.tokenizer.encode(
            prompt, out_type=int, add_bos=True, add_eos=False
        ) for prompt in prompts]

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
        pad_id = self.tokenizer.pad_id()
        # create a tensor of size (batch_size, total_len) filled with padding token
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.args.device)
        for k, t in enumerate(prompt_tokens):
            # fill the initial tokens with the prompt tokens to start with
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.args.device)
        # to keep track of if eos token as been reached in each batch (aka any of the prompts)
        eos_reached = torch.tensor([False] * batch_size, device=self.args.device)
        prompt_tokens_mask = tokens != pad_id  # True if the current token is a prompt token, False otherwise

        # Generate the tokens one at a time
        for cur_pos in tqdm(range(1, total_len), desc="Generating tokens"):
            with torch.no_grad():
                # pass in 1 token at a time (only current token) for all batches, at current position
                # b.c. using KV Cache, the model will output logits for this 1 token only at a time => for inference
                logits = self.model.forward(tokens[:, cur_pos - 1:cur_pos], cur_pos)
            # NOTE: in Mistral, we also have top-p in addition to temperature (Llama no)
            if temperature > 0:
                # The temperature is applied before the softmax for scaling the distribution control randomness
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedily select the token with the max probability
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # Only replace token if it is a padding token (b.c. already have tokens that are from the prompt)
            # note: running the tokens in the prompt is solely for building the initial cache
            # torch.where(cond, a, b): if cond then a else b => if pad then next_token else curr_token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found an EOS token for a padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
            # if all prompts have reached EOS, we are done, break from the loop
            if all(eos_reached): break

        # prepare the output
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token if present, the prev part represents the generated output
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)    # output tokens
            out_text.append(self.tokenizer.decode(current_prompt_tokens))   # output texts
        return (out_tokens, out_text)

    def _sample_top_p(self, probs, p):
        # Top-P Sampling for Inference (Mistral also used Temperature, but here no such support)
        # sort in descending order, (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # get cumulative sum of probs, (B, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # binary mask for which tokens are selected, (B, vocab_size)
        # (substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
        mask = probs_sum - probs_sort > p
        # zeros out all the probabilities of tokens that are not selected by the Top-P (tail probs to ignore)
        probs_sort[mask] = 0.0
        # re-distribute the probabilities so that they sum up to 1.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # sample a token (its index) from these top-p distribution based on their distribution
        next_token = torch.multinomial(probs_sort, num_samples=1)   # torch.multinomial: (B, 1)
        # get the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token)        # torch.gather: (B, 1)
        return next_token


def main():
    torch.manual_seed(0)
    allow_cuda = False
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"
    model = LLaMA.build(
        checkpoints_dir='llama-2-7b/',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=2,
        device=device
    )
    print("all good", model)


if __name__ == "__main__":
    main()
