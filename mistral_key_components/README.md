# Mistral_From_Scratch - Key Components
More details about components used in Mistral.

Note that as the official Mistral_inference requires xformer, which I cannot use,
as I am using a M2 Chip with no server, 
some code cannot be easily tested. 

As a result, I started from LLaMA model, which also contains many key components
used in Mistral (including RoPE, RMSNorm, GQA, KVCache, SiLU). 

However, there are some other key components in Mistral that is missing in LLaMA 
(including MoE, LoRA, and most importantly the sliding window attention + rolling buffer cache).
Currently, I am trying to implement these myself for demo, which can allow testing 
without relying on xformer. 
