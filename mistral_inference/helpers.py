import torch
import torch.nn.functional as F
import math


def simple_scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute the scaled dot-product attention.
    Args:
        query (torch.Tensor): Queries (Q) of shape (batch_size, num_heads, seq_length, head_dim)
        key (torch.Tensor): Keys (K) of shape (batch_size, num_heads, seq_length, head_dim)
        value (torch.Tensor): Values (V) of shape (batch_size, num_heads, seq_length, head_dim)
        mask (torch.Tensor, optional): Mask for the attention. Broadcastable to the shape (batch_size, 1, seq_length, seq_length). Defaults to None.
    Returns:
        torch.Tensor: The result of the attention mechanism, of shape
                      (batch_size, num_heads, seq_length, head_dim)
    """
    # query, key, value shape = (1, B, n_heads, head_dim) => (B, n_heads_q, 1, Head_Dim)
    seq_length, B, n_heads, head_dim = query.shape
    print("seq_length, B, n_heads, head_dim", seq_length, B, n_heads, head_dim)
    query = query.view(B, n_heads, -1, head_dim)
    key = key.view(B, n_heads, -1, head_dim)
    value = value.view(B, n_heads, -1, head_dim)

    scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(head_dim)
    print("score shape", scores.shape)
    scores = F.softmax(scores.float(), dim=-1).type_as(query)
    print("score shape", scores.shape)
    output = torch.matmul(scores, value)
    output = output.transpose(1, 2).contiguous()
    output = output.view(B, seq_length, -1)     # (B, 1, n_heads * head_dim)

    print("output shape XXX", output.shape)
    print("mask._window_size", mask)

    #
    #
    # print("new QKV shape", query.shape, key.shape, value.shape)
    # print("mask._window_size", mask._window_size)
    # print("mask.k_seqinfo", mask.k_seqinfo)
    # print("mask.q_seqinfo", mask.q_seqinfo)

    return output





    # Calculate the scores (Q * K^T) scaled by the dimension's square root
    d_k = query.size(-1)  # Extract the dimensionality of the keys/queries
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=query.dtype))

    # Apply mask, setting masked positions to a large negative value
    # so that they result in a near-zero probability after the softmax
    if mask is not None:
        print(type(mask==0), (mask==0).shape)
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Apply softmax to get the attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Multiply the attention weights with the values (V)
    output = torch.matmul(attention_weights, value)

    return output
