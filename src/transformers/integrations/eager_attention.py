from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _attention_compute_scores(
    query: torch.Tensor,
    key: torch.Tensor,
) -> torch.Tensor:
    nh_q = query.shape[1]
    nh_k = key.shape[1]
    # - query: (bs, nh_q, T_q, hs)
    # - key: (bs, nh_k, T_k, hs)
    q_per_kv = nh_q // nh_k
    key_transposed = key.mT  # (bs, nh_k, hs, T_k)
    if q_per_kv == 1:
        return query @ key_transposed
    else:
        assert q_per_kv > 1
        if nh_k > 1:
            q_shape = query.shape[:1] + (nh_k, q_per_kv) + query.shape[2:]
            _query = query.view(*q_shape)
            key_transposed = key_transposed.unsqueeze(2)
        else:
            _query = query
        # At this point:
        # - _query: (bs, nh_k, q_per_kv, T_q, hs)
        # - key_transposed: (bs, nh_k, 1, hs, T_k)
        # - scores: (bs, nh_k, q_per_kv, T_q, T_k) -> (bs, nh_q, T_q, T_k)
        scores = torch.matmul(_query, key_transposed)
        s_shape = query.shape[:-1] + (key.shape[2],)
        return scores.view(*s_shape)


def _attention_compute_weighted_values(
    scores: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    nh_q = scores.shape[1]
    nh_k = value.shape[1]
    # - scores: (bs, nh_q, T_q, T_k)
    # - value: (bs, nh_k, T_k, hs)
    q_per_kv = nh_q // nh_k
    if q_per_kv == 1:
        return scores @ value
    else:
        if nh_k > 1:
            s_shape = scores.shape[:1] + (nh_k, q_per_kv) + scores.shape[2:]
            _scores = scores.view(*s_shape)
            _value = value.unsqueeze(2)
        else:
            _scores = scores
            _value = value
        # At this point:
        # - _scores: (bs, nh_k, q_per_kv, T_q, T_k)
        # - _value: (bs, nh_k, 1, T_k, hs)
        # - result: (bs, nh_k, q_per_kv, T_q, hs) -> (bs, nh_q, T_q, hs)
        result = torch.matmul(_scores, _value)
        r_shape = scores.shape[:-1] + (value.shape[-1],)
        return result.view(*r_shape)


def eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    `query` has shape `(batch, num_heads, q_len, head_dim)`, while `key`,
    `value` have shape `(batch, num_key_value_groups, kv_len, head_dim)`. Here,
    `num_key_value_groups <= num_heads` and
    `num_heads % num_key_value_groups == 0`.

    """
    assert query.ndim == key.ndim == value.ndim == 4
    _, num_heads, q_len, _ = query.shape
    _, num_key_value_groups, kv_len, _ = key.shape
    assert query.shape[0] == key.shape[0] == value.shape[0]  # batch_size
    assert value.shape[1] == num_key_value_groups and value.shape[2] == kv_len
    assert num_heads % num_key_value_groups == 0 and num_heads >= num_key_value_groups

    attn_weights = _attention_compute_scores(query, key) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :kv_len]
        attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = _attention_compute_weighted_values(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    # attn_output: (batch, q_len, num_heads, head_dim)
    # attn_weights: (batch, num_heads, q_len, kv_len)

    return attn_output, attn_weights
