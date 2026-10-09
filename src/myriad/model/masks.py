"""Attention masks for a chunk of new positions.

Same rules as Transformers' `masking_utils`: a global layer sees every earlier
position and itself; a sliding layer sees the last `sliding_window` positions,
itself included.

Sliding layers attend only over the keys from `sliding_kv_offset` on (the stage
trims them), the keys Transformers' sliding-window cache would hold. Attending
over the whole history with masked-out keys is the same maths but different
bf16 sums: on E2B past the window, that alone moved logits by up to 0.8.

A mask is replaced by None where Transformers does the same
(`_ignore_causal_mask_sdpa`). This lets SDPA use its causal and flash kernels
and keeps the kernel choice the same as Transformers. On the 3080 it made no
difference to the results.
"""

import torch


def sliding_kv_offset(start_pos: int, sliding_window: int) -> int:
    """First position a chunk starting at `start_pos` can see in a sliding layer."""
    return max(start_pos - sliding_window + 1, 0)


def _sdpa_can_skip(n_new: int, kv_len: int, start_pos: int, window: int | None = None) -> bool:
    # Transformers' rule without padding: with no mask, SDPA attends to every key when
    # n_new == 1 and is exactly causal (`is_causal=True`) when the chunk starts the cache.
    if window is not None and kv_len >= window:
        return False
    return n_new == 1 or kv_len == n_new or start_pos == 0


def build_masks(
    start_pos: int, n_new: int, sliding_window: int, device: torch.device | str = "cpu"
) -> dict[str, torch.Tensor | None]:
    """Boolean masks (True = may attend), keyed by layer type; None means plain causal, left to SDPA.

    full_attention: ``[1, 1, n_new, start_pos + n_new]`` over keys ``0 ..``.
    sliding_attention: ``[1, 1, n_new, start_pos + n_new - offset]`` over keys ``offset ..``.
    """
    q = torch.arange(start_pos, start_pos + n_new, device=device)[:, None]
    k_full = torch.arange(start_pos + n_new, device=device)[None, :]
    k_sliding = k_full[:, sliding_kv_offset(start_pos, sliding_window) :]
    causal = k_full <= q
    sliding = (k_sliding <= q) & (k_sliding > q - sliding_window)
    kv_full, kv_sliding = k_full.shape[1], k_sliding.shape[1]
    return {
        "full_attention": None if _sdpa_can_skip(n_new, kv_full, start_pos) else causal[None, None],
        "sliding_attention": None if _sdpa_can_skip(n_new, kv_sliding, start_pos, sliding_window) else sliding[None, None],
    }
