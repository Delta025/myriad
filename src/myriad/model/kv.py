"""Per-session key/value cache for one stage.

Unlike Transformers' sliding-window cache, this keeps every position for every
layer, so `truncate(n)` is always an exact slice, including after the sliding
window has been passed. That is what rejected draft tokens need. The sliding
window is applied by the attention mask instead (see `masks.py`).
"""

import torch


class KVCache:
    def __init__(self) -> None:
        self._layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.length = 0  # number of positions cached, set by the stage after each forward

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V ``[batch, heads, new, head_dim]`` and return the full history.

        Same signature as Transformers' ``Cache.update``, which the attention layers call.
        """
        if layer_idx in self._layers:
            past_k, past_v = self._layers[layer_idx]
            key_states = torch.cat([past_k, key_states], dim=-2)
            value_states = torch.cat([past_v, value_states], dim=-2)
        self._layers[layer_idx] = (key_states, value_states)
        return key_states, value_states

    def truncate(self, length: int) -> None:
        """Forget every position from `length` on."""
        if length < 0:
            raise ValueError(f"cannot truncate to {length}")
        if length >= self.length:
            return
        self._layers = {i: (k[:, :, :length], v[:, :, :length]) for i, (k, v) in self._layers.items()}
        self.length = length

    def layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._layers[layer_idx]
