"""A contiguous range of decoder layers that can run on its own.

The maths is Transformers' own `Gemma4TextDecoderLayer`. This module owns only
what changes when the model is split: which weights are loaded, the KV cache,
positions, masks, rotary embeddings, and the K/V hand-off between KV-shared layers.
"""

import torch
from torch import nn
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextRotaryEmbedding

from myriad.model.checkpoint import Checkpoint
from myriad.model.kv import KVCache
from myriad.model.masks import build_masks, sliding_kv_offset


class _WindowedCache:
    """What the attention layers see: the stage's cache, with sliding layers given only their window.

    The cache itself keeps every position so it can always be truncated.
    """

    def __init__(self, cache: KVCache, sliding_layers: set[int], offset: int):
        self.cache, self.sliding_layers, self.offset = cache, sliding_layers, offset

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        keys, values = self.cache.update(key_states, value_states, layer_idx)
        if layer_idx in self.sliding_layers:
            return keys[:, :, self.offset :], values[:, :, self.offset :]
        return keys, values


class Stage(nn.Module):
    def __init__(self, config, start: int, end: int, state_dict: dict[str, torch.Tensor], device: torch.device | str):
        """Build layers ``[start, end)`` from `state_dict`, whose keys look like ``layers.{i}.mlp.up_proj.weight``."""
        super().__init__()
        self.config = config
        self.start, self.end = start, end
        self.device = torch.device(device)

        # Build on the meta device so no memory is spent on random init, then adopt the loaded tensors.
        with torch.device("meta"):
            self.layers = nn.ModuleList(Gemma4TextDecoderLayer(config, i) for i in range(start, end))
        local = {}
        for name, tensor in state_dict.items():
            idx, rest = name.removeprefix("layers.").split(".", 1)
            if start <= int(idx) < end:
                local[f"{int(idx) - start}.{rest}"] = tensor
        # Released checkpoints still carry K/V weights for KV-shared layers, which have no use for them.
        missing, _unexpected = self.layers.load_state_dict(local, strict=False, assign=True)
        if missing:
            raise ValueError(f"checkpoint is missing weights for layers {start}-{end - 1}: {missing[:5]}")

        # Compute the rotary frequencies on CPU, then move them, as Transformers' from_pretrained does:
        # CUDA rounds one of E2B's global-layer frequencies differently.
        self.rotary = Gemma4TextRotaryEmbedding(config).to(self.device)
        self.layer_types = [config.layer_types[i] for i in range(start, end)]
        self.sliding_layers = {i for i in range(start, end) if config.layer_types[i] == "sliding_attention"}

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Checkpoint, start: int, end: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.bfloat16
    ) -> "Stage":
        names = [n for i in range(start, end) for n in checkpoint.names(f"layers.{i}.")]
        return cls(checkpoint.text_config(), start, end, checkpoint.load(names, device, dtype), device)

    @torch.inference_mode()
    def forward(
        self,
        hidden: torch.Tensor,
        start_pos: int,
        cache: KVCache,
        per_layer_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run positions ``start_pos .. start_pos + n`` through the layers.

        hidden: ``[1, n, hidden_size]``.
        per_layer_inputs: ``[1, n, end - start, ple_dim]`` on models with per-layer embeddings, else None.
        Anything cached from `start_pos` on is dropped first, so re-sending a position
        (for example after rejected draft tokens) overwrites it.
        """
        cache.truncate(start_pos)
        if cache.length != start_pos:
            raise ValueError(f"stage {self.start}-{self.end - 1} has {cache.length} positions cached, got start_pos {start_pos}")

        hidden = hidden.to(self.device)
        if per_layer_inputs is not None:
            per_layer_inputs = per_layer_inputs.to(self.device)
        n_new = hidden.shape[1]
        position_ids = torch.arange(start_pos, start_pos + n_new, device=self.device)[None]
        masks = build_masks(start_pos, n_new, self.config.sliding_window, self.device)
        position_embeddings = {t: self.rotary(hidden, position_ids, t) for t in set(self.layer_types)}
        shared_kv_states: dict = {}  # filled by KV-source layers, read by KV-shared layers of this stage
        windowed = _WindowedCache(cache, self.sliding_layers, sliding_kv_offset(start_pos, self.config.sliding_window))

        for j, (layer, layer_type) in enumerate(zip(self.layers, self.layer_types)):
            hidden = layer(
                hidden,
                per_layer_inputs[:, :, j, :] if per_layer_inputs is not None else None,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[layer_type],
                attention_mask=masks[layer_type],
                position_ids=position_ids,
                past_key_values=windowed,
            )

        cache.length = start_pos + n_new
        return hidden
