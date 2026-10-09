"""The `Pipeline` interface that generation and speculative decoding are written against.

A pipeline turns a chunk of token ids at a given position into logits for each
of them. It keeps the state of earlier positions, and sending a chunk at
``start_pos`` discards anything it held from that position on. That is how
rejected draft tokens are rolled back.

- `LocalPipeline`: every stage in this process (also the base for the client's local ends).
- `HFReference`: stock Transformers, recomputing from scratch; the ground truth in tests.
"""

from collections.abc import Sequence
from typing import Protocol

import torch

from myriad.model.checkpoint import Checkpoint
from myriad.model.ends import Embedder, Head
from myriad.model.kv import KVCache
from myriad.model.split import Split, validate_split
from myriad.model.stage import Stage


def as_ids(token_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
    """Token ids as a ``[1, n]`` long tensor."""
    ids = torch.as_tensor(token_ids, dtype=torch.long)
    return ids.reshape(1, -1)


class Pipeline(Protocol):
    length: int

    def forward(self, token_ids: Sequence[int] | torch.Tensor, start_pos: int) -> torch.Tensor:
        """Logits ``[n, vocab]`` (float32, on CPU) for `token_ids` placed at ``start_pos .. start_pos + n``."""
        ...

    def truncate(self, length: int) -> None: ...


class LocalPipeline:
    def __init__(self, embedder: Embedder, stages: list[Stage], head: Head):
        self.embedder, self.stages, self.head = embedder, stages, head
        self.caches = [KVCache() for _ in stages]
        self.length = 0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Checkpoint | str,
        split: Split | None = None,
        dtype: torch.dtype = torch.bfloat16,
        stage_devices: Sequence[str] | str = "cpu",
        ends_device: str = "cpu",
        ple_device: str | None = None,
    ) -> "LocalPipeline":
        """Load a pipeline; `split` defaults to one stage holding every layer.

        `ple_device` (default: `ends_device`) is where the per-layer-embedding table of E2B/E4B lives.
        """
        if not isinstance(checkpoint, Checkpoint):
            checkpoint = Checkpoint(checkpoint)
        config = checkpoint.text_config()
        split = split or [(0, config.num_hidden_layers)]
        validate_split(config, split)
        if isinstance(stage_devices, str):
            stage_devices = [stage_devices] * len(split)

        embedder = Embedder.from_checkpoint(checkpoint, ends_device, dtype, ple_device)
        head = Head.from_checkpoint(checkpoint, ends_device, dtype, embedder=embedder)
        stages = [Stage.from_checkpoint(checkpoint, a, b, dev, dtype) for (a, b), dev in zip(split, stage_devices)]
        return cls(embedder, stages, head)

    def forward(self, token_ids, start_pos: int) -> torch.Tensor:
        if start_pos > self.length:
            raise ValueError(f"start_pos {start_pos} is past the {self.length} positions processed")
        ids = as_ids(token_ids)
        hidden, per_layer_inputs = self.embedder(ids)
        for stage, cache in zip(self.stages, self.caches):
            stage_inputs = per_layer_inputs[:, :, stage.start : stage.end] if per_layer_inputs is not None else None
            hidden = stage(hidden, start_pos, cache, stage_inputs)
        self.length = start_pos + ids.shape[1]
        return self.head(hidden)[0].float().cpu()

    def truncate(self, length: int) -> None:
        for cache in self.caches:
            cache.truncate(length)
        self.length = min(self.length, length)


class HFReference:
    """Stock Transformers model behind the `Pipeline` interface.

    Recomputes the whole sequence on every call, with no cache, so it shares no
    state-handling code with the pipelines it checks.
    """

    def __init__(self, model):
        self.model = model.eval()
        self.tokens: list[int] = []

    @property
    def length(self) -> int:
        return len(self.tokens)

    @torch.inference_mode()
    def forward(self, token_ids, start_pos: int) -> torch.Tensor:
        new = as_ids(token_ids)[0].tolist()
        self.tokens = self.tokens[:start_pos] + new
        logits = self.model(input_ids=torch.tensor([self.tokens], device=self.model.device), use_cache=False).logits
        return logits[0, -len(new) :].float().cpu()

    def truncate(self, length: int) -> None:
        self.tokens = self.tokens[:length]
