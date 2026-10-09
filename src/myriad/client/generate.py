"""Plain (non-speculative) generation over any `Pipeline`: one target call per token."""

from collections.abc import Callable, Sequence

import torch

from myriad.client.sampling import GREEDY, Sampling, pick
from myriad.model.pipeline import Pipeline


def generate(
    pipeline: Pipeline,
    prompt: Sequence[int],
    max_new_tokens: int,
    sampling: Sampling = GREEDY,
    stop_ids: Sequence[int] = (),
    generator: torch.Generator | None = None,
    on_token: Callable[[int], None] | None = None,
) -> list[int]:
    """Generate up to `max_new_tokens` tokens. Returns only the new tokens. `on_token` sees each one."""
    logits = pipeline.forward(list(prompt), start_pos=0)
    generated: list[int] = []
    while True:
        token = pick(logits[-1], sampling, generator)
        generated.append(token)
        if on_token is not None:
            on_token(token)
        if token in stop_ids or len(generated) == max_new_tokens:
            return generated
        logits = pipeline.forward([token], start_pos=len(prompt) + len(generated) - 1)


def greedy_generate(
    pipeline: Pipeline, prompt: Sequence[int], max_new_tokens: int, stop_ids: Sequence[int] = ()
) -> list[int]:
    return generate(pipeline, prompt, max_new_tokens, GREEDY, stop_ids)
