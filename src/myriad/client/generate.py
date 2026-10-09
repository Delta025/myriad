"""Plain (non-speculative) generation over any `Pipeline`."""

from collections.abc import Sequence

from myriad.model.pipeline import Pipeline


def greedy_generate(
    pipeline: Pipeline, prompt: Sequence[int], max_new_tokens: int, stop_ids: Sequence[int] = ()
) -> list[int]:
    """Generate up to `max_new_tokens` tokens, one round trip each. Returns only the new tokens."""
    logits = pipeline.forward(list(prompt), start_pos=0)
    generated: list[int] = []
    while True:
        token = int(logits[-1].argmax())
        generated.append(token)
        if token in stop_ids or len(generated) == max_new_tokens:
            return generated
        logits = pipeline.forward([token], start_pos=len(prompt) + len(generated) - 1)
