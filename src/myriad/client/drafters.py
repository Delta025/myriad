"""Draft models: cheap local guesses of the next few tokens, checked by the target in one call.

A drafter keeps its own state between rounds. It is handed the whole accepted
sequence each time and works out what it hasn't seen yet, so rejected guesses
are rolled back the same way as on the target: by re-sending from the first
position that changed.
"""

from collections.abc import Sequence
from typing import Protocol

import torch

from myriad.client.sampling import Sampling, probabilities
from myriad.model.pipeline import Pipeline


class Drafter(Protocol):
    def propose(
        self, seq: Sequence[int], k: int, sampling: Sampling, generator: torch.Generator | None = None
    ) -> tuple[list[int], torch.Tensor | None]:
        """Guess the `k` tokens that follow `seq`.

        Returns the tokens and, when sampling, the distributions ``[k, vocab]`` they were drawn from.
        """
        ...


def _common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class ModelDrafter:
    """Drafts with any `Pipeline`, typically a small model of the same family (Gemma 4 E2B for 31B)."""

    def __init__(self, pipeline: Pipeline):
        self.pipe = pipeline
        self.fed: list[int] = []  # tokens whose positions the draft pipeline has cached

    def propose(self, seq, k, sampling, generator=None):
        # Re-send from the first position that differs from what was fed; always at least the
        # last token, whose logits we need.
        start = min(_common_prefix(self.fed, seq), len(seq) - 1)
        logits = self.pipe.forward(list(seq[start:]), start_pos=start)[-1]
        self.fed = list(seq)

        tokens, dists = [], []
        for i in range(k):
            if sampling.greedy:
                token = int(logits.argmax())
            else:
                dist = probabilities(logits[None], sampling)[0]
                token = int(torch.multinomial(dist, 1, generator=generator))
                dists.append(dist)
            tokens.append(token)
            if i < k - 1:  # the last guess is never fed: the target checks it
                logits = self.pipe.forward([token], start_pos=len(self.fed))[-1]
                self.fed.append(token)
        return tokens, (torch.stack(dists) if dists else None)
