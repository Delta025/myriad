"""Turning logits into tokens: greedy, or sampling with temperature, top-k and top-p.

Speculative sampling needs the actual probabilities the draft sampled from
and the target's probabilities under the same settings, so the transform is a
separate function from the draw.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.0  # 0 means greedy
    top_k: int = 0  # 0 means no limit
    top_p: float = 1.0

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


GREEDY = Sampling()


def probabilities(logits: torch.Tensor, sampling: Sampling) -> torch.Tensor:
    """Sampling distribution ``[n, vocab]`` (float32) for logits ``[n, vocab]``. Not for greedy."""
    logits = logits.float() / sampling.temperature
    if sampling.top_k:
        kth = logits.topk(min(sampling.top_k, logits.shape[-1]), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = logits.softmax(dim=-1)
    if sampling.top_p < 1.0:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        # keep the smallest set of top tokens whose mass reaches top_p (always at least one)
        drop = sorted_probs.cumsum(dim=-1) - sorted_probs >= sampling.top_p
        sorted_probs = sorted_probs.masked_fill(drop, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def pick(logits: torch.Tensor, sampling: Sampling, generator: torch.Generator | None = None) -> int:
    """One token from a single row of logits ``[vocab]``."""
    if sampling.greedy:
        return int(logits.argmax())
    return int(torch.multinomial(probabilities(logits[None], sampling)[0], 1, generator=generator))
