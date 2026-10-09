"""Speculative decoding across the swarm (chain, not tree).

Each round:

1. The local drafter guesses `k` tokens after the accepted sequence.
2. One target call carries the last accepted token plus the `k` guesses through
   the whole swarm, giving the target's prediction after each of them.
3. The longest prefix of guesses the target agrees with is accepted, plus one
   token from the target itself: a correction at the first disagreement, or a
   bonus token when every guess was accepted. So each round produces 1 to k+1
   tokens for one trip through the swarm.
4. Rejected guesses are dropped implicitly: the next call starts at the first
   position that changed, and every stage overwrites from there.

Greedy: a guess is accepted when it equals the target's argmax, so the output
is the target's own greedy output.
Sampling: the acceptance rule of Leviathan et al. (2023) and Chen et al. (2023)
keeps the output distributed exactly as sampling from the target alone.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from myriad.client.drafters import Drafter
from myriad.client.sampling import GREEDY, Sampling, pick, probabilities
from myriad.model.pipeline import Pipeline


@dataclass
class Round:
    proposed: int  # guesses sent for checking
    accepted: int  # guesses the target agreed with
    draft_ms: float
    verify_ms: float  # the target call: the trip through the swarm


@dataclass
class SpeculationStats:
    rounds: list[Round] = field(default_factory=list)
    prefill_ms: float = 0.0
    total_ms: float = 0.0
    tokens: int = 0

    @property
    def acceptance_rate(self) -> float:
        proposed = sum(r.proposed for r in self.rounds)
        return sum(r.accepted for r in self.rounds) / proposed if proposed else 0.0

    @property
    def tokens_per_round(self) -> float:
        """Tokens produced per trip through the swarm (1.0 without speculation)."""
        return sum(r.accepted + 1 for r in self.rounds) / len(self.rounds) if self.rounds else 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / (self.total_ms / 1000) if self.total_ms else 0.0


def verify_greedy(guesses: list[int], target_logits: torch.Tensor) -> tuple[int, int]:
    """Number of guesses accepted, and the target's next token. `target_logits` is ``[len(guesses) + 1, vocab]``."""
    predicted = target_logits.argmax(dim=-1).tolist()
    n = 0
    while n < len(guesses) and guesses[n] == predicted[n]:
        n += 1
    return n, predicted[n]


def verify_sampled(
    guesses: list[int], draft_probs: torch.Tensor, target_probs: torch.Tensor, generator: torch.Generator | None = None
) -> tuple[int, int]:
    """Speculative sampling: accept guess i with probability min(1, p_i(x) / q_i(x)).

    On the first rejection, draw from the residual max(0, p_i - q_i), renormalised;
    if every guess is accepted, draw a bonus token from p_k.
    draft_probs: ``[k, vocab]`` (q), target_probs: ``[k + 1, vocab]`` (p).
    """
    for i, token in enumerate(guesses):
        p, q = target_probs[i, token], draft_probs[i, token]
        if torch.rand((), generator=generator) * q < p:  # u < p/q, without dividing by a possibly tiny q
            continue
        residual = (target_probs[i] - draft_probs[i]).clamp(min=0)
        total = residual.sum()
        # residual is zero only if p == q, where rejection cannot happen; guard against rounding anyway
        dist = residual / total if total > 0 else target_probs[i]
        return i, int(torch.multinomial(dist, 1, generator=generator))
    return len(guesses), int(torch.multinomial(target_probs[len(guesses)], 1, generator=generator))


def speculative_generate(
    target: Pipeline,
    drafter: Drafter,
    prompt: Sequence[int],
    max_new_tokens: int,
    k: int = 4,
    sampling: Sampling = GREEDY,
    stop_ids: Sequence[int] = (),
    generator: torch.Generator | None = None,
    on_round: Callable[[Round, list[int]], None] | None = None,
) -> tuple[list[int], SpeculationStats]:
    """Generate up to `max_new_tokens` tokens. Returns the new tokens and statistics.

    `on_round` is called after each round with its statistics and the tokens it produced,
    and once before the first round for the token that follows the prompt (with no guesses).
    """
    stats = SpeculationStats()
    t_start = time.perf_counter()
    logits = target.forward(list(prompt), start_pos=0)
    stats.prefill_ms = (time.perf_counter() - t_start) * 1000

    seq = list(prompt) + [pick(logits[-1], sampling, generator)]
    out = seq[len(prompt) :]
    done = out[-1] in stop_ids
    if on_round is not None:  # the prefill's token: no guesses involved, not counted as a round
        on_round(Round(proposed=0, accepted=0, draft_ms=0.0, verify_ms=stats.prefill_ms), out[:])

    while not done and len(out) < max_new_tokens:
        # A round yields up to k + 1 tokens; don't guess past the limit.
        k_now = min(k, max_new_tokens - len(out) - 1)

        t0 = time.perf_counter()
        guesses, draft_probs = drafter.propose(seq, k_now, sampling, generator) if k_now else ([], None)
        t1 = time.perf_counter()
        target_logits = target.forward([seq[-1], *guesses], start_pos=len(seq) - 1)
        t2 = time.perf_counter()

        if sampling.greedy:
            n, next_token = verify_greedy(guesses, target_logits)
        else:
            target_probs = probabilities(target_logits, sampling)
            if guesses:
                n, next_token = verify_sampled(guesses, draft_probs, target_probs, generator)
            else:
                n, next_token = 0, int(torch.multinomial(target_probs[0], 1, generator=generator))

        produced = []
        for token in [*guesses[:n], next_token]:
            produced.append(token)
            if token in stop_ids:
                done = True
                break
        seq += produced
        out += produced

        r = Round(proposed=len(guesses), accepted=n, draft_ms=(t1 - t0) * 1000, verify_ms=(t2 - t1) * 1000)
        stats.rounds.append(r)
        if on_round is not None:
            on_round(r, produced)

    stats.total_ms = (time.perf_counter() - t_start) * 1000
    stats.tokens = len(out)
    return out, stats
