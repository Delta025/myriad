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


class MTPDrafter:
    """Gemma 4's official multi-token-prediction drafter ("assistant"), run on the client.

    The drafter is a 4-layer model that has no K/V of its own. For each guess it reads:
    - the target's embedding of the last token;
    - the target's final hidden state at the position before it;
    - the target's cached K/V of its last sliding and last global layer (on 31B,
      layers 58 and 59), which the client holds when it runs the last layers.
    So it needs no model of its own and nothing from the peers. It feeds these to
    Transformers' `Gemma4AssistantForCausalLM` the way Transformers' own candidate
    generator does, with one difference: it passes only the K/V of accepted positions.
    """

    def __init__(self, target, assistant):
        """`target`: the target pipeline (local or remote) whose state the drafter reads.
        `assistant`: a loaded `Gemma4AssistantForCausalLM` on the client's device."""
        from myriad.model.split import last_layer_of_each_type

        self.target, self.assistant = target, assistant.eval()
        self.source_layers = last_layer_of_each_type(target.embedder.config)
        self.device = next(assistant.parameters()).device

    @classmethod
    def from_pretrained(cls, target, name_or_path: str, device="cuda", dtype=torch.bfloat16) -> "MTPDrafter":
        from transformers import Gemma4AssistantForCausalLM

        return cls(target, Gemma4AssistantForCausalLM.from_pretrained(name_or_path, dtype=dtype).to(device))

    @torch.inference_mode()
    def propose(self, seq, k, sampling, generator=None):
        target = self.target
        p = len(seq) - 1  # position of the newest token, which the target has not processed yet
        row = p - 1 - target.last_start_pos  # the target's hidden state at position p - 1
        if target.last_hidden is None or not 0 <= row < target.last_hidden.shape[0]:
            raise RuntimeError("the target's last call does not cover the position before the newest token")
        hidden = target.last_hidden[row].to(self.device).reshape(1, 1, -1)
        shared_kv = {}
        for layer_type, layer in self.source_layers.items():
            keys, values = target.cached_kv(layer)
            shared_kv[layer_type] = (keys[:, :, :p].to(self.device), values[:, :, :p].to(self.device))
        position_ids = torch.tensor([[p]], device=self.device)  # fixed for every guess, as in Transformers

        token, tokens, dists = seq[-1], [], []
        for _ in range(k):
            embedding = target.embedder.token_embedding(torch.tensor([[token]]))
            inputs = torch.cat([embedding.to(self.device, hidden.dtype), hidden], dim=-1)
            out = self.assistant(inputs_embeds=inputs, position_ids=position_ids, shared_kv_states=shared_kv)
            logits = out.logits[0, -1]
            if sampling.greedy:
                token = int(logits.argmax())
            else:
                dist = probabilities(logits[None], sampling)[0]
                token = int(torch.multinomial(dist.cpu(), 1, generator=generator))
                dists.append(dist.cpu())
            tokens.append(token)
            hidden = out.last_hidden_state  # the drafter's guess of the target's next hidden state
        return tokens, (torch.stack(dists) if dists else None)
