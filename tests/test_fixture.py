"""M1: the tiny checkpoints load in stock Transformers and behave like a usable model."""

import torch
from safetensors import safe_open
from transformers import Gemma4ForConditionalGeneration

from myriad.testing import TINY_VOCAB


def test_checkpoint_uses_release_key_layout(tiny_checkpoint):
    with safe_open(tiny_checkpoint / "model.safetensors", "pt") as f:
        keys = set(f.keys())
    assert "model.language_model.embed_tokens.weight" in keys
    assert "model.language_model.layers.0.self_attn.q_proj.weight" in keys


def test_hf_forward_runs_and_ranks_tokens(tiny_checkpoint):
    model = Gemma4ForConditionalGeneration.from_pretrained(tiny_checkpoint, dtype=torch.float32).eval()
    ids = torch.randint(0, TINY_VOCAB, (1, 12), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    assert logits.shape == (1, 12, TINY_VOCAB)
    assert torch.isfinite(logits).all()
    # Greedy equivalence tests need a clear winner, not near-ties decided by rounding.
    top2 = logits.topk(2, dim=-1).values
    assert (top2[..., 0] - top2[..., 1]).median() > 0.05
