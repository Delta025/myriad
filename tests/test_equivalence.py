"""M2: a split pipeline gives the same logits and greedy text as stock Transformers.

Tiny random-weight models in float32, with sequences longer than the sliding window (4).
"""

import pytest
import torch
from transformers import Gemma4ForConditionalGeneration

from myriad.client.generate import greedy_generate
from myriad.model.checkpoint import Checkpoint
from myriad.model.pipeline import HFReference, LocalPipeline
from myriad.testing import TINY_VOCAB

SPLITS = {
    "dense": [
        [(0, 8)],
        [(0, 1), (1, 8)],
        [(0, 3), (3, 4), (4, 7), (7, 8)],
        [(i, i + 1) for i in range(8)],
    ],
    # layers 2-7 share K/V and must stay together
    "ple": [
        [(0, 8)],
        [(0, 2), (2, 8)],
        [(0, 1), (1, 2), (2, 8)],
    ],
}
# chunk sizes: a prefill, single-token steps, and multi-token chunks like draft verification
SCHEDULE = [7, 1, 1, 1, 4, 1, 3, 2]
ATOL = 1e-4


def variant_of(checkpoint):
    config = Checkpoint(checkpoint).text_config()
    return "ple" if config.hidden_size_per_layer_input else "dense"


def all_splits(tiny_checkpoint):
    return SPLITS[variant_of(tiny_checkpoint)]


def reference(checkpoint):
    model = Gemma4ForConditionalGeneration.from_pretrained(checkpoint, dtype=torch.float32, attn_implementation="sdpa")
    return HFReference(model)


def tokens(n, seed=0):
    return torch.randint(0, TINY_VOCAB, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def run_schedule(pipeline, ids, schedule):
    out, pos = [], 0
    for size in schedule:
        out.append(pipeline.forward(ids[pos : pos + size], start_pos=pos))
        pos += size
    return torch.cat(out)


def test_chunked_logits_match_hf(tiny_checkpoint):
    ids = tokens(sum(SCHEDULE))
    expected = reference(tiny_checkpoint).forward(ids, start_pos=0)
    for split in all_splits(tiny_checkpoint):
        pipe = LocalPipeline.from_checkpoint(str(tiny_checkpoint), split, dtype=torch.float32)
        got = run_schedule(pipe, ids, SCHEDULE)
        torch.testing.assert_close(got, expected, atol=ATOL, rtol=0, msg=f"split {split}")


def test_rollback_matches_never_seeing_junk(tiny_checkpoint):
    ids, junk = tokens(16), tokens(6, seed=1)
    expected = reference(tiny_checkpoint).forward(ids, start_pos=0)
    for split in all_splits(tiny_checkpoint):
        pipe = LocalPipeline.from_checkpoint(str(tiny_checkpoint), split, dtype=torch.float32)
        pipe.forward(ids[:9], start_pos=0)
        pipe.forward(junk, start_pos=9)  # rejected drafts, crossing the window boundary
        got_a = pipe.forward(ids[9:12], start_pos=9)  # implicit rollback via start_pos
        pipe.forward(junk[:3], start_pos=12)
        pipe.truncate(12)  # explicit rollback
        got_b = pipe.forward(ids[12:], start_pos=12)
        torch.testing.assert_close(torch.cat([got_a, got_b]), expected[9:], atol=ATOL, rtol=0, msg=f"split {split}")


def test_greedy_text_matches_hf_generate(tiny_checkpoint):
    prompt = tokens(6, seed=2)
    model = Gemma4ForConditionalGeneration.from_pretrained(tiny_checkpoint, dtype=torch.float32, attn_implementation="sdpa")
    out = model.generate(torch.tensor([prompt]), max_new_tokens=24, do_sample=False, eos_token_id=None, pad_token_id=0)
    expected = out[0, len(prompt) :].tolist()
    for split in all_splits(tiny_checkpoint):
        pipe = LocalPipeline.from_checkpoint(str(tiny_checkpoint), split, dtype=torch.float32)
        assert greedy_generate(pipe, prompt, max_new_tokens=24) == expected, f"split {split}"


def test_stage_loads_only_its_layers(tiny_checkpoint):
    from myriad.model.stage import Stage

    stage = Stage.from_checkpoint(Checkpoint(tiny_checkpoint), 1, 2, dtype=torch.float32)
    assert len(stage.layers) == 1
    assert all(p.device.type == "cpu" for p in stage.parameters())


def test_start_pos_gap_is_rejected(tiny_checkpoint):
    pipe = LocalPipeline.from_checkpoint(str(tiny_checkpoint), dtype=torch.float32)
    pipe.forward(tokens(3), start_pos=0)
    with pytest.raises(ValueError):
        pipe.forward(tokens(1), start_pos=5)


def test_per_layer_embeddings_read_from_disk_match(tiny_checkpoint):
    from myriad.model.ends import Embedder

    ckpt = Checkpoint(tiny_checkpoint)
    if not ckpt.text_config().hidden_size_per_layer_input:
        pytest.skip("no per-layer embeddings")
    ids = torch.tensor([[5, 9, 5, 300, 0]])
    in_memory = Embedder.from_checkpoint(ckpt, "cpu", torch.float32)(ids)
    from_disk = Embedder.from_checkpoint(ckpt, "cpu", torch.float32, ple_device="disk")(ids)
    assert torch.equal(in_memory[0], from_disk[0]) and torch.equal(in_memory[1], from_disk[1])
