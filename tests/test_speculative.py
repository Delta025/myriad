"""M4: speculative decoding gives the target's own output: identical text (greedy), identical distribution (sampling)."""

import pytest
import torch

from myriad.client.drafters import ModelDrafter
from myriad.client.generate import generate, greedy_generate
from myriad.client.remote import RemotePipeline
from myriad.client.sampling import Sampling
from myriad.client.speculative import speculative_generate, verify_sampled
from myriad.model.checkpoint import Checkpoint
from myriad.model.pipeline import LocalPipeline
from myriad.testing import TINY_VOCAB, ThreadedSwarm, make_tiny_checkpoint


def tokens(n, seed=0):
    return torch.randint(0, TINY_VOCAB, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def pipeline(checkpoint):
    return LocalPipeline.from_checkpoint(str(checkpoint), dtype=torch.float32)


@pytest.fixture(scope="module")
def other_model(tmp_path_factory):
    """An unrelated random model with the same vocabulary: a draft that is often wrong."""
    return make_tiny_checkpoint(tmp_path_factory.mktemp("tiny-other"), "dense", seed=1)


# --- the acceptance rule on its own ---


def test_sampled_acceptance_reproduces_the_target_distribution():
    """Two tokens via speculative rounds, with position-only distributions: their joint must be P0 x P1."""
    # rows: positions 0..3 (a round starting at position 1 with k=2 reads up to position 3)
    P = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4], [0.2, 0.2, 0.6]])  # target
    Q = torch.tensor([[0.1, 0.1, 0.8], [0.6, 0.2, 0.2], [0.3, 0.4, 0.3], [0.5, 0.4, 0.1]])  # draft, deliberately different
    g = torch.Generator().manual_seed(0)
    counts = torch.zeros(3, 3)
    n_trials, k = 30000, 2
    for _ in range(n_trials):
        out = []
        while len(out) < 2:
            pos = len(out)
            guesses = [int(torch.multinomial(Q[pos + i], 1, generator=g)) for i in range(k)]
            n, nxt = verify_sampled(guesses, Q[pos : pos + k], P[pos : pos + k + 1], g)
            out += [*guesses[:n], nxt]
        counts[out[0], out[1]] += 1
    expected = torch.outer(P[0], P[1])
    assert (counts / n_trials - expected).abs().max() < 0.012


# --- end to end on tiny models ---


@pytest.mark.parametrize("k", [1, 3, 5])
def test_greedy_matches_plain_greedy_with_perfect_draft(tiny_checkpoint, k):
    prompt = tokens(6, seed=3)
    expected = greedy_generate(pipeline(tiny_checkpoint), prompt, 20)
    out, stats = speculative_generate(pipeline(tiny_checkpoint), ModelDrafter(pipeline(tiny_checkpoint)), prompt, 20, k=k)
    assert out == expected
    assert stats.acceptance_rate == 1.0
    assert len(stats.rounds) < 20  # fewer trips than tokens


@pytest.mark.parametrize("k", [2, 4])
def test_greedy_matches_plain_greedy_with_wrong_draft(tiny_checkpoint, other_model, k):
    prompt = tokens(6, seed=4)
    expected = greedy_generate(pipeline(tiny_checkpoint), prompt, 24)
    out, stats = speculative_generate(pipeline(tiny_checkpoint), ModelDrafter(pipeline(other_model)), prompt, 24, k=k)
    assert out == expected
    assert stats.acceptance_rate < 0.5  # an unrelated model guesses badly, and rollback is exercised


def test_stop_token_and_length_limit(tiny_checkpoint):
    prompt = tokens(5, seed=5)
    full = greedy_generate(pipeline(tiny_checkpoint), prompt, 30)
    stop = full[7]
    out, _ = speculative_generate(pipeline(tiny_checkpoint), ModelDrafter(pipeline(tiny_checkpoint)), prompt, 30, k=4, stop_ids=[stop])
    assert out == full[: full.index(stop) + 1]
    for limit in (1, 2, 9):
        out, _ = speculative_generate(pipeline(tiny_checkpoint), ModelDrafter(pipeline(tiny_checkpoint)), prompt, limit, k=4)
        assert out == full[:limit]


def test_sampling_with_perfect_draft_accepts_nearly_everything(tiny_checkpoint):
    sampling = Sampling(temperature=0.8, top_p=0.95)
    prompt = tokens(6, seed=6)
    g = torch.Generator().manual_seed(0)
    out, stats = speculative_generate(
        pipeline(tiny_checkpoint), ModelDrafter(pipeline(tiny_checkpoint)), prompt, 30, k=4, sampling=sampling, generator=g
    )
    assert len(out) == 30 and stats.acceptance_rate > 0.95


def test_sampling_with_wrong_draft_runs_and_is_reproducible(tiny_checkpoint, other_model):
    sampling = Sampling(temperature=1.0)
    prompt = tokens(6, seed=7)

    def run():
        g = torch.Generator().manual_seed(42)
        return speculative_generate(pipeline(tiny_checkpoint), ModelDrafter(pipeline(other_model)), prompt, 25, k=3, sampling=sampling, generator=g)

    (a, stats), (b, _) = run(), run()
    assert a == b and len(a) == 25 and stats.acceptance_rate < 1.0
    plain = generate(pipeline(tiny_checkpoint), prompt, 25, sampling, generator=torch.Generator().manual_seed(42))
    assert len(plain) == 25


def test_speculation_over_the_network(tiny_checkpoint, other_model):
    n = Checkpoint(tiny_checkpoint).text_config().num_hidden_layers
    ple = bool(Checkpoint(tiny_checkpoint).text_config().hidden_size_per_layer_input)
    peers = [(1, 2), (2, 8)] if ple else [(1, 4), (4, 7)]
    last = 0 if ple else 1
    prompt = tokens(6, seed=8)
    expected = greedy_generate(pipeline(tiny_checkpoint), prompt, 24)
    with ThreadedSwarm(tiny_checkpoint, peers, model="tiny") as swarm:
        target = RemotePipeline.connect(str(tiny_checkpoint), swarm.tracker_url, model="tiny", first_layers=1, last_layers=last, dtype=torch.float32)
        with target:
            out, stats = speculative_generate(target, ModelDrafter(pipeline(other_model)), prompt, 24, k=3)
            trips = len(target.events.history)
    assert out == expected
    assert trips == 1 + len(stats.rounds)  # one prefill, then one trip per round
    assert n == 8
