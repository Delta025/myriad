"""M3: a client plus three peers over WebSockets gives exactly the in-process pipeline's results."""

import time

import pytest
import torch

from myriad.client.generate import greedy_generate
from myriad.client.remote import PeerError, RemotePipeline
from myriad.model.checkpoint import Checkpoint
from myriad.model.pipeline import LocalPipeline
from myriad.testing import TINY_VOCAB, ThreadedSwarm

# The client runs layer 0 (and on the dense model also the last layer); three peers split the rest.
LAYOUTS = {
    "dense": dict(first=1, last=1, peers=[(1, 3), (3, 5), (5, 7)]),
    "ple": dict(first=1, last=0, peers=[(1, 2), (2, 8)]),  # layers 2-7 share K/V: one peer, and no last layers
}
SCHEDULE = [7, 1, 1, 1, 4, 1, 3, 2]


def tokens(n, seed=0):
    return torch.randint(0, TINY_VOCAB, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


@pytest.fixture
def swarm_and_layout(tiny_checkpoint):
    variant = "ple" if Checkpoint(tiny_checkpoint).text_config().hidden_size_per_layer_input else "dense"
    layout = LAYOUTS[variant]
    with ThreadedSwarm(tiny_checkpoint, layout["peers"], model="tiny") as swarm:
        yield swarm, layout, tiny_checkpoint


def connect(swarm, layout, checkpoint):
    return RemotePipeline.connect(
        str(checkpoint), swarm.tracker_url, model="tiny", first_layers=layout["first"], last_layers=layout["last"],
        dtype=torch.float32,
    )


def local(layout, checkpoint):
    n = Checkpoint(checkpoint).text_config().num_hidden_layers
    split = [(0, layout["first"]), *layout["peers"]] + ([(n - layout["last"], n)] if layout["last"] else [])
    return LocalPipeline.from_checkpoint(str(checkpoint), split, dtype=torch.float32)


def test_remote_logits_are_bit_identical_to_local(swarm_and_layout):
    swarm, layout, ckpt = swarm_and_layout
    ids = tokens(sum(SCHEDULE))
    ref = local(layout, ckpt)
    with connect(swarm, layout, ckpt) as remote:
        pos = 0
        for size in SCHEDULE:
            chunk = ids[pos : pos + size]
            assert torch.equal(remote.forward(chunk, pos), ref.forward(chunk, pos)), f"chunk at {pos}"
            pos += size


def test_rollback_over_the_network(swarm_and_layout):
    swarm, layout, ckpt = swarm_and_layout
    ids, junk = tokens(14), tokens(5, seed=1)

    def run(pipe):
        pipe.forward(ids[:8], 0)
        pipe.forward(junk, 8)  # rejected drafts
        got_a = pipe.forward(ids[8:11], 8)  # implicit rollback: start_pos 8
        pipe.forward(junk[:2], 11)
        pipe.truncate(11)  # explicit rollback
        return torch.cat([got_a, pipe.forward(ids[11:], 11)])

    expected = local(layout, ckpt)
    clean = expected.forward(ids, 0)[8:]  # never saw the junk (different chunking: close, not equal)
    expected = run(expected)
    with connect(swarm, layout, ckpt) as remote:
        got = run(remote)
    assert torch.equal(got, expected)
    torch.testing.assert_close(got, clean, atol=1e-4, rtol=0)


def test_greedy_text_and_hop_events(swarm_and_layout):
    swarm, layout, ckpt = swarm_and_layout
    prompt = tokens(6, seed=2)
    expected = greedy_generate(local(layout, ckpt), prompt, 16)
    with connect(swarm, layout, ckpt) as remote:
        assert greedy_generate(remote, prompt, 16) == expected
        events = remote.events.history
    assert len(events) == 16
    assert [h["layers"] for h in events[-1]["hops"]] == [list(p) for p in layout["peers"]]
    # the tracker received them too (sent from a background thread)
    import httpx

    deadline = time.monotonic() + 5
    while len([e for e in httpx.get(f"{swarm.tracker_url}/events").json() if e["type"] == "forward"]) < 16:
        assert time.monotonic() < deadline, "events did not reach the tracker"
        time.sleep(0.05)


def test_injected_delay_shows_in_round_trip(tiny_checkpoint):
    layout = dict(first=1, last=0, peers=[(1, 8)])
    if Checkpoint(tiny_checkpoint).text_config().hidden_size_per_layer_input:
        layout["peers"] = [(1, 8)]
    with ThreadedSwarm(tiny_checkpoint, layout["peers"], model="tiny", delays_ms=[50]) as swarm:
        with connect(swarm, layout, tiny_checkpoint) as remote:
            remote.forward(tokens(3), 0)
            hop = remote.events.history[-1]["hops"][0]
    assert hop["rtt_ms"] >= 50 and hop["compute_ms"] < 50


def test_unknown_session_is_reported(swarm_and_layout):
    swarm, layout, ckpt = swarm_and_layout
    with connect(swarm, layout, ckpt) as remote:
        with pytest.raises(PeerError, match="unknown_session"):
            remote.peers[0].call({"type": "truncate", "session": "nope", "length": 0})


def test_missing_layers_give_a_clear_error(tiny_checkpoint):
    with ThreadedSwarm(tiny_checkpoint, [(1, 3)], model="tiny") as swarm:
        with pytest.raises(PeerError, match="no live peers cover"):
            RemotePipeline.connect(str(tiny_checkpoint), swarm.tracker_url, model="tiny", dtype=torch.float32)
