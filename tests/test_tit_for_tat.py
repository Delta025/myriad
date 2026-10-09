"""M5 over the network: receipts flow both ways, cheating is caught, contributors are served first."""

import statistics
import threading
import time

import pytest
import torch

from myriad.client.generate import greedy_generate
from myriad.client.remote import PeerError, RemotePipeline
from myriad.ledger.identity import Identity, Receipt
from myriad.ledger.store import Ledger
from myriad.model.checkpoint import Checkpoint
from myriad.testing import TINY_VOCAB, ThreadedSwarm


def tokens(n, seed=0):
    return torch.randint(0, TINY_VOCAB, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


@pytest.fixture
def dense(tiny_checkpoint):
    if Checkpoint(tiny_checkpoint).text_config().hidden_size_per_layer_input:
        pytest.skip("one variant is enough for the credit logic")
    return tiny_checkpoint


def connect(swarm, ckpt, identity=None, ledger=None, parts=None):
    return RemotePipeline.connect(str(ckpt), swarm.tracker_url, model="tiny", first_layers=1, last_layers=0,
                                  dtype=torch.float32, identity=identity, ledger=ledger, parts=parts)


def test_receipts_are_recorded_and_countersigned(dense):
    peer_id, client_id = Identity.generate("peer"), Identity.generate("alice")
    peer_ledger, client_ledger = Ledger(peer_id.public_key), Ledger(client_id.public_key)
    with ThreadedSwarm(dense, [(1, 8)], model="tiny", identities=[peer_id], ledgers=[peer_ledger]) as swarm:
        with connect(swarm, dense, client_id, client_ledger) as pipe:
            greedy_generate(pipe, tokens(5), 6)  # prefill of 5 positions, then 5 single steps
        time.sleep(0.1)
    work = 7 * (5 + 5)  # 7 layers x 10 positions
    assert client_ledger.received_from(peer_id.public_key) == work
    assert peer_ledger.given_to(client_id.public_key) == work
    assert client_ledger.name(peer_id.public_key) == "peer" and peer_ledger.name(client_id.public_key) == "alice"
    # every receipt the client got, it countersigned; the peer has all but the last countersignature
    signed = peer_ledger._db.execute("SELECT COUNT(*) FROM receipts WHERE client_sig IS NOT NULL").fetchone()[0]
    assert signed == 5


def test_a_peer_that_inflates_its_receipts_is_rejected(dense):
    with ThreadedSwarm(dense, [(1, 8)], model="tiny") as swarm:
        peer = swarm.peers[0]
        honest = peer._receipt

        def inflated(session_id, session, positions):
            return honest(session_id, session, positions * 10)  # claims ten times the work

        peer._receipt = inflated
        with connect(swarm, dense) as pipe:
            with pytest.raises(PeerError, match="invalid receipt"):
                pipe.forward(tokens(3), 0)


def test_contributor_is_served_before_freeloaders_under_load(dense):
    """One busy peer, four clients at once. The peer's node has received work from alice before."""
    node, alice = Identity.generate("node"), Identity.generate("alice")
    ledger = Ledger(node.public_key)
    past = Receipt(alice.public_key, node.public_key, "earlier", 1, 8, 100, 1, time.time())  # alice served the node
    ledger.record(past, alice.sign(past.payload()), node.sign(past.payload()))

    with ThreadedSwarm(dense, [(1, 8)], model="tiny", identities=[node], ledgers=[ledger], unchoke=0.0) as swarm:
        peer = swarm.peers[0]
        compute = peer._run_stage

        def slow(session, request, received):  # make the peer the bottleneck
            time.sleep(0.02)
            return compute(session, request, received)

        peer._run_stage = slow
        clients = {"alice": alice} | {f"free{i}": Identity.generate(f"free{i}") for i in range(3)}
        waits = {name: [] for name in clients}
        parts = RemotePipeline.load_client_parts(Checkpoint(dense), 1, 0, torch.float32)

        def run(name):
            with connect(swarm, dense, clients[name], parts=parts) as pipe:
                greedy_generate(pipe, tokens(4, seed=len(name)), 25)
                waits[name] = [e["hops"][0]["queue_ms"] for e in pipe.events.history[1:]]

        threads = [threading.Thread(target=run, args=(name,)) for name in clients]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    alice_wait = statistics.mean(waits["alice"])
    free_wait = statistics.mean(w for name in clients if name != "alice" for w in waits[name])
    # Scheduling is not preemptive: alice still waits for the job already running when she arrives
    # (measured: about 43 ms vs 94 ms for the freeloaders), but never behind a queue of them.
    assert alice_wait < free_wait / 1.5, (alice_wait, free_wait)
    assert all(len(w) == 24 for w in waits.values())  # everyone finished: freeloaders are slowed, not stopped
