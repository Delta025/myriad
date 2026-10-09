"""M5: identities, signed receipts, the local ledger and the tit-for-tat scheduler."""

import asyncio
import time

from myriad.ledger.identity import Identity, Receipt, verify
from myriad.ledger.store import Ledger
from myriad.peer.scheduler import Scheduler


def receipt(peer: Identity, client: Identity, positions=5, seq=1, session="s") -> Receipt:
    return Receipt(peer.public_key, client.public_key, session, 2, 6, positions, seq, time.time())


def test_receipts_are_signed_and_tamper_evident():
    alice, bob = Identity.generate("alice"), Identity.generate("bob")
    r = receipt(alice, bob)
    sig = alice.sign(r.payload())
    assert verify(alice.public_key, sig, r.payload())
    assert not verify(bob.public_key, sig, r.payload())  # wrong signer
    inflated = Receipt(**(r.to_dict() | {"positions": 500}))
    assert not verify(alice.public_key, sig, inflated.payload())  # edited after signing
    assert r.units == 4 * 5


def test_identity_persists(tmp_path):
    a = Identity.load_or_create(tmp_path / "node", "alice")
    b = Identity.load_or_create(tmp_path / "node")
    assert a.public_key == b.public_key


def test_ledger_balances():
    me, alice, bob = Identity.generate("me"), Identity.generate("alice"), Identity.generate("bob")
    ledger = Ledger(me.public_key)
    ledger.set_name(alice.public_key, "alice")
    r1 = receipt(alice, me, positions=10)  # alice worked for me: 40 units
    ledger.record(r1, alice.sign(r1.payload()), me.sign(r1.payload()))
    r2 = receipt(me, bob, positions=3, session="t")  # I worked for bob: 12 units
    ledger.record(r2, me.sign(r2.payload()))
    assert ledger.received_from(alice.public_key) == 40 and ledger.given_to(alice.public_key) == 0
    assert ledger.given_to(bob.public_key) == 12 and ledger.received_from(bob.public_key) == 0
    names = {b["name"]: (b["received"], b["given"]) for b in ledger.balances()}
    assert names["alice"] == (40, 0) and names[bob.public_key[:8]] == (0, 12)
    ledger.record(r1, alice.sign(r1.payload()))  # the same receipt again does not double count
    assert ledger.received_from(alice.public_key) == 40


def run_scheduler(scores: dict[str, float], requesters: list[str], unchoke: float, seed=0, rounds=1) -> list[str]:
    """Queue all requests at once, then let the scheduler run them; returns the order they ran in."""
    async def main():
        scheduler = Scheduler(lambda r: scores.get(r, 0.0), unchoke, seed)
        order = []
        for _ in range(rounds):
            await asyncio.gather(*(scheduler.run(r, lambda r=r: order.append(r)) for r in requesters))
        scheduler.shutdown()
        return order, scheduler.picks
    return asyncio.run(main())


def test_scheduler_serves_contributors_first_and_ties_in_arrival_order():
    order, _ = run_scheduler({"alice": 50, "bob": 10}, ["f1", "bob", "f2", "alice", "f3"], unchoke=0.0)
    assert order == ["alice", "bob", "f1", "f2", "f3"]


def test_optimistic_unchoke_gives_others_a_share():
    requesters = ["alice", "f1", "f2", "f3"]
    order, picks = run_scheduler({"alice": 100}, requesters, unchoke=0.25, seed=1, rounds=400)
    firsts = [order[i] for i in range(0, len(order), len(requesters))]  # who went first in each round
    share = 1 - firsts.count("alice") / len(firsts)
    # alice goes first unless the first slot is an unchoke that picks someone else: 0.25 * 3/4 of rounds
    assert 0.12 < share < 0.26
    # 3 of every 4 picks have a choice (the last one in a round doesn't), so about 0.25 * 3/4 are optimistic
    assert 0.15 < sum(optimistic for _, optimistic in picks) / len(picks) < 0.23
