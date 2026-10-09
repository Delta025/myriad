"""Show tit-for-tat: a node that contributes gets faster service than freeloaders when peers are busy.

    uv run python scripts/tit_for_tat_demo.py google/gemma-4-E2B-it --freeloaders 4 --delay-ms 20

alice and bob each run a peer (half of the model's layers each), as separate
processes. Each node's peer and client share an identity directory, hence one
ledger. The freeloaders only run clients. To fit on one GPU, the clients run as
threads in this process and share one copy of the client-side weights.

1. Warm-up: alice and bob each generate once, so each has done work for the other
   (alice's peer served bob, bob's peer served alice), recorded as signed receipts.
2. Contention, twice: alice and the freeloaders generate at the same time, more
   than the peers can serve at once. First with first-come-first-served peers,
   then with tit-for-tat peers (contributors first, one slot in five to a random
   request).

The dashboard is live at http://127.0.0.1:8765/ while it runs.
"""

import argparse
import statistics
import tempfile
import threading
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from myriad.client.generate import greedy_generate
from myriad.client.remote import RemotePipeline
from myriad.client.reporting import Reporter
from myriad.ledger.identity import Identity
from myriad.ledger.store import Ledger
from myriad.localswarm import TRACKER_URL, LocalSwarm
from myriad.model.checkpoint import Checkpoint
from myriad.model.split import even_split

PROMPTS = [
    "Explain in detail how a rainbow forms.",
    "Describe the water cycle step by step.",
    "Explain in detail how airplanes stay in the air.",
    "Describe in detail how the heart pumps blood.",
    "Explain how volcanoes form and erupt.",
    "Describe how bees make honey, step by step.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--freeloaders", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=50)
    parser.add_argument("--delay-ms", type=float, default=20.0, help="simulated round trip per peer")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    ckpt = Checkpoint(args.model)
    tokenizer = AutoTokenizer.from_pretrained(ckpt.path)
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                             tokenize=True, return_dict=False) for p in PROMPTS]
    base = Path(tempfile.mkdtemp(prefix="myriad-tft-"))
    names = ["alice", "bob"] + [f"freeloader{i + 1}" for i in range(args.freeloaders)]
    identities = {n: Identity.load_or_create(base / n, n) for n in names}
    split = even_split(ckpt.text_config(), 2, start=1)
    parts = RemotePipeline.load_client_parts(ckpt, 1, 0, torch.bfloat16, args.device, "cpu")
    print(f"alice serves layers {split[0]}, bob serves {split[1]}; {args.delay_ms:.0f} ms simulated round trip per peer")

    def client(name: str, prompt: list[int], results: dict | None = None) -> None:
        identity = identities[name]
        ledger = Ledger(identity.public_key, base / name / "ledger.sqlite") if name in ("alice", "bob") else None
        pipe = RemotePipeline.connect(ckpt, TRACKER_URL, model=args.model, first_layers=1, last_layers=0,
                                      identity=identity, ledger=ledger, parts=parts)
        reporter = Reporter(pipe, args.model)
        with pipe:
            reporter.start("plain")
            t = time.perf_counter()
            out = greedy_generate(pipe, prompt, args.tokens)
            seconds = time.perf_counter() - t
            reporter.finish(out, seconds)
        forwards = [e for e in pipe.events.history if e["type"] == "forward"][1:]  # skip the prefill
        if results is not None:
            results[name] = (len(out) / seconds, statistics.mean(h["queue_ms"] for e in forwards for h in e["hops"]))

    results = {}
    for phase, extra in (("first come, first served", ["--first-come-first-served"]), ("tit-for-tat", [])):
        peer_args = [["--identity", str(base / "alice"), "--node-name", "alice", *extra],
                     ["--identity", str(base / "bob"), "--node-name", "bob", *extra]]
        with LocalSwarm(args.model, split, base, args.delay_ms, peer_args=peer_args):
            if not results:
                print("warm-up: alice and bob each generate once (each one's peer serves the other)")
                client("alice", prompts[0])
                client("bob", prompts[1])
                for node, other in (("bob", "alice"), ("alice", "bob")):
                    ledger = Ledger(identities[node].public_key, base / node / "ledger.sqlite")
                    work = ledger.received_from(identities[other].public_key)
                    print(f"  {node}'s ledger: {other} did {work:,} layer-positions of work for {node}")
            phase_results = {}
            threads = [threading.Thread(target=client, args=(name, prompts[2 + i % 4], phase_results))
                       for i, name in enumerate(["alice", *names[2:]])]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            results[phase] = phase_results
            print(f"\n{phase}:")
            for name in ["alice", *names[2:]]:
                rate, wait = phase_results[name]
                print(f"  {name:<12} {rate:5.2f} tok/s   average wait in peer queues {wait:6.1f} ms")

    fcfs, tft = results["first come, first served"], results["tit-for-tat"]
    free = names[2:]
    avg = lambda r: statistics.mean(r[n][0] for n in free)  # noqa: E731
    print(f"\nalice:       {fcfs['alice'][0]:.2f} -> {tft['alice'][0]:.2f} tok/s ({tft['alice'][0] / fcfs['alice'][0]:.2f}x)")
    print(f"freeloaders: {avg(fcfs):.2f} -> {avg(tft):.2f} tok/s on average (slowed, still progressing)")


if __name__ == "__main__":
    main()
