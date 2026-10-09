"""Check that a swarm of separate peer processes matches the in-process pipeline on a real model.

    uv run python scripts/check_network.py google/gemma-4-E2B-it --peers 3
    uv run python scripts/check_network.py google/gemma-4-E2B-it --peers 3 --delay-ms 20

Phase 1 starts a tracker and `--peers` peer processes (`myriad peer`) on this
machine, and generates through them with the client holding layer 0, the
embedding and the head. Phase 2 stops them and runs the same split in one
process with the same devices. PASS means identical tokens and bit-identical logits.
"""

import argparse
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import torch
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from myriad.client.remote import RemotePipeline
from myriad.model.checkpoint import Checkpoint
from myriad.model.pipeline import LocalPipeline
from myriad.model.split import even_split

PROMPT = "Explain in three sentences why the sky is blue."
TRACKER_PORT, FIRST_PEER_PORT = 8765, 9765


def greedy_with_logits(pipe, prompt_ids: list[int], n_new: int):
    """Greedy decoding (as in myriad.client.generate), keeping each step's last-position logits."""
    logits = [pipe.forward(prompt_ids, start_pos=0)[-1:]]
    tokens = [int(logits[-1][-1].argmax())]
    while len(tokens) < n_new:
        logits.append(pipe.forward([tokens[-1]], start_pos=len(prompt_ids) + len(tokens) - 1))
        tokens.append(int(logits[-1][-1].argmax()))
    return tokens, torch.cat(logits)


def start_swarm(model: str, split, delay_ms: float, log_dir: Path) -> list[subprocess.Popen]:
    cli = [sys.executable, "-m", "myriad.cli"]
    tracker_url = f"http://127.0.0.1:{TRACKER_PORT}"

    def launch(name, args):
        log = open(log_dir / f"{name}.log", "w")
        return subprocess.Popen(cli + args, stdout=log, stderr=subprocess.STDOUT)

    procs = [launch("tracker", ["tracker", "--host", "127.0.0.1", "--port", str(TRACKER_PORT)])]
    for i, (a, b) in enumerate(split):
        procs.append(
            launch(
                f"peer{i}",
                ["-v", "peer", model, "--layers", f"{a}:{b}", "--host", "127.0.0.1", "--port", str(FIRST_PEER_PORT + i),
                 "--tracker", tracker_url, "--region", f"local-{i}", "--delay-ms", str(delay_ms)],
            )
        )

    deadline = time.monotonic() + 300
    while True:
        if any(p.poll() is not None for p in procs):
            raise RuntimeError(f"a swarm process exited early; see logs in {log_dir}")
        try:
            if len(httpx.get(f"{tracker_url}/peers", timeout=2).json()) == len(split):
                return procs
        except httpx.HTTPError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(f"peers did not register; see logs in {log_dir}")
        time.sleep(0.5)


def stop_swarm(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        p.terminate()
    for p in procs:
        p.wait(timeout=30)


def print_hops(events: list[dict]) -> None:
    decode = events[1:]  # skip the prefill
    print("  per-hop latency over decode steps (median ms):")
    for i in range(len(decode[0]["hops"])):
        hops = [e["hops"][i] for e in decode]
        rtt, compute = statistics.median(h["rtt_ms"] for h in hops), statistics.median(h["compute_ms"] for h in hops)
        a, b = hops[0]["layers"]
        print(f"    peer {i} (layers {a}-{b - 1}): round trip {rtt:6.1f}   compute {compute:6.1f}   network+other {rtt - compute:6.1f}")
    print(f"    whole step (client + all hops): {statistics.median(e['total_ms'] for e in decode):.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--peers", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--delay-ms", type=float, default=0.0, help="simulated round trip added by each peer")
    parser.add_argument("--stage-device", default="cuda")
    args = parser.parse_args()
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    ckpt = Checkpoint(args.model)
    config = ckpt.text_config()
    peer_split = even_split(config, args.peers, start=1)
    tokenizer = AutoTokenizer.from_pretrained(ckpt.path)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=True, return_dict=False
    )
    print(f"{args.model}: client layers 0-0, peers {peer_split}, delay {args.delay_ms} ms per peer")

    log_dir = Path(tempfile.mkdtemp(prefix="myriad-swarm-"))
    print(f"Phase 1: swarm of {args.peers} peer processes (logs in {log_dir})")
    procs = start_swarm(args.model, peer_split, args.delay_ms, log_dir)
    try:
        pipe = RemotePipeline.connect(
            ckpt, f"http://127.0.0.1:{TRACKER_PORT}", model=args.model, first_layers=1, last_layers=0,
            stage_device=args.stage_device, ends_device="cpu",
        )
        with pipe:
            t = time.perf_counter()
            net_tokens, net_logits = greedy_with_logits(pipe, prompt, args.tokens)
            print(f"  {len(net_tokens)} tokens in {time.perf_counter() - t:.1f}s")
            print_hops(pipe.events.history)
        del pipe
    finally:
        stop_swarm(procs)

    print("Phase 2: same split in one process")
    local = LocalPipeline.from_checkpoint(ckpt, [(0, 1), *peer_split], torch.bfloat16, args.stage_device, "cpu")
    t = time.perf_counter()
    loc_tokens, loc_logits = greedy_with_logits(local, prompt, args.tokens)
    print(f"  {len(loc_tokens)} tokens in {time.perf_counter() - t:.1f}s")

    print("\nSwarm:", repr(tokenizer.decode(net_tokens)))
    identical = net_tokens == loc_tokens and torch.equal(net_logits, loc_logits)
    print(f"max |logit difference|: {(net_logits - loc_logits).abs().max().item():.4f}")
    if identical:
        print(f"PASS: {len(net_tokens)} tokens and their logits are bit-identical")
    else:
        print("FAIL: the swarm and the in-process pipeline differ")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
