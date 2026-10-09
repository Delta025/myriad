"""Benchmark speculative decoding through a swarm, with and without injected latency.

    uv run python scripts/bench_speculative.py google/gemma-4-E2B-it --draft google/gemma-4-E2B-it --peers 2
    uv run python scripts/bench_speculative.py google/gemma-4-E4B-it --mtp google/gemma-4-E4B-it-assistant --last-layers 20

With --mtp, the draft is Gemma 4's official multi-token-prediction drafter. It reads
the target's last sliding and last global layer, which must run on the client: pass
--last-layers accordingly (2 on 31B; the whole KV-sharing block on E2B/E4B).

For each per-peer delay, start a local swarm (tracker plus peer processes),
then generate with plain greedy decoding and with speculative greedy decoding
for each `k`. Prints a markdown table and, with --json, saves the raw numbers.

Speculative greedy output equals plain greedy output in float32. In bfloat16 the
k + 1 token verification call rounds differently from single-token calls, so the
outputs can part where two tokens are tied within bf16 noise; the table shows
from which token (scripts/check_chunking.py measures this directly).
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from myriad.client.drafters import ModelDrafter, MTPDrafter
from myriad.client.generate import greedy_generate
from myriad.client.remote import RemotePipeline
from myriad.client.speculative import speculative_generate
from myriad.localswarm import TRACKER_URL, LocalSwarm
from myriad.model.checkpoint import Checkpoint
from myriad.model.pipeline import LocalPipeline
from myriad.model.split import even_split

PROMPT = "Write a short paragraph about how bees communicate."


def same_text(out: list[int], plain: list[int]) -> str:
    """'yes', or where the outputs part. In bf16 that happens only at near-ties (see check_chunking.py)."""
    if out == plain:
        return "yes"
    first = next((i for i, (a, b) in enumerate(zip(out, plain)) if a != b), min(len(out), len(plain)))
    return f"from token {first}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    draft = parser.add_mutually_exclusive_group(required=True)
    draft.add_argument("--draft", help="a draft model, e.g. google/gemma-4-E2B-it")
    draft.add_argument("--mtp", help="an MTP drafter, e.g. google/gemma-4-E4B-it-assistant")
    parser.add_argument("--peers", type=int, default=2)
    parser.add_argument("--first-layers", type=int, default=1)
    parser.add_argument("--last-layers", type=int, default=0)
    parser.add_argument("--delays", type=float, nargs="+", default=[0, 20, 50, 100], help="ms per peer")
    parser.add_argument("--k", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path, help="save results here")
    args = parser.parse_args()
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    ckpt = Checkpoint(args.model)
    n_layers = ckpt.text_config().num_hidden_layers
    split = even_split(ckpt.text_config(), args.peers, start=args.first_layers, end=n_layers - args.last_layers)
    tokenizer = AutoTokenizer.from_pretrained(ckpt.path)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=True, return_dict=False
    )
    print(f"target {args.model} (client layers 0-{args.first_layers - 1} and {n_layers - args.last_layers}-{n_layers - 1}, "
          f"peers {split}), draft {args.draft or args.mtp} {'(MTP)' if args.mtp else ''}, {args.tokens} tokens")
    if args.draft:
        # The draft runs on the client for every guessed token, so its embedding and head go on the GPU too;
        # only the per-layer-embedding table of E2B/E4B stays in CPU RAM.
        drafter = ModelDrafter(
            LocalPipeline.from_checkpoint(args.draft, None, torch.bfloat16, args.device, args.device, ple_device="cpu")
        )

    # Load the client's own weights once; each latency setting gets a fresh swarm and session.
    parts = RemotePipeline.load_client_parts(ckpt, args.first_layers, args.last_layers, torch.bfloat16, args.device, "cpu")
    rows = []
    for delay in args.delays:
        log_dir = Path(tempfile.mkdtemp(prefix="myriad-bench-"))
        with LocalSwarm(args.model, split, log_dir, delay):
            target = RemotePipeline.connect(
                ckpt, TRACKER_URL, model=args.model, first_layers=args.first_layers, last_layers=args.last_layers,
                stage_device=args.device, ends_device="cpu", parts=parts,
            )
            if args.mtp:
                drafter = MTPDrafter.from_pretrained(target, args.mtp, args.device)
            with target:
                greedy_generate(target, prompt, 4)  # warm up kernels and connections
                t = time.perf_counter()
                plain = greedy_generate(target, prompt, args.tokens)
                plain_s = time.perf_counter() - t
                rows.append(dict(delay_ms=delay, k=0, tok_s=len(plain) / plain_s, speedup=1.0, acceptance=None,
                                 tokens_per_trip=1.0, identical=True, same="yes"))
                print(f"delay {delay:>5.0f} ms  plain        {len(plain) / plain_s:6.2f} tok/s")
                for k in args.k:
                    out, stats = speculative_generate(target, drafter, prompt, args.tokens, k=k)
                    tok_s = stats.tokens_per_second
                    draft_ms = sum(r.draft_ms for r in stats.rounds) / len(stats.rounds)
                    verify_ms = sum(r.verify_ms for r in stats.rounds) / len(stats.rounds)
                    rows.append(dict(delay_ms=delay, k=k, tok_s=tok_s, speedup=tok_s / (len(plain) / plain_s),
                                     acceptance=stats.acceptance_rate, tokens_per_trip=stats.tokens_per_round,
                                     identical=out == plain, same=same_text(out, plain), draft_ms=draft_ms,
                                     verify_ms=verify_ms))
                    print(f"delay {delay:>5.0f} ms  speculative k={k}  {tok_s:6.2f} tok/s  "
                          f"x{rows[-1]['speedup']:.2f}  acceptance {stats.acceptance_rate:.0%}  "
                          f"{stats.tokens_per_round:.2f} tok/trip  draft {draft_ms:.0f} ms + trip {verify_ms:.0f} ms per round  "
                          f"{same_text(out, plain)}")

    print("\n| Delay per peer | Mode | Tokens/s | Speedup | Acceptance | Tokens per trip | Same output |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        mode = "plain" if r["k"] == 0 else f"speculative, k={r['k']}"
        acc = "" if r["acceptance"] is None else f"{r['acceptance']:.0%}"
        print(f"| {r['delay_ms']:.0f} ms | {mode} | {r['tok_s']:.2f} | {r['speedup']:.2f}x | {acc} | "
              f"{r['tokens_per_trip']:.2f} | {r['same']} |")
    if args.json:
        args.json.write_text(json.dumps(dict(model=args.model, draft=args.draft or args.mtp, split=split, rows=rows), indent=2))


if __name__ == "__main__":
    main()
