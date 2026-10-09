"""Command line: run a tracker, a peer, or generate text through the swarm.

    myriad tracker --port 8000
    myriad peer google/gemma-4-E2B-it --layers 1:13 --port 9001 --tracker http://127.0.0.1:8000
    myriad generate google/gemma-4-E2B-it --tracker http://127.0.0.1:8000 --prompt "Hello"
    myriad generate google/gemma-4-31B-it --tracker ... --draft google/gemma-4-E2B-it --k 4 --prompt "Hello"
"""

import argparse
import asyncio
import logging
import statistics
import time

import torch

DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def _layers(text: str) -> tuple[int, int]:
    start, end = text.split(":")
    return int(start), int(end)


def run_tracker(args) -> None:
    import uvicorn

    from myriad.tracker.app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")


def run_peer(args) -> None:
    from myriad.model.checkpoint import Checkpoint
    from myriad.model.stage import Stage
    from myriad.peer.server import PeerServer

    start, end = args.layers
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stage = Stage.from_checkpoint(Checkpoint(args.model), start, end, device, DTYPES[args.dtype])
    gpu = torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu"
    server = PeerServer(stage, args.name or args.model, region=args.region, delay_ms=args.delay_ms)
    asyncio.run(server.serve(args.host, args.port, args.public_url, args.tracker, gpu))


def run_generate(args) -> None:
    from transformers import AutoTokenizer, GenerationConfig

    from myriad.client.drafters import ModelDrafter
    from myriad.client.generate import generate
    from myriad.client.remote import RemotePipeline
    from myriad.client.sampling import Sampling
    from myriad.client.speculative import speculative_generate
    from myriad.model.checkpoint import Checkpoint
    from myriad.model.pipeline import LocalPipeline

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Checkpoint(args.model)
    pipe = RemotePipeline.connect(
        checkpoint, args.tracker, model=args.name or args.model, first_layers=args.first_layers,
        last_layers=args.last_layers, dtype=DTYPES[args.dtype], stage_device=device, ends_device=args.ends_device,
    )
    drafter = None
    if args.draft:
        # the draft runs once per guessed token, so its head goes on the GPU too (PLE table stays in RAM)
        draft = LocalPipeline.from_checkpoint(args.draft, None, DTYPES[args.dtype], device, device, ple_device="cpu")
        drafter = ModelDrafter(draft)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint.path)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], add_generation_prompt=True, tokenize=True, return_dict=False
    )
    route = " -> ".join(f"{p.peer_id}[{p.start}-{p.end - 1}]@{p.region}" for p in pipe.peers)
    print(f"route: client -> {route} -> client")
    sampling = Sampling(args.temperature, args.top_k, args.top_p)
    generator = torch.Generator().manual_seed(args.seed)

    def report_round(r, produced):
        # counts and timings only: the tracker never sees the text
        pipe.events.emit(
            {"type": "speculation", "time": time.time(), "client_id": pipe.client_id, "session": pipe.session,
             "proposed": r.proposed, "accepted": r.accepted, "produced": len(produced),
             "draft_ms": round(r.draft_ms, 2), "verify_ms": round(r.verify_ms, 2)}
        )

    with pipe:
        eos = GenerationConfig.from_pretrained(checkpoint.path).eos_token_id
        stop = eos if isinstance(eos, list) else [eos]
        t = time.perf_counter()
        if drafter is None:
            tokens = generate(pipe, prompt, args.max_tokens, sampling, stop, generator)
        else:
            tokens, stats = speculative_generate(
                pipe, drafter, prompt, args.max_tokens, args.k, sampling, stop, generator, on_round=report_round
            )
        elapsed = time.perf_counter() - t
    print(tokenizer.decode(tokens, skip_special_tokens=True))
    print()
    print(f"{len(tokens)} tokens in {elapsed:.1f}s ({len(tokens) / elapsed:.1f} tok/s)")
    if drafter is not None:
        print(
            f"speculation: k={args.k}, acceptance {stats.acceptance_rate:.0%}, "
            f"{stats.tokens_per_round:.2f} tokens per trip, {len(stats.rounds)} trips"
        )
    forwards = [e for e in pipe.events.history if e["type"] == "forward"]
    _print_hop_summary(forwards[1:])  # skip the prefill


def _print_hop_summary(events: list[dict]) -> None:
    if not events:
        return
    print("per-hop latency over decode steps (median ms):")
    for i in range(len(events[0]["hops"])):
        hops = [e["hops"][i] for e in events]
        rtt = statistics.median(h["rtt_ms"] for h in hops)
        compute = statistics.median(h["compute_ms"] for h in hops)
        a, b = hops[0]["layers"]
        print(f"  {hops[0]['peer_id']} layers {a}-{b - 1}: round trip {rtt:.1f}, compute {compute:.1f}, other {rtt - compute:.1f}")
    print(f"  whole step: {statistics.median(e['total_ms'] for e in events):.1f}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="myriad", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tracker", help="run the tracker")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=run_tracker)

    p = sub.add_parser("peer", help="serve a range of layers")
    p.add_argument("model", help="Hugging Face repo id or local checkpoint directory")
    p.add_argument("--layers", type=_layers, required=True, help="start:end, end exclusive (e.g. 1:13)")
    p.add_argument("--name", help="model name to register under (default: the model argument)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--public-url", help="URL clients should use, e.g. behind a proxy (default ws://host:port)")
    p.add_argument("--tracker", help="tracker URL, e.g. http://127.0.0.1:8000")
    p.add_argument("--region", default="local")
    p.add_argument("--delay-ms", type=float, default=0.0, help="simulated round-trip latency added to every reply")
    p.add_argument("--device", help="default: cuda if available")
    p.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    p.set_defaults(func=run_peer)

    p = sub.add_parser("generate", help="generate text through the swarm")
    p.add_argument("model")
    p.add_argument("--tracker", required=True)
    p.add_argument("--name", help="model name peers registered under (default: the model argument)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--first-layers", type=int, default=1, help="layers the client runs before the swarm")
    p.add_argument("--last-layers", type=int, default=0, help="layers the client runs after the swarm")
    p.add_argument("--device", help="device for the client's layers (default: cuda if available)")
    p.add_argument("--ends-device", default="cpu", help="device for embedding and output head")
    p.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    p.add_argument("--draft", help="draft model for speculative decoding, e.g. google/gemma-4-E2B-it")
    p.add_argument("--k", type=int, default=4, help="tokens drafted per round")
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=run_generate)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
