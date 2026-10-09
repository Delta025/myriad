"""Command line: run a tracker, a peer, or generate text through the swarm.

    myriad tracker --port 8000
    myriad peer google/gemma-4-E2B-it --layers 1:13 --port 9001 --tracker http://127.0.0.1:8000
    myriad generate google/gemma-4-E2B-it --tracker http://127.0.0.1:8000 --prompt "Hello"
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

    from myriad.client.generate import greedy_generate
    from myriad.client.remote import RemotePipeline
    from myriad.model.checkpoint import Checkpoint

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Checkpoint(args.model)
    pipe = RemotePipeline.connect(
        checkpoint, args.tracker, model=args.name or args.model, first_layers=args.first_layers,
        last_layers=args.last_layers, dtype=DTYPES[args.dtype], stage_device=device, ends_device=args.ends_device,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.path)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], add_generation_prompt=True, tokenize=True, return_dict=False
    )
    route = " -> ".join(f"{p.peer_id}[{p.start}-{p.end - 1}]@{p.region}" for p in pipe.peers)
    print(f"route: client -> {route} -> client")

    with pipe:
        eos = GenerationConfig.from_pretrained(checkpoint.path).eos_token_id
        stop = eos if isinstance(eos, list) else [eos]
        t = time.perf_counter()
        tokens = greedy_generate(pipe, prompt, args.max_tokens, stop_ids=stop)
        elapsed = time.perf_counter() - t
    print(tokenizer.decode(tokens, skip_special_tokens=True))
    print(f"\n{len(tokens)} tokens in {elapsed:.1f}s ({len(tokens) / elapsed:.1f} tok/s)")
    _print_hop_summary(pipe.events.history[1:])  # skip the prefill


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
    p.set_defaults(func=run_generate)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
