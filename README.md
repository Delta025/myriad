# Myriad

Myriad runs large open-weight language models across a peer-to-peer swarm of consumer GPUs, with no blockchain.

The model is split by layers across peers. The client keeps the embeddings, the first and last layers, the output head and a small draft model, so peers only ever see intermediate activations. Two ideas are being built first:

- **Speculative decoding across the swarm.** The client drafts several tokens locally, and the swarm checks them all in one network round trip.
- **Tit-for-tat credits.** Peers that contribute compute get priority over peers that only consume, tracked with signed receipts instead of a token.

The first target is Gemma 4 31B split across GPUs in several regions. The draft model is Gemma 4 E2B, then Google's official Gemma 4 multi-token-prediction drafter, which only needs state the client already holds.

> Status: early prototype, under construction for MLH Hack Day at UQAM.

## Progress

| Milestone | Status |
| --- | --- |
| Project skeleton, tiny random-weight Gemma 4 test models | Done |
| Run a model as a chain of layer-range stages, with a cache that can roll back | **Done.** Bit-identical to Transformers on Gemma 4 E2B and E4B |
| Stages on separate peers over the network, tracker | **Done.** Bit-identical to the in-process pipeline |
| Speculative decoding across the swarm | Next |
| Tit-for-tat credits | Planned |
| Live dashboard | Planned |
| Multi-region deployment and benchmarks | Planned |

### What works today

A Gemma 4 model can be cut into stages, each holding a range of decoder layers and loading only those layers' weights. Each stage keeps its own key/value cache per session. Every call carries the position it starts at, and a stage drops anything it cached from that position on, so rejected draft tokens can be rolled back without an extra message.

Correctness is checked against stock Hugging Face Transformers:

- **Tiny models (CPU, seconds):** logits match Transformers for every split, chunk size and rollback pattern tested, including sequences longer than the sliding window.
- **Real models (RTX 3080):** Gemma 4 E2B and E4B split into 3 stages produce 560 greedy tokens identical to Transformers, with a maximum logit difference of exactly 0.0. That run goes past the 512-token sliding window.

Two details were needed for bit-exact results:

- Sliding-window layers attend over exactly the keys Transformers' cache holds.
- Rotary frequencies are computed on CPU, then moved to the GPU.

E2B and E4B share key/value state between their last ~20 layers, so those layers must stay in one stage. Split validation enforces this. Gemma 4 31B has no such constraint.

Stages can also run on separate peers:

- Each peer is a process that serves a range of layers over WebSockets. Messages are msgpack, and tensors travel as raw bytes, so they arrive bit-identical.
- Peers register with a tracker. The client asks the tracker for a route of peers covering the layers it doesn't run itself.
- The client runs the embedding, its own first (and optionally last) layers and the output head. It calls the peers one after another and times every hop.
- A swarm of three peer processes produces tokens and logits bit-identical to the in-process pipeline, on tiny models in the tests and on Gemma 4 E2B on an RTX 3080.
- Peers can add a simulated round-trip delay (`--delay-ms`) for latency experiments. `scripts/netem.sh` adds real delay on Linux.

## Development

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

The tests use tiny Gemma 4 models with random weights and run on CPU in a few seconds.

To check a real model against Transformers (downloads the weights from Hugging Face; the decoder layers go on the GPU, the embedding and output head on CPU):

```bash
uv run python scripts/check_equivalence.py google/gemma-4-E4B-it --stages 3
```

## Running a swarm

```bash
uv run myriad tracker --port 8000
uv run myriad peer google/gemma-4-E2B-it --layers 1:13 --port 9001 --tracker http://127.0.0.1:8000
uv run myriad peer google/gemma-4-E2B-it --layers 13:35 --port 9002 --tracker http://127.0.0.1:8000
uv run myriad generate google/gemma-4-E2B-it --tracker http://127.0.0.1:8000 --prompt "Why is the sky blue?"
```

Each command runs in its own terminal, and peers can run on different machines (pass `--public-url` if a peer sits behind a proxy). On E2B and E4B, the last ~20 layers must be served by a single peer. `generate` prints the route, the text, and the median round trip and compute time of each hop.

To check that a swarm of local peer processes matches the in-process pipeline exactly:

```bash
uv run python scripts/check_network.py google/gemma-4-E2B-it --peers 3 --delay-ms 20
```

## Layout

```
src/myriad/
  model/      checkpoint loading, split rules, stages, KV cache, masks, embedding/head, pipelines
  client/     generation loops, RemotePipeline (client side of a swarm); speculative decoding to come
  peer/       peer server: one stage, one KV cache per session
  tracker/    peer registry, route selection, event stream for the dashboard
  protocol/   wire messages (msgpack, raw tensor bytes)
  ledger/     identities, receipts, credits (to come)
  dashboard/  live web view (to come)
  cli.py      `myriad tracker | peer | generate`
  testing.py  tiny random-weight Gemma 4 checkpoints, in-process test swarm
scripts/      equivalence checks, netem helper, later deployment and benchmarks
tests/
```

## Prior work

Layer-sharded inference across machines is well established ([Petals](https://github.com/bigscience-workshop/petals), [exo](https://github.com/exo-explore/exo)). Speculative decoding over a pipeline of machines has been studied in [PipeInfer](https://arxiv.org/abs/2407.11798) and [FlowSpec](https://arxiv.org/abs/2507.02620). Myriad's focus is an open swarm of untrusted peers, with incentives and with the client keeping the output head.

## License

Apache-2.0. See [LICENSE](LICENSE).
