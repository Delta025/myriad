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
| Speculative decoding across the swarm | **Done.** Up to 1.97x faster under latency, same output |
| Official Gemma 4 multi-token-prediction drafter | **Done.** Runs entirely on the client |
| Live dashboard | **Done** |
| Tit-for-tat credits | **Done.** Signed receipts, local ledgers, contributors served first |
| Multi-region deployment and benchmarks | Next |

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

### Speculative decoding

The client drafts `k` tokens with a small local model. One call then carries the last token plus all `k` guesses through every peer. The client checks them against the target's predictions, which it can do because it holds the output head. It keeps the longest agreeing prefix plus one token from the target, so every trip through the swarm yields 1 to `k+1` tokens. Rejected guesses cost no extra message: the next call starts at the first changed position, and every peer overwrites from there.

- **Greedy:** a guess is accepted only if it equals the target's own choice.
- **Sampling:** uses the acceptance rule of [Leviathan et al.](https://arxiv.org/abs/2211.17192) and [Chen et al.](https://arxiv.org/abs/2302.01318), which keeps the output distributed exactly as sampling from the target alone. A Monte Carlo test checks this.

Benchmark on one RTX 3080: Gemma 4 E2B as the target, split over 2 peer processes, with simulated round-trip latency per peer (`scripts/bench_speculative.py`, 64 tokens). For this test the draft is E2B itself, which is as expensive as the target, so it is a pessimistic case for draft cost. A real setup drafts for a much larger target (31B).

| Latency per peer | Plain | Speculative, k=2 | Speculative, k=4 |
| --- | --- | --- | --- |
| 0 ms | 7.20 tok/s | 8.05 tok/s (1.12x) | 7.87 tok/s (1.09x) |
| 20 ms | 5.87 tok/s | 7.29 tok/s (1.24x) | 7.31 tok/s (1.25x) |
| 50 ms | 4.30 tok/s | 6.18 tok/s (1.44x) | 6.59 tok/s (1.53x) |
| 100 ms | 2.77 tok/s | 5.23 tok/s (1.89x) | 5.44 tok/s (1.97x) |

The more latency between peers, the more speculation helps, because it removes round trips.

#### The official Gemma 4 drafter

Google publishes a small multi-token-prediction drafter for each Gemma 4 model (`google/gemma-4-31B-it-assistant` and so on). It has no key/value cache of its own. Each guess reads:

- the target's embedding of the last token
- the target's final hidden state
- the target's cached keys and values of its last sliding-window layer and last global layer

On 31B those are layers 58 and 59, which the client runs anyway, so the drafter needs nothing from the peers. Myriad feeds these inputs to Transformers' drafter module exactly as Transformers' own assisted generation does; a test checks the guesses match. Use it with `--mtp google/gemma-4-31B-it-assistant --last-layers 2`.

Results on the RTX 3080 with Gemma 4 E4B as the target (2 peer processes) and its 150 MB drafter. On E4B, the drafter's source layers sit in the block of layers that share key/value state, so for this test the client runs layers 22–41.

| Latency per peer | Plain | MTP drafter, k=2 |
| --- | --- | --- |
| 0 ms | 6.34 tok/s | 6.79 tok/s (1.07x) |
| 20 ms | 5.05 tok/s | 5.73 tok/s (1.14x) |
| 50 ms | 3.87 tok/s | 4.75 tok/s (1.23x) |
| 100 ms | 2.76 tok/s | 3.62 tok/s (1.31x) |

The small E4B drafter guesses right less often than E2B drafting for itself: 1.57 tokens per trip against 3.0. Transformers' own assisted generation gets 1.64 with the same drafter and prompt. Each guess, though, costs about 10 ms instead of about 70 ms. Measurements with 31B and its larger drafter are next.

About exactness: speculative greedy output is identical to plain greedy output in float32, on the tiny test models and on E2B (`scripts/check_chunking.py`). In bfloat16, checking `k+1` tokens in one call rounds slightly differently from one token at a time, so the two can part where the top two tokens are tied within bf16 noise. In the k=4 runs above that happened once, at token 11, where " dances" and " intricate" were 0.016 apart.

## Running a swarm

```bash
uv run myriad tracker --port 8000
uv run myriad peer google/gemma-4-E2B-it --layers 1:13 --port 9001 --tracker http://127.0.0.1:8000
uv run myriad peer google/gemma-4-E2B-it --layers 13:35 --port 9002 --tracker http://127.0.0.1:8000
uv run myriad generate google/gemma-4-E2B-it --tracker http://127.0.0.1:8000 --prompt "Why is the sky blue?"
```

Add `--draft google/gemma-4-E2B-it --k 4` (or `--mtp <drafter> --last-layers <n>`) to `generate` for speculative decoding, and `--temperature`, `--top-p`, `--top-k`, `--seed` for sampling.

Each command runs in its own terminal, and peers can run on different machines (pass `--public-url` if a peer sits behind a proxy). On E2B and E4B, the last ~20 layers must be served by a single peer. `generate` prints the route, the text, and the median round trip and compute time of each hop.

### Tit-for-tat credits

Peers that contribute get served first, without a token or a blockchain:

- **Identities and receipts.** Every node has an Ed25519 key, shared by its peer and its client. After each call, the peer returns a receipt saying which layers it ran, for how many positions, for which client. The peer signs it; the client checks that it describes exactly that call, and countersigns it with its next request. Both sides keep doubly signed records, as in Tribler's [TrustChain](https://doi.org/10.1016/j.future.2017.08.048). A peer that inflates its receipts is rejected.
- **Local ledgers.** Each node keeps its own SQLite ledger of the receipts it is party to. There is no global ledger.
- **Tit-for-tat queues.** When several requests wait for a peer's GPU, it first serves its own node's client, then requesters by how much work they have done for its node. One slot in five goes to a random waiting request (BitTorrent's "optimistic unchoke"), so newcomers and freeloaders still make progress.

`scripts/tit_for_tat_demo.py` runs two contributing nodes (each serves half of Gemma 4 E2B as a separate process) and four freeloaders on one RTX 3080, all generating at once:

| | First come, first served | Tit-for-tat |
| --- | --- | --- |
| Contributor: wait in peer queues | 52 ms | 26 ms |
| Contributor: speed | 2.80 tok/s | 3.17 tok/s (1.13x) |
| Freeloaders: speed (average) | 2.88 tok/s | 2.73 tok/s |

The contributor's queueing time halves. The overall speed gain is smaller here because the clients share one machine and queueing is only part of each token's time. Freeloaders are slowed, not stopped.

Use `--identity <dir> --node-name <name>` on `myriad peer` and `myriad generate` to give a node a persistent key and ledger.

### Dashboard

The tracker serves a live dashboard at its root URL (for example http://127.0.0.1:8000/). It shows:

- the latest request moving from the client through each peer and back, with per-hop round-trip and compute times
- which peers serve which layers
- tokens per second, acceptance rate and tokens per trip, with recent runs compared against plain decoding
- the token stream, colored by draft tokens accepted, tokens corrected by the swarm, and bonus tokens
- the peers with their region and GPU
- each peer's credits: the work others have done for its node, and the order of its queue

By default the client sends the tracker only counts and timings. Pass `--share-text` to `generate` to show the text too: whoever runs the tracker can then read the output.

To try it on one machine, `uv run python scripts/local_swarm.py google/gemma-4-E2B-it --peers 2 --delay-ms 40` starts a tracker and peers. The dashboard is then at http://127.0.0.1:8765/.

To check that a swarm of local peer processes matches the in-process pipeline exactly:

```bash
uv run python scripts/check_network.py google/gemma-4-E2B-it --peers 3 --delay-ms 20
```

To benchmark speculative decoding under latency:

```bash
uv run python scripts/bench_speculative.py google/gemma-4-E2B-it --draft google/gemma-4-E2B-it --peers 2 --delays 0 20 50 100 --k 2 4
```

## Layout

```
src/myriad/
  model/      checkpoint loading, split rules, stages, KV cache, masks, embedding/head, pipelines
  client/     generation, sampling, speculative decoding, drafters, RemotePipeline (client side of a swarm)
  peer/       peer server (one stage, one KV cache per session) and tit-for-tat scheduler
  tracker/    peer registry, route selection, event stream for the dashboard
  protocol/   wire messages (msgpack, raw tensor bytes)
  ledger/     Ed25519 identities, signed receipts, per-node SQLite ledger
  dashboard/  live web view, served by the tracker
  cli.py      `myriad tracker | peer | generate`
  testing.py  tiny random-weight Gemma 4 checkpoints, in-process test swarm
scripts/      equivalence checks, netem helper, later deployment and benchmarks
tests/
```

## Known limitations and next optimisations

- **Kernel-launch overhead.** On a consumer GPU under Windows, a decode step is dominated by the cost of launching thousands of small kernels: an E2B step does about 12 ms of GPU work but takes about 74 ms. Capturing each stage's step as a **CUDA graph**, with a static KV cache, is the planned fix.
- **Sliding-window layers keep their whole history,** so rollback is always exact. They could keep only the window plus the draft length.
- **bf16 speculative output can part from plain output at near-ties** (see above). Batch-invariant kernels would remove this.
- **The client calls peers one after another.** Relaying peer to peer would halve the network legs per trip.

## Prior work

Layer-sharded inference across machines is well established ([Petals](https://github.com/bigscience-workshop/petals), [exo](https://github.com/exo-explore/exo)). Speculative decoding over a pipeline of machines has been studied in [PipeInfer](https://arxiv.org/abs/2407.11798) and [FlowSpec](https://arxiv.org/abs/2507.02620). Myriad's focus is an open swarm of untrusted peers, with incentives and with the client keeping the output head.

## License

Apache-2.0. See [LICENSE](LICENSE).
