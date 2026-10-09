# Myriad

Myriad runs large open-weight language models across a peer-to-peer swarm of consumer GPUs, with no blockchain.

The model is split by layers across peers. The client keeps the embeddings, the first and last layers, the output head and a small draft model, so peers only ever see intermediate activations. Two ideas are being built first:

- **Speculative decoding across the swarm.** The client drafts several tokens locally, and the swarm checks them all in one network round trip.
- **Tit-for-tat credits.** Peers that contribute compute get priority over peers that only consume, tracked with signed receipts instead of a token.

The first target is Gemma 4 31B split across GPUs in several regions, with Gemma 4 E2B as the draft model.

> Status: early prototype, under construction for MLH Hack Day at UQAM.

## Development

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

Tests use a tiny Gemma 4 model with random weights and run on CPU. Tests marked `gpu` need a CUDA GPU and downloaded weights; skip them with `uv run pytest -m "not gpu"`.

## License

Apache-2.0. See [LICENSE](LICENSE).
