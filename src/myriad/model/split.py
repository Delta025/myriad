"""Rules for where a Gemma 4 model can be cut into stages.

A stage is a half-open layer range ``(start, end)``. A split is a list of stages
that covers every layer once, in order.

E2B and E4B add one rule. Their last ``num_kv_shared_layers`` layers have no
K/V projections: each reuses the K/V of the last non-shared layer of the same
type (sliding or global), handed over inside a single forward call. So the
shared layers and the layers they read from must be in the same stage.
"""

Split = list[tuple[int, int]]


def kv_source_layers(config) -> dict[str, int]:
    """For each layer type, the layer whose K/V the shared layers reuse. Empty if no sharing."""
    n_layers = config.num_hidden_layers
    n_shared = getattr(config, "num_kv_shared_layers", 0) or 0
    if n_shared == 0:
        return {}
    first_shared = n_layers - n_shared
    earlier = config.layer_types[:first_shared]
    return {
        layer_type: first_shared - 1 - earlier[::-1].index(layer_type)
        for layer_type in set(config.layer_types[first_shared:])
    }


def unsplittable_block(config) -> tuple[int, int] | None:
    """The layer range ``(start, end)`` that must stay in one stage, if any."""
    sources = kv_source_layers(config)
    if not sources:
        return None
    return min(sources.values()), config.num_hidden_layers


def validate_split(config, split: Split) -> None:
    """Raise ValueError unless `split` covers all layers in order and keeps the KV-sharing block whole."""
    n_layers = config.num_hidden_layers
    if not split:
        raise ValueError("split is empty")
    expected_start = 0
    for start, end in split:
        if start != expected_start or end <= start:
            raise ValueError(f"stages must be contiguous and non-empty, got {split}")
        expected_start = end
    if expected_start != n_layers:
        raise ValueError(f"split {split} does not cover all {n_layers} layers")

    block = unsplittable_block(config)
    if block is not None:
        for _, end in split[:-1]:
            if block[0] < end < block[1]:
                raise ValueError(
                    f"split point {end} falls inside layers {block[0]}-{block[1] - 1}, "
                    "which share K/V and must stay in one stage"
                )


def even_split(config, n_stages: int) -> Split:
    """Cut the model into `n_stages` stages of roughly equal layer counts, respecting the KV-sharing block.

    The KV-sharing block counts as a single unit, so a stage that contains it may be larger.
    """
    n_layers = config.num_hidden_layers
    block = unsplittable_block(config)
    cut_points = [i for i in range(1, n_layers) if block is None or not (block[0] < i < block[1])]
    if n_stages - 1 > len(cut_points):
        raise ValueError(f"cannot cut {n_layers} layers into {n_stages} stages")

    chosen: list[int] = []
    for k in range(1, n_stages):
        target = k * n_layers / n_stages
        best = min((c for c in cut_points if c not in chosen and (not chosen or c > chosen[-1])), key=lambda c: abs(c - target))
        chosen.append(best)

    bounds = [0, *chosen, n_layers]
    split = list(zip(bounds[:-1], bounds[1:]))
    validate_split(config, split)
    return split
