import pytest
from transformers import Gemma4TextConfig

from myriad.model.split import even_split, kv_source_layers, unsplittable_block, validate_split
from myriad.testing import VARIANTS


def e4b_like():
    # Layer pattern and KV sharing of the released E4B config.
    return Gemma4TextConfig(num_hidden_layers=42, num_kv_shared_layers=18, hidden_size_per_layer_input=256)


def test_e4b_kv_sharing_block_matches_release():
    config = e4b_like()
    assert kv_source_layers(config) == {"sliding_attention": 22, "full_attention": 23}
    assert unsplittable_block(config) == (22, 42)


def test_dense_model_can_be_cut_anywhere():
    config = Gemma4TextConfig(**VARIANTS["dense"])
    assert unsplittable_block(config) is None
    validate_split(config, [(i, i + 1) for i in range(8)])


@pytest.mark.parametrize(
    "split",
    [
        [(0, 4), (4, 8)],  # cuts the KV-sharing block (layers 2-7)
        [(0, 3), (3, 8)],
        [(0, 4), (5, 8)],  # gap
        [(0, 4)],  # incomplete
        [(0, 4), (4, 4), (4, 8)],  # empty stage
    ],
)
def test_invalid_splits_are_rejected(split):
    with pytest.raises(ValueError):
        validate_split(Gemma4TextConfig(**VARIANTS["ple"]), split)


def test_even_split_respects_block():
    split = even_split(e4b_like(), 3)
    assert split[0][0] == 0 and split[-1][1] == 42
    assert all(not (22 < end < 42) for _, end in split[:-1])
    assert len(even_split(Gemma4TextConfig(**VARIANTS["dense"]), 4)) == 4


def test_even_split_of_a_sub_range():
    config = e4b_like()
    split = even_split(config, 3, start=1, end=42)
    assert split[0][0] == 1 and split[-1][1] == 42 and len(split) == 3
    validate_split(config, [(0, 1), *split])
