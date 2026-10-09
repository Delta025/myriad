import torch

from myriad.model.kv import KVCache
from myriad.model.masks import build_masks


def test_masks_match_window_rules():
    masks = build_masks(start_pos=3, n_new=2, sliding_window=3)
    # queries at positions 3 and 4; global keys 0..4, sliding keys 1..4 (position 0 is out of every window)
    assert masks["full_attention"][0, 0].int().tolist() == [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]
    assert masks["sliding_attention"][0, 0].int().tolist() == [[1, 1, 1, 0], [0, 1, 1, 1]]


def test_plain_causal_masks_are_left_to_sdpa():
    decode = build_masks(start_pos=10, n_new=1, sliding_window=4)
    assert decode["full_attention"] is None  # one query sees every key
    assert decode["sliding_attention"] is not None  # 4 keys >= window: Transformers keeps this mask
    prefill = build_masks(start_pos=0, n_new=3, sliding_window=4)
    assert prefill["full_attention"] is None and prefill["sliding_attention"] is None
    assert build_masks(start_pos=0, n_new=5, sliding_window=4)["sliding_attention"] is not None


def test_cache_appends_and_truncates():
    cache = KVCache()
    k = torch.arange(5.0).reshape(1, 1, 5, 1)
    full_k, _ = cache.update(k, k, layer_idx=3)
    cache.length = 5
    full_k, _ = cache.update(k[:, :, :2] + 10, k[:, :, :2] + 10, layer_idx=3)
    assert full_k.flatten().tolist() == [0, 1, 2, 3, 4, 10, 11]
    cache.length = 7
    cache.truncate(4)
    assert cache.length == 4
    assert cache.layer(3)[0].flatten().tolist() == [0, 1, 2, 3]
    cache.truncate(6)  # longer than what is cached: no-op
    assert cache.length == 4
