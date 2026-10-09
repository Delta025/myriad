import pytest
import torch

from myriad.protocol.messages import decode, encode


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_tensors_round_trip_bit_exact(dtype):
    t = torch.randn(1, 5, 7, generator=torch.Generator().manual_seed(0)).to(dtype)
    t[0, 0, 0] = float("inf")
    back = decode(encode({"type": "forward", "hidden": t}))["hidden"]
    assert back.dtype == dtype and back.shape == t.shape
    assert torch.equal(back.view(torch.uint8), t.view(torch.uint8))  # same bits, not just close


def test_non_contiguous_tensor_and_plain_fields():
    t = torch.arange(12.0).reshape(3, 4).T  # transposed view
    msg = decode(encode({"type": "x", "id": 3, "start_pos": 17, "hidden": t, "name": "é"}))
    assert torch.equal(msg["hidden"], t) and msg["start_pos"] == 17 and msg["name"] == "é"


def test_rejects_messages_without_type():
    import msgpack

    with pytest.raises(ValueError):
        decode(msgpack.packb([1, 2]))
