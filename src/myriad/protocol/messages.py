"""Messages between the client and peers.

Every message is one msgpack-encoded WebSocket frame holding a dict with a
``type`` and an ``id``; the reply echoes the ``id``. Tensors travel as raw
bytes (msgpack extension type 1), so they arrive bit-identical, bf16 included.

Client → peer                                  Peer → client
-------------                                  -------------
hello         {client_id}                      hello_ok   {peer_id, model, start, end, region, protocol}
open_session  {session}                        ok
forward       {session, start_pos, hidden,     forward_ok {hidden, queue_ms, compute_ms}
               per_layer_inputs?}
truncate      {session, length}                ok
close_session {session}                        ok
                                               error      {code, message}

``forward`` runs positions ``start_pos .. start_pos + n`` through the peer's
layers. ``hidden`` is ``[1, n, hidden_size]``; ``per_layer_inputs`` (only on
models with per-layer embeddings) is ``[1, n, layers on this peer, ple_dim]``.
The peer first drops anything it cached for the session from ``start_pos`` on,
so rolling back rejected draft tokens needs no extra message.
"""

import msgpack
import torch

PROTOCOL_VERSION = 1

_TENSOR_EXT = 1
_DTYPES = {str(d).removeprefix("torch."): d for d in (torch.float32, torch.bfloat16, torch.float16, torch.int64)}


def _pack_tensor(t: torch.Tensor) -> bytes:
    t = t.detach().cpu().contiguous()
    data = t.reshape(-1).view(torch.uint8).numpy().tobytes()
    return msgpack.packb([str(t.dtype).removeprefix("torch."), list(t.shape), data])


def _unpack_tensor(payload: bytes) -> torch.Tensor:
    dtype, shape, data = msgpack.unpackb(payload)
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported tensor dtype {dtype!r}")
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).view(_DTYPES[dtype]).reshape(shape)


def _default(obj):
    if isinstance(obj, torch.Tensor):
        return msgpack.ExtType(_TENSOR_EXT, _pack_tensor(obj))
    raise TypeError(f"cannot encode {type(obj).__name__}")


def _ext_hook(code: int, data: bytes):
    if code == _TENSOR_EXT:
        return _unpack_tensor(data)
    return msgpack.ExtType(code, data)


def encode(message: dict) -> bytes:
    return msgpack.packb(message, default=_default)


def decode(frame: bytes) -> dict:
    message = msgpack.unpackb(frame, ext_hook=_ext_hook)
    if not isinstance(message, dict) or "type" not in message:
        raise ValueError("a message must be a dict with a 'type'")
    return message


def error(request: dict, code: str, text: str) -> dict:
    return {"type": "error", "id": request.get("id"), "code": code, "message": text}
