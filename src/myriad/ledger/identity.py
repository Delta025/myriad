"""Node identities and signed work receipts.

A node is an Ed25519 key pair. Its peer and its client use the same key, so
work the node does for others and work others do for it land in one ledger.

A receipt says "peer P ran layers a..b for n positions of client C's session S".
The peer signs it and the client countersigns it, so each side holds proof that
the other agreed (in the style of Tribler's TrustChain).
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import msgpack
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

RECEIPT_VERSION = 1


class Identity:
    def __init__(self, private_key: Ed25519PrivateKey, name: str = ""):
        self._key = private_key
        raw = private_key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.public_key = raw.hex()  # 64 hex characters; also the node's id on the wire
        self.name = name or self.public_key[:8]

    @classmethod
    def generate(cls, name: str = "") -> "Identity":
        return cls(Ed25519PrivateKey.generate(), name)

    @classmethod
    def load_or_create(cls, directory: Path, name: str = "") -> "Identity":
        """The key in `directory/identity.key`, created on first use."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "identity.key"
        if path.exists():
            key = serialization.load_pem_private_key(path.read_bytes(), password=None)
        else:
            key = Ed25519PrivateKey.generate()
            path.write_bytes(
                key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
            )
        return cls(key, name)

    def sign(self, data: bytes) -> bytes:
        return self._key.sign(data)


def verify(public_key_hex: str, signature: bytes, data: bytes) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex)).verify(signature, data)
        return True
    except (InvalidSignature, ValueError):
        return False


@dataclass(frozen=True)
class Receipt:
    peer: str  # public key of the node that did the work
    client: str  # public key of the node it was done for
    session: str
    start: int  # layers start..end-1
    end: int
    positions: int  # token positions processed in this call
    seq: int  # per session, so receipts can't be replayed
    time: float

    @property
    def units(self) -> int:
        """Work done, in layer-positions."""
        return (self.end - self.start) * self.positions

    def payload(self) -> bytes:
        """The exact bytes both sides sign."""
        return msgpack.packb([RECEIPT_VERSION, self.peer, self.client, self.session, self.start, self.end,
                              self.positions, self.seq, self.time])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Receipt":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})
