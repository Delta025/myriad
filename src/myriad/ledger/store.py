"""A node's local ledger: the receipts it signed or received, in SQLite.

Each node keeps its own. There is no global ledger: a peer decides whom to
serve first from its own records of who has worked for it.
"""

import sqlite3
import threading
from pathlib import Path

from myriad.ledger.identity import Receipt


class Ledger:
    def __init__(self, owner: str, path: Path | str = ":memory:"):
        """`owner`: this node's public key."""
        self.owner = owner
        self._lock = threading.Lock()  # the peer's event loop and its client's thread share it
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS receipts (
                peer TEXT, client TEXT, session TEXT, seq INTEGER,
                start INTEGER, "end" INTEGER, positions INTEGER, units INTEGER, time REAL,
                peer_sig BLOB, client_sig BLOB,
                PRIMARY KEY (peer, session, seq)
            );
            CREATE TABLE IF NOT EXISTS names (key TEXT PRIMARY KEY, name TEXT);
            """
        )

    def record(self, receipt: Receipt, peer_sig: bytes, client_sig: bytes | None = None) -> None:
        """Store a receipt this node is party to (verify signatures before calling)."""
        if self.owner not in (receipt.peer, receipt.client):
            raise ValueError("this node is not party to the receipt")
        with self._lock:
            self._db.execute(
                """INSERT INTO receipts VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT (peer, session, seq) DO UPDATE SET client_sig = COALESCE(excluded.client_sig, client_sig)""",
                (receipt.peer, receipt.client, receipt.session, receipt.seq, receipt.start, receipt.end,
                 receipt.positions, receipt.units, receipt.time, peer_sig, client_sig),
            )
            self._db.commit()

    def add_countersignature(self, peer: str, session: str, seq: int, client_sig: bytes) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE receipts SET client_sig = ? WHERE peer = ? AND session = ? AND seq = ?",
                (client_sig, peer, session, seq),
            )
            self._db.commit()

    def received_from(self, other: str) -> int:
        """Work `other` has done for this node, in layer-positions."""
        return self._sum("SELECT SUM(units) FROM receipts WHERE peer = ? AND client = ?", other, self.owner)

    def given_to(self, other: str) -> int:
        """Work this node has done for `other`."""
        return self._sum("SELECT SUM(units) FROM receipts WHERE peer = ? AND client = ?", self.owner, other)

    def _sum(self, query: str, *args) -> int:
        with self._lock:
            return self._db.execute(query, args).fetchone()[0] or 0

    def set_name(self, key: str, name: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO names VALUES (?, ?)", (key, name))
            self._db.commit()

    def name(self, key: str) -> str:
        with self._lock:
            row = self._db.execute("SELECT name FROM names WHERE key = ?", (key,)).fetchone()
        return row[0] if row else key[:8]

    def balances(self) -> list[dict]:
        """Per counterparty: work received from them and given to them."""
        with self._lock:
            rows = self._db.execute(
                """SELECT other, SUM(recv), SUM(given) FROM (
                     SELECT peer AS other, units AS recv, 0 AS given FROM receipts WHERE client = ? AND peer != ?
                     UNION ALL
                     SELECT client AS other, 0, units FROM receipts WHERE peer = ? AND client != ?)
                   GROUP BY other""",
                (self.owner, self.owner, self.owner, self.owner),
            ).fetchall()
        return [{"key": k, "name": self.name(k), "received": r, "given": g} for k, r, g in rows]
