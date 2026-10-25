"""XOR ID-space geometry for the 8-bit Kademlia-like DHT.

Pure functions over integer node IDs. Imports only stdlib and numpy.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

ID_BITS = 8
ID_SPACE = 1 << ID_BITS


def xor_dist(a: int, b: int) -> int:
    """Kademlia distance between two IDs."""
    return a ^ b


def bucket_index(f: int, x: int) -> int:
    """Bucket of ``x`` relative to focal node ``f``: floor(log2(f ^ x)).

    For x != f the result lies in [0, id_bits - 1]. By convention a lookup
    key equal to F is attributed to bucket 0 (Requirement 3.4).
    """
    d = f ^ x
    if d == 0:
        return 0
    return d.bit_length() - 1


def closest(target: int, ids: Iterable[int], n: int) -> list[int]:
    """Return the ``n`` IDs closest to ``target`` by increasing XOR distance.

    IDs are unique, so XOR distances to a fixed target are unique and no
    tie-breaking is needed.
    """
    if n <= 0:
        return []
    return sorted(ids, key=lambda i: i ^ target)[:n]


def draw_ids(rng: np.random.Generator, n: int = 128, bits: int = ID_BITS) -> list[int]:
    """Draw ``n`` unique IDs uniformly from the 2**bits possible values."""
    space = 1 << bits
    if not 0 < n <= space:
        raise ValueError(f"cannot draw {n} unique ids from {space} values")
    return [int(v) for v in rng.choice(space, size=n, replace=False)]
