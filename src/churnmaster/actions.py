"""Maintenance action identifiers (observable-only; stdlib only)."""
from __future__ import annotations
from enum import IntEnum
__all__ = ["Action"]
class Action(IntEnum):
    KEEP = 0
    PROBE = 1
    REPLACE = 2
