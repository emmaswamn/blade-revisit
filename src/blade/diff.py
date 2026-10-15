"""Change detection stubs — week 2 fills absolute diff + connected components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DiffStats:
    blade_width_px: float | None = None
    area_ratio_2017: float | None = None
    area_ratio_2018: float | None = None
    area_delta: float | None = None


def estimate_change_stub() -> DiffStats:
    return DiffStats()
