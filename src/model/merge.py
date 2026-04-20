from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from src.model.labeling import LayerPattern


@dataclass(frozen=True)
class MergedPattern:
    deck: str
    quantities: Dict[int, int]
    reduced_cost: float
    upper: LayerPattern
    lower: LayerPattern


def merge_first_feasible(
    upper_patterns: List[LayerPattern],
    lower_patterns: List[LayerPattern],
    max_total_by_type: Dict[int, int],
    require_negative_reduced_cost: bool = True,
) -> Optional[MergedPattern]:
    """Early-stop merge as requested.

    Rule:
    1) Check the pair (best upper rc, best lower rc) first.
    2) If invalid, iterate remaining pairs until found or exhausted.
    3) Validity checks:
       - merged reduced cost < 0 (if enabled)
       - quantity sum by type <= max_total_by_type
    """

    if not upper_patterns or not lower_patterns:
        return None

    up = sorted(upper_patterns, key=lambda p: p.reduced_cost)
    low = sorted(lower_patterns, key=lambda p: p.reduced_cost)

    for i, j in _pair_order(len(up), len(low)):
        u = up[i]
        l = low[j]
        merged_q = _merge_quantities(u.quantities, l.quantities)

        if any(merged_q.get(t, 0) > max_total_by_type.get(t, 0) for t in merged_q):
            continue

        rc = u.reduced_cost + l.reduced_cost
        if require_negative_reduced_cost and rc >= 0.0:
            continue

        deck = str(u.shape_params.get("deck", ""))
        if deck and str(l.shape_params.get("deck", "")) and deck != str(l.shape_params.get("deck", "")):
            continue

        return MergedPattern(deck=deck, quantities=merged_q, reduced_cost=rc, upper=u, lower=l)

    return None


def _merge_quantities(a: Dict[int, int], b: Dict[int, int]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for k in set(a.keys()) | set(b.keys()):
        out[k] = int(a.get(k, 0) + b.get(k, 0))
    return out


def _pair_order(n_up: int, n_low: int) -> Iterable[Tuple[int, int]]:
    # First try the two best reduced-cost subpatterns directly.
    yield 0, 0
    for i in range(n_up):
        for j in range(n_low):
            if i == 0 and j == 0:
                continue
            yield i, j
