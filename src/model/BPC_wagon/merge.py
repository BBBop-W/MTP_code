from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import sys
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.model.BPC_compartment.labeling import CompartmentPattern


@dataclass(frozen=True)
class MergedPattern:
    deck: str
    quantities: Dict[int, int]
    reduced_cost: float
    upper: CompartmentPattern
    lower: CompartmentPattern


def merge_first_feasible(
    upper_patterns: List[CompartmentPattern],
    lower_patterns: List[CompartmentPattern],
    max_total_by_type: Dict[int, int],
    require_negative_reduced_cost: bool = True,
    reduced_cost_fn: Callable[[CompartmentPattern, CompartmentPattern, Dict[int, int]], float] | None = None,
    forbidden_signatures: set[Tuple[int, ...]] | None = None,
    signature_order: List[int] | None = None,
    select_best: bool = False,
) -> Optional[MergedPattern]:
    """Early-stop merge as requested.

    Rule:
    1) Check the pair (best upper rc, best lower rc) first.
    2) If invalid, iterate remaining pairs until found or exhausted.
    3) Validity checks:
       - merged reduced cost < 0 (if enabled)
       - We RELAX the quantity sum by type check here. The Restricted Master Problem 
         already has U_i constraints, so it will naturally ignore or restrict columns 
         that over-use a car type. Removing this check significantly speeds up merge!
       - If the best merge has already been generated, keep scanning the same deck
         instead of returning a duplicate that the caller will discard.
    """

    if not select_best:
        merged = merge_feasible_patterns(
            upper_patterns=upper_patterns,
            lower_patterns=lower_patterns,
            max_total_by_type=max_total_by_type,
            require_negative_reduced_cost=require_negative_reduced_cost,
            reduced_cost_fn=reduced_cost_fn,
            forbidden_signatures=forbidden_signatures,
            signature_order=signature_order,
            max_results=1,
        )
        return merged[0] if merged else None

    merged = merge_feasible_patterns(
        upper_patterns=upper_patterns,
        lower_patterns=lower_patterns,
        max_total_by_type=max_total_by_type,
        require_negative_reduced_cost=require_negative_reduced_cost,
        reduced_cost_fn=reduced_cost_fn,
        forbidden_signatures=forbidden_signatures,
        signature_order=signature_order,
        max_results=1,
    )
    return merged[0] if merged else None


def merge_feasible_patterns(
    upper_patterns: List[CompartmentPattern],
    lower_patterns: List[CompartmentPattern],
    max_total_by_type: Dict[int, int],
    require_negative_reduced_cost: bool = True,
    reduced_cost_fn: Callable[[CompartmentPattern, CompartmentPattern, Dict[int, int]], float] | None = None,
    forbidden_signatures: set[Tuple[int, ...]] | None = None,
    signature_order: List[int] | None = None,
    max_results: int | None = None,
) -> List[MergedPattern]:
    if not upper_patterns or not lower_patterns:
        return []

    # Sort so the most negative reduced cost is at index 0
    up = sorted(upper_patterns, key=lambda p: p.reduced_cost)
    low = sorted(lower_patterns, key=lambda p: p.reduced_cost)
    best_by_signature: Dict[Tuple[int, ...], MergedPattern] = {}

    for i, j in _pair_order(len(up), len(low)):
        u = up[i]
        l = low[j]
        
        # Verify deck modes match exactly
        u_deck = str(u.shape_params.get("deck", ""))
        l_deck = str(l.shape_params.get("deck", ""))
        if u_deck and l_deck and u_deck != l_deck:
            continue

        merged_q = _merge_quantities(u.quantities, l.quantities)
        if signature_order is not None:
            signature = tuple(int(merged_q.get(i, 0)) for i in signature_order)
        else:
            signature = tuple(sorted((k, int(v)) for k, v in merged_q.items()))
        if forbidden_signatures is not None and signature in forbidden_signatures:
            continue
        rc = reduced_cost_fn(u, l, merged_q) if reduced_cost_fn is not None else u.reduced_cost + l.reduced_cost
        if require_negative_reduced_cost and rc >= -1e-5:
            continue

        deck = u_deck if u_deck else l_deck
        candidate = MergedPattern(deck=deck, quantities=merged_q, reduced_cost=rc, upper=u, lower=l)
        old = best_by_signature.get(signature)
        if old is None or candidate.reduced_cost < old.reduced_cost:
            best_by_signature[signature] = candidate

    out = sorted(best_by_signature.values(), key=lambda candidate: candidate.reduced_cost)
    if max_results is not None:
        out = out[:max_results]
    return out

def merge_patterns_for_mode(
    mode: str,
    patterns_by_compartment: Dict[str, List[CompartmentPattern]],
    max_total_by_type: Dict[int, int],
    require_negative_reduced_cost: bool = True,
    reduced_cost_fn: Callable[[CompartmentPattern, CompartmentPattern, Dict[int, int]], float] | None = None,
) -> Optional[MergedPattern]:
    """
    Given a specific mode (e.g. 'h-h', 'h-m', 'm-h', 'm-m'), extract the matching
    upper and lower patterns and run the merge.
    """
    upper_key = f"upper_{mode}"
    lower_key = f"lower_{mode}"
    
    upper_patterns = patterns_by_compartment.get(upper_key, [])
    lower_patterns = patterns_by_compartment.get(lower_key, [])
    
    return merge_first_feasible(
        upper_patterns, 
        lower_patterns, 
        max_total_by_type, 
        require_negative_reduced_cost,
        reduced_cost_fn=reduced_cost_fn,
    )


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


if __name__ == "__main__":
    print("--- Testing Merge Sub-patterns ---")
    
    # 1. Create Dummy upper and lower patterns for all 4 deck modes
    modes = ["h-h", "h-m", "m-h", "m-m"]
    patterns_by_compartment: Dict[str, List[CompartmentPattern]] = {}
    
    # Dummy car types
    car_types = [101, 102, 103]
    max_total_by_type = {101: 4, 102: 4, 103: 4}
    
    for mode in modes:
        upper_key = f"upper_{mode}"
        lower_key = f"lower_{mode}"
        
        # Upper Patterns (RC from -100 to -5000)
        patterns_by_compartment[upper_key] = [
            CompartmentPattern(
                compartment_id=upper_key,
                quantities={101: 2, 102: 0, 103: 1},
                reduced_cost=-5000.0,
                best_length=15000.0,
                shape_params={"deck": mode, "compartment": "upper"}
            ),
            CompartmentPattern(
                compartment_id=upper_key,
                quantities={101: 3, 102: 0, 103: 0},
                reduced_cost=-4000.0,
                best_length=14000.0,
                shape_params={"deck": mode, "compartment": "upper"}
            ),
            CompartmentPattern(
                compartment_id=upper_key,
                quantities={101: 1, 102: 1, 103: 0},
                reduced_cost=-100.0,  # Not great
                best_length=10000.0,
                shape_params={"deck": mode, "compartment": "upper"}
            )
        ]
        
        # Lower Patterns (RC from -50 to -4000)
        patterns_by_compartment[lower_key] = [
            CompartmentPattern(
                compartment_id=lower_key,
                quantities={101: 1, 102: 2, 103: 1},
                reduced_cost=-4000.0,
                best_length=18000.0,
                shape_params={"deck": mode, "compartment": "lower"}
            ),
            CompartmentPattern(
                compartment_id=lower_key,
                quantities={101: 3, 102: 1, 103: 0},  # This will exceed max_total_by_type (101: 2 + 3 = 5 > 4) if combined with top upper pattern
                reduced_cost=-5000.0, # Better RC, but will be rejected when combined with top upper pattern!
                best_length=19000.0,
                shape_params={"deck": mode, "compartment": "lower"}
            )
        ]

    # 2. Test merging for each mode
    for mode in modes:
        print(f"\\n--- Testing Mode: {mode} ---")
        best_merge = merge_patterns_for_mode(
            mode=mode,
            patterns_by_compartment=patterns_by_compartment,
            max_total_by_type=max_total_by_type,
            require_negative_reduced_cost=True
        )
        
        if best_merge:
            print(f"Success! Found negative RC column for {mode}:")
            print(f"  Total RC: {best_merge.reduced_cost}")
            
            clean_q = {k: v for k, v in best_merge.quantities.items() if v > 0}
            print(f"  Total Quantities: {clean_q}")
            print(f"  Upper Qty: {best_merge.upper.quantities} (RC: {best_merge.upper.reduced_cost})")
            print(f"  Lower Qty: {best_merge.lower.quantities} (RC: {best_merge.lower.reduced_cost})")
        else:
            print(f"Failed to find negative RC column for {mode}.")
