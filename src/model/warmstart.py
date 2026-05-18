from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from src.model.BPC_compartment.feasibility_check import ExactFeasibilityEvaluator
from src.model.BPC_compartment.labeling import CompartmentSpec
from src.utility.config import config as Config


@dataclass
class WarmstartLoadResult:
    added: int = 0
    skipped_infeasible: int = 0
    skipped_duplicate: int = 0
    skipped_empty: int = 0
    skipped_unknown: int = 0
    skipped_over_limit: int = 0

    def __iadd__(self, other: "WarmstartLoadResult") -> "WarmstartLoadResult":
        self.added += other.added
        self.skipped_infeasible += other.skipped_infeasible
        self.skipped_duplicate += other.skipped_duplicate
        self.skipped_empty += other.skipped_empty
        self.skipped_unknown += other.skipped_unknown
        self.skipped_over_limit += other.skipped_over_limit
        return self


def load_wagon_warmstart_columns(master, json_paths: Iterable[Tuple[Path, str]]) -> WarmstartLoadResult:
    from src.model.BPC_wagon.CG import PatternColumn

    result = WarmstartLoadResult()
    seen = {
        tuple(int(col.q.get(i, 0)) for i in master.I)
        for col in master.columns.values()
    }
    for json_path, prefix in json_paths:
        data = _load_json(json_path)
        if data is None:
            continue
        single = WarmstartLoadResult()
        name_map = _build_name_map(master)
        evaluator = ExactFeasibilityEvaluator(compute_reachable_types=False)
        for idx, carriage in enumerate(data.get("carriage", [])):
            deck = str(carriage.get("position", "h-h")).strip()
            upper, upper_unknown = _quantities_from_names(carriage.get("top", []), name_map)
            lower, lower_unknown = _quantities_from_names(carriage.get("bottom", []), name_map)
            single.skipped_unknown += upper_unknown + lower_unknown
            if not upper and not lower:
                single.skipped_empty += 1
                continue
            if not _compartment_is_valid(master, evaluator, "upper", deck, upper):
                single.skipped_infeasible += 1
                continue
            if not _compartment_is_valid(master, evaluator, "lower", deck, lower):
                single.skipped_infeasible += 1
                continue
            q = {i: upper.get(i, 0) + lower.get(i, 0) for i in master.I}
            q = {i: qty for i, qty in q.items() if qty > 0}
            if any(qty > int(master.U[i]) for i, qty in q.items()):
                single.skipped_over_limit += 1
                continue
            signature = tuple(int(q.get(i, 0)) for i in master.I)
            if signature in seen:
                single.skipped_duplicate += 1
                continue
            seen.add(signature)
            master.add_column(
                PatternColumn(
                    column_id=f"{prefix}_c{idx}",
                    q=q,
                    cost=-sum(master.length[i] * qty for i, qty in q.items()),
                    metadata={"source": prefix, "deck": deck, "warmstart": "json"},
                )
            )
            single.added += 1
        _print_result(json_path, prefix, single, "wagon")
        result += single
    return result


def load_compartment_warmstart_columns(master, json_paths: Iterable[Tuple[Path, str]]) -> WarmstartLoadResult:
    from src.model.BPC_compartment.CG import PatternColumn

    result = WarmstartLoadResult()
    seen = {
        (col.compartment, col.deck_mode, tuple(int(col.q.get(i, 0)) for i in master.I))
        for col in master.columns.values()
    }
    for json_path, prefix in json_paths:
        data = _load_json(json_path)
        if data is None:
            continue
        single = WarmstartLoadResult()
        name_map = _build_name_map(master)
        evaluator = ExactFeasibilityEvaluator(compute_reachable_types=False)
        for idx, carriage in enumerate(data.get("carriage", [])):
            deck = str(carriage.get("position", "h-h")).strip()
            for compartment, key in [("upper", "top"), ("lower", "bottom")]:
                q, unknown = _quantities_from_names(carriage.get(key, []), name_map)
                single.skipped_unknown += unknown
                if not q:
                    single.skipped_empty += 1
                    continue
                if any(qty > int(master.U[i]) for i, qty in q.items()):
                    single.skipped_over_limit += 1
                    continue
                if not _compartment_is_valid(master, evaluator, compartment, deck, q):
                    single.skipped_infeasible += 1
                    continue
                signature = (compartment, deck, tuple(int(q.get(i, 0)) for i in master.I))
                if signature in seen:
                    single.skipped_duplicate += 1
                    continue
                seen.add(signature)
                master.add_column(
                    PatternColumn(
                        column_id=f"{prefix}_c{idx}_{compartment}",
                        q=q,
                        cost=-sum(master.length[i] * qty for i, qty in q.items()),
                        compartment=compartment,
                        deck_mode=deck,
                        metadata={"source": prefix, "warmstart": "json"},
                    )
                )
                single.added += 1
        _print_result(json_path, prefix, single, "compartment")
        result += single
    return result


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"[Warmstart] missing JSON: {path}")
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_name_map(master) -> Dict[str, int]:
    candidates: Dict[str, set[int]] = defaultdict(set)
    for i in master.I:
        row = master.car_info.iloc[i - 1]
        brand = str(row["program"]).strip()
        model = str(row["model"]).strip()
        candidates[f"{brand} {model}"].add(i)
        candidates[f"{brand}-{model}"].add(i)
        candidates[model].add(i)
    return {name: next(iter(ids)) for name, ids in candidates.items() if len(ids) == 1}


def _quantities_from_names(names: List[str], name_map: Dict[str, int]) -> Tuple[Dict[int, int], int]:
    quantities: Dict[int, int] = {}
    unknown = 0
    for raw_name in names:
        name = str(raw_name).strip()
        car_id = name_map.get(name)
        if car_id is None:
            unknown += 1
            continue
        quantities[car_id] = quantities.get(car_id, 0) + 1
    return quantities, unknown


def _compartment_is_valid(master, evaluator: ExactFeasibilityEvaluator, compartment: str, deck: str, q: Dict[int, int]) -> bool:
    if not q:
        return True
    if sum(q.values()) > Config.max_units_per_compartment:
        return False
    compartment_spec = CompartmentSpec(
        compartment_id=f"{compartment}_{deck}",
        car_types=master.I,
        car_lengths=master.length,
        compartment_length_limit=Config.top_len if compartment == "upper" else Config.bottom_len,
        car_heights={i: float(master.car_info.iloc[i - 1]["height"]) for i in master.I},
        shape_params={"compartment": compartment, "deck": deck},
        max_quantity_by_type={
            i: min(Config.max_units_per_compartment, int(master.U[i]))
            for i in master.I
        },
    )
    return evaluator.evaluate(compartment_spec, q).feasible


def _print_result(path: Path, prefix: str, result: WarmstartLoadResult, family: str) -> None:
    print(
        f"[Warmstart] {family} {prefix} from {path.name}: "
        f"added={result.added}, infeasible={result.skipped_infeasible}, "
        f"duplicate={result.skipped_duplicate}, empty={result.skipped_empty}, "
        f"unknown={result.skipped_unknown}, over_limit={result.skipped_over_limit}"
    )
