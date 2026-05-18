from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import src.model.BPC_layer.feasibility_check as feasibility_check
from src.experiments.compare_ex_gr_equal_lengths import (
    ExperimentResourceModel,
    _build_full_resource_model,
    _choice_count_vectors,
    _prune_outer_dominated_choices,
)
from src.model.BPC_layer.BBtree import BBTree, normalize_car_table
from src.model.BPC_layer.labeling import LayerSpec
from src.utility.config import config as Config


Residual = Tuple[float, ...]


@dataclass(frozen=True)
class LayerBenchCase:
    name: str
    instance_dir: Path
    compartment: str
    deck: str
    splits: int
    quantity_cap: int


@dataclass(frozen=True)
class MethodConfig:
    name: str
    placement_mode: str
    use_dominance: bool
    use_local_skyline: bool = True


@dataclass(frozen=True)
class ResidualLabel:
    stage: int
    reduced_cost: float
    quantities: Tuple[int, ...]
    nominal_length: float
    residual: Residual


@dataclass
class BenchStats:
    placement_assignments_attempted: int = 0
    placement_assignments_feasible_raw: int = 0
    labels_created: int = 0
    labels_feasible: int = 0
    labels_pruned_by_dominance: int = 0
    labels_pruned_by_local_skyline: int = 0
    labels_after_dominance: int = 0
    stage_attempted: List[int] | None = None
    stage_raw_feasible: List[int] | None = None
    stage_created: List[int] | None = None
    stage_kept: List[int] | None = None

    def __post_init__(self) -> None:
        self.stage_attempted = []
        self.stage_raw_feasible = []
        self.stage_created = []
        self.stage_kept = []


def _instance_layer(case: LayerBenchCase) -> LayerSpec:
    car_info = normalize_car_table(case.instance_dir / "cars.csv")
    car_types = list(range(1, len(car_info) + 1))
    return LayerSpec(
        layer_id=f"{case.name}_{case.compartment}_{case.deck}",
        car_types=car_types,
        car_lengths={i: float(car_info.iloc[i - 1]["length"]) for i in car_types},
        car_heights={i: float(car_info.iloc[i - 1]["height"]) for i in car_types},
        layer_length_limit=Config.top_len if case.compartment == "upper" else Config.bottom_len,
        shape_params={"compartment": case.compartment, "deck": case.deck},
        max_quantity_by_type={
            i: min(
                case.quantity_cap,
                int(car_info.iloc[i - 1]["mandatory"] + car_info.iloc[i - 1]["optional"]),
            )
            for i in car_types
        },
    )


def _resource_model(layer: LayerSpec, placement_mode: str) -> ExperimentResourceModel:
    full = _build_full_resource_model(layer, interval_profile="full")
    if placement_mode == "ex":
        return full
    if placement_mode == "gr":
        return _prune_outer_dominated_choices(full)
    raise ValueError(f"unknown placement_mode: {placement_mode}")


def _residual_dominates(a: Residual, b: Residual, eps: float = 1e-9) -> bool:
    return all(x + eps >= y for x, y in zip(a, b)) and any(x > y + eps for x, y in zip(a, b))


def _residual_weakly_dominates(a: Residual, b: Residual, eps: float = 1e-9) -> bool:
    return all(x + eps >= y for x, y in zip(a, b))


def _local_skyline(residuals: Iterable[Residual], stats: BenchStats, eps: float = 1e-9) -> List[Residual]:
    kept: List[Residual] = []
    seen = set()
    for residual in residuals:
        key = tuple(round(v, 8) for v in residual)
        if key in seen:
            stats.labels_pruned_by_local_skyline += 1
            continue
        seen.add(key)
        dominated = False
        remove_idx: List[int] = []
        for idx, old in enumerate(kept):
            if _residual_weakly_dominates(old, residual, eps):
                dominated = True
                break
            if _residual_dominates(residual, old, eps):
                remove_idx.append(idx)
        if dominated:
            stats.labels_pruned_by_local_skyline += 1
            continue
        if remove_idx:
            stats.labels_pruned_by_local_skyline += len(remove_idx)
            remove_set = set(remove_idx)
            kept = [value for idx, value in enumerate(kept) if idx not in remove_set]
        kept.append(residual)
    return kept


def _apply_consumption(residual: Residual, hits: Sequence[int], amount: float, eps: float = 1e-9) -> Residual | None:
    updated = list(residual)
    for idx in hits:
        updated[idx] -= amount
        if updated[idx] < -eps:
            return None
    return tuple(updated)


def _placement_residuals(
    resource_model: ExperimentResourceModel,
    residual: Residual,
    car_type: int,
    quantity: int,
    unit_resource: float,
    stats: BenchStats,
) -> Iterator[Residual]:
    if quantity == 0:
        stats.placement_assignments_attempted += 1
        stats.placement_assignments_feasible_raw += 1
        yield residual
        return
    choices = resource_model.choices_by_type.get(car_type, ())
    if not choices:
        return
    for counts in _choice_count_vectors(quantity, len(choices)):
        stats.placement_assignments_attempted += 1
        updated = residual
        feasible = True
        for count, choice in zip(counts, choices):
            if count == 0:
                continue
            updated = _apply_consumption(updated, choice.hits, count * unit_resource)
            if updated is None:
                feasible = False
                break
        if feasible:
            stats.placement_assignments_feasible_raw += 1
            yield updated


def _label_dominates(a: ResidualLabel, b: ResidualLabel, eps: float = 1e-9) -> bool:
    if a.reduced_cost > b.reduced_cost + eps:
        return False
    if not _residual_weakly_dominates(a.residual, b.residual, eps):
        return False
    return a.reduced_cost < b.reduced_cost - eps or _residual_dominates(a.residual, b.residual, eps)


def _apply_label_dominance(labels: List[ResidualLabel], stats: BenchStats, eps: float = 1e-9) -> List[ResidualLabel]:
    kept: List[ResidualLabel] = []
    for cand in sorted(labels, key=lambda label: label.reduced_cost):
        dominated = False
        for old in kept:
            if _label_dominates(old, cand, eps):
                dominated = True
                break
        if dominated:
            stats.labels_pruned_by_dominance += 1
            continue
        kept.append(cand)
    stats.labels_after_dominance += len(kept)
    return kept


def _quantity_key(car_types: Sequence[int], quantities: Sequence[int]) -> Tuple[Tuple[int, int], ...]:
    return tuple((car_type, q) for car_type, q in zip(car_types, quantities) if q > 0)


def _generate_residual_labels(layer: LayerSpec, resource_model: ExperimentResourceModel, method: MethodConfig) -> Tuple[List[ResidualLabel], BenchStats]:
    car_types = sorted(layer.car_types, key=lambda i: (-float(layer.car_heights.get(i, 0.0)), i))
    root = ResidualLabel(
        stage=0,
        reduced_cost=0.0,
        quantities=tuple(0 for _ in car_types),
        nominal_length=0.0,
        residual=resource_model.capacities,
    )
    current = [root]
    stats = BenchStats()

    for stage, car_type in enumerate(car_types, start=1):
        next_labels: List[ResidualLabel] = []
        attempted_before_stage = stats.placement_assignments_attempted
        raw_feasible_before_stage = stats.placement_assignments_feasible_raw
        max_q = int(layer.max_quantity_by_type.get(car_type, 0))
        unit_resource = layer.car_lengths[car_type] + resource_model.delta
        for label in current:
            for q in range(0, max_q + 1):
                child_quantities = list(label.quantities)
                child_quantities[stage - 1] = q
                residuals = list(_placement_residuals(resource_model, label.residual, car_type, q, unit_resource, stats))
                if method.use_local_skyline:
                    residuals = _local_skyline(residuals, stats)
                for residual in residuals:
                    nominal_length = label.nominal_length + layer.car_lengths[car_type] * q
                    next_labels.append(
                        ResidualLabel(
                            stage=stage,
                            reduced_cost=label.reduced_cost - layer.car_lengths[car_type] * q,
                            quantities=tuple(child_quantities),
                            nominal_length=nominal_length,
                            residual=residual,
                        )
                    )
        stats.labels_created += len(next_labels)
        stats.labels_feasible += len(next_labels)
        stats.stage_attempted.append(stats.placement_assignments_attempted - attempted_before_stage)
        stats.stage_raw_feasible.append(stats.placement_assignments_feasible_raw - raw_feasible_before_stage)
        stats.stage_created.append(len(next_labels))
        if method.use_dominance:
            next_labels = _apply_label_dominance(next_labels, stats)
        stats.stage_kept.append(len(next_labels))
        current = next_labels
        if not current:
            break

    return current, stats


def run_layer_method(case: LayerBenchCase, method: MethodConfig) -> Dict[str, object]:
    feasibility_check.GLOBAL_NUM_SPLITS = case.splits
    feasibility_check.GLOBAL_INDEP_MODE = True
    feasibility_check._SEGMENTS_CACHE.clear()

    layer = _instance_layer(case)
    model = _resource_model(layer, method.placement_mode)
    t0 = time.perf_counter()
    labels, stats = _generate_residual_labels(layer, model, method)
    wall_time = time.perf_counter() - t0
    final_labels = [
        label for label in labels
        if label.stage == len(layer.car_types) and label.nominal_length <= layer.layer_length_limit + 1e-9
    ]
    keys = {_quantity_key(sorted(layer.car_types, key=lambda i: (-float(layer.car_heights.get(i, 0.0)), i)), label.quantities) for label in final_labels}
    nonempty_keys = {key for key in keys if key}
    return {
        "case": case.name,
        "instance": case.instance_dir.name,
        "compartment": case.compartment,
        "deck": case.deck,
        "splits": case.splits,
        "quantity_cap": case.quantity_cap,
        "method": method.name,
        "placement_mode": method.placement_mode,
        "dominance": method.use_dominance,
        "local_skyline": method.use_local_skyline,
        "resources": len(model.capacities),
        "choices_total": sum(len(v) for v in model.choices_by_type.values()),
        "choices_max_per_type": max((len(v) for v in model.choices_by_type.values()), default=0),
        "final_labels": len(final_labels),
        "nonempty_quantity_keys": len(nonempty_keys),
        "placement_assignments_attempted": stats.placement_assignments_attempted,
        "placement_assignments_feasible_raw": stats.placement_assignments_feasible_raw,
        "labels_created": stats.labels_created,
        "labels_feasible": stats.labels_feasible,
        "labels_pruned_by_dominance": stats.labels_pruned_by_dominance,
        "labels_pruned_by_local_skyline": stats.labels_pruned_by_local_skyline,
        "labels_after_dominance": stats.labels_after_dominance,
        "dominance_prune_rate": stats.labels_pruned_by_dominance / stats.labels_feasible if stats.labels_feasible else 0.0,
        "local_skyline_prune_rate_raw": stats.labels_pruned_by_local_skyline / stats.placement_assignments_feasible_raw if stats.placement_assignments_feasible_raw else 0.0,
        "stage_attempted": "|".join(str(v) for v in stats.stage_attempted),
        "stage_raw_feasible": "|".join(str(v) for v in stats.stage_raw_feasible),
        "stage_created": "|".join(str(v) for v in stats.stage_created),
        "stage_kept": "|".join(str(v) for v in stats.stage_kept),
        "wall_time": wall_time,
    }


def _print_row(row: Dict[str, object]) -> None:
    print(
        f"{row['case']:20s} {row['method']:14s} "
        f"time={float(row['wall_time']):8.3f}s "
        f"choices={row['choices_total']:4} keys={row['nonempty_quantity_keys']:5} "
        f"attempts={row['placement_assignments_attempted']:10} labels={row['labels_feasible']:9} dom={row['labels_pruned_by_dominance']:8} "
        f"local={row['labels_pruned_by_local_skyline']:8}"
    )


def _write_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def default_methods() -> List[MethodConfig]:
    return [
        MethodConfig("EX_no_dom", "ex", use_dominance=False, use_local_skyline=True),
        MethodConfig("EX_dom", "ex", use_dominance=True, use_local_skyline=True),
        MethodConfig("EX_dom_no_local", "ex", use_dominance=True, use_local_skyline=False),
        MethodConfig("GR_no_dom", "gr", use_dominance=False, use_local_skyline=True),
        MethodConfig("GR_dom", "gr", use_dominance=True, use_local_skyline=True),
        MethodConfig("GR_dom_no_local", "gr", use_dominance=True, use_local_skyline=False),
    ]


def default_layer_cases() -> List[LayerBenchCase]:
    root = PROJECT_ROOT / "data" / "Instance"
    return [
        LayerBenchCase("m5_lower_hh_s3", root / "m5c5", "lower", "h-h", 3, 4),
        LayerBenchCase("m6_lower_hh_s3", root / "m6c6", "lower", "h-h", 3, 4),
        LayerBenchCase("m6_upper_hm_s3", root / "m6c6", "upper", "h-m", 3, 4),
        LayerBenchCase("m7_lower_hh_s3", root / "m7c7", "lower", "h-h", 3, 3),
    ]


def default_root_cases() -> List[Tuple[Path, int]]:
    root = PROJECT_ROOT / "data" / "Instance"
    return [
        (root / "m5c5", 3),
        (root / "m6c6", 3),
    ]


def run_root_gr(instance_dir: Path, splits: int, use_dominance: bool, max_cg_iters: int) -> Dict[str, object]:
    feasibility_check.GLOBAL_NUM_SPLITS = splits
    feasibility_check.GLOBAL_INDEP_MODE = True
    feasibility_check._SEGMENTS_CACHE.clear()

    tree = BBTree(
        instance_dir=instance_dir,
        output_root=PROJECT_ROOT / "result" / "label_pruning_benchmark_tmp",
        max_nodes=1,
        max_cg_iters=max_cg_iters,
        log_to_console=False,
        use_dominance=use_dominance,
        use_cuts=False,
        use_rc_bound=True,
        use_residual_profile=True,
        use_height_order=True,
        use_local_residual_skyline=True,
        residual_profile_mode="full",
        compute_reachable_types=False,
        print_bb_progress=False,
        print_subproblem_progress=False,
    )
    t0 = time.perf_counter()
    solution = tree.cg_engine.solve()
    wall_time = time.perf_counter() - t0
    stats = tree.cg_engine.stats
    return {
        "case": f"{instance_dir.name}_splits{splits}",
        "instance": instance_dir.name,
        "splits": splits,
        "method": "GR_dom" if use_dominance else "GR_no_dom",
        "objective": solution.objective if solution.objective is not None else "",
        "generated_columns": tree.cg_engine.generated_columns,
        "labels_feasible": stats.labels_feasible,
        "labels_pruned_by_bound": stats.labels_pruned_by_bound,
        "labels_pruned_by_dominance": stats.labels_pruned_by_dominance,
        "labels_pruned_by_local_skyline": stats.labels_pruned_by_local_skyline,
        "labels_after_dominance": stats.labels_after_dominance,
        "dominance_prune_rate": stats.labels_pruned_by_dominance / stats.labels_feasible if stats.labels_feasible else 0.0,
        "local_skyline_prune_rate": stats.labels_pruned_by_local_skyline / stats.labels_feasible if stats.labels_feasible else 0.0,
        "master_time": stats.master_time,
        "pricing_time": stats.pricing_time,
        "labeling_time": stats.labeling_time,
        "wall_time": wall_time,
    }


def _print_root_row(row: Dict[str, object]) -> None:
    print(
        f"{row['case']:14s} {row['method']:9s} "
        f"time={float(row['wall_time']):7.3f}s obj={row['objective']} cols={row['generated_columns']:5} "
        f"labels={row['labels_feasible']:8} dom={row['labels_pruned_by_dominance']:7} "
        f"local={row['labels_pruned_by_local_skyline']:7}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["layer", "root", "both"], default="both")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "result" / "label_pruning_benchmark" / "ex_gr_summary.csv")
    parser.add_argument("--root-out", type=Path, default=PROJECT_ROOT / "result" / "label_pruning_benchmark" / "root_gr_dominance_summary.csv")
    parser.add_argument("--max-cg-iters", type=int, default=120)
    args = parser.parse_args()

    if args.scope in {"layer", "both"}:
        layer_rows: List[Dict[str, object]] = []
        for case in default_layer_cases():
            for method in default_methods():
                row = run_layer_method(case, method)
                layer_rows.append(row)
                _print_row(row)
        _write_csv(layer_rows, args.out)
        print(f"\nwrote {args.out}")

    if args.scope in {"root", "both"}:
        root_rows: List[Dict[str, object]] = []
        for instance_dir, splits in default_root_cases():
            for use_dominance in (False, True):
                row = run_root_gr(instance_dir, splits, use_dominance, args.max_cg_iters)
                root_rows.append(row)
                _print_root_row(row)
        _write_csv(root_rows, args.root_out)
        print(f"\nwrote {args.root_out}")


if __name__ == "__main__":
    main()
