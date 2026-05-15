from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import src.model.BPC_LayerMaster.feasibility_check as feasibility_check
from src.experiments.compare_ex_gr_equal_lengths import (
    ExperimentResourceModel,
    ToyCase,
    _apply_choice,
    _build_full_resource_model,
    _choice_count_vectors,
    _prune_outer_dominated_choices,
    _quantity_vectors,
    _skyline,
)
from src.model.BPC_LayerMaster.labeling import LayerSpec


Residual = Tuple[float, ...]
QuantityKey = Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class TinyLabel:
    stage: int
    quantities: Tuple[int, ...]
    residual: Residual


@dataclass
class LabelStats:
    labels_created: int = 0
    labels_after_stage: List[int] | None = None
    profile_states_after_stage: List[int] | None = None

    def __post_init__(self) -> None:
        self.labels_after_stage = []
        self.profile_states_after_stage = []


def _quantity_key(car_types: Sequence[int], quantities: Sequence[int]) -> QuantityKey:
    return tuple((car_type, q) for car_type, q in zip(car_types, quantities) if q > 0)


def _extend_residuals(
    resource_model: ExperimentResourceModel,
    residual: Residual,
    car_type: int,
    quantity: int,
    unit_resource: float,
) -> List[Residual]:
    choices = resource_model.choices_by_type.get(car_type, ())
    children: List[Residual] = []
    for counts in _choice_count_vectors(quantity, len(choices)):
        updated = residual
        feasible = True
        for count, choice in zip(counts, choices):
            if count == 0:
                continue
            updated = _apply_choice(updated, choice.hits, count * unit_resource)
            if updated is None:
                feasible = False
                break
        if feasible:
            children.append(updated)
    return _skyline(children)


def _apply_quantity_group_skyline(labels: Iterable[TinyLabel], car_types: Sequence[int]) -> List[TinyLabel]:
    grouped: Dict[Tuple[int, ...], List[Residual]] = {}
    stage_by_quantity: Dict[Tuple[int, ...], int] = {}
    for label in labels:
        grouped.setdefault(label.quantities, []).append(label.residual)
        stage_by_quantity[label.quantities] = label.stage

    kept: List[TinyLabel] = []
    for quantities, residuals in grouped.items():
        for residual in _skyline(residuals):
            kept.append(TinyLabel(stage=stage_by_quantity[quantities], quantities=quantities, residual=residual))
    return kept


def generate_label_quantity_set(
    resource_model: ExperimentResourceModel,
    car_types: Sequence[int],
    max_per_type: int,
    max_total: int,
    unit_resource: float,
) -> Tuple[set[QuantityKey], LabelStats]:
    stats = LabelStats()
    root = TinyLabel(
        stage=0,
        quantities=tuple(0 for _ in car_types),
        residual=resource_model.capacities,
    )
    labels: List[TinyLabel] = [root]

    for stage, car_type in enumerate(car_types, start=1):
        next_labels: List[TinyLabel] = []
        for label in labels:
            used_before = sum(label.quantities)
            for q in range(min(max_per_type, max_total - used_before) + 1):
                child_quantities = list(label.quantities)
                child_quantities[stage - 1] = q
                child_quantities_tuple = tuple(child_quantities)
                for residual in _extend_residuals(resource_model, label.residual, car_type, q, unit_resource):
                    next_labels.append(TinyLabel(stage=stage, quantities=child_quantities_tuple, residual=residual))
        stats.labels_created += len(next_labels)
        labels = _apply_quantity_group_skyline(next_labels, car_types)
        stats.labels_after_stage.append(len(labels))
        stats.profile_states_after_stage.append(sum(1 for _label in labels))
        if not labels:
            break

    feasible = {
        _quantity_key(car_types, label.quantities)
        for label in labels
        if any(label.quantities)
    }
    return feasible, stats


def gurobi_quantity_set(case: ToyCase, layer: LayerSpec) -> set[QuantityKey]:
    segments = feasibility_check._get_segments_for_layer(layer.car_heights)
    car_types = sorted(case.heights)
    feasible: set[QuantityKey] = set()
    for quantities in _quantity_vectors(car_types, case.max_per_type, case.max_total):
        result = feasibility_check.check_layer_gurobi(
            case.compartment,
            case.deck,
            quantities,
            layer.car_lengths,
            layer.car_heights,
            segments,
        )
        if result is not None:
            feasible.add(tuple(sorted((car_type, q) for car_type, q in quantities.items() if q > 0)))
    return feasible


def run_case(case: ToyCase) -> None:
    feasibility_check.GLOBAL_NUM_SPLITS = case.num_splits
    feasibility_check._SEGMENTS_CACHE.clear()

    car_types = sorted(case.heights)
    layer = LayerSpec(
        layer_id=case.name,
        car_types=car_types,
        car_lengths={car_type: case.equal_length for car_type in car_types},
        layer_length_limit=10**9,
        car_heights=case.heights,
        shape_params={"deck": case.deck, "compartment": case.compartment},
        max_quantity_by_type={car_type: case.max_per_type for car_type in car_types},
    )
    full_model = _build_full_resource_model(layer, interval_profile="full")
    gr_model = _prune_outer_dominated_choices(full_model)
    unit_resource = case.equal_length + full_model.delta
    total_vectors = sum(1 for _ in _quantity_vectors(car_types, case.max_per_type, case.max_total))

    print(f"\n=== {case.name} ===")
    print(f"compartment={case.compartment}, deck={case.deck}, splits={case.num_splits}, types={len(car_types)}")

    t0 = time.perf_counter()
    ex_set, ex_stats = generate_label_quantity_set(full_model, car_types, case.max_per_type, case.max_total, unit_resource)
    t_ex = time.perf_counter() - t0

    t0 = time.perf_counter()
    gr_set, gr_stats = generate_label_quantity_set(gr_model, car_types, case.max_per_type, case.max_total, unit_resource)
    t_gr = time.perf_counter() - t0

    t0 = time.perf_counter()
    gp_set = gurobi_quantity_set(case, layer)
    t_gp = time.perf_counter() - t0

    print(
        "sets: "
        f"candidates={total_vectors}, EX={len(ex_set)}, GR={len(gr_set)}, Gurobi={len(gp_set)}, "
        f"EX==Gurobi={ex_set == gp_set}, GR==Gurobi={gr_set == gp_set}, EX==GR={ex_set == gr_set}"
    )
    print(
        "time: "
        f"EX_label={t_ex:.4f}s, GR_label={t_gr:.4f}s, Gurobi_enum={t_gp:.4f}s"
    )
    print(
        "labels after stage: "
        f"EX={ex_stats.labels_after_stage}, GR={gr_stats.labels_after_stage}"
    )

    missing_ex = sorted(gp_set - ex_set)[:5]
    extra_ex = sorted(ex_set - gp_set)[:5]
    missing_gr = sorted(gp_set - gr_set)[:5]
    extra_gr = sorted(gr_set - gp_set)[:5]
    if missing_ex or extra_ex or missing_gr or extra_gr:
        print(f"mismatch samples: missing_ex={missing_ex}, extra_ex={extra_ex}, missing_gr={missing_gr}, extra_gr={extra_gr}")


def main() -> None:
    cases = [
        ToyCase(
            name="lower_hh_splits1_3types_label_vs_gurobi",
            compartment="lower",
            deck="h-h",
            num_splits=1,
            heights={1: 1500.0, 2: 1700.0, 3: 1900.0},
            max_per_type=2,
            max_total=5,
        ),
        ToyCase(
            name="lower_hh_splits3_4types_label_vs_gurobi",
            compartment="lower",
            deck="h-h",
            num_splits=3,
            heights={1: 1500.0, 2: 1650.0, 3: 1800.0, 4: 2000.0},
            max_per_type=2,
            max_total=5,
        ),
        ToyCase(
            name="upper_hm_splits3_4types_label_vs_gurobi",
            compartment="upper",
            deck="h-m",
            num_splits=3,
            heights={1: 1500.0, 2: 1650.0, 3: 1780.0, 4: 1950.0},
            max_per_type=2,
            max_total=5,
        ),
        ToyCase(
            name="lower_hh_splits3_4types_long_label_vs_gurobi",
            compartment="lower",
            deck="h-h",
            num_splits=3,
            heights={1: 1500.0, 2: 1650.0, 3: 1800.0, 4: 2000.0},
            max_per_type=2,
            max_total=8,
            equal_length=3600.0,
        ),
        ToyCase(
            name="upper_hm_splits3_4types_long_label_vs_gurobi",
            compartment="upper",
            deck="h-m",
            num_splits=3,
            heights={1: 1500.0, 2: 1650.0, 3: 1780.0, 4: 1950.0},
            max_per_type=2,
            max_total=8,
            equal_length=3600.0,
        ),
    ]
    for case in cases:
        run_case(case)


if __name__ == "__main__":
    main()
