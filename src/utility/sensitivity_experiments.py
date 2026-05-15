#!/usr/bin/env python3
"""Generate and run sensitivity-analysis instances for the motorail model."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
import traceback
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.gurobi import build_and_solve
from src.utility.config import config as Config
from src.utility.generate_instance import load_candidates


RAW_EXCEL = PROJECT_ROOT / "data/raw data/尺寸整理.xlsx"
INSTANCE_ROOT = PROJECT_ROOT / "data/Instance"
RESULT_ROOT = PROJECT_ROOT / "result"

BASE_SIZES = {
    "small": [(6, 6), (7, 7), (8, 8)],
    "medium": [(10, 10), (11, 11), (12, 12)],
    "large": [(14, 14), (15, 15), (16, 16)],
}
SCALE_ORDER = ["small", "medium", "large"]
SEED_IDS = [1]
PROPORTION_LEVELS = [round(v / 10.0, 1) for v in range(11)]
OPTIONAL_RATIO_LEVELS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
CHUNK_LEVELS = list(range(2, 9))

DEFAULT_MIP_GAP = 0.001
DEFAULT_TIME_LIMIT = 3600.0
POST_SOLVE_GRACE_SEC = 600.0
FAMILY_RUN_ORDER = {
    "optional_ratio": 0,
    "proportion": 1,
    "objective": 2,
    "chunking": 3,
}


@dataclass(frozen=True)
class BaseCase:
    scale: str
    num_types: int
    num_wagons: int
    seed_id: int

    @property
    def base_id(self) -> str:
        return f"{self.scale}_I{self.num_types:02d}_J{self.num_wagons:02d}_R{self.seed_id}"

    @property
    def seed(self) -> int:
        return 100_000 + self.num_types * 1000 + self.num_wagons * 100 + self.seed_id


def load_scaled_candidates() -> pd.DataFrame:
    candidates = load_candidates(RAW_EXCEL).copy()
    length_min = float(candidates["长"].min())
    length_max = float(candidates["长"].max())
    height_min = float(candidates["高"].min())
    height_max = float(candidates["高"].max())

    length_score = (candidates["长"] - length_min) / (length_max - length_min)
    height_score = (candidates["高"] - height_min) / (height_max - height_min)
    candidates["scale_score"] = pd.concat([length_score, height_score], axis=1).max(axis=1)

    cut_small, cut_medium = candidates["scale_score"].quantile([1.0 / 3.0, 2.0 / 3.0])
    candidates["vehicle_size"] = "large"
    candidates.loc[candidates["scale_score"] <= cut_small, "vehicle_size"] = "small"
    candidates.loc[
        (candidates["scale_score"] > cut_small) & (candidates["scale_score"] <= cut_medium),
        "vehicle_size",
    ] = "medium"
    candidates["flat_compatible"] = candidates["高"] <= Config.A_height_h
    candidates.attrs["scale_info"] = {
        "length_min": length_min,
        "length_max": length_max,
        "height_min": height_min,
        "height_max": height_max,
        "cut_small": float(cut_small),
        "cut_medium": float(cut_medium),
    }
    return candidates


def allocate_counts(total: int, proportions: Dict[str, float]) -> Dict[str, int]:
    positive_keys = [k for k, v in proportions.items() if v > 0]
    out = {k: 0 for k in proportions}
    if total >= len(positive_keys):
        for key in positive_keys:
            out[key] = 1

    remaining = total - sum(out.values())
    positive_sum = sum(proportions[k] for k in positive_keys)
    if remaining <= 0 or positive_sum <= 0:
        return out

    raw = {k: remaining * proportions[k] / positive_sum for k in proportions}
    add = {k: int(math.floor(v)) for k, v in raw.items()}
    for key, val in add.items():
        out[key] += val
    remaining = total - sum(out.values())
    order = sorted(raw, key=lambda k: (raw[k] - add[k], proportions[k]), reverse=True)
    for key in order[:remaining]:
        out[key] += 1
    return out


def allocate_quantity(total: int, n: int, rng: random.Random, min_each: int = 0) -> List[int]:
    if n <= 0:
        return []
    if total <= 0:
        return [0] * n
    if min_each * n > total:
        min_each = 0

    values = [min_each] * n
    remaining = total - min_each * n
    weights = [rng.random() + 0.25 for _ in range(n)]
    weight_sum = sum(weights)
    raw = [remaining * w / weight_sum for w in weights]
    add = [int(math.floor(v)) for v in raw]
    for idx, val in enumerate(add):
        values[idx] += val
    leftover = total - sum(values)
    order = sorted(range(n), key=lambda idx: raw[idx] - add[idx], reverse=True)
    for idx in order[:leftover]:
        values[idx] += 1
    return values


def allocate_weighted_quantity(total: int, weights: List[float], min_each: int = 0) -> List[int]:
    n = len(weights)
    if n <= 0:
        return []
    if total <= 0:
        return [0] * n
    if min_each * n > total:
        min_each = 0

    values = [min_each] * n
    remaining = total - min_each * n
    weight_sum = sum(weights)
    if weight_sum <= 0:
        weights = [1.0] * n
        weight_sum = float(n)

    raw = [remaining * w / weight_sum for w in weights]
    add = [int(math.floor(v)) for v in raw]
    for idx, val in enumerate(add):
        values[idx] += val
    leftover = total - sum(values)
    order = sorted(range(n), key=lambda idx: raw[idx] - add[idx], reverse=True)
    for idx in order[:leftover]:
        values[idx] += 1
    return values


def allocate_monotone_quantity_levels(totals: Iterable[int], n: int, rng: random.Random) -> Dict[int, List[int]]:
    """
    Allocate nested integer quantities for increasing total levels.
    Each positive increase is spread to every type at least once, so every
    type's quantity strictly increases from one positive level to the next.
    """
    current = [0] * n
    previous_total = 0
    out: Dict[int, List[int]] = {}
    for total in totals:
        total = int(total)
        if total < previous_total:
            raise ValueError("Quantity levels must be nondecreasing")
        increment = total - previous_total
        if increment > 0 and increment < n:
            raise ValueError(
                f"Cannot increase every type: increment {increment} is smaller than {n} types"
            )
        additions = allocate_quantity(increment, n, rng, min_each=1 if increment > 0 else 0)
        current = [value + add for value, add in zip(current, additions)]
        out[total] = current.copy()
        previous_total = total
    return out


def allocate_monotone_weighted_quantity_levels(totals: Iterable[int], weights: List[float]) -> Dict[int, List[int]]:
    current = [0] * len(weights)
    previous_total = 0
    out: Dict[int, List[int]] = {}
    for total in sorted({int(value) for value in totals}):
        if total < previous_total:
            raise ValueError("Quantity levels must be nondecreasing")
        increment = total - previous_total
        additions = allocate_weighted_quantity(
            increment,
            weights,
            min_each=1 if increment > 0 and increment >= len(weights) else 0,
        )
        current = [value + add for value, add in zip(current, additions)]
        out[total] = current.copy()
        previous_total = total
    return out


def allocate_totals_by_group(total: int, groups: Dict[str, List[int]], proportions: Dict[str, float]) -> Dict[str, int]:
    active = {group: proportions.get(group, 0.0) for group, idxs in groups.items() if idxs and proportions.get(group, 0.0) > 0}
    out = {group: 0 for group in groups}
    if total <= 0 or not active:
        return out

    active_sum = sum(active.values())
    raw = {group: total * weight / active_sum for group, weight in active.items()}
    base = {group: int(math.floor(value)) for group, value in raw.items()}
    for group, value in base.items():
        out[group] = value
    remaining = total - sum(out.values())
    order = sorted(active, key=lambda group: (raw[group] - base[group], active[group]), reverse=True)
    for group in order[:remaining]:
        out[group] += 1
    return out


def allocate_quantity_by_vehicle_size(
    total: int,
    sampled: pd.DataFrame,
    proportions: Dict[str, float],
    rng: random.Random,
) -> List[int]:
    groups = {
        size: [idx for idx, value in enumerate(sampled["vehicle_size"].astype(str)) if value == size]
        for size in ["small", "medium", "large"]
    }
    group_totals = allocate_totals_by_group(total, groups, proportions)
    values = [0] * len(sampled)
    for size, idxs in groups.items():
        if not idxs:
            continue
        group_values = allocate_quantity(
            group_totals[size],
            len(idxs),
            rng,
            min_each=1 if group_totals[size] >= len(idxs) and group_totals[size] > 0 else 0,
        )
        for idx, value in zip(idxs, group_values):
            values[idx] = value
    return values


def build_proportion_optional_vectors(
    total: int,
    sampled: pd.DataFrame,
    levels: Iterable[float],
    rng: random.Random,
) -> Dict[float, List[int]]:
    groups = {
        size: [idx for idx, value in enumerate(sampled["vehicle_size"].astype(str)) if value == size]
        for size in ["small", "medium", "large"]
    }
    group_weights = {
        size: [rng.random() + 0.25 for _ in idxs]
        for size, idxs in groups.items()
    }
    proportions_by_level = {
        level: {
            "small": level,
            "medium": (1.0 - level) / 2.0,
            "large": (1.0 - level) / 2.0,
        }
        for level in levels
    }
    group_totals_by_level = {
        level: allocate_totals_by_group(total, groups, proportions)
        for level, proportions in proportions_by_level.items()
    }
    allocations_by_size = {
        size: allocate_monotone_weighted_quantity_levels(
            [group_totals[size] for group_totals in group_totals_by_level.values()],
            group_weights[size],
        )
        for size in ["small", "medium", "large"]
    }

    out: Dict[float, List[int]] = {}
    for level in levels:
        values = [0] * len(sampled)
        for size, idxs in groups.items():
            group_total = group_totals_by_level[level][size]
            group_values = allocations_by_size[size][group_total]
            for idx, value in zip(idxs, group_values):
                values[idx] = value
        out[level] = values
    return out


def sample_vehicle_types(
    candidates: pd.DataFrame,
    num_types: int,
    proportions: Dict[str, float],
    rng: random.Random,
) -> pd.DataFrame:
    counts = allocate_counts(num_types, proportions)
    sampled_parts = []
    for size in ["small", "medium", "large"]:
        pool = candidates[candidates["vehicle_size"] == size]
        count = counts.get(size, 0)
        if count <= 0:
            continue
        if len(pool) < count:
            raise ValueError(f"Not enough {size} vehicles: need {count}, have {len(pool)}")
        sampled_parts.append(pool.sample(n=count, random_state=rng.randint(1, 2**31 - 1)))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=rng.randint(1, 2**31 - 1)).reset_index(drop=True)


def instance_stats(cars: pd.DataFrame) -> Dict[str, float | int]:
    stats: Dict[str, float | int] = {}
    for size in ["small", "medium", "large"]:
        size_rows = cars[cars["vehicle_size"] == size]
        stats[f"n_{size}_types"] = int(len(size_rows))
        stats[f"p_{size}_types"] = float(len(size_rows) / len(cars)) if len(cars) else 0.0
        stats[f"D_{size}"] = int(size_rows["mandatory"].sum())
        stats[f"C_{size}"] = int(size_rows["optional"].sum())

    for col, prefix in [("length", "length"), ("height", "height"), ("scale_score", "scale_score")]:
        stats[f"{prefix}_min"] = float(cars[col].min())
        stats[f"{prefix}_mean"] = float(cars[col].mean())
        stats[f"{prefix}_max"] = float(cars[col].max())
        if col != "scale_score":
            stats[f"{prefix}_std"] = float(cars[col].std(ddof=0))

    stats["flat_compatible_types"] = int(cars["flat_compatible"].sum())
    stats["D_total"] = int(cars["mandatory"].sum())
    stats["C_total"] = int(cars["optional"].sum())
    total_types = len(cars)
    entropy = 0.0
    for size in ["small", "medium", "large"]:
        p = len(cars[cars["vehicle_size"] == size]) / total_types if total_types else 0.0
        if p > 0:
            entropy -= p * math.log(p)
    stats["mix_entropy"] = float(entropy / math.log(3.0)) if total_types else 0.0
    return stats


def write_instance(
    instance_root: Path,
    instance_id: str,
    base: BaseCase,
    family: str,
    sampled: pd.DataFrame,
    mandatory_total: int,
    optional_total: int,
    rng: random.Random,
    extra_meta: Dict[str, object],
    mandatory_values: List[int] | None = None,
    optional_values: List[int] | None = None,
) -> Path:
    mandatory = (
        list(mandatory_values)
        if mandatory_values is not None
        else allocate_quantity(mandatory_total, len(sampled), rng, min_each=1 if mandatory_total >= len(sampled) else 0)
    )
    optional = (
        list(optional_values)
        if optional_values is not None
        else allocate_quantity(optional_total, len(sampled), rng, min_each=0)
    )
    if len(mandatory) != len(sampled) or len(optional) != len(sampled):
        raise ValueError("Explicit quantity vectors must match the sampled vehicle count")
    if sum(mandatory) != mandatory_total:
        raise ValueError(f"Mandatory vector sums to {sum(mandatory)}, expected {mandatory_total}")
    if sum(optional) != optional_total:
        raise ValueError(f"Optional vector sums to {sum(optional)}, expected {optional_total}")
    cars = pd.DataFrame(
        {
            "program": sampled["项目"].astype(str),
            "model": sampled["车型"].astype(str),
            "length": sampled["长"].round(0).astype(int),
            "height": sampled["高"].round(0).astype(int),
            "optional": optional,
            "mandatory": mandatory,
            "vehicle_size": sampled["vehicle_size"].astype(str),
            "scale_score": sampled["scale_score"].round(6),
            "flat_compatible": sampled["flat_compatible"].astype(bool),
        }
    )

    instance_dir = instance_root / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    cars.to_csv(instance_dir / "cars.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"carriage_num": [base.num_wagons]}).to_csv(
        instance_dir / "carriage.csv", index=False, encoding="utf-8-sig"
    )

    stats = instance_stats(cars)
    meta = {
        "instance_id": instance_id,
        "experiment_family": family,
        "base_id": base.base_id,
        "problem_scale": base.scale,
        "num_types_I": base.num_types,
        "num_wagons_J": base.num_wagons,
        "seed": base.seed,
        "mandatory_total_target": int(mandatory_total),
        "optional_total_target": int(optional_total),
        "demand_per_wagon_D": float(stats["D_total"] / base.num_wagons),
        "demand_per_wagon_C": float(stats["C_total"] / base.num_wagons),
        "available_per_wagon": float((stats["D_total"] + stats["C_total"]) / base.num_wagons),
        **extra_meta,
        **stats,
    }
    with open(instance_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return instance_dir


def base_cases(scale: str) -> List[BaseCase]:
    if scale not in BASE_SIZES:
        raise ValueError(f"Unknown scale: {scale}")
    return [
        BaseCase(scale, num_types, num_wagons, seed_id)
        for num_types, num_wagons in BASE_SIZES[scale]
        for seed_id in SEED_IDS
    ]


def generate_scale_sensitivity(date_tag: str, scale: str) -> tuple[pd.DataFrame, Path, Path]:
    candidates = load_scaled_candidates()
    instance_root = INSTANCE_ROOT / f"sensitivity_{date_tag}" / scale
    result_root = RESULT_ROOT / f"sensitivity_{date_tag}" / scale
    plan_dir = result_root / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)

    plan_rows: List[Dict[str, object]] = []

    for base in base_cases(scale):
        base_rng = random.Random(base.seed)
        balanced = {"small": 1 / 3, "medium": 1 / 3, "large": 1 / 3}
        base_sample = sample_vehicle_types(candidates, base.num_types, balanced, base_rng)

        common_id = f"{base.base_id}_balanced_D5_C10"
        write_instance(
            instance_root,
            common_id,
            base,
            "common_balanced",
            base_sample,
            mandatory_total=5 * base.num_wagons,
            optional_total=10 * base.num_wagons,
            rng=random.Random(base.seed + 11),
            extra_meta={"vehicle_mix_target": balanced, "rho_optional": 2.0},
        )

        for chunks in CHUNK_LEVELS:
            plan_rows.append(
                make_plan_row(
                    date_tag,
                    "chunking",
                    base,
                    common_id,
                    num_splits=chunks,
                    objective_type="length",
                    varied_parameter="num_splits",
                    varied_value=chunks,
                )
            )

        # Reuse chunking num_splits=3 length run; only quantity objective needs a separate solve.
        plan_rows.append(
            make_plan_row(
                date_tag,
                "objective",
                base,
                common_id,
                num_splits=3,
                objective_type="quantity",
                varied_parameter="objective_type",
                varied_value="quantity",
            )
        )

        proportion_sample = base_sample.copy()
        optional_by_p_small = build_proportion_optional_vectors(
            15 * base.num_wagons,
            proportion_sample,
            PROPORTION_LEVELS,
            random.Random(base.seed + 211),
        )
        for p_small in PROPORTION_LEVELS:
            proportions = {
                "small": p_small,
                "medium": (1.0 - p_small) / 2.0,
                "large": (1.0 - p_small) / 2.0,
            }
            instance_id = f"{base.base_id}_prop_{int(p_small * 100):03d}"
            write_instance(
                instance_root,
                instance_id,
                base,
                "proportion",
                proportion_sample,
                mandatory_total=0,
                optional_total=15 * base.num_wagons,
                rng=random.Random(base.seed + 211),
                extra_meta={
                    "vehicle_mix_target": proportions,
                    "p_small": p_small,
                    "rho_optional": None,
                    "optional_quantity_matches_composition": True,
                    "vehicle_type_pool_fixed_across_p": True,
                    "within_size_optional_ratio_fixed": True,
                },
                optional_values=optional_by_p_small[p_small],
            )
            plan_rows.append(
                make_plan_row(
                    date_tag,
                    "proportion",
                    base,
                    instance_id,
                    num_splits=3,
                    objective_type="length",
                    varied_parameter="p_small",
                    varied_value=p_small,
                )
            )

        optional_total_by_rho = {
            rho_level: round(rho_level * 5 * base.num_wagons)
            for rho_level in OPTIONAL_RATIO_LEVELS
        }
        optional_by_total = allocate_monotone_quantity_levels(
            optional_total_by_rho.values(),
            len(base_sample),
            random.Random(base.seed + 311),
        )
        mandatory_for_optional_ratio = allocate_quantity(
            5 * base.num_wagons,
            len(base_sample),
            random.Random(base.seed + 411),
            min_each=1 if 5 * base.num_wagons >= len(base_sample) else 0,
        )

        for rho in OPTIONAL_RATIO_LEVELS:
            optional_total = optional_total_by_rho[rho]
            instance_id = f"{base.base_id}_rho_{str(rho).replace('.', 'p')}"
            write_instance(
                instance_root,
                instance_id,
                base,
                "optional_ratio",
                base_sample,
                mandatory_total=5 * base.num_wagons,
                optional_total=optional_total,
                rng=random.Random(base.seed + 511),
                extra_meta={
                    "vehicle_mix_target": balanced,
                    "rho_optional": rho,
                    "optional_monotone_by_type": True,
                    "mandatory_fixed_across_rho": True,
                },
                mandatory_values=mandatory_for_optional_ratio,
                optional_values=optional_by_total[optional_total],
            )
            plan_rows.append(
                make_plan_row(
                    date_tag,
                    "optional_ratio",
                    base,
                    instance_id,
                    num_splits=3,
                    objective_type="length",
                    varied_parameter="rho_optional",
                    varied_value=rho,
                )
            )

    plan = pd.DataFrame(plan_rows)
    plan_path = plan_dir / f"sensitivity_{scale}_plan_{date_tag}.csv"
    plan.to_csv(plan_path, index=False)
    return plan, instance_root, result_root


def make_plan_row(
    date_tag: str,
    family: str,
    base: BaseCase,
    instance_id: str,
    num_splits: int,
    objective_type: str,
    varied_parameter: str,
    varied_value: object,
) -> Dict[str, object]:
    run_id = f"{family}_{base.base_id}_{varied_parameter}_{varied_value}_{objective_type}_k{num_splits}"
    run_id = str(run_id).replace("/", "_").replace(" ", "_")
    return {
        "date_tag": date_tag,
        "experiment_family": family,
        "base_id": base.base_id,
        "problem_scale": base.scale,
        "num_types_I": base.num_types,
        "num_wagons_J": base.num_wagons,
        "seed_id": base.seed_id,
        "seed": base.seed,
        "instance_id": instance_id,
        "run_id": run_id,
        "num_splits": int(num_splits),
        "independent_mode_split": False,
        "objective_type": objective_type,
        "varied_parameter": varied_parameter,
        "varied_value": varied_value,
        "time_limit": DEFAULT_TIME_LIMIT,
        "mip_gap": DEFAULT_MIP_GAP,
    }


def flatten_meta(instance_dir: Path) -> Dict[str, object]:
    meta_path = instance_dir / "meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    out = {}
    for key, value in meta.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                out[f"{key}_{sub_key}"] = sub_value
        else:
            out[key] = value
    return out


def solve_worker(payload: Dict[str, object], queue: mp.Queue) -> None:
    try:
        summary = build_and_solve(
            instance_dir=Path(str(payload["instance_dir"])),
            output_dir=Path(str(payload["output_dir"])),
            log_to_console=False,
            num_splits=int(payload["num_splits"]),
            independent_mode_split=bool(payload["independent_mode_split"]),
            objective_type=str(payload["objective_type"]),
            time_limit=float(payload["time_limit"]),
            mip_gap=float(payload["mip_gap"]),
        )
        queue.put({"ok": True, "summary": summary})
    except BaseException as exc:
        queue.put(
            {
                "ok": False,
                "summary": {
                    "status": "ERROR",
                    "sol_count": 0,
                    "objective_type": str(payload["objective_type"]),
                    "num_splits": int(payload["num_splits"]),
                    "independent_mode_split": bool(payload["independent_mode_split"]),
                    "time_limit": float(payload["time_limit"]),
                    "mip_gap_target": float(payload["mip_gap"]),
                    "runner_status": "worker_exception",
                    "error_message": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )


def load_summary_if_available(summary_path: Path) -> Dict[str, object] | None:
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def terminate_process(process: mp.Process) -> None:
    process.terminate()
    process.join(timeout=30)
    if process.is_alive():
        process.kill()
        process.join(timeout=30)


def solve_with_watchdog(row: Dict[str, object], instance_dir: Path, output_dir: Path) -> Dict[str, object]:
    summary_path = output_dir / "solver" / "solve_summary.json"
    timeout_sec = float(row["time_limit"]) + POST_SOLVE_GRACE_SEC
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    payload = {
        "instance_dir": str(instance_dir),
        "output_dir": str(output_dir),
        "num_splits": int(row["num_splits"]),
        "independent_mode_split": bool(row["independent_mode_split"]),
        "objective_type": str(row["objective_type"]),
        "time_limit": float(row["time_limit"]),
        "mip_gap": float(row["mip_gap"]),
    }
    process = ctx.Process(target=solve_worker, args=(payload, queue), daemon=False)
    process.start()
    process.join(timeout=timeout_sec)

    if process.is_alive():
        print(
            f"[watchdog] {row['run_id']} exceeded {timeout_sec:g}s; terminating worker",
            flush=True,
        )
        terminate_process(process)
        summary = load_summary_if_available(summary_path)
        if summary is not None:
            summary["runner_status"] = "worker_timeout_after_summary"
            summary["worker_timeout_sec"] = timeout_sec
            return summary
        return {
            "status": "ERROR",
            "sol_count": 0,
            "objective_type": str(row["objective_type"]),
            "num_splits": int(row["num_splits"]),
            "independent_mode_split": bool(row["independent_mode_split"]),
            "time_limit": float(row["time_limit"]),
            "mip_gap_target": float(row["mip_gap"]),
            "runner_status": "worker_timeout_no_summary",
            "worker_timeout_sec": timeout_sec,
        }

    if not queue.empty():
        payload_out = queue.get()
        summary = payload_out.get("summary", {})
        if payload_out.get("ok"):
            summary["runner_status"] = "worker_completed"
            return summary
        return summary

    summary = load_summary_if_available(summary_path)
    if summary is not None:
        summary["runner_status"] = "worker_exit_after_summary"
        return summary
    return {
        "status": "ERROR",
        "sol_count": 0,
        "objective_type": str(row["objective_type"]),
        "num_splits": int(row["num_splits"]),
        "independent_mode_split": bool(row["independent_mode_split"]),
        "time_limit": float(row["time_limit"]),
        "mip_gap_target": float(row["mip_gap"]),
        "runner_status": "worker_exit_no_summary",
        "worker_exitcode": process.exitcode,
    }


def run_plan(
    plan: pd.DataFrame,
    date_tag: str,
    scale: str,
    max_runs: int | None = None,
    families: set[str] | None = None,
    skip_existing: bool = True,
) -> Path:
    instance_root = INSTANCE_ROOT / f"sensitivity_{date_tag}" / scale
    result_root = RESULT_ROOT / f"sensitivity_{date_tag}" / scale
    result_root.mkdir(parents=True, exist_ok=True)
    results_path = result_root / f"sensitivity_{scale}_results_{date_tag}.csv"

    completed = 0
    run_plan_df = plan.copy()
    run_plan_df["_family_order"] = run_plan_df["experiment_family"].map(FAMILY_RUN_ORDER).fillna(99)
    run_plan_df = run_plan_df.sort_values(
        ["_family_order", "base_id", "varied_parameter", "varied_value", "objective_type"],
        kind="stable",
    )

    for row in run_plan_df.drop(columns=["_family_order"]).to_dict("records"):
        if families and row["experiment_family"] not in families:
            continue
        if max_runs is not None and completed >= max_runs:
            break

        instance_dir = instance_root / str(row["instance_id"])
        output_dir = result_root / "runs" / str(row["run_id"])
        summary_path = output_dir / "solver" / "solve_summary.json"
        if skip_existing and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            print(
                f"[run] {row['run_id']} instance={row['instance_id']} "
                f"k={row['num_splits']} objective={row['objective_type']}",
                flush=True,
            )
            try:
                summary = solve_with_watchdog(row, instance_dir, output_dir)
            except Exception as exc:
                summary = {
                    "status": "ERROR",
                    "sol_count": 0,
                    "objective_type": str(row["objective_type"]),
                    "num_splits": int(row["num_splits"]),
                    "independent_mode_split": bool(row["independent_mode_split"]),
                    "time_limit": float(row["time_limit"]),
                    "mip_gap_target": float(row["mip_gap"]),
                    "error_message": repr(exc),
                }
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_dir / "run_error.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
        record = {
            **flatten_meta(instance_dir),
            **row,
            **summary,
            "result_dir": str(output_dir),
        }
        append_result(results_path, record)
        completed += 1
    return results_path


def append_result(path: Path, record: Dict[str, object]) -> None:
    row = pd.DataFrame([record])
    if path.exists():
        existing = pd.read_csv(path)
        if "run_id" in existing.columns and "run_id" in record:
            existing = existing[existing["run_id"] != record["run_id"]]
        if existing.empty:
            combined = row
        else:
            combined = pd.concat([existing, row], ignore_index=True, sort=False)
        combined.to_csv(path, index=False)
    else:
        row.to_csv(path, index=False)


def load_or_generate_plan(date_tag: str) -> tuple[pd.DataFrame, Path, Path]:
    return load_or_generate_scale_plan(date_tag, "small")


def load_or_generate_scale_plan(date_tag: str, scale: str) -> tuple[pd.DataFrame, Path, Path]:
    result_root = RESULT_ROOT / f"sensitivity_{date_tag}" / scale
    plan_path = result_root / "plans" / f"sensitivity_{scale}_plan_{date_tag}.csv"
    if plan_path.exists():
        plan = pd.read_csv(plan_path)
        return plan, INSTANCE_ROOT / f"sensitivity_{date_tag}" / scale, result_root
    return generate_scale_sensitivity(date_tag, scale)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate/run sensitivity experiments.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Date tag for plans/results.")
    parser.add_argument("--generate", action="store_true", help="Generate instances and plan.")
    parser.add_argument("--run", action="store_true", help="Run Gurobi according to the plan.")
    parser.add_argument(
        "--scales",
        nargs="*",
        default=["small"],
        choices=SCALE_ORDER,
        help="Scales to process in order. Default: small.",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap for this trial run.")
    parser.add_argument(
        "--families",
        nargs="*",
        default=None,
        help="Optional subset: chunking proportion optional_ratio objective.",
    )
    parser.add_argument("--no-skip-existing", action="store_true", help="Rerun rows even if summaries exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.generate and not args.run:
        args.generate = True

    if args.generate:
        for scale in args.scales:
            plan, instance_root, result_root = generate_scale_sensitivity(args.date, scale)
            print(f"Generated {len(plan)} {scale} sensitivity run rows")
            print(f"Instances: {instance_root}")
            print(f"Results: {result_root}")
    else:
        pass

    if args.run:
        families = set(args.families) if args.families else None
        for scale in args.scales:
            plan, _instance_root, _result_root = load_or_generate_scale_plan(args.date, scale)
            results_path = run_plan(
                plan,
                date_tag=args.date,
                scale=scale,
                max_runs=args.max_runs,
                families=families,
                skip_existing=not args.no_skip_existing,
            )
            print(f"Results CSV: {results_path}")


if __name__ == "__main__":
    main()
