#!/usr/bin/env python3
"""Generate instance data under data/Instance from raw Excel Sheet1.

Usage example:
  python src/utility/generate_instance.py --models 10 --carriages 10
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utility.config import config as Config

# IDE debug switches.
# 0 = False, 1 = True.
DBG_OVERWRITE = 1
DBG_PRINT_SUMMARY = 1

# Runtime settings (edit directly when debugging).
RUN_MODELS = 10
RUN_CARRIAGES = 10
RUN_SEED = None
RUN_EXCEL = Path("data/raw data/尺寸整理.xlsx")
RUN_OUTPUT_ROOT = Path("data/Instance")


def load_candidates(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name="Sheet1")
    need_cols = ["项目", "车型", "长", "高"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Sheet1 missing columns: {missing}")

    data = df[need_cols].copy()
    data["项目"] = data["项目"].ffill()
    data = data.dropna(subset=need_cols)

    data["长"] = pd.to_numeric(data["长"], errors="coerce")
    data["高"] = pd.to_numeric(data["高"], errors="coerce")
    data = data.dropna(subset=["长", "高"])

    # Exclude vehicles higher than the wagon's highest allowed height.
    max_allowed_height = max(
        Config.A_height_h,
        Config.A_height_m,
        Config.B_height,
        Config.C_height,
        Config.D_height_h,
        Config.D_height_m,
        Config.E_height,
    )
    data = data[data["高"] <= max_allowed_height]

    data = data.drop_duplicates(subset=["项目", "车型"]).reset_index(drop=True)
    return data


def generate_counts(model_count: int, carriage_count: int, rng: random.Random) -> tuple[list[int], list[int]]:
    # Retry until global constraints are satisfied.
    for _ in range(10000):
        optional = [rng.randint(8, 20) for _ in range(model_count)]
        mandatory = [rng.randint(0, min(10, o)) for o in optional]
        if sum(optional) >= carriage_count * 12 and sum(mandatory) <= carriage_count * 7:
            return optional, mandatory
    raise RuntimeError("Failed to generate optional/mandatory counts under constraints.")


def build_instance(models: int, carriages: int, excel_path: Path, output_root: Path, seed: int | None, overwrite: bool) -> Path:
    if models < 1:
        raise ValueError("--models must be >= 1")
    if carriages < 1:
        raise ValueError("--carriages must be >= 1")

    rng = random.Random(seed)
    candidates = load_candidates(excel_path)

    if len(candidates) < models:
        raise ValueError(f"Only {len(candidates)} unique models available, cannot sample {models}.")

    sampled = candidates.sample(n=models, random_state=seed).reset_index(drop=True)
    optional, mandatory = generate_counts(models, carriages, rng)

    out = pd.DataFrame(
        {
            "program": sampled["项目"].astype(str),
            "model": sampled["车型"].astype(str),
            "length": sampled["长"].round(0).astype(int),
            "height": sampled["高"].round(0).astype(int),
            "optional": optional,
            "mandatory": mandatory,
        }
    )

    instance_dir = output_root / f"m{models}c{carriages}"
    if instance_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Target already exists: {instance_dir}. Use --overwrite to replace.")
        shutil.rmtree(instance_dir)
    instance_dir.mkdir(parents=True, exist_ok=True)

    out.to_csv(instance_dir / "cars.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"carriage_num": [carriages]}).to_csv(
        instance_dir / "carriage.csv", index=False, encoding="utf-8-sig"
    )

    return instance_dir


def main() -> None:
    models = int(RUN_MODELS)
    carriages = int(RUN_CARRIAGES)
    seed = RUN_SEED
    excel_path = Path(RUN_EXCEL)
    output_root = Path(RUN_OUTPUT_ROOT)
    overwrite = bool(DBG_OVERWRITE)

    instance_dir = build_instance(
        models=models,
        carriages=carriages,
        excel_path=excel_path,
        output_root=output_root,
        seed=seed,
        overwrite=overwrite,
    )

    if bool(DBG_PRINT_SUMMARY):
        cars = pd.read_csv(instance_dir / "cars.csv")
        print(f"Generated: {instance_dir}")
        print(f"rows(models): {len(cars)}")
        print(f"sum(optional): {int(cars['optional'].sum())}")
        print(f"sum(mandatory): {int(cars['mandatory'].sum())}")
        print(f"constraint optional >= {carriages * 12}")
        print(f"constraint mandatory <= {carriages * 7}")


if __name__ == "__main__":
    main()
