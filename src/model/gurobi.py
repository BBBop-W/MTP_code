import json
import pickle
import sys
from pathlib import Path

import gurobipy as gp
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utility.config import config as Config
from src.utility.dynamic_segmentation import get_model_segments

# IDE debug switches.
# 0 = False, 1 = True.
DBG_LOG_TO_CONSOLE = 1
DBG_PRINT_SUMMARY = 1

# Runtime settings (edit directly when debugging).
RUN_INSTANCE_DIR = "data/Instance/m12c12"
RUN_OUTPUT_DIR = "result"


def normalize_car_table(cars_path: Path) -> pd.DataFrame:
    car_info = pd.read_csv(cars_path)
    col_map = {
        "Brand": "program",
        "Model": "model",
        "Length": "length",
        "Height": "height",
        "Optional#": "optional",
        "Mandatory#": "mandatory",
    }
    car_info = car_info.rename(columns=col_map)
    required = ["program", "model", "length", "height", "optional", "mandatory"]
    missing = [c for c in required if c not in car_info.columns]
    if missing:
        raise ValueError(f"cars.csv missing required columns: {missing}")

    car_info["length"] = pd.to_numeric(car_info["length"])
    car_info["height"] = pd.to_numeric(car_info["height"])
    car_info["optional"] = pd.to_numeric(car_info["optional"]).astype(int)
    car_info["mandatory"] = pd.to_numeric(car_info["mandatory"]).astype(int)
    return car_info


def build_and_solve(
    instance_dir: Path, 
    output_dir: Path, 
    log_to_console: bool = False,
    num_splits: int = 1,
    independent_mode_split: bool = True,
    objective_type: str = "length",
    time_limit: float | None = None,
    mip_gap: float | None = None,
    threads: int | None = None,
) -> dict:
    cars_path = instance_dir / "cars.csv"
    carriage_path = instance_dir / "carriage.csv"
    if not cars_path.exists():
        raise FileNotFoundError(f"Missing file: {cars_path}")
    if not carriage_path.exists():
        raise FileNotFoundError(f"Missing file: {carriage_path}")

    car_info = normalize_car_table(cars_path)
    carriage_num = int(pd.read_csv(carriage_path)["carriage_num"].iloc[0])
    objective_type = objective_type.lower().strip()
    if objective_type not in {"length", "quantity"}:
        raise ValueError("objective_type must be 'length' or 'quantity'")

    segments_data = get_model_segments(car_info, num_splits, independent_mode_split)

    I = list(range(1, len(car_info) + 1))
    J = list(range(1, carriage_num + 1))
    P = ["h-h", "h-m", "m-h", "m-m"]
    deck_side_mode = {
        "h-h": {"left": "h", "right": "h"},
        "h-m": {"left": "h", "right": "m"},
        "m-h": {"left": "m", "right": "h"},
        "m-m": {"left": "m", "right": "m"},
    }

    lengths = {i: float(car_info.iloc[i - 1]["length"]) for i in I}
    mandatory = {i: int(car_info.iloc[i - 1]["mandatory"]) for i in I}
    optional = {i: int(car_info.iloc[i - 1]["optional"]) for i in I}
    heights = {i: float(car_info.iloc[i - 1]["height"]) for i in I}

    H = []
    L = {}
    h_h_limits = {}
    h_m_limits = {}
    
    H_lower = []
    H_upper = []
    component_side = {}

    # Process lower deck segments
    lower_central = "lower_central"
    H.append(lower_central)
    H_lower.append(lower_central)
    component_side[lower_central] = "central"
    L[lower_central] = segments_data["lower"]["central"]["len"]
    h_h_limits[lower_central] = segments_data["lower"]["central"]["h_h"]
    h_m_limits[lower_central] = segments_data["lower"]["central"]["h_m"]

    num_lower_blocks = len(segments_data["lower"]["blocks"])
    for block in segments_data["lower"]["blocks"]:
        b_name = block["name"] # e.g. block_1
        for side in ["left", "right"]:
            h_name = f"lower_{b_name}_{side}"
            H.append(h_name)
            H_lower.append(h_name)
            component_side[h_name] = side
            L[h_name] = block["len"]
            h_h_limits[h_name] = block["h_h"]
            h_m_limits[h_name] = block["h_m"]

    # Process upper deck segments
    upper_central = "upper_central"
    H.append(upper_central)
    H_upper.append(upper_central)
    component_side[upper_central] = "central"
    L[upper_central] = segments_data["upper"]["central"]["len"]
    h_h_limits[upper_central] = segments_data["upper"]["central"]["h_h"]
    h_m_limits[upper_central] = segments_data["upper"]["central"]["h_m"]

    num_upper_blocks = len(segments_data["upper"]["blocks"])
    for block in segments_data["upper"]["blocks"]:
        b_name = block["name"]
        for side in ["left", "right"]:
            h_name = f"upper_{b_name}_{side}"
            H.append(h_name)
            H_upper.append(h_name)
            component_side[h_name] = side
            L[h_name] = block["len"]
            h_h_limits[h_name] = block["h_h"]
            h_m_limits[h_name] = block["h_m"]

    Delta = 400.0
    N = 10

    def height_limit(h: str, side_mode: str) -> float:
        return h_m_limits[h] if side_mode == "m" else h_h_limits[h]

    def eta(i: int, p: str, h: str) -> int:
        side = component_side[h]
        if side == "central":
            limit = min(height_limit(h, side_mode) for side_mode in deck_side_mode[p].values())
        else:
            limit = height_limit(h, deck_side_mode[p][side])
        return int(heights[i] <= limit)

    eta_iph = {(i, p, h): eta(i, p, h) for i in I for p in P for h in H}

    model = gp.Model("motorail_mlp_ic")
    model.Params.OutputFlag = 1 if log_to_console else 0
    model.Params.TimeLimit = Config.timelimit if time_limit is None else float(time_limit)
    model.Params.MIPGap = Config.gap if mip_gap is None else float(mip_gap)
    if threads is not None:
        model.Params.Threads = int(threads)

    z = model.addVars(J, P, vtype=gp.GRB.BINARY, name="z")
    x = model.addVars(I, J, H, vtype=gp.GRB.INTEGER, lb=0, ub=N, name="x")

    # Objective
    if objective_type == "length":
        obj_expr = gp.quicksum(x[i, j, h] * lengths[i] for i in I for j in J for h in H)
    else:
        obj_expr = gp.quicksum(x[i, j, h] for i in I for j in J for h in H)
    model.setObjective(obj_expr, gp.GRB.MAXIMIZE)

    # Demand constraints
    model.addConstrs(
        gp.quicksum(x[i, j, h] for j in J for h in H) <= optional[i] + mandatory[i]
        for i in I
    )
    model.addConstrs(
        gp.quicksum(x[i, j, h] for j in J for h in H) >= mandatory[i]
        for i in I
    )

    # Deck-position selection
    model.addConstrs(
        gp.quicksum(z[j, p] for p in P) == 1
        for j in J
    )

    # Interval capacity generation
    def get_intervals(compartment_prefix, num_blocks):
        intervals = []
        for l in range(num_blocks + 1):
            for r in range(num_blocks + 1):
                blocks = [f"{compartment_prefix}_central"]
                for i in range(1, l + 1):
                    blocks.append(f"{compartment_prefix}_block_{i}_left")
                for i in range(1, r + 1):
                    blocks.append(f"{compartment_prefix}_block_{i}_right")
                
                if l == num_blocks and r == num_blocks:
                    mod = -Delta
                elif l == num_blocks or r == num_blocks:
                    mod = 0
                else:
                    mod = Delta
                intervals.append((blocks, mod))
        return intervals

    lower_intervals = get_intervals("lower", num_lower_blocks)
    upper_intervals = get_intervals("upper", num_upper_blocks)

    # Nested length-checking regions
    for blocks, mod in lower_intervals + upper_intervals:
        cap = sum(L[h] for h in blocks) + mod
        model.addConstrs(
            gp.quicksum(x[i, j, h] * (lengths[i] + Delta) for i in I for h in blocks)
            <= gp.quicksum(cap * z[j, p] for p in P)
            for j in J
        )

    # Component-position placement feasibility
    model.addConstrs(
        x[i, j, h] <= N * gp.quicksum(eta_iph[i, p, h] * z[j, p] for p in P)
        for i in I for j in J for h in H
    )

    # Maximum loading quantity by compartment
    model.addConstrs(
        gp.quicksum(x[i, j, h] for i in I for h in H_lower) <= N
        for j in J
    )
    model.addConstrs(
        gp.quicksum(x[i, j, h] for i in I for h in H_upper) <= N
        for j in J
    )

    model.optimize()

    output_dir.mkdir(parents=True, exist_ok=True)
    solution_dir = output_dir / "solver"
    diagnostics_dir = output_dir / "diagnostics"
    solution_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "status": int(model.status),
        "sol_count": int(model.SolCount),
        "objective_type": objective_type,
        "num_splits": int(num_splits),
        "independent_mode_split": bool(independent_mode_split),
        "time_limit": float(model.Params.TimeLimit),
        "mip_gap_target": float(model.Params.MIPGap),
        "threads": int(model.Params.Threads),
        "runtime_sec": float(model.Runtime),
        "node_count": float(model.NodeCount),
        "num_vars": int(model.NumVars),
        "num_constrs": int(model.NumConstrs),
        "num_qconstrs": int(model.NumQConstrs),
        "num_types": int(len(I)),
        "num_wagons": int(carriage_num),
        "num_lower_side_blocks": int(num_lower_blocks),
        "num_upper_side_blocks": int(num_upper_blocks),
        "num_lower_components": int(len(H_lower)),
        "num_upper_components": int(len(H_upper)),
        "num_components_total": int(len(H)),
        "num_lower_length_regions": int((num_lower_blocks + 1) ** 2),
        "num_upper_length_regions": int((num_upper_blocks + 1) ** 2),
        "mandatory_total": int(sum(mandatory.values())),
        "optional_total": int(sum(optional.values())),
    }

    if model.status == gp.GRB.Status.INFEASIBLE:
        if bool(DBG_PRINT_SUMMARY):
            print(f"Optimization was stopped with status {model.status}")
        model.computeIIS()
        model.write(str(diagnostics_dir / "model_iis.ilp"))
        with open(solution_dir / "solve_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    if model.SolCount <= 0:
        if bool(DBG_PRINT_SUMMARY):
            print(f"Optimization terminated with status {model.status}, no feasible incumbent found.")
        with open(solution_dir / "solve_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    if model.status == gp.GRB.Status.OPTIMAL:
        if bool(DBG_PRINT_SUMMARY):
            print(f"Optimal objective value is {model.objVal:g}")
    else:
        if bool(DBG_PRINT_SUMMARY):
            print(f"Optimization terminated with status {model.status}, exporting incumbent solution.")
            print(f"Incumbent objective value is {model.objVal:g}")

    x_sol = model.getAttr("X", x)
    z_sol = model.getAttr("X", z)
    loaded_by_type = {
        i: sum(x_sol[i, j, h] for j in J for h in H)
        for i in I
    }
    loaded_quantity = sum(loaded_by_type.values())
    loaded_length = sum(lengths[i] * loaded_by_type[i] for i in I)
    mandatory_loaded = sum(min(loaded_by_type[i], mandatory[i]) for i in I)
    optional_loaded = sum(max(loaded_by_type[i] - mandatory[i], 0.0) for i in I)

    car_sol = pd.DataFrame(
        {
            "program": [car_info.iloc[i - 1]["program"] for i in I],
            "model": [car_info.iloc[i - 1]["model"] for i in I],
            "length": [lengths[i] for i in I],
            "height": [heights[i] for i in I],
            "optional": [optional[i] for i in I],
            "mandatory": [mandatory[i] for i in I],
            "num": [loaded_by_type[i] for i in I],
        }
    )
    for extra_col in ["vehicle_size", "scale_score", "flat_compatible"]:
        if extra_col in car_info.columns:
            car_sol[extra_col] = list(car_info[extra_col])
    car_sol.to_csv(solution_dir / "car_sol.csv", index=False)

    carriage_list = []
    deck_counts = {p: 0 for p in P}
    for j in J:
        compartment = {h: {} for h in H}
        selected_position = max(P, key=lambda p: z_sol[j, p])
        deck_counts[selected_position] += 1
        compartment["position"] = selected_position
        compartment["pi_left"] = 1 if deck_side_mode[selected_position]["left"] == "m" else 0
        compartment["pi_right"] = 1 if deck_side_mode[selected_position]["right"] == "m" else 0
        for i in I:
            key = f"{car_info.iloc[i - 1]['program']}-{car_info.iloc[i - 1]['model']}"
            for h in H:
                val = x_sol[i, j, h]
                if val > 1e-6:
                    compartment[h][key] = val
        carriage_list.append(compartment)

    with open(solution_dir / "carriage_info.json", "w", encoding="utf-8") as f:
        json.dump({"carriage": carriage_list}, f, indent=2, ensure_ascii=False)

    lower_loaded_qty = sum(x_sol[i, j, h] for i in I for j in J for h in H_lower)
    upper_loaded_qty = sum(x_sol[i, j, h] for i in I for j in J for h in H_upper)
    lower_loaded_len = sum(x_sol[i, j, h] * lengths[i] for i in I for j in J for h in H_lower)
    upper_loaded_len = sum(x_sol[i, j, h] * lengths[i] for i in I for j in J for h in H_upper)

    summary.update(
        {
            "obj_val": float(model.objVal),
            "obj_bound": float(model.objBound),
            "mip_gap": float(model.MIPGap) if model.IsMIP else None,
            "loaded_quantity": float(loaded_quantity),
            "loaded_length_mm": float(loaded_length),
            "loaded_length_m": float(loaded_length / 1000.0),
            "avg_qty_per_wagon": float(loaded_quantity / carriage_num),
            "avg_len_per_wagon_mm": float(loaded_length / carriage_num),
            "avg_len_per_vehicle_mm": float(loaded_length / loaded_quantity) if loaded_quantity else None,
            "nominal_utilization": float(loaded_length / (carriage_num * (Config.bottom_len + Config.top_len))),
            "nominal_idle_mm_per_wagon": float(
                (carriage_num * (Config.bottom_len + Config.top_len) - loaded_length) / carriage_num
            ),
            "mandatory_loaded": float(mandatory_loaded),
            "optional_loaded": float(optional_loaded),
            "optional_acceptance_rate": float(optional_loaded / sum(optional.values())) if sum(optional.values()) else None,
            "lower_loaded_qty": float(lower_loaded_qty),
            "upper_loaded_qty": float(upper_loaded_qty),
            "lower_loaded_len_mm": float(lower_loaded_len),
            "upper_loaded_len_mm": float(upper_loaded_len),
        }
    )
    for p, count in deck_counts.items():
        summary[f"deck_{p.replace('-', '')}_count"] = int(count)

    if "vehicle_size" in car_info.columns:
        for size in ["small", "medium", "large"]:
            idxs = [i for i in I if str(car_info.iloc[i - 1]["vehicle_size"]) == size]
            summary[f"loaded_{size}_qty"] = float(sum(loaded_by_type[i] for i in idxs))
            summary[f"loaded_{size}_len_mm"] = float(sum(lengths[i] * loaded_by_type[i] for i in idxs))

    car_records = car_info.to_dict(orient="records")
    car_solution_records = car_sol.to_dict(orient="records")
    positive_x = []
    for i in I:
        car_row = car_info.iloc[i - 1]
        for j in J:
            selected_position = max(P, key=lambda p: z_sol[j, p])
            for h in H:
                val = float(x_sol[i, j, h])
                if val > 1e-6:
                    positive_x.append(
                        {
                            "type_index": int(i),
                            "wagon_index": int(j),
                            "component": h,
                            "deck_position": selected_position,
                            "quantity": val,
                            "program": str(car_row["program"]),
                            "model": str(car_row["model"]),
                            "length": float(lengths[i]),
                            "height": float(heights[i]),
                            "mandatory": int(mandatory[i]),
                            "optional": int(optional[i]),
                            "vehicle_size": str(car_row["vehicle_size"]) if "vehicle_size" in car_info.columns else None,
                        }
                    )
    solution_detail = {
        "summary": summary,
        "indices": {
            "I": I,
            "J": J,
            "H": H,
            "H_lower": H_lower,
            "H_upper": H_upper,
            "P": P,
        },
        "components": {
            h: {
                "length_capacity": float(L[h]),
                "side": component_side[h],
                "height_h": float(h_h_limits[h]),
                "height_m": float(h_m_limits[h]),
            }
            for h in H
        },
        "deck_positions": {
            int(j): {
                "position": max(P, key=lambda p: z_sol[j, p]),
                "z": {p: float(z_sol[j, p]) for p in P},
            }
            for j in J
        },
        "positive_x": positive_x,
        "cars": car_records,
        "car_solution": car_solution_records,
        "carriage": carriage_list,
    }
    with open(solution_dir / "solution_detail.pkl", "wb") as f:
        pickle.dump(solution_detail, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(solution_dir / "solve_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    instance_dir = Path(RUN_INSTANCE_DIR)
    output_dir = Path(RUN_OUTPUT_DIR) / instance_dir.name
    
    # Configuration parameters for dynamic segmentation
    num_splits = 1
    independent_mode_split = True

    print(f"Running instance {instance_dir.name} with num_splits={num_splits}, independent_mode_split={independent_mode_split}")
    
    build_and_solve(
        instance_dir, 
        output_dir, 
        log_to_console=bool(DBG_LOG_TO_CONSOLE),
        num_splits=num_splits,
        independent_mode_split=independent_mode_split
    )


if __name__ == "__main__":
    main()
