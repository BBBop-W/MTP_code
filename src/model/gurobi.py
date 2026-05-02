import json
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
    independent_mode_split: bool = True
) -> None:
    cars_path = instance_dir / "cars.csv"
    carriage_path = instance_dir / "carriage.csv"
    if not cars_path.exists():
        raise FileNotFoundError(f"Missing file: {cars_path}")
    if not carriage_path.exists():
        raise FileNotFoundError(f"Missing file: {carriage_path}")

    car_info = normalize_car_table(cars_path)
    carriage_num = int(pd.read_csv(carriage_path)["carriage_num"].iloc[0])

    segments_data = get_model_segments(car_info, num_splits, independent_mode_split)

    I = list(range(1, len(car_info) + 1))
    J = list(range(1, carriage_num + 1))
    K = ["left", "right"]

    lengths = {i: float(car_info.iloc[i - 1]["length"]) for i in I}
    mandatory = {i: int(car_info.iloc[i - 1]["mandatory"]) for i in I}
    optional = {i: int(car_info.iloc[i - 1]["optional"]) for i in I}
    heights = {i: float(car_info.iloc[i - 1]["height"]) for i in I}

    H = []
    L = {}
    h_h_limits = {}
    h_m_limits = {}
    
    H_lower_k = {"left": [], "right": []}
    H_upper_k = {"left": [], "right": []}

    # Process lower deck segments
    lower_central = "lower_central"
    H.append(lower_central)
    L[lower_central] = segments_data["lower"]["central"]["len"]
    h_h_limits[lower_central] = segments_data["lower"]["central"]["h_h"]
    h_m_limits[lower_central] = segments_data["lower"]["central"]["h_m"]
    H_lower_k["left"].append(lower_central)
    H_lower_k["right"].append(lower_central)

    num_lower_blocks = len(segments_data["lower"]["blocks"])
    for idx, block in enumerate(segments_data["lower"]["blocks"]):
        b_name = block["name"] # e.g. block_1
        for side in ["left", "right"]:
            h_name = f"lower_{b_name}_{side}"
            H.append(h_name)
            L[h_name] = block["len"]
            h_h_limits[h_name] = block["h_h"]
            h_m_limits[h_name] = block["h_m"]
            H_lower_k[side].append(h_name)

    # Process upper deck segments
    upper_central = "upper_central"
    H.append(upper_central)
    L[upper_central] = segments_data["upper"]["central"]["len"]
    h_h_limits[upper_central] = segments_data["upper"]["central"]["h_h"]
    h_m_limits[upper_central] = segments_data["upper"]["central"]["h_m"]
    H_upper_k["left"].append(upper_central)
    H_upper_k["right"].append(upper_central)

    num_upper_blocks = len(segments_data["upper"]["blocks"])
    for idx, block in enumerate(segments_data["upper"]["blocks"]):
        b_name = block["name"]
        for side in ["left", "right"]:
            h_name = f"upper_{b_name}_{side}"
            H.append(h_name)
            L[h_name] = block["len"]
            h_h_limits[h_name] = block["h_h"]
            h_m_limits[h_name] = block["h_m"]
            H_upper_k[side].append(h_name)

    H_k = {
        "left": H_lower_k["left"] + H_upper_k["left"],
        "right": H_lower_k["right"] + H_upper_k["right"]
    }

    Delta = 400.0
    N = 10
    BigM = 25000.0  # Safe large number for length relaxation

    phi = {(i, h): int(heights[i] <= h_m_limits[h]) for i in I for h in H}
    epsilon = {(i, h): int(heights[i] <= h_h_limits[h]) for i in I for h in H}

    model = gp.Model("motorail_mlp_ic")
    model.Params.OutputFlag = 1 if log_to_console else 0
    model.Params.TimeLimit = Config.timelimit
    model.Params.MIPGap = Config.gap

    pi = model.addVars(J, K, vtype=gp.GRB.BINARY, name="pi")
    x = model.addVars(I, J, H, vtype=gp.GRB.INTEGER, lb=0, ub=N, name="x")

    # Objective
    model.setObjective(gp.quicksum(x[i, j, h] * lengths[i] for i in I for j in J for h in H), gp.GRB.MAXIMIZE)

    # Demand constraints
    model.addConstrs(
        gp.quicksum(x[i, j, h] for j in J for h in H) <= optional[i] + mandatory[i]
        for i in I
    )
    model.addConstrs(
        gp.quicksum(x[i, j, h] for j in J for h in H) >= mandatory[i]
        for i in I
    )

    # Interval capacity generation
    def get_intervals(layer_prefix, num_blocks):
        intervals = []
        for l in range(num_blocks + 1):
            for r in range(num_blocks + 1):
                blocks = [f"{layer_prefix}_central"]
                for i in range(1, l + 1):
                    blocks.append(f"{layer_prefix}_block_{i}_left")
                for i in range(1, r + 1):
                    blocks.append(f"{layer_prefix}_block_{i}_right")
                
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

    # Lower deck capacities (always enforced due to floor slope)
    for blocks, mod in lower_intervals:
        cap = sum(L[h] for h in blocks) + mod
        model.addConstrs(
            gp.quicksum(x[i, j, h] * (lengths[i] + Delta) for i in I for h in blocks) <= cap
            for j in J
        )

    # Upper deck capacities (internal boundaries relaxed when pi=0 i.e. flat deck)
    for blocks, mod in upper_intervals:
        cap = sum(L[h] for h in blocks) + mod
        is_full_deck = (len(blocks) == 2 * num_upper_blocks + 1)
        
        for j in J:
            if is_full_deck:
                model.addConstr(
                    gp.quicksum(x[i, j, h] * (lengths[i] + Delta) for i in I for h in blocks) <= cap
                )
            else:
                for k in K:
                    model.addConstr(
                        gp.quicksum(x[i, j, h] * (lengths[i] + Delta) for i in I for h in blocks) 
                        <= cap + BigM * (1 - pi[j, k])
                    )

    # Height limitations based on pi
    # mode 'h' (pi=0) => epsilon applies
    model.addConstrs(
        x[i, j, h] <= N * epsilon[i, h] + N * pi[j, k]
        for i in I for j in J for k in K for h in H_k[k]
    )

    # mode 'm' (pi=1) => phi applies
    model.addConstrs(
        x[i, j, h] <= N * phi[i, h] + N * (1 - pi[j, k])
        for i in I for j in J for k in K for h in H_k[k]
    )

    model.optimize()

    output_dir.mkdir(parents=True, exist_ok=True)
    solution_dir = output_dir / "solver"
    diagnostics_dir = output_dir / "diagnostics"
    solution_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    if model.status == gp.GRB.Status.INFEASIBLE:
        if bool(DBG_PRINT_SUMMARY):
            print(f"Optimization was stopped with status {model.status}")
        model.computeIIS()
        model.write(str(diagnostics_dir / "model_iis.ilp"))
        return

    if model.SolCount <= 0:
        if bool(DBG_PRINT_SUMMARY):
            print(f"Optimization terminated with status {model.status}, no feasible incumbent found.")
        return

    if model.status == gp.GRB.Status.OPTIMAL:
        if bool(DBG_PRINT_SUMMARY):
            print(f"Optimal objective value is {model.objVal:g}")
    else:
        if bool(DBG_PRINT_SUMMARY):
            print(f"Optimization terminated with status {model.status}, exporting incumbent solution.")
            print(f"Incumbent objective value is {model.objVal:g}")

    x_sol = model.getAttr("X", x)
    pi_sol = model.getAttr("X", pi)

    car_sol = pd.DataFrame(
        {
            "program": [car_info.iloc[i - 1]["program"] for i in I],
            "model": [car_info.iloc[i - 1]["model"] for i in I],
            "num": [
                sum(x_sol[i, j, h] for j in J for h in H) for i in I
            ],
        }
    )
    car_sol.to_csv(solution_dir / "car_sol.csv", index=False)

    carriage_list = []
    for j in J:
        compartment = {h: {} for h in H}
        compartment["pi_left"] = int(round(pi_sol[j, "left"]))
        compartment["pi_right"] = int(round(pi_sol[j, "right"]))
        for i in I:
            key = f"{car_info.iloc[i - 1]['program']}-{car_info.iloc[i - 1]['model']}"
            for h in H:
                val = x_sol[i, j, h]
                if val > 1e-6:
                    compartment[h][key] = val
        carriage_list.append(compartment)

    with open(solution_dir / "carriage_info.json", "w", encoding="utf-8") as f:
        json.dump({"carriage": carriage_list}, f, indent=2, ensure_ascii=False)

    summary = {
        "status": int(model.status),
        "sol_count": int(model.SolCount),
        "obj_val": float(model.objVal),
        "obj_bound": float(model.objBound),
        "mip_gap": float(model.MIPGap) if model.IsMIP else None,
    }
    with open(solution_dir / "solve_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


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
