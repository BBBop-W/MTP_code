import json
import sys
from pathlib import Path

import gurobipy as gp
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utility.config import config as Config

# IDE debug switches.
# 0 = False, 1 = True.
DBG_LOG_TO_CONSOLE = 1
DBG_PRINT_SUMMARY = 1

# Runtime settings (edit directly when debugging).
RUN_INSTANCE_DIR = "data/Instance/m7c7"
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


def build_and_solve(instance_dir: Path, output_dir: Path, log_to_console: bool = False) -> None:
    cars_path = instance_dir / "cars.csv"
    carriage_path = instance_dir / "carriage.csv"
    if not cars_path.exists():
        raise FileNotFoundError(f"Missing file: {cars_path}")
    if not carriage_path.exists():
        raise FileNotFoundError(f"Missing file: {carriage_path}")

    car_info = normalize_car_table(cars_path)
    carriage_num = int(pd.read_csv(carriage_path)["carriage_num"].iloc[0])

    I = list(range(1, len(car_info) + 1))
    J = list(range(1, carriage_num + 1))
    K = ["left", "right"]

    A = {"left": "A_left", "right": "A_right"}
    B = {"left": "B_left", "right": "B_right"}
    D = {"left": "D_left", "right": "D_right"}
    C = "C"
    E = "E"

    H = [A["left"], A["right"], B["left"], B["right"], C, D["left"], D["right"], E]
    H_k = {
        k: [A[k], B[k], C, D[k], E] for k in K
    }
    H1 = [A["left"], A["right"], B["left"], B["right"], C]
    H2 = [B["left"], B["right"], C]
    H3 = [D["left"], D["right"], E]
    H4 = {k: [A[k], B[k], C] for k in K}
    H5 = {k: [B[k], C] for k in K}
    H6 = {k: [D[k], E] for k in K}

    L = {
        A["left"]: Config.A_len,
        A["right"]: Config.A_len,
        B["left"]: Config.B_len,
        B["right"]: Config.B_len,
        C: Config.C_len,
        D["left"]: Config.D_len,
        D["right"]: Config.D_len,
        E: Config.E_len,
    }

    Delta = 400.0
    N = 10
    m_h3_minus_h6 = max((sum(L[h] for h in H3 if h not in H6[k]) - Delta) for k in K)
    m_h3_minus_e = sum(L[h] for h in H3 if h != E) - 2 * Delta
    BigM = max(m_h3_minus_h6, m_h3_minus_e)

    lengths = {i: float(car_info.iloc[i - 1]["length"]) for i in I}
    mandatory = {i: int(car_info.iloc[i - 1]["mandatory"]) for i in I}
    optional = {i: int(car_info.iloc[i - 1]["optional"]) for i in I}
    heights = {i: float(car_info.iloc[i - 1]["height"]) for i in I}

    def height_limit(h: str, middle: bool) -> float:
        if h in (A["left"], A["right"]):
            return Config.A_height_m if middle else Config.A_height_h
        if h in (B["left"], B["right"]):
            return Config.B_height
        if h == C:
            return Config.C_height
        if h in (D["left"], D["right"]):
            return Config.D_height_m if middle else Config.D_height_h
        if h == E:
            return Config.E_height
        raise ValueError(f"Unknown component: {h}")

    phi = {(i, h): int(heights[i] <= height_limit(h, middle=True)) for i in I for h in H}
    epsilon = {(i, h): int(heights[i] <= height_limit(h, middle=False)) for i in I for h in H}

    model = gp.Model("motorail_mlp_ic")
    model.Params.OutputFlag = 1 if log_to_console else 0
    model.Params.TimeLimit = Config.timelimit
    model.Params.MIPGap = Config.gap

    pi = model.addVars(J, K, vtype=gp.GRB.BINARY, name="pi")
    x = model.addVars(I, J, H, vtype=gp.GRB.INTEGER, lb=0, ub=N, name="x")

    # Objective (eq_obj)
    model.setObjective(gp.quicksum(x[i, j, h] * lengths[i] for i in I for j in J for h in H), gp.GRB.MAXIMIZE)

    # (2)
    model.addConstrs(
        gp.quicksum(x[i, j, h] for j in J for h in H) <= optional[i] + mandatory[i]
        for i in I
    )

    # (3)
    model.addConstrs(
        gp.quicksum(x[i, j, h] for j in J for h in H) >= mandatory[i]
        for i in I
    )

    # (4)
    model.addConstrs(
        gp.quicksum(x[i, j, h] * (lengths[i] + Delta) for i in I for h in H1) <= sum(L[h] for h in H1) - Delta
        for j in J
    )
    model.addConstrs(
        gp.quicksum(x[i, j, h] * (lengths[i] + Delta) for i in I for h in H3) <= sum(L[h] for h in H3) - Delta
        for j in J
    )

    # (5)
    model.addConstrs(
        gp.quicksum(x[i, j, h] * (lengths[i] + Delta) for i in I for h in H4[k]) <= sum(L[h] for h in H4[k])
        for j in J for k in K
    )

    # (6)
    model.addConstrs(
        gp.quicksum(x[i, j, h] * (lengths[i] + Delta) for i in I for h in H5[k]) <= sum(L[h] for h in H5[k]) + Delta
        for j in J for k in K
    )
    model.addConstrs(
        gp.quicksum(x[i, j, h] * (lengths[i] + Delta) for i in I for h in H2) <= sum(L[h] for h in H2) + Delta
        for j in J for k in K
    )
    model.addConstrs(
        gp.quicksum(x[i, j, C] * (lengths[i] + Delta) for i in I) <= L[C] + Delta
        for j in J for k in K
    )

    # (7)
    model.addConstrs(
        gp.quicksum(x[i, j, h] * (lengths[i] + Delta) for i in I for h in H6[k])
        <= sum(L[h] for h in H6[k]) + BigM * (1 - pi[j, k])
        for j in J for k in K
    )

    # (8)
    model.addConstrs(
        gp.quicksum(x[i, j, E] * (lengths[i] + Delta) for i in I) <= L[E] + Delta + BigM * (1 - pi[j, k])
        for j in J for k in K
    )

    # (9)
    model.addConstrs(
        x[i, j, h] <= N * epsilon[i, h] + N * pi[j, k]
        for i in I for j in J for k in K for h in H_k[k]
    )

    # (10)
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
        compartment = {
            "A_left": {},
            "A_right": {},
            "B_left": {},
            "B_right": {},
            "C": {},
            "D_left": {},
            "D_right": {},
            "E": {},
            "pi_left": int(round(pi_sol[j, "left"])),
            "pi_right": int(round(pi_sol[j, "right"])),
        }
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
    build_and_solve(instance_dir, output_dir, bool(DBG_LOG_TO_CONSOLE))


if __name__ == "__main__":
    main()
