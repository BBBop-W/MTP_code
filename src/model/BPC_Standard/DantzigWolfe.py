import sys
import time
import math
import json
from collections import deque
from pathlib import Path

import gurobipy as gp
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utility.dynamic_segmentation import get_model_segments

class BBNode:
    def __init__(self, node_id, depth, branch_a_bounds, branch_q_bounds):
        self.node_id = node_id
        self.depth = depth
        self.branch_a_bounds = branch_a_bounds
        self.branch_q_bounds = branch_q_bounds

class DWMasterProblem:
    def __init__(self, car_info: pd.DataFrame, carriage_num: int, penalty_unmet: float = 1e6):
        self.car_info = car_info.reset_index(drop=True)
        self.carriage_num = int(carriage_num)
        self.penalty_unmet = float(penalty_unmet)

        self.I = list(range(1, len(self.car_info) + 1))
        self.length = {i: float(self.car_info.iloc[i - 1]["length"]) for i in self.I}
        self.D = {i: int(self.car_info.iloc[i - 1]["mandatory"]) for i in self.I}
        self.U = {i: int(self.car_info.iloc[i - 1]["mandatory"] + self.car_info.iloc[i - 1]["optional"]) for i in self.I}

        self.columns = [] 
        
    def add_column(self, q_dict, cost):
        # Only add if not already in pool
        col_key = tuple(sorted(q_dict.items()))
        for c in self.columns:
            if tuple(sorted(c["q"].items())) == col_key:
                return False
        self.columns.append({"q": q_dict, "cost": cost})
        return True

    def solve_lp(self, node: BBNode):
        branch_a_bounds = node.branch_a_bounds or {}
        branch_q_bounds = node.branch_q_bounds or {}
        
        model = gp.Model("DW_Master")
        model.Params.OutputFlag = 0

        theta = {}
        for idx, col in enumerate(self.columns):
            theta[idx] = model.addVar(lb=0.0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, obj=col["cost"], name=f"theta_{idx}")

        unmet = {i: model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, obj=self.penalty_unmet, name=f"unmet_{i}") for i in self.I}

        mandatory_constr = {}
        optional_constr = {}
        for i in self.I:
            mandatory_constr[i] = model.addConstr(
                gp.quicksum(col["q"].get(i, 0) * theta[idx] for idx, col in enumerate(self.columns)) + unmet[i] >= self.D[i],
                name=f"mandatory_{i}"
            )
            optional_constr[i] = model.addConstr(
                gp.quicksum(col["q"].get(i, 0) * theta[idx] for idx, col in enumerate(self.columns)) <= self.U[i],
                name=f"optional_{i}"
            )

        wagon_constr = model.addConstr(
            gp.quicksum(theta[idx] for idx in range(len(self.columns))) <= self.carriage_num,
            name="wagon_limit"
        )

        branch_a_constr = {}
        for i, (lb, ub) in branch_a_bounds.items():
            expr = gp.quicksum(theta[idx] for idx, col in enumerate(self.columns) if col["q"].get(i, 0) > 0)
            if lb is not None: branch_a_constr[(i, 'lb')] = model.addConstr(expr >= lb)
            if ub is not None: branch_a_constr[(i, 'ub')] = model.addConstr(expr <= ub)
            
        branch_q_constr = {}
        for i, (lb, ub) in branch_q_bounds.items():
            expr = gp.quicksum(col["q"].get(i, 0) * theta[idx] for idx, col in enumerate(self.columns))
            if lb is not None: branch_q_constr[(i, 'lb')] = model.addConstr(expr >= lb)
            if ub is not None: branch_q_constr[(i, 'ub')] = model.addConstr(expr <= ub)

        model.ModelSense = gp.GRB.MINIMIZE
        model.optimize()

        if model.Status not in {gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL}:
            return None

        dual_branch_a = {i: 0.0 for i in self.I}
        for (i, btype), c in branch_a_constr.items():
            dual_branch_a[i] += float(c.Pi)
            
        dual_branch_q = {i: 0.0 for i in self.I}
        for (i, btype), c in branch_q_constr.items():
            dual_branch_q[i] += float(c.Pi)

        return {
            "obj": model.ObjVal,
            "theta": {idx: float(theta[idx].X) for idx in range(len(self.columns))},
            "duals": {
                "alpha": {i: float(mandatory_constr[i].Pi) for i in self.I},
                "beta": {i: float(optional_constr[i].Pi) for i in self.I},
                "kappa": float(wagon_constr.Pi),
                "branch_a": dual_branch_a,
                "branch_q": dual_branch_q
            }
        }

    def choose_branch_var(self, theta_vals, eps=1e-5):
        a_sums = {i: 0.0 for i in self.I}
        q_sums = {i: 0.0 for i in self.I}
        
        for idx, val in theta_vals.items():
            if val > eps:
                for i, q in self.columns[idx]["q"].items():
                    if q > 0:
                        a_sums[i] += val
                    q_sums[i] += q * val
                    
        for i in self.I:
            if abs(a_sums[i] - round(a_sums[i])) > eps:
                return 'a', i, a_sums[i]

        for i in self.I:
            if abs(q_sums[i] - round(q_sums[i])) > eps:
                return 'q', i, q_sums[i]

        return None

class DWPricingProblem:
    def __init__(self, car_info: pd.DataFrame, num_splits: int, independent_mode: bool):
        self.car_info = car_info
        self.I = list(range(1, len(self.car_info) + 1))
        self.lengths = {i: float(self.car_info.iloc[i - 1]["length"]) for i in self.I}
        self.heights = {i: float(self.car_info.iloc[i - 1]["height"]) for i in self.I}
        
        df = pd.DataFrame({"height": list(sorted(set(self.heights.values())))})
        self.segments_data = get_model_segments(df, num_splits, independent_mode)
        
        self.H = []
        self.L = {}
        self.h_h_limits = {}
        self.h_m_limits = {}
        self.H_k = {"left": [], "right": []}
        
        lower_central = "lower_central"
        self.H.append(lower_central)
        self.L[lower_central] = self.segments_data["lower"]["central"]["len"]
        self.h_h_limits[lower_central] = self.segments_data["lower"]["central"]["h_h"]
        self.h_m_limits[lower_central] = self.segments_data["lower"]["central"]["h_m"]
        self.H_k["left"].append(lower_central)
        self.H_k["right"].append(lower_central)

        self.num_lower_blocks = len(self.segments_data["lower"]["blocks"])
        for idx, block in enumerate(self.segments_data["lower"]["blocks"]):
            b_name = block["name"]
            for side in ["left", "right"]:
                h_name = f"lower_{b_name}_{side}"
                self.H.append(h_name)
                self.L[h_name] = block["len"]
                self.h_h_limits[h_name] = block["h_h"]
                self.h_m_limits[h_name] = block["h_m"]
                self.H_k[side].append(h_name)

        upper_central = "upper_central"
        self.H.append(upper_central)
        self.L[upper_central] = self.segments_data["upper"]["central"]["len"]
        self.h_h_limits[upper_central] = self.segments_data["upper"]["central"]["h_h"]
        self.h_m_limits[upper_central] = self.segments_data["upper"]["central"]["h_m"]
        self.H_k["left"].append(upper_central)
        self.H_k["right"].append(upper_central)

        self.num_upper_blocks = len(self.segments_data["upper"]["blocks"])
        for idx, block in enumerate(self.segments_data["upper"]["blocks"]):
            b_name = block["name"]
            for side in ["left", "right"]:
                h_name = f"upper_{b_name}_{side}"
                self.H.append(h_name)
                self.L[h_name] = block["len"]
                self.h_h_limits[h_name] = block["h_h"]
                self.h_m_limits[h_name] = block["h_m"]
                self.H_k[side].append(h_name)

        self.epsilon = {(i, h): int(self.heights[i] <= self.h_h_limits[h]) for i in self.I for h in self.H}
        self.phi = {(i, h): int(self.heights[i] <= self.h_m_limits[h]) for i in self.I for h in self.H}

    def get_intervals(self, layer_prefix, num_blocks):
        intervals = []
        Delta = 400.0
        for l in range(num_blocks + 1):
            for r in range(num_blocks + 1):
                blocks = [f"{layer_prefix}_central"]
                for i in range(1, l + 1):
                    blocks.append(f"{layer_prefix}_block_{i}_left")
                for i in range(1, r + 1):
                    blocks.append(f"{layer_prefix}_block_{i}_right")
                
                if l == num_blocks and r == num_blocks: mod = -Delta
                elif l == num_blocks or r == num_blocks: mod = 0
                else: mod = Delta
                intervals.append((blocks, mod))
        return intervals

    def solve(self, duals):
        model = gp.Model("DW_Pricing")
        model.Params.OutputFlag = 0
        model.Params.MIPGap = 1e-4

        K = ["left", "right"]
        pi = model.addVars(K, vtype=gp.GRB.BINARY, name="pi")
        x = model.addVars(self.I, self.H, vtype=gp.GRB.INTEGER, lb=0, name="x")
        
        q = {i: gp.quicksum(x[i, h] for h in self.H) for i in self.I}
        a = model.addVars(self.I, vtype=gp.GRB.BINARY, name="a")
        
        for i in self.I:
            model.addConstr(q[i] <= 100 * a[i])
            model.addConstr(q[i] >= a[i])
            
        expr = gp.LinExpr(-duals["kappa"])
        for i in self.I:
            expr += (-self.lengths[i] - duals["alpha"][i] - duals["beta"][i] - duals["branch_q"][i]) * q[i]
            expr += (-duals["branch_a"][i]) * a[i]
            
        model.setObjective(expr, gp.GRB.MINIMIZE)
        
        Delta = 400.0
        BigM = 25000.0
        N = 10

        lower_intervals = self.get_intervals("lower", self.num_lower_blocks)
        for blocks, mod in lower_intervals:
            cap = sum(self.L[h] for h in blocks) + mod
            model.addConstr(gp.quicksum(x[i, h] * (self.lengths[i] + Delta) for i in self.I for h in blocks) <= cap)

        upper_intervals = self.get_intervals("upper", self.num_upper_blocks)
        for blocks, mod in upper_intervals:
            cap = sum(self.L[h] for h in blocks) + mod
            is_full_deck = (len(blocks) == 2 * self.num_upper_blocks + 1)
            if is_full_deck:
                model.addConstr(gp.quicksum(x[i, h] * (self.lengths[i] + Delta) for i in self.I for h in blocks) <= cap)
            else:
                for k in K:
                    model.addConstr(gp.quicksum(x[i, h] * (self.lengths[i] + Delta) for i in self.I for h in blocks) <= cap + BigM * (1 - pi[k]))

        for k in K:
            for h in self.H_k[k]:
                for i in self.I:
                    model.addConstr(x[i, h] <= N * self.epsilon[i, h] + N * pi[k])
                    model.addConstr(x[i, h] <= N * self.phi[i, h] + N * (1 - pi[k]))

        model.optimize()
        
        if model.Status == gp.GRB.OPTIMAL:
            rc = model.ObjVal
            if rc < -1e-5:
                q_res = {i: int(round(q[i].getValue())) for i in self.I}
                cost = -sum(self.lengths[i] * q_res[i] for i in self.I)
                return {"q": q_res, "cost": cost, "rc": rc}
        return None

def run_standard_dw(instance_name: str, num_splits: int, independent_mode: bool):
    instance_dir = PROJECT_ROOT / "data/Instance" / instance_name
    cars_path = instance_dir / "cars.csv"
    carriage_path = instance_dir / "carriage.csv"
    
    car_info = pd.read_csv(cars_path)
    col_map = {"Brand": "program", "Model": "model", "Length": "length", "Height": "height", "Optional#": "optional", "Mandatory#": "mandatory"}
    car_info = car_info.rename(columns=col_map)
    carriage_num = int(pd.read_csv(carriage_path)["carriage_num"].iloc[0])

    print(f"--- Starting Standard Dantzig-Wolfe FULL BPC on {instance_name} ---")
    
    master = DWMasterProblem(car_info, carriage_num)
    pricing = DWPricingProblem(car_info, num_splits, independent_mode)

    for i in master.I:
        master.add_column({i: 1}, -master.length[i])

    vns_json = PROJECT_ROOT / "result" / instance_name / "VNS" / "carriage_info.json"
    if vns_json.exists():
        with open(vns_json, 'r') as f:
            vns_data = json.load(f)["carriage"]
            for c_info in vns_data:
                q_dict = {}
                cars_list = c_info.get("top", []) + c_info.get("bottom", [])
                from collections import Counter
                counts = Counter(cars_list)
                for car_name, qty in counts.items():
                    idx = None
                    for i in master.I:
                        model_name = master.car_info.iloc[i-1]["model"]
                        if model_name in car_name or car_name in model_name:
                            idx = i; break
                    if idx: q_dict[idx] = q_dict.get(idx, 0) + qty
                if q_dict:
                    cost = -sum(master.length[i] * q for i, q in q_dict.items())
                    master.add_column(q_dict, cost)
        print("Loaded VNS warmstart columns into Standard DW Master!")

    root_node = BBNode(0, 0, {}, {})
    queue = deque([root_node])
    global_lb = -math.inf
    best_obj = None
    node_count = 0
    t0 = time.time()
    
    print(f"{'Node':>6}  {'Depth':>6}  {'Left':>6}  {'Global LB':>14}  {'Current Node':>14}  {'Best Incumbent':>14}  {'Gap':>8}  {'Time(s)':>8}")

    while queue and node_count < 1000:
        node = queue.popleft()
        
        # CG Loop
        sol = None
        for it in range(100):
            sol = master.solve_lp(node)
            if not sol: break
            new_col = pricing.solve(sol["duals"])
            if new_col:
                master.add_column(new_col["q"], new_col["cost"])
            else:
                break
                
        node_count += 1
        elapsed = time.time() - t0
        
        if not sol:
            print(f"{node.node_id:6d}  {node.depth:6d}  {len(queue):6d}  {global_lb:14.2f}  {'Infeasible':>14}  {best_obj if best_obj else '-':>14}  {'-':>8}  {elapsed:8.2f}")
            continue
            
        cur_obj = sol["obj"]
        
        if best_obj is not None and cur_obj >= best_obj - 1e-5:
            continue
            
        branch_var = master.choose_branch_var(sol["theta"])
        gap_str = f"{abs(best_obj - global_lb)/abs(best_obj):.2%}" if best_obj and global_lb != -math.inf else "-"
        print(f"{node.node_id:6d}  {node.depth:6d}  {len(queue):6d}  {global_lb:14.2f}  {cur_obj:14.2f}  {best_obj if best_obj else '-':>14}  {gap_str:>8}  {elapsed:8.2f}")
        
        if branch_var is None:
            if best_obj is None or cur_obj < best_obj:
                best_obj = cur_obj
            continue
            
        btype, i, val = branch_var
        floor_val = math.floor(val)
        ceil_val = math.ceil(val)
        
        left_a = dict(node.branch_a_bounds)
        left_q = dict(node.branch_q_bounds)
        right_a = dict(node.branch_a_bounds)
        right_q = dict(node.branch_q_bounds)
        
        if btype == 'a':
            left_a[i] = (left_a.get(i, (None, None))[0], floor_val)
            right_a[i] = (ceil_val, right_a.get(i, (None, None))[1])
        else:
            left_q[i] = (left_q.get(i, (None, None))[0], floor_val)
            right_q[i] = (ceil_val, right_q.get(i, (None, None))[1])
            
        queue.append(BBNode(node_count, node.depth + 1, left_a, left_q))
        queue.append(BBNode(node_count+1, node.depth + 1, right_a, right_q))
        
        node_count += 1
        
        valid_nodes = []
        for n in queue:
            s = master.solve_lp(n)
            if s: valid_nodes.append(s["obj"])
            
        if valid_nodes:
            global_lb = min([cur_obj] + valid_nodes)
        else:
            global_lb = cur_obj

    print(f"\nFinal BPC Objective: {best_obj}")
    return best_obj

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_standard_dw(sys.argv[1], 1, True)
    else:
        run_standard_dw("m5c5", 1, True)