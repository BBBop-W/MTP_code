import sys
from pathlib import Path
import random
import time
from typing import Dict, List, Tuple, Iterator, Optional, Set, TYPE_CHECKING
import itertools
from math import inf
from dataclasses import dataclass

import gurobipy as gp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.utility.config import config as Config

if TYPE_CHECKING:
    from src.model.BPC.labeling import BSResult, LayerSpec

def _recurse_check_layer_bs(
    compartment: str, 
    deck_mode: str, 
    quantities: Dict[int, int], 
    car_lengths: Dict[int, float], 
    car_heights: Dict[int, float]
) -> float | None:
    cars = []
    for cid, q in quantities.items():
        for _ in range(q):
            cars.append(cid)
    
    if not cars:
        return 0.0

    if len(cars) > 8:
        return None

    mode_left = deck_mode.split("-")[0]
    mode_right = deck_mode.split("-")[1]
    delta = 400.0

    pi_left = 1 if mode_left == "m" else 0
    pi_right = 1 if mode_right == "m" else 0

    total_length = sum(car_lengths[c] for c in cars)

    if compartment == "lower":
        h_a_left = Config.A_height_m if pi_left == 1 else Config.A_height_h
        h_a_right = Config.A_height_m if pi_right == 1 else Config.A_height_h
        h_b = Config.B_height
        h_c = Config.C_height
        h_a_max = max(h_a_left, h_a_right)
        
        L_a = Config.A_len
        L_b = Config.B_len
        L_c = Config.C_len
        
        if any(car_heights[c] > h_c for c in cars):
            return None
            
        cars_1 = [c for c in cars if car_heights[c] > h_b]
        cars_2 = [c for c in cars if h_a_max < car_heights[c] <= h_b]
        cars_3 = [c for c in cars if car_heights[c] <= h_a_max]
        
        sum_C = sum(car_lengths[c] + delta for c in cars_1)
        if sum_C > L_c + delta + 1e-5:
            return None
            
        for assignment2 in itertools.product(["left", "right"], repeat=len(cars_2)):
            sum_B_L_2, sum_B_R_2 = 0.0, 0.0
            
            for c, side in zip(cars_2, assignment2):
                l = car_lengths[c] + delta
                if side == "left":
                    sum_B_L_2 += l
                else:
                    sum_B_R_2 += l
                    
            if sum_C + sum_B_L_2 + sum_B_R_2 > L_c + 2*L_b + delta + 1e-5:
                continue
                
            feasible2 = True
            if pi_left == 1 and sum_C + sum_B_L_2 > L_c + L_b + delta + 1e-5:
                feasible2 = False
            if pi_right == 1 and sum_C + sum_B_R_2 > L_c + L_b + delta + 1e-5:
                feasible2 = False
                
            if not feasible2:
                continue
                
            layer3_feasible = False
            for assignment3 in itertools.product(["left", "right"], repeat=len(cars_3)):
                sum_B_L_3, sum_A_L = 0.0, 0.0
                sum_B_R_3, sum_A_R = 0.0, 0.0
                
                for c, side in zip(cars_3, assignment3):
                    h = car_heights[c]
                    l = car_lengths[c] + delta
                    if side == "left":
                        if h > h_a_left: sum_B_L_3 += l
                        else: sum_A_L += l
                    else:
                        if h > h_a_right: sum_B_R_3 += l
                        else: sum_A_R += l
                        
                sum_B_L = sum_B_L_2 + sum_B_L_3
                sum_B_R = sum_B_R_2 + sum_B_R_3
                
                if sum_C + sum_B_L + sum_B_R > L_c + 2*L_b + delta + 1e-5: continue
                if sum_C + sum_B_L + sum_B_R + sum_A_L + sum_A_R > L_c + 2*L_b + 2*L_a - delta + 1e-5: continue
                
                feasible3 = True
                if pi_left == 1:
                    if sum_C + sum_B_L > L_c + L_b + delta + 1e-5: feasible3 = False
                    if sum_C + sum_B_L + sum_A_L > L_c + L_b + L_a + 1e-5: feasible3 = False
                if pi_right == 1:
                    if sum_C + sum_B_R > L_c + L_b + delta + 1e-5: feasible3 = False
                    if sum_C + sum_B_R + sum_A_R > L_c + L_b + L_a + 1e-5: feasible3 = False
                    
                if feasible3:
                    layer3_feasible = True
                    break
                    
            if layer3_feasible:
                return total_length

    else: # upper
        h_d_left = Config.D_height_m if pi_left == 1 else Config.D_height_h
        h_d_right = Config.D_height_m if pi_right == 1 else Config.D_height_h
        h_e = Config.E_height
        h_d_max = max(h_d_left, h_d_right)
        
        L_d = Config.D_len
        L_e = Config.E_len
        
        if any(car_heights[c] > h_e for c in cars):
            return None
            
        cars_1 = [c for c in cars if car_heights[c] > h_d_max]
        cars_2 = [c for c in cars if car_heights[c] <= h_d_max]
        
        sum_E_init = sum(car_lengths[c] + delta for c in cars_1)
        if sum_E_init > L_e + delta + 1e-5:
            return None
            
        for assignment in itertools.product(["left", "right"], repeat=len(cars_2)):
            sum_E_2 = 0.0
            sum_D_L = 0.0
            sum_D_R = 0.0
            
            for c, side in zip(cars_2, assignment):
                h = car_heights[c]
                l = car_lengths[c] + delta
                if side == "left":
                    if h > h_d_left: sum_E_2 += l
                    else: sum_D_L += l
                else:
                    if h > h_d_right: sum_E_2 += l
                    else: sum_D_R += l
                    
            sum_E = sum_E_init + sum_E_2
            if sum_E + sum_D_L + sum_D_R > L_e + 2*L_d - delta + 1e-5: continue
            
            feasible = True
            if pi_left == 1:
                if sum_E > L_e + delta + 1e-5: feasible = False
                if sum_E + sum_D_L > L_e + L_d + 1e-5: feasible = False
            if pi_right == 1:
                if sum_E > L_e + delta + 1e-5: feasible = False
                if sum_E + sum_D_R > L_e + L_d + 1e-5: feasible = False
                
            if feasible:
                return total_length

    return None


def _simple_check_layer_bs(
    compartment: str, 
    deck_mode: str, 
    quantities: Dict[int, int], 
    car_lengths: Dict[int, float], 
    car_heights: Dict[int, float]
) -> float | None:
    """
    检查单层车厢是否能够装载给定的车辆组合。
    逻辑：从内到外（从高度最高层到高度较低层），先把只能停在最中间层的车提取出来，
    然后剩下的车进行左右分配 (Left / Right)。
    这个完全贴合了你的物理业务逻辑：最核心的区域（C/E）是不分左右且连通的，
    而外围区域则严格区分为左半边和右半边。
    """
    cars = []
    for cid, q in quantities.items():
        for _ in range(q):
            cars.append(cid)
    
    if not cars:
        return 0.0

    # 限制车辆数量以保证 2^N 搜索极快
    if len(cars) > 8:
        return None

    if "-" in deck_mode:
        mode_left = deck_mode.split("-")[0]
        mode_right = deck_mode.split("-")[1]
    else:
        m = "h" if deck_mode == "horizontal" else "m"
        mode_left = m
        mode_right = m

    delta = 400.0

    # Pi = 1 表示 MIDDLE 模式 (中间有限高折损)，Pi = 0 表示 HORIZONTAL (平放)
    pi_left = 1 if mode_left == "m" else 0
    pi_right = 1 if mode_right == "m" else 0

    total_length = sum(car_lengths[c] for c in cars)

    if compartment == "lower":
        # 下层高度与长度参数
        h_a_left = Config.A_height_m if pi_left == 1 else Config.A_height_h
        h_a_right = Config.A_height_m if pi_right == 1 else Config.A_height_h
        h_b = Config.B_height
        h_c = Config.C_height
        
        L_a = Config.A_len
        L_b = Config.B_len
        L_c = Config.C_len
        
        # 预先检查：如果有车超过最高限制(C区限高)，直接不可行
        if any(car_heights[c] > h_c for c in cars):
            return None
            
        # 最核心部分：把必须停在中间的车辆单独挑出
        # 只要高度 > B_height，就只能停在中间(C)，这部分完全不分左右
        fixed_C = [c for c in cars if car_heights[c] > h_b]
        
        # 剩下的车辆将参与 Left/Right 的分割 (因为A,B区严格分左右)
        rem_cars = [c for c in cars if car_heights[c] <= h_b]
        
        # 枚举剩下的车的所有左右分配方案
        for assignment in itertools.product(["left", "right"], repeat=len(rem_cars)):
            # 统计各个子区域所需的长度
            sum_C = sum(car_lengths[c] + delta for c in fixed_C)
            sum_B_L, sum_B_R = 0.0, 0.0
            sum_A_L, sum_A_R = 0.0, 0.0
            
            for c, side in zip(rem_cars, assignment):
                h = car_heights[c]
                l = car_lengths[c] + delta
                if side == "left":
                    # 如果分到左边且高度 > 左边最外侧A区限高，则它只能向内退避到B区
                    if h > h_a_left: sum_B_L += l
                    else: sum_A_L += l
                else:
                    # 分到右边同理
                    if h > h_a_right: sum_B_R += l
                    else: sum_A_R += l
                    
            # 基础容量检查（匹配 Gurobi 默认的全局约束）
            # 1. 纯C区容量
            if sum_C > L_c + delta + 1e-5: continue
            # 2. 内层全量(C + B_L + B_R)
            if sum_C + sum_B_L + sum_B_R > L_c + 2*L_b + delta + 1e-5: continue
            # 3. 全局总长，需扣除两端Delta
            if sum_C + sum_B_L + sum_B_R + sum_A_L + sum_A_R > L_c + 2*L_b + 2*L_a - delta + 1e-5: continue
            
            feasible = True
            # 当左侧处于 Middle 状态时，检查左侧边界容量
            if pi_left == 1:
                if sum_C + sum_B_L > L_c + L_b + delta + 1e-5: feasible = False
                if sum_C + sum_B_L + sum_A_L > L_c + L_b + L_a + 1e-5: feasible = False
                
            # 当右侧处于 Middle 状态时，检查右侧边界容量
            if pi_right == 1:
                if sum_C + sum_B_R > L_c + L_b + delta + 1e-5: feasible = False
                if sum_C + sum_B_R + sum_A_R > L_c + L_b + L_a + 1e-5: feasible = False
                
            if feasible:
                return total_length

    else: # upper
        # 上层高度与长度参数
        h_d_left = Config.D_height_m if pi_left == 1 else Config.D_height_h
        h_d_right = Config.D_height_m if pi_right == 1 else Config.D_height_h
        h_e = Config.E_height
        
        L_d = Config.D_len
        L_e = Config.E_len
        
        # 预先检查：如果高度超最高(E区)，不可行
        if any(car_heights[c] > h_e for c in cars):
            return None
            
        # 上层只有中心区(E)和外侧区(D)，因为只有两层，所以所有车都参与左右分配。
        # 如果一辆车高 > 外侧区(D) 限高，那么它无论分到哪边都会去“占用”中心的 E 区资源
        for assignment in itertools.product(["left", "right"], repeat=len(cars)):
            sum_E = 0.0
            sum_D_L = 0.0
            sum_D_R = 0.0
            
            for c, side in zip(cars, assignment):
                h = car_heights[c]
                l = car_lengths[c] + delta
                if side == "left":
                    # 高于外侧，退避到中心E
                    if h > h_d_left: sum_E += l
                    else: sum_D_L += l
                else:
                    # 分别处理右侧
                    if h > h_d_right: sum_E += l
                    else: sum_D_R += l
                    
            # 基础容量检查：全局总长
            if sum_E + sum_D_L + sum_D_R > L_e + 2*L_d - delta + 1e-5: continue
            
            feasible = True
            # 当左侧处于 Middle 状态时，检查对应的局部截断约束
            if pi_left == 1:
                if sum_E > L_e + delta + 1e-5: feasible = False
                if sum_E + sum_D_L > L_e + L_d + 1e-5: feasible = False
                
            # 当右侧处于 Middle 状态时
            if pi_right == 1:
                if sum_E > L_e + delta + 1e-5: feasible = False
                if sum_E + sum_D_R > L_e + L_d + 1e-5: feasible = False
                
            if feasible:
                return total_length

    return None


class HierarchicalBSEvaluator:
    """Simplified 2^N Search Evaluator for maximum 8 cars.
    Wrapper to match the labeling.py API logic.
    """
    def __init__(self):
        self.accumulated_time = 0.0
        self._cache = {}

    def evaluate(self, layer: 'LayerSpec', quantities: Dict[int, int]) -> 'BSResult':
        from src.model.BPC.labeling import BSResult
        import time
        t0 = time.time()
        
        mode = layer.shape_params.get("deck", "h-h")
        compartment = layer.shape_params.get("compartment", "lower")
        
        # Cleanup zero quantities
        clean_q = {i: q for i, q in quantities.items() if q > 0}
        
        cache_key = (compartment, mode, tuple(sorted(clean_q.items())))
        if cache_key in self._cache:
            self.accumulated_time += (time.time() - t0)
            return self._cache[cache_key]
        
        res = _recurse_check_layer_bs(
            compartment=compartment,
            deck_mode=mode,
            quantities=clean_q,
            car_lengths=layer.car_lengths,
            car_heights=layer.car_heights
        )
        
        if res is not None:
            # Check reachable types (one-step extension)
            reachable = set()
            for t in layer.car_types:
                max_q = int(layer.max_quantity_by_type.get(t, 6))
                if clean_q.get(t, 0) >= max_q:
                    continue
                probe = dict(clean_q)
                probe[t] = probe.get(t, 0) + 1
                
                probe_key = (compartment, mode, tuple(sorted(probe.items())))
                if probe_key in self._cache:
                    if self._cache[probe_key].feasible:
                        reachable.add(t)
                    continue

                probe_res = _recurse_check_layer_bs(
                    compartment=compartment,
                    deck_mode=mode,
                    quantities=probe,
                    car_lengths=layer.car_lengths,
                    car_heights=layer.car_heights
                )
                if probe_res is not None:
                    reachable.add(t)
                    # We can't cache probe fully as BSResult since we don't know its reachable set yet,
                    # but we can cache feasibility. We'll just rely on main cache for full hits.
                    
            bs_result = BSResult(feasible=True, best_length=res, reachable_types=reachable)
            self._cache[cache_key] = bs_result
            self.accumulated_time += (time.time() - t0)
            return bs_result
            
        bs_result = BSResult(feasible=False, best_length=inf, reachable_types=set())
        self._cache[cache_key] = bs_result
        self.accumulated_time += (time.time() - t0)
        return bs_result


def check_layer_bs(
    compartment: str, 
    deck_mode: str, 
    quantities: Dict[int, int], 
    car_lengths: Dict[int, float], 
    car_heights: Dict[int, float]
) -> float | None:
    # Direct wrapper for new simple check
    return _simple_check_layer_bs(compartment, deck_mode, quantities, car_lengths, car_heights)

def check_layer_gurobi(
    compartment: str, 
    deck_mode: str, 
    quantities: Dict[int, int], 
    car_lengths: Dict[int, float], 
    car_heights: Dict[int, float]
) -> float | None:
    model = gp.Model("single_layer_check")
    model.Params.OutputFlag = 0
    
    I = [i for i, q in quantities.items() if q > 0]
    if not I:
        return 0.0

    K = ["left", "right"]
    A = {"left": "A_left", "right": "A_right"}
    B = {"left": "B_left", "right": "B_right"}
    D = {"left": "D_left", "right": "D_right"}
    C = "C"
    E = "E"

    mode_left = deck_mode.split("-")[0]
    mode_right = deck_mode.split("-")[1]
    
    # Pi = 1 means MIDDLE, Pi = 0 means HORIZONTAL (matching gurobi.py)
    pi = {"left": 1 if mode_left == "m" else 0, "right": 1 if mode_right == "m" else 0}

    if compartment == "lower":
        H = [A["left"], A["right"], B["left"], B["right"], C]
        H_k = {k: [A[k], B[k], C] for k in K}
        H1 = [A["left"], A["right"], B["left"], B["right"], C]
        H4 = {k: [A[k], B[k], C] for k in K}
        H5 = {k: [B[k], C] for k in K}
        H2 = [B["left"], B["right"], C]
    else:
        H = [D["left"], D["right"], E]
        H_k = {k: [D[k], E] for k in K}
        H3 = [D["left"], D["right"], E]
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

    def get_height_limit(h: str, pi_val: int) -> float:
        middle_flag = (pi_val == 1)
        if h in (A["left"], A["right"]):
            return Config.A_height_m if middle_flag else Config.A_height_h
        if h in (B["left"], B["right"]):
            return Config.B_height
        if h == C:
            return Config.C_height
        if h in (D["left"], D["right"]):
            return Config.D_height_m if middle_flag else Config.D_height_h
        if h == E:
            return Config.E_height
        raise ValueError(f"Unknown component: {h}")

    x = model.addVars(I, H, vtype=gp.GRB.INTEGER, lb=0, name="x")
    model.addConstrs((gp.quicksum(x[i, h] for h in H) == quantities[i] for i in I), name="qty")

    Delta = 400.0

    for i in I:
        for k in K:
            for h in H_k[k]:
                hl = get_height_limit(h, pi[k])
                if car_heights[i] > hl:
                    model.addConstr(x[i, h] == 0)

    if compartment == "lower":
        model.addConstr(gp.quicksum(x[i, h] * (car_lengths[i] + Delta) for i in I for h in H1) <= sum(L[h] for h in H1) - Delta)
        for k in K:
            # Enforce exactly like gurobi.py: Pi=1 (middle) -> enforced, Pi=0 (horizontal) -> relaxed
            if pi[k] == 1:
                model.addConstr(gp.quicksum(x[i, h] * (car_lengths[i] + Delta) for i in I for h in H4[k]) <= sum(L[h] for h in H4[k]))
                model.addConstr(gp.quicksum(x[i, h] * (car_lengths[i] + Delta) for i in I for h in H5[k]) <= sum(L[h] for h in H5[k]) + Delta)
        model.addConstr(gp.quicksum(x[i, h] * (car_lengths[i] + Delta) for i in I for h in H2) <= sum(L[h] for h in H2) + Delta)
        model.addConstr(gp.quicksum(x[i, C] * (car_lengths[i] + Delta) for i in I) <= L[C] + Delta)
    else:
        model.addConstr(gp.quicksum(x[i, h] * (car_lengths[i] + Delta) for i in I for h in H3) <= sum(L[h] for h in H3) - Delta)
        for k in K:
            if pi[k] == 1:
                model.addConstr(gp.quicksum(x[i, h] * (car_lengths[i] + Delta) for i in I for h in H6[k]) <= sum(L[h] for h in H6[k]))
                model.addConstr(gp.quicksum(x[i, E] * (car_lengths[i] + Delta) for i in I) <= L[E] + Delta)

    model.setObjective(0.0, gp.GRB.MAXIMIZE)
    model.optimize()

    if model.status == gp.GRB.OPTIMAL:
        return sum(car_lengths[i] * quantities[i] for i in I)
    return None

if __name__ == "__main__":
    from src.utility.generate_instance import load_candidates
    
    print("--- Feasibility Check Testing (Simplified BS) ---")
    excel_path = Path("data/raw data/尺寸整理.xlsx")
    if not excel_path.exists():
        print(f"Cannot find data at {excel_path}. Please verify path.")
        sys.exit(1)
        
    candidates = load_candidates(excel_path)
    
    car_lengths = {i: row["长"] for i, row in candidates.iterrows()}
    car_heights = {i: row["高"] for i, row in candidates.iterrows()}
    car_types = list(car_lengths.keys())
    
    tests = 100000
    rng = random.Random(42)
    
    print(f"Running {tests} random single-layer verification tests...")
    
    match_count = 0
    grb_time = 0.0
    bs_time = 0.0
    bs_rc_time = 0.0

    for case_id in range(tests):
        compartment = rng.choice(["upper", "lower"])
        deck_mode = rng.choice(["h-h", "h-m", "m-h", "m-m"])
        
        quantities = {i: 0 for i in car_types}
        total_cars = rng.randint(6, 8)
        for _ in range(total_cars):
            quantities[rng.choice(car_types)] += 1
        quantities = {i: q for i, q in quantities.items() if q > 0}
            
        t0 = time.time()
        gurobi_res = check_layer_gurobi(compartment, deck_mode, quantities, car_lengths, car_heights)
        t1 = time.time()
        bs_res = check_layer_bs(compartment, deck_mode, quantities, car_lengths, car_heights)
        t2 = time.time()
        bs_rc_res = _recurse_check_layer_bs(compartment, deck_mode, quantities, car_lengths, car_heights)
        t3 = time.time()
        
        grb_time += (t1 - t0)
        bs_time += (t2 - t1)
        bs_rc_time += (t3 - t2)
        
        match = False
        if bs_res is None and gurobi_res is None and bs_rc_res is None:
            match = True
        elif bs_res is not None and gurobi_res is not None and bs_rc_res is not None and abs(bs_res - gurobi_res) < 1e-5 and abs(bs_rc_res - gurobi_res) < 1e-5:
            match = True
            
        if match:
            match_count += 1
        else:
            print(f"\nTest {case_id} FAILED: {compartment} {deck_mode}")
            print("Cars:")
            for cid, q in quantities.items():
                print(f"  ID {cid}: Qty={q}, Len={car_lengths[cid]}, Height={car_heights[cid]}")
            print(f"Gurobi: {gurobi_res}, BS: {bs_res}, BS Recursive: {bs_rc_res}")

    print(f"\nMatch rate: {match_count}/{tests}")
    print(f"Gurobi total time: {grb_time:.4f} s")
    print(f"BS total time: {bs_time:.4f} s")
    print(f"BS Recursive total time: {bs_rc_time:.4f} s")
