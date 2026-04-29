import sys
import subprocess
from pathlib import Path
import pandas as pd
import time
import json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.model.BPC_LayerMaster.BBtree import BBTree
from src.model.BPC_LayerMaster.CG import PatternColumn

def load_vns_columns_layer_master(master, json_path: Path, prefix: str):
    if not json_path.exists():
        print(f"[Warn] JSON file not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    car_type_map = {}
    for i in master.I:
        row = master.car_info.iloc[i - 1]
        brand = row["program"]
        model = row["model"]
        car_type_map[(str(brand).strip(), str(model).strip())] = i
        car_type_map[str(model).strip()] = i 
        
    added_count = 0
    for idx, c in enumerate(data.get("carriage", [])):
        deck_mode = c.get("position", "h-h")
        
        for comp, car_list in [("upper", c.get("top", [])), ("lower", c.get("bottom", []))]:
            q_dict = {}
            for m in car_list:
                m_clean = m.strip()
                car_id = car_type_map.get(m_clean)
                if car_id is None:
                    # Try to find by matching model substring
                    for k, v in car_type_map.items():
                        if isinstance(k, tuple) and k[1] == m_clean:
                            car_id = v
                            break
                if car_id is not None:
                    q_dict[car_id] = q_dict.get(car_id, 0) + 1
            
            if q_dict:
                cost = -sum(master.length[i] * q for i, q in q_dict.items())
                col_id = f"{prefix}_c{idx}_{comp}"
                pattern = PatternColumn(
                    column_id=col_id,
                    q=q_dict,
                    cost=cost,
                    compartment=comp,
                    deck_mode=deck_mode,
                    metadata={"source": prefix}
                )
                master.add_column(pattern)
                added_count += 1
                
    print(f"Loaded {added_count} layer columns from {json_path.name} ({prefix})")

def run_bpc(instance_name: str, use_warmstart: bool = True):
    instance_dir = PROJECT_ROOT / "data/Instance" / instance_name
    output_dir = PROJECT_ROOT / "result" / instance_name
    
    if use_warmstart:
        print(f"--- 1. Running VNS (C++) for {instance_name} ---")
        vns_exe = PROJECT_ROOT / "VNS_cpp" / "vns_solver"
        if not vns_exe.exists():
            print(f"[Error] VNS executable not found at {vns_exe}. Compiling...")
            sys.exit(1)
        try:
            subprocess.run([str(vns_exe), instance_name], cwd=str(PROJECT_ROOT / "VNS_cpp"), check=True)
        except subprocess.CalledProcessError as e:
            print(f"[Error] VNS execution failed: {e}")
            sys.exit(1)
    else:
        print(f"--- 1. Skipping VNS Warmstart for {instance_name} ---")
        
    print(f"--- 2. Initializing LayerMaster BPC for {instance_name} ---")
    bbtree = BBTree(
        instance_dir=instance_dir,
        output_root=PROJECT_ROOT / "result",
        max_nodes=5000,
        max_cg_iters=3000,
        log_to_console=False,
        use_dominance=True,
        use_cuts=False,
        print_bb_progress=True,
        print_subproblem_progress=False
    )
    
    if use_warmstart:
        print("--- 3. Injecting Warmstart Columns ---")
        bi_json = output_dir / "BI" / "carriage_info.json"
        vns_json = output_dir / "VNS" / "carriage_info.json"
        
        load_vns_columns_layer_master(bbtree.master, bi_json, "BI")
        load_vns_columns_layer_master(bbtree.master, vns_json, "VNS")
        
        print(f"Total layer columns in pool before BPC: {len(bbtree.master.columns)}")

    print(f"--- 4. Running LayerMaster Branch-Price-and-Cut ---")
    t0 = time.time()
    result = bbtree.solve()
    total_time = time.time() - t0

    print("\n==============================================")
    print("      LayerMaster BPC Execution Summary       ")
    print("==============================================")
    print(f"Instance           : {instance_name}")
    print(f"Explored Nodes     : {result.explored_nodes}")
    print(f"Generated Columns  : {result.generated_columns}")
    if result.best_objective is not None:
        print(f"Best Objective     : {result.best_objective:.2f}")
    else:
        print(f"Best Objective     : INFEASIBLE")
    
    print("\n--- Time Profiling ---")
    print(f"Total BPC Time     : {total_time:.2f} s")
    print(f"Master Solve Time  : {bbtree.cg_engine.stats.master_time:.2f} s")
    print(f"Pricing Total Time : {bbtree.cg_engine.stats.pricing_time:.2f} s")
    print(f"  ├─ Labeling Time : {bbtree.cg_engine.stats.labeling_time:.2f} s")
    print(f"  ├─ Feas Check BS : {bbtree.cg_engine.stats.bs_time:.2f} s")
    print("==============================================\n")

if __name__ == "__main__":
    target_instance = "m11c11"
    use_ws = True
    
    args = sys.argv[1:]
    if args:
        target_instance = args[0]
        if len(args) > 1 and args[1].lower() in ['false', '0', 'no', 'off']:
            use_ws = False

    run_bpc(target_instance, use_warmstart=use_ws)
