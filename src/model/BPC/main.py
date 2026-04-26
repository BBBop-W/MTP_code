import sys
import subprocess
from pathlib import Path
import pandas as pd
import time

# Set up project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.model.BPC.BBtree import BBTree, normalize_car_table
from src.model.BPC.CG import PatternColumn

def load_vns_columns(master, csv_path: Path, prefix: str):
    """
    Read column.csv output from C++ VNS and inject them into the MasterProblem.
    Format: brand, model, 1, 2, 3, 4, 5
    """
    if not csv_path.exists():
        print(f"[Warn] Column file not found: {csv_path}")
        return

    # Read the column file
    df = pd.read_csv(csv_path)
    
    # We need to map (brand, model) to the master problem's car type index (1 to N).
    # master.car_info has 'program' (brand) and 'model'
    car_type_map = {}
    for i in master.I:
        row = master.car_info.iloc[i - 1]
        brand = row["program"]
        model = row["model"]
        car_type_map[(str(brand).strip(), str(model).strip())] = i

    # The dataframe has columns: brand, model, '1', '2', ... 'N_carriages'
    # We extract carriage counts
    carriage_cols = [c for c in df.columns if c not in ["brand", "model"]]
    
    added_count = 0
    for col_name in carriage_cols:
        q_dict = {}
        for _, row in df.iterrows():
            brand = str(row["brand"]).strip()
            model = str(row["model"]).strip()
            qty = int(row[col_name])
            if qty > 0:
                key = (brand, model)
                if key in car_type_map:
                    car_type_idx = car_type_map[key]
                    q_dict[car_type_idx] = qty
                else:
                    print(f"[Warn] Unknown car in VNS output: {brand} {model}")
        
        if q_dict:
            # Calculate cost (negative total length)
            cost = -sum(master.length[i] * q for i, q in q_dict.items())
            
            # Create PatternColumn
            col_id = f"{prefix}_{col_name}"
            pattern = PatternColumn(
                column_id=col_id,
                q=q_dict,
                cost=cost,
                metadata={"source": prefix}
            )
            master.add_column(pattern)
            added_count += 1
            
    print(f"Loaded {added_count} columns from {csv_path.name} ({prefix})")


def run_bpc(instance_name: str, use_warmstart: bool = True):
    instance_dir = PROJECT_ROOT / "data/Instance" / instance_name
    output_dir = PROJECT_ROOT / "result" / instance_name
    
    if use_warmstart:
        # 1. Run C++ VNS Solver
        print(f"--- 1. Running VNS (C++) for {instance_name} ---")
        vns_exe = PROJECT_ROOT / "VNS_cpp" / "vns_solver"
        
        if not vns_exe.exists():
            print(f"[Error] VNS executable not found at {vns_exe}. Please compile it first.")
            sys.exit(1)
            
        vns_cmd = [str(vns_exe), instance_name]
        try:
            # Run VNS and pipe output to console
            subprocess.run(vns_cmd, cwd=str(PROJECT_ROOT / "VNS_cpp"), check=True)
        except subprocess.CalledProcessError as e:
            print(f"[Error] VNS execution failed: {e}")
            sys.exit(1)
    else:
        print(f"--- 1. Skipping VNS Warmstart for {instance_name} ---")
        
    print(f"--- 2. Initializing BPC for {instance_name} ---")
    # Initialize BBTree
    bbtree = BBTree(
        instance_dir=instance_dir,
        output_root=PROJECT_ROOT / "result",
        max_nodes=5000,
        max_cg_iters=3000,
        log_to_console=False,
        use_dominance=True,
        use_cuts=True,
        print_bb_progress=True,
        print_subproblem_progress=False
    )
    
    if use_warmstart:
        # 3. Load BI and VNS columns to warmstart MasterProblem
        print("--- 3. Injecting Warmstart Columns ---")
        bi_csv = output_dir / "BI" / "column.csv"
        vns_csv = output_dir / "VNS" / "column.csv"
        
        load_vns_columns(bbtree.master, bi_csv, "BI")
        load_vns_columns(bbtree.master, vns_csv, "VNS")
        
        print(f"Total columns in pool before BPC: {len(bbtree.master.columns)}")

    # 4. Run Branch-Price-and-Cut
    print(f"--- 4. Running Branch-Price-and-Cut (BPC) ---")
    t0 = time.time()
    result = bbtree.solve()
    total_time = time.time() - t0

    # 5. Output Summary
    print("\n==============================================")
    print("           BPC Execution Summary              ")
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
    print(f"  └─ Merging Time  : {bbtree.cg_engine.stats.merge_time:.2f} s")
    print("==============================================\n")

if __name__ == "__main__":
    target_instance = "m7c7"
    use_ws = True
    
    args = sys.argv[1:]
    if args:
        target_instance = args[0]
        if len(args) > 1 and args[1].lower() in ['false', '0', 'no', 'off']:
            use_ws = False

    run_bpc(target_instance, use_warmstart=use_ws)