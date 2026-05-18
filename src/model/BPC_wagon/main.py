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

from src.model.BPC_wagon.BBtree import BBTree
from src.model.warmstart import load_wagon_warmstart_columns


def run_bpc(
    instance_name: str,
    use_warmstart: bool = True,
    num_splits: int = 1,
    independent_mode_split: bool = True,
    pricing_method: str = "merging",
):
    instance_dir = PROJECT_ROOT / "data/Instance" / instance_name
    output_dir = PROJECT_ROOT / "result" / instance_name

    import src.model.BPC_wagon.feasibility_check as feas
    feas.GLOBAL_NUM_SPLITS = num_splits
    feas.GLOBAL_INDEP_MODE = independent_mode_split
    feas._SEGMENTS_CACHE.clear()
    import src.model.BPC_layer.feasibility_check as layer_feas
    layer_feas.GLOBAL_NUM_SPLITS = num_splits
    layer_feas.GLOBAL_INDEP_MODE = independent_mode_split
    layer_feas._SEGMENTS_CACHE.clear()
    
    if use_warmstart:
        # 1. Run C++ VNS Solver
        print(f"--- 1. Running VNS (C++) for {instance_name} ---")
        vns_exe = PROJECT_ROOT / "VNS_cpp" / "vns_solver"
        
        if not vns_exe.exists():
            print(f"[Error] VNS executable not found at {vns_exe}. Please compile it first.")
            sys.exit(1)
            
        vns_cmd = [str(vns_exe), instance_name, str(num_splits), str(int(independent_mode_split))]
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
        use_cuts=False,
        pricing_method=pricing_method,
        num_splits=num_splits,
        independent_mode_split=independent_mode_split,
        print_bb_progress=True,
        print_subproblem_progress=False
    )
    
    if use_warmstart:
        # 3. Load BI and VNS columns to warmstart MasterProblem
        print("--- 3. Injecting Warmstart Columns ---")
        bi_json = output_dir / "BI" / "carriage_info.json"
        vns_json = output_dir / "VNS" / "carriage_info.json"
        load_wagon_warmstart_columns(
            bbtree.master,
            [(bi_json, "BI"), (vns_json, "VNS")],
        )
        
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
    print(f"Pricing Method     : {pricing_method}")
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
        if len(args) > 2:
            try:
                num_splits = int(args[2])
            except ValueError:
                num_splits = 1
        else:
            num_splits = 1
        if len(args) > 3:
            independent_mode_split = args[3].lower() not in ['false', '0', 'no', 'off']
        else:
            independent_mode_split = True
        if len(args) > 4:
            pricing_method = args[4].lower()
        else:
            pricing_method = "merging"
    else:
        num_splits = 1
        independent_mode_split = True
        pricing_method = "merging"

    run_bpc(
        target_instance,
        use_warmstart=use_ws,
        num_splits=num_splits,
        independent_mode_split=independent_mode_split,
        pricing_method=pricing_method,
    )
