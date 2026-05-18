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

from src.model.BPC_layer.BBtree import BBTree
from src.model.warmstart import load_compartment_warmstart_columns

def run_bpc(
    instance_name: str,
    use_warmstart: bool = True,
    num_splits: int = 1,
    independent_mode_split: bool = True,
    use_rc_bound: bool = True,
    use_residual_profile: bool = True,
    use_height_order: bool = True,
    use_local_residual_skyline: bool = True,
    residual_profile_mode: str = "full",
    profile_generator_mode: str = "hyb",
    compute_reachable_types: bool = False,
    pricing_method: str = "labeling",
):
    instance_dir = PROJECT_ROOT / "data/Instance" / instance_name
    output_dir = PROJECT_ROOT / "result" / instance_name

    # Pass dynamic split configuration to BPC's feasibility check
    import src.model.BPC_layer.feasibility_check as feas
    feas.GLOBAL_NUM_SPLITS = num_splits
    feas.GLOBAL_INDEP_MODE = independent_mode_split
    feas._SEGMENTS_CACHE.clear()

    if use_warmstart:
        print(f"--- 1. Running VNS (C++) for {instance_name} ---")
        vns_exe = PROJECT_ROOT / "VNS_cpp" / "vns_solver"
        if not vns_exe.exists():
            print(f"[Error] VNS executable not found at {vns_exe}. Compiling...")
            sys.exit(1)
        try:
            subprocess.run([str(vns_exe), instance_name, str(num_splits), str(int(independent_mode_split))], cwd=str(PROJECT_ROOT / "VNS_cpp"), check=True)
        except subprocess.CalledProcessError as e:
            print(f"[Error] VNS execution failed: {e}")
            sys.exit(1)
    else:
        print(f"--- 1. Skipping VNS Warmstart for {instance_name} ---")
        
    print(f"--- 2. Initializing CompartmentMaster BPC for {instance_name} ---")
    bbtree = BBTree(
        instance_dir=instance_dir,
        output_root=PROJECT_ROOT / "result",
        max_nodes=5000,
        max_cg_iters=3000,
        log_to_console=False,
        use_dominance=True,
        use_cuts=False,
        pricing_method=pricing_method,
        use_rc_bound=use_rc_bound,
        use_residual_profile=use_residual_profile,
        use_height_order=use_height_order,
        use_local_residual_skyline=use_local_residual_skyline,
        residual_profile_mode=residual_profile_mode,
        profile_generator_mode=profile_generator_mode,
        compute_reachable_types=compute_reachable_types,
        print_bb_progress=True,
        print_subproblem_progress=False
    )
    
    if use_warmstart:
        print("--- 3. Injecting Warmstart Columns ---")
        bi_json = output_dir / "BI" / "carriage_info.json"
        vns_json = output_dir / "VNS" / "carriage_info.json"
        load_compartment_warmstart_columns(
            bbtree.master,
            [(bi_json, "BI"), (vns_json, "VNS")],
        )
        
        print(f"Total compartment columns in pool before BPC: {len(bbtree.master.columns)}")

    print(f"--- 4. Running CompartmentMaster Branch-Price-and-Cut ---")
    t0 = time.time()
    result = bbtree.solve()
    total_time = time.time() - t0

    print("\n==============================================")
    print("   CompartmentMaster BPC Execution Summary    ")
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
    print(f"  ├─ Bound Pruned  : {bbtree.cg_engine.stats.labels_pruned_by_bound}")
    print(f"  ├─ Dom. Pruned   : {bbtree.cg_engine.stats.labels_pruned_by_dominance}")
    print(f"  ├─ Local Pruned  : {bbtree.cg_engine.stats.labels_pruned_by_local_skyline}")
    print(f"  ├─ Kept Labels   : {bbtree.cg_engine.stats.labels_after_dominance}")
    print(f"  ├─ Reach Probes  : {bbtree.cg_engine.stats.reachability_probes}")
    print(f"  ├─ Residual Prof.: {use_residual_profile}")
    print(f"  ├─ Height Order  : {use_height_order}")
    print(f"  ├─ Local Skyline : {use_local_residual_skyline}")
    print(f"  ├─ Profile Mode  : {residual_profile_mode}")
    print(f"  ├─ Generator     : {profile_generator_mode}")
    print("==============================================\n")

if __name__ == "__main__":
    target_instance = "m11c11"
    use_ws = True
    
    args = sys.argv[1:]
    if args:
        target_instance = args[0]
        if len(args) > 1 and args[1].lower() in ['false', '0', 'no', 'off']:
            use_ws = False
        if len(args) > 2:
            pricing_method = args[2].lower()
        else:
            pricing_method = "labeling"
    else:
        pricing_method = "labeling"

    run_bpc(target_instance, use_warmstart=use_ws, pricing_method=pricing_method)
