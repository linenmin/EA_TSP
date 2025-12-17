
import numpy as np
import os
import sys
from optimized_thread_LocalSearch_inf import r0123456
from run_island_model import apply_role, DEFAULT_CONFIG

def test_scout_config():
    print("Testing Scout Config in apply_role...")
    cfg = apply_role(DEFAULT_CONFIG, 1) # Role 1 = Scout
    # Check key scout params
    success = True
    if cfg["lam"] != 100:
        print(f"FAIL: Scout lam should be 100, got {cfg['lam']}")
        success = False
    if cfg["ls_max_steps"] != 50:
        print(f"FAIL: Scout ls_max_steps should be 50, got {cfg['ls_max_steps']}")
        success = False
    if cfg["stagnation_limit"] != 50:
        print(f"FAIL: Scout stagnation_limit should be 50, got {cfg['stagnation_limit']}")
        success = False
        
    if success:
        print("PASS: Scout config correct.")

def test_asymmetry_detection_and_run():
    print("\nTesting Asymmetry Detection & Running short loop...")
    # Create asymmetric matrix (3 cities)
    # 0->1: 10, 1->0: 100
    D = np.array([
        [0.0, 10.0, 100.0],
        [100.0, 0.0, 10.0], # 1->2 is 10, 2->1 is 100 (if 2->1 is 20 in previous thought)
        [10.0, 100.0, 0.0]
    ], dtype=np.float64)
    
    csv_name = "test_asym_verify.csv"
    np.savetxt(csv_name, D, delimiter=",")
    
    try:
        # Run extremely short optimization
        # N_RUNS=5 is enough to trigger init, mutation, crossover loop logic
        solver = r0123456(lam=10, N_RUNS=5, ls_max_steps=2) # Minimal setup
        
        # Optimize
        print("Starting solver...")
        solver.optimize(csv_name)
        
        # Check symmetry flag
        if hasattr(solver, "_is_symmetric"):
            if solver._is_symmetric is False:
                print("PASS: _is_symmetric is False (Correct).")
            else:
                print(f"FAIL: _is_symmetric is {solver._is_symmetric} (Expected False).")
        else:
            print("FAIL: _is_symmetric attribute missing.")
            
    except Exception as e:
        print(f"CRASH During Execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(csv_name):
            os.remove(csv_name)

if __name__ == "__main__":
    test_scout_config()
    test_asymmetry_detection_and_run()
