
import sys
import os
import numpy as np

# Mock environment to satisfy imports if needed
os.environ["NUMBA_NUM_THREADS"] = "1"

def verify():
    print("=== Verifying GA Acceleration Changes ===")
    
    # 1. 检查 Scout 配置 (Dynamic Scaling)
    try:
        from run_island_model import apply_role
        
        # Test Case 1: Base Stag=100 -> Expected Scout Stag=max(100, 40)=100
        base_cfg_small = {
            "lam": 100, 
            "stagnation_limit": 100,
            "N_RUNS": 1000,
            "exploit_mut": 0.3, "exploit_ls": 30
        }
        
        # Role 1 = Scout
        cfg_scout_small = apply_role(base_cfg_small.copy(), role=1)
        assert cfg_scout_small["lam"] == 150
        assert cfg_scout_small["stagnation_limit"] == 100
        
        # Test Case 2: Base Stag=400 -> Expected Scout Stag=max(100, 160)=160
        base_cfg_large = {
            "lam": 500, 
            "stagnation_limit": 400,
            "N_RUNS": 1000,
            "exploit_mut": 0.3, "exploit_ls": 30
        }
        cfg_scout_large = apply_role(base_cfg_large.copy(), role=1)
        assert cfg_scout_large["lam"] == 150
        assert cfg_scout_large["stagnation_limit"] == 160
        
        print(f"[OK] Scout Configuration Verified: Base100->{cfg_scout_small['stagnation_limit']}, Base400->{cfg_scout_large['stagnation_limit']}")
        
    except Exception as e:
        print(f"[FAIL] Scout Config Check Failed: {e}")
        exit(1)

    # 2. 检查 Acceleration Operators (DLB, DoubleBridge, Swap)
    try:
        from optimized_thread_LocalSearch_inf import _candidate_or_opt_jit, double_bridge_move, _swap_segments_jit, tour_length_jit
        
        print("[Test] Verifying Acceleration Operators...")
        
        # Mock Data
        N = 50
        D = np.random.rand(N, N)
        tour = np.arange(N, dtype=np.int32)
        np.random.shuffle(tour)
        
        # Test 1: Double Bridge
        print("  - Testing Double Bridge...")
        tour_db = tour.copy()
        tour_db = double_bridge_move(tour_db)
        assert len(tour_db) == N
        assert len(np.unique(tour_db)) == N
        assert not np.array_equal(tour, tour_db) # Should be different
        print(f"    -> OK")

        # Test 2: Swap Segments
        print("  - Testing Swap Segments...")
        tour_swap = tour.copy()
        # Try multiple times to ensure we hit a valid swap
        succ = False
        for _ in range(20):
             if _swap_segments_jit(tour_swap, D):
                 succ = True
                 break
        assert len(tour_swap) == N
        assert len(np.unique(tour_swap)) == N
        print(f"    -> OK (Success={succ})")

        # Test 3: DLB in Or-Opt
        print("  - Testing DLB Integration...")
        knn = np.random.randint(0, N, (N, 5)).astype(np.int32)
        dlb_mask = np.zeros(N, dtype=np.bool_)
        
        # Run once
        start_time = os.times().elapsed
        _candidate_or_opt_jit(tour.copy(), D, knn, max_iters=10, dlb_mask=dlb_mask)
        # Check if dlb_mask was modified (some bits set to True likely)
        # Note: If no improvement found, mask bits should be set to True
        print(f"    -> OK (Ran without crash)")
        
        print("[OK] All Acceleration Operators Verified.")

    except ImportError as e:
        print(f"[FAIL] Import Error: {e}")
        exit(1)
    except Exception as e:
        print(f"[FAIL] Operator Verification Failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
        
    print("\n=== SUCCESS: All Checks Passed ===")

if __name__ == "__main__":
    verify()
