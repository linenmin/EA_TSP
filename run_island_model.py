
import multiprocessing
import time
import os
import random
from optimized_thread_LocalSearch_inf import r0123456

# Configuration (Same as main script or custom)
# Target file
CSV_FILE = "tour750.csv"

# Configuration for Island 0: "The Exploiter" (精耕细作型)
# 低变异，高选择压力，深度局部搜索 => 负责挖掘局部最优
CONFIG_EXPLOIT = {
    "N_RUNS": 10_000_000,
    "lam": 200,
    "mu": 200,           # RTR ignore this but good for record
    "k_tournament": 5,    # High pressure
    "mutation_rate": 0.3, # Low mutation
    "local_rate": 0.3,    # Frequent LS
    "ls_max_steps": 50,   # Deep LS
    "stagnation_limit": 150 # Less tolerance for stagnation
}

# Configuration for Island 1: "The Explorer" (疯狂探索型)
# 高变异，低选择压力，快速局部搜索 => 负责跳出局部最优，寻找新大陆
CONFIG_EXPLORE = {
    "N_RUNS": 10_000_000,
    "lam": 200,
    "mu": 200,
    "k_tournament": 2,    # Low pressure (almost random)
    "mutation_rate": 0.8, # Very High mutation
    "local_rate": 0.1,    # Sparse LS
    "ls_max_steps": 20,   # Shallow LS
    "stagnation_limit": 300 # High tolerance, let it wander
}

def island_worker(island_id, config, csv_file, mig_queue, recv_queue):
    """
    Worker function for a single island process.
    """
    role = "Exploiter" if island_id == 0 else "Explorer"
    print(f"[Launcher] Island {island_id} ({role}) starting (PID: {os.getpid()})...")
    
    # Initialize solver with unique seed offset
    seed = random.randint(0, 100000) + island_id * 999
    
    solver = r0123456(
        N_RUNS=config["N_RUNS"],
        lam=config["lam"],
        mu=config["mu"],
        k_tournament=config["k_tournament"],
        mutation_rate=config["mutation_rate"],
        local_rate=config["local_rate"],
        ls_max_steps=config["ls_max_steps"],
        stagnation_limit=config["stagnation_limit"],
        rng_seed=seed
    )
    
    # Run optimization (blocking)
    solver.optimize(csv_file, mig_queue=mig_queue, recv_queue=recv_queue, island_id=island_id)
    
    print(f"[Launcher] Island {island_id} finished.")

def main():
    print("==================================================")
    print(f"Starting Parallel Island Model (2 Islands)")
    print(f"Target: {CSV_FILE}")
    print("==================================================")
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        return

    # Create communication queues
    # q1: Island 0 -> Island 1
    # q2: Island 1 -> Island 0
    q1 = multiprocessing.Queue(maxsize=10)
    q2 = multiprocessing.Queue(maxsize=10)
    
    # Define processes
    p1 = multiprocessing.Process(
        target=island_worker,
        args=(0, CONFIG_EXPLOIT, CSV_FILE, q1, q2) # Island 0 (Exploit) sends to q1
    )
    
    p2 = multiprocessing.Process(
        target=island_worker,
        args=(1, CONFIG_EXPLORE, CSV_FILE, q2, q1) # Island 1 (Explore) sends to q2
    )

    
    try:
        p1.start()
        p2.start()
        
        # Wait for them? Or just let them run?
        # Typically we wait
        p1.join()
        p2.join()
        
    except KeyboardInterrupt:
        print("\n[Launcher] Stopping islands...")
        p1.terminate()
        p2.terminate()
        p1.join()
        p2.join()
        print("[Launcher] Stopped.")

if __name__ == "__main__":
    # Windows support for multiprocessing
    multiprocessing.freeze_support()
    main()
