
import multiprocessing
import time
import os
import random
from optimized_thread_LocalSearch_inf import r0123456


# ==============================================================================
# 1. 基础问题配置 (Base Configs per Problem)
# ==============================================================================
# 种群大小随城市数量线性或超线性增长
# ==============================================================================
# 1. 基础问题配置 (Base Configs per Problem)
# ==============================================================================
# 策略：反比配置 (Inverse Scaling)
# 小图计算快，可以使用超大种群来保证多样性。
# 大图计算慢，必须减小种群规模以换取迭代代数。
# ==============================================================================
# 1. 基础问题配置 (Base Configs per Problem)
# ==============================================================================
# 策略：Memetic Evolution (文化基因进化)
# 为了支持高频局部搜索 (Quality)，必须大幅缩减种群 (Quantity)。
PROBLEM_CONFIGS = {
    "tour50.csv": {
        "N_RUNS": 10_000_000, "lam": 10000, "mu": 10000, "stagnation_limit": 500
    },
    "tour250.csv": {
        "N_RUNS": 10_000_000, "lam": 1000, "mu": 1000, "stagnation_limit": 300
    },
    "tour500.csv": {
        "N_RUNS": 10_000_000, "lam": 300, "mu": 300, "stagnation_limit": 200
    },
    "tour750.csv": {
        "N_RUNS": 10_000_000, "lam": 150, "mu": 150, "stagnation_limit": 150
    },
    "tour1000.csv": {
        "N_RUNS": 10_000_000, "lam": 100, "mu": 100, "stagnation_limit": 150
    }
}

# 默认配置 (如果不匹配)
DEFAULT_CONFIG = {"N_RUNS": 10_000_000, "lam": 200, "mu": 200, "stagnation_limit": 200}

# ==============================================================================
# 2. 岛屿角色修正 (Role Modifiers)
# ==============================================================================
def apply_role(base_config, role):
    """
    根据岛屿角色 (0=Exploiter, 1=Explorer) 修改基础配置。
    """
    cfg = base_config.copy()
    
    if role == 0: # Exploiter (精耕细作)
        cfg["k_tournament"] = 5     # 高压力
        cfg["mutation_rate"] = 0.3  # 低变异
        # 核心修改：全民皆兵，每个子代都必须经过局部搜索优化
        cfg["local_rate"] = 1.0     
        cfg["ls_max_steps"] = 40    # 中深度
        # Stagnation 稍低，为了尽快重启
        cfg["stagnation_limit"] = int(cfg["stagnation_limit"] * 0.8)
        
    else: # Explorer (疯狂探索)
        cfg["k_tournament"] = 2     # 低压力 (接近随机)
        cfg["mutation_rate"] = 0.8  # 极高变异
        # 探索者也要优化，否则生成的垃圾解没有意义，只是步数少点
        cfg["local_rate"] = 0.6     
        cfg["ls_max_steps"] = 15    # 浅搜
        # Stagnation 稍高，允许流浪
        cfg["stagnation_limit"] = int(cfg["stagnation_limit"] * 1.5)
        
    return cfg

def island_worker(island_id, config, csv_file, mig_queue, recv_queue):
    """
    Worker function for a single island process.
    """
    role_name = "Exploiter" if island_id == 0 else "Explorer"
    print(f"[Launcher] Island {island_id} ({role_name}) starting (PID: {os.getpid()})...")
    
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
    # --- 用户在此处选择目标文件 ---
    TARGET_CSV = "tour500.csv"
    # TARGET_CSV = "tour500.csv"
    
    print("==================================================")
    print(f"Starting Parallel Island Model (2 Islands)")
    print(f"Target: {TARGET_CSV}")
    print("==================================================")
    
    if not os.path.exists(TARGET_CSV):
        print(f"Error: {TARGET_CSV} not found!")
        return

    # 获取基础配置
    base_cfg = PROBLEM_CONFIGS.get(TARGET_CSV, DEFAULT_CONFIG)
    
    # 生成岛屿专属配置
    cfg0 = apply_role(base_cfg, 0)
    cfg1 = apply_role(base_cfg, 1)
    
    print(f"Config 0 (Exploiter): lam={cfg0['lam']}, mut={cfg0['mutation_rate']}, ls={cfg0['ls_max_steps']}")
    print(f"Config 1 (Explorer):  lam={cfg1['lam']}, mut={cfg1['mutation_rate']}, ls={cfg1['ls_max_steps']}")

    # Create communication queues
    # q1: Island 0 -> Island 1
    # q2: Island 1 -> Island 0
    q1 = multiprocessing.Queue(maxsize=10)
    q2 = multiprocessing.Queue(maxsize=10)
    
    # Define processes
    p1 = multiprocessing.Process(
        target=island_worker,
        args=(0, cfg0, TARGET_CSV, q1, q2) 
    )
    
    p2 = multiprocessing.Process(
        target=island_worker,
        args=(1, cfg1, TARGET_CSV, q2, q1) 
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
