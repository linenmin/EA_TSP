import multiprocessing
import time
import os
import random
import csv
import pandas as pd
import numpy as np

# ⭐ 关键修复：限制 Numba 线程数，避免多进程争抢资源
# 在 import 优化模块之前设置，每个进程只用 1 个 Numba 线程
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from optimized_thread_LocalSearch_inf import r0123456

# ==============================================================================
# Hyperparameter Search Logic
# ==============================================================================
# We have 36 cores. Each Experiment uses 2 cores (Exploiter + Explorer).
# Max Parallel Experiments = 18.
# Target: tour1000.csv (The hardest one)


# ==============================================================================
# Hyperparameter Search Logic
# ==============================================================================

# 用户指定需要搜索的文件列表
TARGET_FILES = [
    # "tour50.csv", 
    # "tour250.csv",
    # "tour500.csv",
    # "tour750.csv",
    "tour1000.csv"
]

LOG_FILE = "hpc_grid_search_results.csv"
NUM_TRIALS_PER_FILE = 36

# 不同文件大小对应不同的搜索范围
# 定则：小图搜大种群，大图搜小种群 (Memetic)

# ... (PARAM_GRIDS_BY_FILE definition)
# 不同文件大小对应不同的搜索范围
# 定则：小图搜大种群，大图搜小种群 (Memetic)
PARAM_GRIDS_BY_FILE = {
    # Tiny & Small (50-250): brute force diversity
    "tour50.csv": {
        "lam": [2000, 5000, 10000, 20000],
        "exploit_mut": [0.2, 0.3], "explore_mut": [0.7, 0.9],
        "exploit_ls_step": [20, 30], "explore_ls_step": [10, 20],
        "exploit_ls_rate": [0.2, 0.4], "explore_ls_rate": [0.1, 0.2]
    },
    "tour250.csv": {
        "lam": [1000, 2000, 3000, 5000],
        "exploit_mut": [0.2, 0.3], "explore_mut": [0.7, 0.9],
        "exploit_ls_step": [30, 40], "explore_ls_step": [10, 20],
        "exploit_ls_rate": [0.2, 0.4], "explore_ls_rate": [0.1, 0.2]
    },
    # Medium (500): Transition zone
    "tour500.csv": {
        # 修正：包含更小的种群 (150-500) 以支持 Memetic 搜索
        "lam": [150, 300, 500, 800],
        "exploit_mut": [0.2, 0.3], "explore_mut": [0.7, 0.9],
        "exploit_ls_step": [30, 40, 50], "explore_ls_step": [15, 25],
        "exploit_ls_rate": [0.5, 0.8, 1.0], "explore_ls_rate": [0.2, 0.4, 0.6]
    },
    # Large (750-1000): Memetic (High Quality, Low Quantity)
    "tour750.csv": {
        "lam": [100, 120, 150, 200],
        "exploit_mut": [0.2, 0.3, 0.4], "explore_mut": [0.7, 0.8, 0.9],
        "exploit_ls_step": [40, 50, 60], "explore_ls_step": [15, 20, 30],
        "exploit_ls_rate": [0.8, 1.0], "explore_ls_rate": [0.4, 0.6]
    },
    "tour1000.csv": {
         # 1000城非常吃力，尝试极简种群
        "lam": [50, 80, 100, 150],
        "exploit_mut": [0.2, 0.3, 0.4], "explore_mut": [0.7, 0.8, 0.9],
        "exploit_ls_step": [40, 50, 60], "explore_ls_step": [15, 20, 30],
        "exploit_ls_rate": [0.8, 1.0], "explore_ls_rate": [0.4, 0.6]
    }
}

# 默认回退 (Fallback)
DEFAULT_GRID = PARAM_GRIDS_BY_FILE["tour1000.csv"]

def worker_island_wrapper(island_id, config, csv_file, mig_queue, recv_queue, result_queue):
    """
    岛屿进程工作函数。
    ⭐ 关键修复：stagnation_limit 根据岛屿角色动态调整
    """
    seed = random.randint(0, 1_000_000)
    
    # ⭐ 动态设置 stagnation_limit（与 run_island_model.py 保持一致）
    base_stag = 150
    if island_id == 0:  # Exploiter: 更快重启
        stag_limit = int(base_stag * 0.8)  # 120
    else:  # Explorer: 允许更长时间探索
        stag_limit = int(base_stag * 1.5)  # 225
    
    solver = r0123456(
        N_RUNS=10_000_000,
        lam=int(config["lam"]),
        mu=int(config["lam"]),
        k_tournament=config["k_tournament"],
        mutation_rate=config["mutation_rate"],
        local_rate=config["local_rate"],
        ls_max_steps=config["ls_max_steps"],
        stagnation_limit=stag_limit,  # ⭐ 使用动态值
        rng_seed=seed
    )
    
    try:
        # Time limit is handled internally by Reporter/Optimize loop (300s)
        solver.optimize(csv_file, mig_queue=mig_queue, recv_queue=recv_queue, island_id=island_id)
        best_fit = solver.best_ever_fitness
    except Exception as e:
        print(f"Island {island_id} Error: {e}")
        best_fit = float('inf')
        
    result_queue.put((island_id, best_fit))


# ... (imports)

def generate_random_configs(grid, n=100):
    """从给定的网格中随机生成 N 组配置"""
    configs = []
    for i in range(n):
        c = {}
        for k, v in grid.items():
            c[k] = random.choice(v)
        configs.append(c)
    return configs

def main():
    print(f"Starting HPC Hyperparameter Search (支持断点续传)")
    print(f"Targets: {TARGET_FILES}")
    
    # --- 1. 初始化日志与断点恢复 ---
    
    # 默认表头键值 (基于tour1000的网格，假设所有网格键一致)
    keys = list(DEFAULT_GRID.keys())
    # CSV 表头: 文件名, 任务ID, 最优解, 耗时, ...参数...
    headers = ["filename", "run_id", "best_fitness", "duration"] + keys
    
    # 统计已完成的任务数量
    completed_counts = {fname: 0 for fname in TARGET_FILES}
    
    if os.path.exists(LOG_FILE):
        print(f"Found existing log: {LOG_FILE}, analyzing progress...")
        try:
            df = pd.read_csv(LOG_FILE)
            # 统计每个文件跑了多少行
            counts = df['filename'].value_counts()
            for fname in TARGET_FILES:
                if fname in counts:
                    completed_counts[fname] = counts[fname]
                    print(f"  - {fname}: 已完成 {completed_counts[fname]} / {NUM_TRIALS_PER_FILE}")
        except Exception as e:
            print(f"Error reading log file: {e}. Starting fresh.")
    else:
        # 如果文件不存在，创建并写入表头
        print("Creating new log file...")
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
    # MAX_PAIRS: 并行度控制
    # 原来是 18 (36核满载)，但这会导致内存带宽争抢，降低每核的有效算力
    # 导致 300秒内跑不了几代。降低到 14 (28核) 以留有余地。
    MAX_PAIRS = 14  
    
    # --- 2. 遍历每个文件进行搜索 ---
    for filename in TARGET_FILES:
        print(f"\n>>> Processing {filename} <<<")
        
        # 计算还需要跑多少组
        done = completed_counts.get(filename, 0)
        needed = NUM_TRIALS_PER_FILE - done
        
        if needed <= 0:
            print(f"Skipping {filename} (Already ran {done} trials).")
            continue
            
        print(f"Generating {needed} new random configs...")
        
        # 加载对应的参数网格
        grid = PARAM_GRIDS_BY_FILE.get(filename, DEFAULT_GRID)
        # 生成所需的随机配置
        experiments = generate_random_configs(grid, needed)
        
        # 封装为待处理列表 (继承之前的 ID 编号，从 done+1 开始)
        pending_experiments = [(done + i + 1, cfg) for i, cfg in enumerate(experiments)]
        active = [] 
        
        # --- 3. 并发执行循环 ---
        while pending_experiments or active:
            # A. 填充空闲槽位
            while len(active) < MAX_PAIRS and pending_experiments:
                run_id, params = pending_experiments.pop(0)
                print(f"[{filename}] Launching Run {run_id}/{NUM_TRIALS_PER_FILE}...")
                
                # 构建两个岛屿的配置
                # ⭐ 关键修复：local_rate 固定为与 run_island_model.py 一致的值
                # Exploiter: local_rate=1.0 (全民皆兵，每个子代必做局部搜索)
                # Explorer: local_rate=0.6 (保持探索性，但也要优化)
                cfg0 = {
                    "lam": params["lam"],
                    "k_tournament": 5, 
                    "mutation_rate": params["exploit_mut"],
                    "local_rate": 1.0,  # ⭐ 固定为 1.0，这是 Memetic 的核心！
                    "ls_max_steps": params["exploit_ls_step"]
                }
                cfg1 = {
                    "lam": params["lam"],
                    "k_tournament": 2, 
                    "mutation_rate": params["explore_mut"],
                    "local_rate": 0.6,  # ⭐ 固定为 0.6，与单跑一致
                    "ls_max_steps": params["explore_ls_step"]
                }
                
                # 创建通信队列
                q1 = multiprocessing.Queue(maxsize=10)
                q2 = multiprocessing.Queue(maxsize=10)
                res_q = multiprocessing.Queue()
                
                # 启动两个子进程
                p0 = multiprocessing.Process(target=worker_island_wrapper, args=(0, cfg0, filename, q1, q2, res_q))
                p1 = multiprocessing.Process(target=worker_island_wrapper, args=(1, cfg1, filename, q2, q1, res_q))
                
                p0.start()
                p1.start()
                
                # 记录活跃任务信息
                active.append({
                    "p0": p0, "p1": p1, "id": run_id, 
                    "cfg": params, "q": res_q, 
                    "start": time.time(), "file": filename
                })
                
            # B. 检查任务完成情况
            still_active = []
            for job in active:
                # 如果两个子进程都结束了
                if not job["p0"].is_alive() and not job["p1"].is_alive():
                    # 确保资源释放
                    job["p0"].join(); job["p1"].join()
                    
                    # 获取结果 (取两个岛中的最小值)
                    res = []
                    while not job["q"].empty(): res.append(job["q"].get())
                    best_fit = min([r[1] for r in res]) if res else float('inf')
                    duration = time.time() - job["start"]
                    
                    print(f"[{job['file']}] Run {job['id']} Done: {best_fit:.2f} (Time: {duration:.1f}s)")
                    
                    # 写入 CSV 日志 (立即持久化，防止断电丢失)
                    param_values = [job['cfg'][k] for k in keys]
                    row = [job['file'], job['id'], best_fit, duration] + param_values
                    with open(LOG_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                else:
                    still_active.append(job)
            
            # 更新活跃列表并稍作休眠
            active = still_active
            time.sleep(1) 
            
    print("All Searches Completed.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
