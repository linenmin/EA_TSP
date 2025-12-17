
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
# 策略：根据 HPC 超参搜索结果，为每个问题设置最优参数
# 搜索日期: 2024-12-16
# ==============================================================================
PROBLEM_CONFIGS = {
    "tour50.csv": {
        "N_RUNS": 10_000_000, "lam": 5000, "stagnation_limit": 800,
        # 岛屿特定参数 (默认值，小问题不需要精细调参)
        "exploit_mut": 0.3, "explore_mut": 0.8,
        "exploit_ls": 30, "explore_ls": 15
    },
    "tour250.csv": {
        "N_RUNS": 10_000_000, "lam": 500, "stagnation_limit": 400,
        "exploit_mut": 0.3, "explore_mut": 0.8,
        "exploit_ls": 30, "explore_ls": 15
    },
    "tour500.csv": {
        # HPC 最佳: 99562 (run 4)
        "N_RUNS": 10_000_000, "lam": 300, "stagnation_limit": 300,
        "exploit_mut": 0.4, "explore_mut": 0.6,
        "exploit_ls": 40, "explore_ls": 15
    },
    "tour750.csv": {
        # 诊断发现: 400 代不足，需要更多代数
        # 方案: 减小 λ (150→100), 减少 LS 步数 (30→20)
        "N_RUNS": 10_000_000, "lam": 100, "stagnation_limit": 200,
        "exploit_mut": 0.4, "explore_mut": 0.8,
        "exploit_ls": 20, "explore_ls": 10
    },
    "tour1000.csv": {
        # HPC 最佳: 63848 (run 1)
        "N_RUNS": 10_000_000, "lam": 100, "stagnation_limit": 150,
        "exploit_mut": 0.2, "explore_mut": 0.6,
        "exploit_ls": 30, "explore_ls": 15
    }
}

# 默认配置 (如果不匹配)
DEFAULT_CONFIG = {
    "N_RUNS": 10_000_000, "lam": 200, "stagnation_limit": 200,
    "exploit_mut": 0.3, "explore_mut": 0.8, "exploit_ls": 30, "explore_ls": 15
}

# ==============================================================================
# 2. 岛屿角色修正 (Role Modifiers)
# ==============================================================================
# 迁移策略: 分层筛选迁移 (Hierarchical Selection)
# - Explorer: 每 50 代发送 2% 种群 (前 30% 适应度 → 选距离最大的)
# - 发送前做 20 步 Or-Opt
# - 发送后不重启，继续进化
# ==============================================================================
def apply_role(base_config, role):
    """
    根据岛屿角色 (0=Exploiter, 1=Explorer) 应用问题特定的配置。
    
    Exploiter: 高选择压力、低变异率、深度 LS
    Explorer: 低选择压力、高变异率、轻量 LS (5 步)
    """
    cfg = {"N_RUNS": base_config["N_RUNS"], "lam": base_config["lam"], "mu": base_config["lam"]}
    
    if role == 0:  # Exploiter (精耕细作)
        cfg["k_tournament"] = 5
        cfg["mutation_rate"] = base_config["exploit_mut"]
        cfg["local_rate"] = 1.0  # 精英优先策略已内置
        cfg["ls_max_steps"] = base_config["exploit_ls"]
        cfg["stagnation_limit"] = int(base_config["stagnation_limit"] * 0.8)
        
    else:  # Explorer (多样性发生器)
        cfg["k_tournament"] = 2
        cfg["mutation_rate"] = base_config["explore_mut"]
        cfg["local_rate"] = 0.6
        cfg["ls_max_steps"] = base_config["explore_ls"]  # 轻量 LS，实际在 optimize 中固定为 5 步
        cfg["stagnation_limit"] = int(base_config["stagnation_limit"] * 1.5)
        
    return cfg

def island_worker(island_id, config, csv_file, mig_queue, recv_queue, log_file=None):
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
        rng_seed=seed,
        log_file=log_file  # 诊断日志文件
    )
    
    # Run optimization (blocking)
    solver.optimize(csv_file, mig_queue=mig_queue, recv_queue=recv_queue, island_id=island_id)
    
    print(f"[Launcher] Island {island_id} finished.")

def run_single_problem(target_csv, enable_log=True, log_folder="."):
    """运行单个问题的双岛屿模型
    
    Args:
        target_csv: 目标 CSV 文件名
        enable_log: 是否启用诊断日志
        log_folder: 日志存放文件夹路径
    """
    print("=" * 60)
    print(f"Starting Parallel Island Model (2 Islands)")
    print(f"Target: {target_csv}")
    print(f"Log folder: {log_folder}")
    print("=" * 60)
    
    if not os.path.exists(target_csv):
        print(f"Error: {target_csv} not found! Skipping...")
        return
    
    # 获取基础配置
    base_cfg = PROBLEM_CONFIGS.get(target_csv, DEFAULT_CONFIG)
    
    # 生成岛屿专属配置
    cfg0 = apply_role(base_cfg, 0)
    cfg1 = apply_role(base_cfg, 1)
    
    print(f"Config 0 (Exploiter): lam={cfg0['lam']}, mut={cfg0['mutation_rate']}, ls={cfg0['ls_max_steps']}")
    print(f"Config 1 (Explorer):  lam={cfg1['lam']}, mut={cfg1['mutation_rate']}, ls={cfg1['ls_max_steps']}")

    # Create communication queues
    q1 = multiprocessing.Queue(maxsize=10)
    q2 = multiprocessing.Queue(maxsize=10)
    
    # 诊断日志文件路径 (存入时间戳文件夹)
    if enable_log:
        log_0 = os.path.join(log_folder, f"island_0_{target_csv.replace('.csv', '')}_log.csv")
        log_1 = os.path.join(log_folder, f"island_1_{target_csv.replace('.csv', '')}_log.csv")
        print(f"[Diagnostic] Logs: {log_0}, {log_1}")
    else:
        log_0 = None
        log_1 = None
    
    # Define processes
    p1 = multiprocessing.Process(
        target=island_worker,
        args=(0, cfg0, target_csv, q1, q2, log_0) 
    )
    
    p2 = multiprocessing.Process(
        target=island_worker,
        args=(1, cfg1, target_csv, q2, q1, log_1) 
    )

    try:
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        
    except KeyboardInterrupt:
        print("\n[Launcher] Stopping islands...")
        p1.terminate()
        p2.terminate()
        p1.join()
        p2.join()
        print("[Launcher] Stopped.")


def main():
    # ==============================================================================
    # 用户配置区域
    # ==============================================================================
    
    # 选择要运行的文件列表 (并行执行)
    TARGET_FILES = [
        "tour500.csv",
        "tour750.csv",
        "tour1000.csv",
    ]
    
    # 是否启用诊断日志
    ENABLE_DIAGNOSTIC_LOG = True
    
    # ==============================================================================
    
    # 创建时间戳文件夹存放日志
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_folder = f"logs_{timestamp}"
    if ENABLE_DIAGNOSTIC_LOG:
        os.makedirs(log_folder, exist_ok=True)
        print(f"[Diagnostic] Log folder: {log_folder}")
    
    print("=" * 60)
    print(f"并行运行模式: {len(TARGET_FILES)} 个问题同时运行")
    print(f"问题列表: {TARGET_FILES}")
    print(f"总进程数: {len(TARGET_FILES) * 2} (每个问题 2 个岛屿)")
    print("=" * 60)
    
    # 收集所有进程
    all_processes = []
    
    for target_csv in TARGET_FILES:
        if not os.path.exists(target_csv):
            print(f"[Warning] {target_csv} not found! Skipping...")
            continue
        
        # 获取配置
        base_cfg = PROBLEM_CONFIGS.get(target_csv, DEFAULT_CONFIG)
        cfg0 = apply_role(base_cfg, 0)
        cfg1 = apply_role(base_cfg, 1)
        
        print(f"[{target_csv}] Exploiter: lam={cfg0['lam']}, Explorer: lam={cfg1['lam']}")
        
        # 诊断日志 (存入时间戳文件夹)
        if ENABLE_DIAGNOSTIC_LOG:
            log_0 = os.path.join(log_folder, f"island_0_{target_csv.replace('.csv', '')}_log.csv")
            log_1 = os.path.join(log_folder, f"island_1_{target_csv.replace('.csv', '')}_log.csv")
        else:
            log_0 = None
            log_1 = None
        
        # 为每个问题创建独立的通信队列
        q1 = multiprocessing.Queue(maxsize=10)
        q2 = multiprocessing.Queue(maxsize=10)
        
        # 创建进程
        p1 = multiprocessing.Process(
            target=island_worker,
            args=(0, cfg0, target_csv, q1, q2, log_0),
            name=f"{target_csv}_Exploiter"
        )
        p2 = multiprocessing.Process(
            target=island_worker,
            args=(1, cfg1, target_csv, q2, q1, log_1),
            name=f"{target_csv}_Explorer"
        )
        
        all_processes.extend([p1, p2])
    
    print(f"\n启动 {len(all_processes)} 个进程...")
    
    # 收集所有队列用于清理
    all_queues = []
    
    try:
        # 启动所有进程
        for p in all_processes:
            p.start()
        
        # 等待所有进程完成 (带超时)
        for p in all_processes:
            p.join(timeout=330)  # 5.5 分钟超时
            if p.is_alive():
                print(f"[Warning] Process {p.name} still alive, terminating...")
                p.terminate()
                p.join(timeout=5)
        
    except KeyboardInterrupt:
        print("\n[Launcher] 正在停止所有进程...")
        for p in all_processes:
            p.terminate()
        for p in all_processes:
            p.join(timeout=5)
    
    print("\n" + "=" * 60)
    print("所有问题运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    # Windows support for multiprocessing
    multiprocessing.freeze_support()
    main()
