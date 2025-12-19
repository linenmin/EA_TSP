
import multiprocessing
import time
import os
import random
from optimized_thread_LocalSearch_inf import r0123456

# ==============================================================================
# 配置区域 (Configuration)
# ==============================================================================

# 基础问题参数配置 (根据问题规模调整)
PROBLEM_CONFIGS = {
    "tour50.csv": {
        "N_RUNS": 10_000_000, "lam": 5000, "stagnation_limit": 800,
        "exploit_mut": 0.3, "explore_mut": 0.8,
        "exploit_ls": 30, "explore_ls": 15
    },
    "tour250.csv": {
        "N_RUNS": 10_000_000, "lam": 500, "stagnation_limit": 400,
        "exploit_mut": 0.3, "explore_mut": 0.8,
        "exploit_ls": 30, "explore_ls": 15
    },
    "tour500.csv": {
        # 针对 tour500 的特殊优化配置 (降低变异率，增加LS)
        "N_RUNS": 10_000_000, "lam": 300, "stagnation_limit": 300,
        "exploit_mut": 0.1, "explore_mut": 0.6,
        "exploit_ls": 40, "explore_ls": 15
    },
    "tour750.csv": {
        # 减小种群规模以加速迭代，降低变异率
        "N_RUNS": 10_000_000, "lam": 100, "stagnation_limit": 200,
        "exploit_mut": 0.1, "explore_mut": 0.8,
        "exploit_ls": 20, "explore_ls": 10
    },
    "tour1000.csv": {
        # 大规模问题配置
        "N_RUNS": 10_000_000, "lam": 100, "stagnation_limit": 150,
        "exploit_mut": 0.1, "explore_mut": 0.6,
        "exploit_ls": 30, "explore_ls": 15
    }
}

# 默认配置 (兜底)
DEFAULT_CONFIG = {
    "N_RUNS": 10_000_000, "lam": 200, "stagnation_limit": 200,
    "exploit_mut": 0.3, "explore_mut": 0.8, 
    "exploit_ls": 30, "explore_ls": 15
}

# ==============================================================================
# 核心逻辑 (Core Logic)
# ==============================================================================

def apply_role(base_config, role):
    """根据岛屿角色 (0=Exploiter, 1=Scout) 生成特定配置"""
    # 复制基础配置
    cfg = {"N_RUNS": base_config["N_RUNS"], "lam": base_config["lam"], "mu": base_config["lam"]}
    
    if role == 0:  # Exploiter (精耕细作型)
        cfg["k_tournament"] = 5  # 高选择压力
        cfg["mutation_rate"] = base_config["exploit_mut"]  # 低变异率
        cfg["local_rate"] = 1.0  # 全局LS
        cfg["ls_max_steps"] = base_config["exploit_ls"]  # 深度LS
        cfg["stagnation_limit"] = int(base_config["stagnation_limit"] * 0.8) # 较早重启
        
    else:  # Scout (快速侦察型)
        cfg["k_tournament"] = 2  # 低选择压力
        cfg["lam"] = 150  # 固定小种群
        cfg["mu"] = 150
        cfg["mutation_rate"] = 0.5 # 中等变异
        cfg["local_rate"] = 1.0    # 强LS保证质量
        cfg["ls_max_steps"] = 50   # 深度搜索
        # 动态停滞阈值，防止夭折
        cfg["stagnation_limit"] = max(100, int(base_config["stagnation_limit"] * 0.4))
        
    return cfg

def island_worker(island_id, config, csv_file, mig_queue, recv_queue, log_file=None):
    """岛屿工作进程函数"""
    role_name = "Exploiter" if island_id == 0 else "Explorer"
    print(f"[Launcher] Island {island_id} ({role_name}) starting (PID: {os.getpid()})...")
    
    # 随机种子偏移
    seed = random.randint(0, 100000) + island_id * 999
    
    # 初始化求解器
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
        log_file=log_file  # 诊断日志
    )
    
    # 运行优化 (阻塞直到完成)
    solver.optimize(csv_file, mig_queue=mig_queue, recv_queue=recv_queue, island_id=island_id)
    
    print(f"[Launcher] Island {island_id} finished.")

def main():
    """主函数：并行启动岛屿模型"""
    
    # 要运行的目标文件列表
    TARGET_FILES = [
        "tour500.csv",
        "tour750.csv",
        "tour1000.csv",
        # "tour50.csv",
        # "tour250.csv",
    ]
    
    # 是否启用诊断日志
    ENABLE_DIAGNOSTIC_LOG = True
    
    # 创建带时间戳的日志文件夹
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_folder = f"logs_{timestamp}"
    if ENABLE_DIAGNOSTIC_LOG:
        os.makedirs(log_folder, exist_ok=True)
        print(f"[Diagnostic] Log folder: {log_folder}")
    
    print("=" * 60)
    print(f"并行运行模式: {len(TARGET_FILES)} 个问题同时运行")
    print(f"总进程数: {len(TARGET_FILES) * 2} (每个问题 2 个岛屿)")
    print("=" * 60)
    
    all_processes = []
    all_queues = []
    
    # 遍历每个问题启动进程
    for target_csv in TARGET_FILES:
        if not os.path.exists(target_csv):
            print(f"[Warning] {target_csv} not found! Skipping...")
            continue
        
        # 获取配置并生成角色配置
        base_cfg = PROBLEM_CONFIGS.get(target_csv, DEFAULT_CONFIG)
        cfg0 = apply_role(base_cfg, 0) # Exploiter
        cfg1 = apply_role(base_cfg, 1) # Explorer
        
        print(f"[{target_csv}] Exploiter: lam={cfg0['lam']}, Explorer: lam={cfg1['lam']}")
        
        # 设置日志路径
        if ENABLE_DIAGNOSTIC_LOG:
            log_0 = os.path.join(log_folder, f"island_0_{target_csv.replace('.csv', '')}_log.csv")
            log_1 = os.path.join(log_folder, f"island_1_{target_csv.replace('.csv', '')}_log.csv")
        else:
            log_0 = None
            log_1 = None
        
        # 创建通信队列
        q1 = multiprocessing.Queue(maxsize=10)
        q2 = multiprocessing.Queue(maxsize=10)
        all_queues.extend([q1, q2])
        
        # 创建进程 0 (Exploiter)
        p1 = multiprocessing.Process(
            target=island_worker,
            args=(0, cfg0, target_csv, q1, q2, log_0),
            name=f"{target_csv}_Exploiter"
        )
        # 创建进程 1 (Scout)
        p2 = multiprocessing.Process(
            target=island_worker,
            args=(1, cfg1, target_csv, q2, q1, log_1),
            name=f"{target_csv}_Explorer"
        )
        
        p1.daemon = True
        p2.daemon = True
        all_processes.extend([p1, p2])
    
    print(f"\n启动 {len(all_processes)} 个进程...")
    
    try:
        # 启动所有进程
        for p in all_processes:
            p.start()
        
        # 等待所有进程完成
        for p in all_processes:
            p.join(timeout=310)  # 5.5 分钟超时
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
    
    finally:
        # 清理队列资源
        for q in all_queues:
            q.cancel_join_thread()
            try:
                while True:
                    q.get_nowait()
            except:
                pass
    
    print("\n" + "=" * 60)
    print("所有问题运行完成!")
    print("=" * 60)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
