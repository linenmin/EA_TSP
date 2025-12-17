"""
单岛 GA 并行运行器 (Single Island GA Parallel Runner)
同时运行多个 CSV 文件，每个问题一个进程
"""

import multiprocessing
import time
import os
from datetime import datetime
from single_island_ga import SingleIslandGA


# ==============================================================================
# 问题配置
# ==============================================================================
PROBLEM_CONFIGS = {
    "tour50.csv": {
        "lam": 3000,              # 种群大小
        "k_tournament": 5,        # 锦标赛大小
        "mutation_rate": 0.3,     # 变异率
        "ls_max_steps": 30,       # LS 步数
        "stagnation_limit": 120,  # 停滞阈值
    },
    "tour250.csv": {
        "lam": 1000,
        "k_tournament": 5,
        "mutation_rate": 0.3,
        "ls_max_steps": 30,
        "stagnation_limit": 120,
    },
    "tour500.csv": {
        "lam": 300,
        "k_tournament": 5,
        "mutation_rate": 0.3,
        "ls_max_steps": 30,
        "stagnation_limit": 120,
    },
    "tour750.csv": {
        "lam": 150,
        "k_tournament": 5,
        "mutation_rate": 0.3,
        "ls_max_steps": 20,
        "stagnation_limit": 120,
    },
    "tour1000.csv": {
        "lam": 100,
        "k_tournament": 5,
        "mutation_rate": 0.35,
        "ls_max_steps": 15,
        "stagnation_limit": 150,
    },
}


def worker(csv_file, config, log_folder):
    """Worker 进程: 运行单个问题"""
    problem_name = csv_file.replace('.csv', '')
    print(f"[{csv_file}] 开始优化...")
    print(f"[{csv_file}] 配置: lam={config['lam']}, k={config['k_tournament']}, mut={config['mutation_rate']}")
    
    solver = SingleIslandGA(
        lam=config["lam"],
        k_tournament=config["k_tournament"],
        mutation_rate=config["mutation_rate"],
        ls_max_steps=config["ls_max_steps"],
        stagnation_limit=config["stagnation_limit"]
    )
    
    try:
        solver.optimize(csv_file)
    except Exception as e:
        print(f"[{csv_file}] 错误: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"[{csv_file}] 完成")


def main():
    """主函数: 并行运行所有问题"""
    
    # ==============================================================================
    # 用户配置区
    # ==============================================================================
    TARGET_FILES = [
        "tour500.csv",
        "tour750.csv",
        "tour1000.csv",
    ]
    
    ENABLE_LOG = True  # 是否启用日志
    # ==============================================================================
    
    # 创建时间戳文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_folder = f"single_logs_{timestamp}"
    
    if ENABLE_LOG:
        os.makedirs(log_folder, exist_ok=True)
        print(f"[Log] 日志文件夹: {log_folder}")
    
    print("=" * 60)
    print("单岛 GA 并行运行器")
    print(f"问题列表: {TARGET_FILES}")
    print(f"进程数: {len(TARGET_FILES)}")
    print("=" * 60)
    
    # 创建进程
    processes = []
    for csv_file in TARGET_FILES:
        if not os.path.exists(csv_file):
            print(f"[警告] {csv_file} 不存在，跳过")
            continue
        
        # 获取配置 (默认值回退)
        config = PROBLEM_CONFIGS.get(csv_file, {
            "lam": 100,
            "k_tournament": 5,
            "mutation_rate": 0.3,
            "ls_max_steps": 20,
            "stagnation_limit": 120,
        })
        
        p = multiprocessing.Process(
            target=worker,
            args=(csv_file, config, log_folder),
            name=csv_file
        )
        p.daemon = True
        processes.append(p)
    
    # 启动所有进程
    print(f"\n启动 {len(processes)} 个进程...")
    for p in processes:
        p.start()
    
    # 等待完成
    try:
        for p in processes:
            p.join(timeout=310)  # 5 分钟超时
            if p.is_alive():
                print(f"[警告] {p.name} 超时，终止...")
                p.terminate()
                p.join(timeout=5)
    except KeyboardInterrupt:
        print("\n[中断] 正在停止所有进程...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join(timeout=5)
    
    print("\n" + "=" * 60)
    print("所有问题运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
