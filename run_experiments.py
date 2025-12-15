
import os
import time
# 导入优化算法类
from optimized_thread_LocalSearch_inf import r0123456

# 定义实验配置
# 格式: {文件名: {参数字典}}
experiments = [
    {
        "filename": "tour50.csv",
        "params": {
            "N_RUNS": 10_000_000,
            "lam": 20000,
            "mu": 18000,
            "mutation_rate": 0.3,
            "k_tournament": 30,
            "local_rate": 0.2,
            "ls_max_steps": 30,
            "stagnation_limit": 8
        }
    },
    {
        "filename": "tour250.csv",
        "params": {
            "N_RUNS": 10_000_000,
            "lam": 6000,
            "mu": 4000,
            "mutation_rate": 0.3,
            "k_tournament": 30,
            "local_rate": 0.2,
            "ls_max_steps": 30,
            "stagnation_limit": 8
        }
    },
    {
        "filename": "tour500.csv",
        "params": {
            "N_RUNS": 10_000_000,
            "lam": 20000,
            "mu": 15000,
            "mutation_rate": 0.3,
            "k_tournament": 30,
            "local_rate": 0.2,
            "ls_max_steps": 30,
            "stagnation_limit": 8
        }
    },
    {
        "filename": "tour750.csv",
        "params": {
            "N_RUNS": 10_000_000,
            "lam": 2000,
            "mu": 1500,
            "mutation_rate": 0.3,  # 默认值 (用户未指定，参考其他组)
            "k_tournament": 30,
            "local_rate": 0.2,
            "ls_max_steps": 30,
            "stagnation_limit": 8  # 默认值
        }
    },
    {
        "filename": "tour1000.csv",
        "params": {
            "N_RUNS": 10_000_000,
            "lam": 200,
            "mu": 150,
            "mutation_rate": 0.7,  # 特殊配置
            "k_tournament": 30,
            "local_rate": 0.2,
            "ls_max_steps": 30,
            "stagnation_limit": 8
        }
    }
]

def run_all_experiments():
    print("==================================================")
    print("Starting Batch Experiments...")
    print("==================================================")

    for i, exp in enumerate(experiments):
        filename = exp["filename"]
        params = exp["params"]
        
        print(f"\n[{i+1}/{len(experiments)}] Running experiment for: {filename}")
        print(f"Parameters: {params}")
        
        # 检查文件是否存在
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' not found. Skipping...")
            continue
            
        start_time = time.time()
        
        # 实例化求解器
        solver = r0123456(
            N_RUNS=params["N_RUNS"],
            lam=params["lam"],
            mu=params["mu"],
            k_tournament=params["k_tournament"],
            mutation_rate=params["mutation_rate"],
            local_rate=params["local_rate"],
            ls_max_steps=params["ls_max_steps"],
            stagnation_limit=params["stagnation_limit"]
        )
        
        # 运行优化
        try:
            solver.optimize(filename)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Moving to next experiment...")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            
        duration = time.time() - start_time
        print(f"Experiment for {filename} finished in {duration:.2f} seconds.")

    print("\n==================================================")
    print("All experiments completed.")
    print("==================================================")

if __name__ == "__main__":
    run_all_experiments()
