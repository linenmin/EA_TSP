"""
TSP 诊断日志分析与可视化脚本
用于分析岛屿模型运行日志，生成收敛曲线、多样性变化等图表
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def load_logs(island_0_log, island_1_log):
    """加载两个岛屿的日志文件"""
    df0 = pd.read_csv(island_0_log)
    df1 = pd.read_csv(island_1_log)
    df0['island'] = 0
    df1['island'] = 1
    return df0, df1

def plot_convergence(df0, df1, output_file='convergence.png'):
    """绘制收敛曲线对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df0['gen'], df0['best_fit'], label='Island 0 (Exploiter)', alpha=0.8, linewidth=1)
    ax.plot(df1['gen'], df1['best_fit'], label='Island 1 (Explorer)', alpha=0.8, linewidth=1)
    
    # 标记最终最佳
    best0 = df0['best_fit'].min()
    best1 = df1['best_fit'].min()
    overall_best = min(best0, best1)
    
    ax.axhline(y=overall_best, color='r', linestyle='--', label=f'Overall Best: {overall_best:.2f}', alpha=0.7)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness')
    ax.set_title('Convergence Comparison: Exploiter vs Explorer')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"[Saved] {output_file}")
    return fig

def plot_diversity(df0, df1, output_file='diversity.png'):
    """绘制多样性变化图"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Bond Distance
    axes[0].plot(df0['gen'], df0['diversity'], label='Island 0', alpha=0.7)
    axes[0].plot(df1['gen'], df1['diversity'], label='Island 1', alpha=0.7)
    axes[0].set_ylabel('Avg Bond Distance')
    axes[0].set_title('Diversity Metrics Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Edge Entropy
    axes[1].plot(df0['gen'], df0['entropy'], label='Island 0', alpha=0.7)
    axes[1].plot(df1['gen'], df1['entropy'], label='Island 1', alpha=0.7)
    axes[1].set_ylabel('Edge Entropy')
    axes[1].set_xlabel('Generation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"[Saved] {output_file}")
    return fig

def plot_events(df0, df1, output_file='events.png'):
    """绘制事件时间线（迁移、排斥、RTR接受率）"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # RTR Accept Rate (滚动窗口)
    window = 50
    df0['rtr_rate'] = df0['rtr_accepts'].rolling(window).mean()
    df1['rtr_rate'] = df1['rtr_accepts'].rolling(window).mean()
    
    axes[0].plot(df0['gen'], df0['rtr_rate'], label='Island 0', alpha=0.7)
    axes[0].plot(df1['gen'], df1['rtr_rate'], label='Island 1', alpha=0.7)
    axes[0].set_ylabel(f'RTR Accepts (rolling {window})')
    axes[0].set_title('RTR Selection Pressure & Events')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Migration Events
    mig0 = df0[df0['migration'] == 1]
    mig1 = df1[df1['migration'] == 1]
    axes[1].scatter(mig0['gen'], [0.5]*len(mig0), marker='|', s=100, c='blue', label='Island 0 Import')
    axes[1].scatter(mig1['gen'], [1.5]*len(mig1), marker='|', s=100, c='orange', label='Island 1 Import')
    axes[1].set_ylabel('Migration Events')
    axes[1].set_yticks([0.5, 1.5])
    axes[1].set_yticklabels(['Island 0', 'Island 1'])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Repulsion Events (only Island 1)
    rep1 = df1[df1['repulsion'] == 1]
    axes[2].scatter(rep1['gen'], [1]*len(rep1), marker='x', s=80, c='red', label='Repulsion Triggered')
    axes[2].set_ylabel('Repulsion Events')
    axes[2].set_xlabel('Generation')
    axes[2].set_yticks([1])
    axes[2].set_yticklabels(['Island 1'])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"[Saved] {output_file}")
    return fig

def plot_stagnation(df0, df1, output_file='stagnation.png'):
    """绘制停滞计数曲线"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(df0['gen'], df0['stagnation'], label='Island 0 (Exploiter)', alpha=0.7)
    ax.plot(df1['gen'], df1['stagnation'], label='Island 1 (Explorer)', alpha=0.7)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Stagnation Counter')
    ax.set_title('Stagnation Counter Over Time (drops indicate restart)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"[Saved] {output_file}")
    return fig

def print_summary(df0, df1):
    """打印摘要统计"""
    print("\n" + "=" * 60)
    print("诊断摘要")
    print("=" * 60)
    
    print(f"\n[Island 0 - Exploiter]")
    print(f"  总代数: {df0['gen'].max()}")
    print(f"  最佳成绩: {df0['best_fit'].min():.4f}")
    print(f"  迁移次数: {df0['migration'].sum()}")
    print(f"  平均 RTR 接受数: {df0['rtr_accepts'].mean():.2f}")
    
    print(f"\n[Island 1 - Explorer]")
    print(f"  总代数: {df1['gen'].max()}")
    print(f"  最佳成绩: {df1['best_fit'].min():.4f}")
    print(f"  迁移次数: {df1['migration'].sum()}")
    print(f"  排斥次数: {df1['repulsion'].sum()}")
    print(f"  平均 RTR 接受数: {df1['rtr_accepts'].mean():.2f}")
    
    overall_best = min(df0['best_fit'].min(), df1['best_fit'].min())
    print(f"\n[Overall]")
    print(f"  最佳成绩: {overall_best:.4f}")
    print("=" * 60)

def main():
    # ==============================================================================
    # 用户配置区域：只需要填入日志文件夹路径
    # ==============================================================================
    
    LOG_FOLDER = "logs_20251217_115250"  # 填入日志文件夹名称
    
    # ==============================================================================
    
    if not os.path.exists(LOG_FOLDER):
        print(f"错误：文件夹 {LOG_FOLDER} 不存在！")
        return
    
    # 自动检测文件夹中的所有 CSV 文件
    csv_files = [f for f in os.listdir(LOG_FOLDER) if f.endswith('_log.csv')]
    
    if not csv_files:
        print(f"错误：文件夹 {LOG_FOLDER} 中没有找到 CSV 日志文件！")
        return
    
    # 提取问题名称 (例如: island_0_tour500_log.csv -> tour500)
    problem_names = set()
    for f in csv_files:
        # 格式: island_X_<problem>_log.csv
        parts = f.replace('_log.csv', '').split('_')
        if len(parts) >= 3:
            problem_name = '_'.join(parts[2:])  # 支持包含下划线的问题名
            problem_names.add(problem_name)
    
    print(f"检测到 {len(problem_names)} 个问题: {sorted(problem_names)}")
    print("=" * 60)
    
    # 为每个问题生成图表
    for problem in sorted(problem_names):
        log_0 = os.path.join(LOG_FOLDER, f"island_0_{problem}_log.csv")
        log_1 = os.path.join(LOG_FOLDER, f"island_1_{problem}_log.csv")
        
        if not os.path.exists(log_0) or not os.path.exists(log_1):
            print(f"[Skip] {problem}: 缺少日志文件")
            continue
        
        print(f"\n正在分析: {problem}")
        
        df0, df1 = load_logs(log_0, log_1)
        
        # 输出路径 (存入同一文件夹)
        prefix = os.path.join(LOG_FOLDER, f"{problem}_")
        
        # 生成图表
        plot_convergence(df0, df1, f"{prefix}convergence.png")
        plot_diversity(df0, df1, f"{prefix}diversity.png")
        plot_events(df0, df1, f"{prefix}events.png")
        plot_stagnation(df0, df1, f"{prefix}stagnation.png")
        
        # 打印摘要
        print_summary(df0, df1)
    
    print(f"\n所有图表已保存到: {LOG_FOLDER}/")

if __name__ == "__main__":
    main()
