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
    ax.plot(df1['gen'], df1['best_fit'], label='Island 1 (Scout)', alpha=0.8, linewidth=1)
    
    # 标记最终最佳
    best0 = df0['best_fit'].min()
    best1 = df1['best_fit'].min()
    overall_best = min(best0, best1)
    
    ax.axhline(y=overall_best, color='r', linestyle='--', label=f'Overall Best: {overall_best:.2f}', alpha=0.7)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness')
    ax.set_title('Convergence Comparison: Exploiter vs Scout')
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
    axes[0].plot(df0['gen'], df0['diversity'], label='Island 0 (Exploiter)', alpha=0.7)
    axes[0].plot(df1['gen'], df1['diversity'], label='Island 1 (Scout)', alpha=0.7)
    axes[0].set_ylabel('Avg Bond Distance')
    axes[0].set_title('Diversity Metrics Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Edge Entropy
    axes[1].plot(df0['gen'], df0['entropy'], label='Island 0 (Exploiter)', alpha=0.7)
    axes[1].plot(df1['gen'], df1['entropy'], label='Island 1 (Scout)', alpha=0.7)
    axes[1].set_ylabel('Edge Entropy')
    axes[1].set_xlabel('Generation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"[Saved] {output_file}")
    return fig

def plot_events(df0, df1, output_file='events.png'):
    """绘制事件时间线 (RTR接受率 + 迁移事件)
    
    新设计:
    - 子图1: RTR 接受率
    - 子图2: Scout 发送次数统计 (基于重启事件 -> Scout Strategy)
    - 子图3: Exploiter 接收成功次数 (migration=1)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # === RTR Accept Rate (滚动窗口) ===
    window = 50
    df0['rtr_rate'] = df0['rtr_accepts'].rolling(window).mean()
    df1['rtr_rate'] = df1['rtr_accepts'].rolling(window).mean()
    
    axes[0].plot(df0['gen'], df0['rtr_rate'], label='Exploiter (Accepting Scout Seeds)', alpha=0.7, color='blue')
    axes[0].plot(df1['gen'], df1['rtr_rate'], label='Scout (Internal Evolution)', alpha=0.7, color='green')
    axes[0].set_ylabel(f'RTR Accepts (rolling {window})')
    axes[0].set_title('Selection Pressure & Migration Events')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # === Scout 发送: 识别 Reset 事件 ===
    # Scout 每 50 代或更短时间重启一次。检测 stagnation 骤降。
    stag_diff = df1['stagnation'].diff()
    # Stagnation drop > 10 distincts a reset
    restart_events = df1[stag_diff < -10]
    
    if not restart_events.empty:
        axes[1].scatter(restart_events['gen'], [1]*len(restart_events), 
                       marker='^', s=100, c='green', edgecolors='darkgreen', 
                       linewidths=2, label=f'Scout Reset & Send Best ({len(restart_events)} times)', zorder=5)
    
    axes[1].axhline(y=1, color='green', alpha=0.3, linestyle='--')
    axes[1].set_ylabel('Scout Events')
    axes[1].set_ylim(0.5, 1.5)
    axes[1].set_yticks([1])
    axes[1].set_yticklabels(['Reset/Send'])
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # === Exploiter 接收成功 ===
    # migration 字段在 Exploiter 日志中标记为 1 表示有迁移并发生替换(直接替换或RTR胜出)
    imports = df0[df0['migration'] == 1]
    
    if not imports.empty:
        axes[2].scatter(imports['gen'], [1]*len(imports), 
                       marker='v', s=100, c='blue', edgecolors='darkblue',
                       linewidths=2, label=f'Exploiter Absorbed Scout Seed ({len(imports)} times)', zorder=5)
        
        # 尝试画垂直线连接 Scout 发送和 Exploiter 接收 (虽然时间轴上可能不完全对齐，但也很有趣)
        axes[2].vlines(imports['gen'], 0.5, 1.5, color='blue', alpha=0.2)

    axes[2].axhline(y=1, color='blue', alpha=0.3, linestyle='--')
    axes[2].set_ylabel('Exploiter Imports')
    axes[2].set_xlabel('Generation')
    axes[2].set_ylim(0.5, 1.5)
    axes[2].set_yticks([1])
    axes[2].set_yticklabels(['Imported'])
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"[Saved] {output_file}")
    return fig

def plot_stagnation(df0, df1, output_file='stagnation.png'):
    """绘制停滞计数曲线"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(df0['gen'], df0['stagnation'], label='Island 0 (Exploiter)', alpha=0.7)
    ax.plot(df1['gen'], df1['stagnation'], label='Island 1 (Scout)', alpha=0.7, color='green')
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Stagnation Counter')
    ax.set_title('Stagnation Counter (Sawtooth pattern expected for Scout)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"[Saved] {output_file}")
    return fig

def print_summary(df0, df1):
    """打印摘要统计"""
    print("\n" + "=" * 60)
    print("Scout-Exploiter Model Diagnosis")
    print("=" * 60)
    
    print(f"\n[Island 0 - Exploiter (The Master)]")
    print(f"  Generations: {df0['gen'].max()}")
    print(f"  Best Fitness: {df0['best_fit'].min():.4f}")
    print(f"  Scout Seeds Absorbed: {df0['migration'].sum()}")
    print(f"  Avg RTR Acceptance: {df0['rtr_accepts'].mean():.2f}")
    
    print(f"\n[Island 1 - Scout (The Scout)]")
    print(f"  Generations: {df1['gen'].max()}")
    print(f"  Best Fitness (Local): {df1['best_fit'].min():.4f}")
    # Scout 不接收迁移，所以 migration 通常为 0
    # 我们更关心它的重启次数
    stag_diff = df1['stagnation'].diff()
    restarts = (stag_diff < -10).sum()
    print(f"  Resets/Sends Triggered: {restarts}")
    print(f"  Avg RTR Acceptance: {df1['rtr_accepts'].mean():.2f}")
    
    overall_best = min(df0['best_fit'].min(), df1['best_fit'].min())
    print(f"\n[Overall Performance]")
    print(f"  Global Best Fitness: {overall_best:.4f}")
    print("=" * 60)

def main():
    # ==============================================================================
    # 用户配置区域：只需要填入日志文件夹路径
    # ==============================================================================
    
    LOG_FOLDER = "logs_20251217_224448"  # 填入日志文件夹名称
    
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
