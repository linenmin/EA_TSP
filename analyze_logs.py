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
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twiny()
    
    # Exploiter on Primary X (Bottom)
    l1, = ax1.plot(df0['gen'], df0['best_fit'], label='Island 0 (Exploiter)', color='blue', alpha=0.8, linewidth=1.5)
    
    # Scout on Secondary X (Top)
    l2, = ax2.plot(df1['gen'], df1['best_fit'], label='Island 1 (Trauma Center)', color='green', alpha=0.5, linewidth=1, linestyle='--')
    
    # 标记最终最佳
    best0 = df0['best_fit'].min()
    best1 = df1['best_fit'].min()
    overall_best = min(best0, best1)
    
    l3 = ax1.axhline(y=overall_best, color='r', linestyle=':', label=f'Overall Best: {overall_best:.2f}', alpha=0.7)
    
    ax1.set_xlabel('Exploiter Generation')
    ax2.set_xlabel('Scout Iteration (LNS Steps)')
    ax1.set_ylabel('Best Fitness')
    ax1.set_title('Convergence Comparison: Exploiter vs Scout (Dual X-Axis)')
    
    # Combined Legend
    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    # Focus on stable part (zoom in Y-axis)
    # Filter out the initial high values by setting max Y to 30% above best
    y_limit = overall_best * 1.05
    ax1.set_ylim(bottom=overall_best * 0.9999, top=y_limit)
    
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"[Saved] {output_file}")
    return fig

def plot_diversity(df0, df1, output_file='diversity.png'):
    """绘制多样性变化图 (Dual X-Axis)"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    
    # --- Subplot 1: Bond Distance ---
    ax1 = axes[0]
    ax1b = ax1.twiny()
    
    # Exploiter
    l1, = ax1.plot(df0['gen'], df0['diversity'], label='Exploiter Bond Dist', color='blue', alpha=0.7)
    # Scout
    l2, = ax1b.plot(df1['gen'], df1['diversity'], label='Scout Bond Dist', color='green', alpha=0.5, linestyle='--')
    
    ax1.set_ylabel('Avg Bond Distance')
    ax1.set_xlabel('Exploiter Gen')
    ax1b.set_xlabel('Scout Iter')
    ax1.set_title('Diversity Metrics (Bond Distance)')
    
    # Legend
    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # --- Subplot 2: Edge Entropy ---
    ax2 = axes[1]
    ax2b = ax2.twiny()
    
    # Exploiter
    l3, = ax2.plot(df0['gen'], df0['entropy'], label='Exploiter Entropy', color='purple', alpha=0.7)
    # Scout
    l4, = ax2b.plot(df1['gen'], df1['entropy'], label='Scout Entropy', color='orange', alpha=0.5, linestyle='--')
    
    ax2.set_ylabel('Edge Entropy')
    ax2.set_xlabel('Exploiter Gen')
    ax2b.set_xlabel('Scout Iter')
    
    lines2 = [l3, l4]
    ax2.legend(lines2, [l.get_label() for l in lines2], loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"[Saved] {output_file}")
    return fig

def plot_events(df0, df1, output_file='events.png'):
    """绘制事件时间线 (RTR接受率 + 迁移事件) - Independent X Axes"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
    
    # === Subplot 1: RTR Accept Rate (Dual Scale) ===
    ax1 = axes[0]
    ax1b = ax1.twiny()
    
    window = 50
    df0['rtr_rate'] = df0['rtr_accepts'].rolling(window).mean()
    df1['rtr_rate'] = df1['rtr_accepts'].rolling(window).mean()
    
    # Exploiter
    l1, = ax1.plot(df0['gen'], df0['rtr_rate'], label='Exploiter RTR Rate', alpha=0.7, color='blue')
    # Scout
    l2, = ax1b.plot(df1['gen'], df1['rtr_rate'], label='Scout RTR Rate', alpha=0.7, color='green', linestyle='--')
    
    ax1.set_ylabel(f'RTR Accepts (rolling {window})')
    ax1.set_title('Selection Pressure (Exploiter vs Scout)')
    ax1.set_xlabel('Exploiter Gen')
    ax1b.set_xlabel('Scout Iter')
    
    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    
    # === Subplot 2: Trauma Center Activity (Scout Scale) ===
    # Scout Admissions (Recv) & Discharges (Sent)
    # Uses df1 (Scout Scale)
    ax2 = axes[1]
    
    admissions = df1[df1['migration'] == 1]
    if not admissions.empty:
        ax2.scatter(admissions['gen'], [1]*len(admissions), 
                       marker='v', s=80, c='red', label=f'Trauma Admission (In) ({len(admissions)})')
    
    discharges = df1[df1['repulsion'] == 1]
    if not discharges.empty:
        ax2.scatter(discharges['gen'], [1]*len(discharges), 
                       marker='^', s=100, c='green', edgecolors='black', 
                       linewidths=2, label=f'Trauma Discharge (Out) ({len(discharges)})')
                       
    ax2.set_ylabel('Events')
    ax2.set_xlabel('Scout Iteration (LNS Steps)')
    ax2.set_title('Trauma Center Activity (Red=In, Green=Out)')
    ax2.set_yticks([])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # === Subplot 3: Exploiter Reception (Exploiter Scale) ===
    # Exploiter Received Healed
    # Uses df0 (Exploiter Scale)
    ax3 = axes[2]
    
    healed = df0[df0['migration'] == 1]
    if not healed.empty:
        ax3.scatter(healed['gen'], healed['best_fit'], 
                       marker='*', s=150, c='gold', edgecolors='black', 
                       label=f'Exploiter Received Healed ({len(healed)})', zorder=10)
    
    ax3.plot(df0['gen'], df0['best_fit'], label='Exploiter Fitness', alpha=0.5, color='blue')
    ax3.set_ylabel('Fitness')
    ax3.set_xlabel('Exploiter Generation')
    ax3.set_title('Exploiter Reception & Fitness Impact')
    
    # Zoom in Y-axis
    min_fit = df0['best_fit'].min()
    ax3.set_ylim(bottom=min_fit * 0.9999, top=min_fit * 1.05)
    
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    

    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"[Saved] {output_file}")
    return fig

def plot_stagnation(df0, df1, output_file='stagnation.png'):
    """绘制停滞计数曲线 (Dual X-Axis)"""
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twiny()
    
    # Exploiter
    l1, = ax1.plot(df0['gen'], df0['stagnation'], label='Exploiter Stagnation', alpha=0.8, color='blue')
    
    # Scout (Usually 0 for LNS)
    l2, = ax2.plot(df1['gen'], df1['stagnation'], label='Scout Stagnation', alpha=0.6, color='green', linestyle='--')
    
    ax1.set_xlabel('Exploiter Generation')
    ax2.set_xlabel('Scout Iteration')
    ax1.set_ylabel('Stagnation Counter')
    ax1.set_title('Stagnation Counter (Exploiter vs Scout)')
    
    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')
    ax1.grid(True, alpha=0.3)
    
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
    
    LOG_FOLDER = "logs_20251218_101252"  # 填入日志文件夹名称
    
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
