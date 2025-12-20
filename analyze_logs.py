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

def plot_convergence(df0, df1, output_file=None, problem_name=None, ax=None):
    """绘制收敛曲线对比图"""
    if ax is None:
        fig, ax1 = plt.subplots(figsize=(12, 6))
    else:
        ax1 = ax
        fig = ax.get_figure()

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
    title = 'Convergence Comparison: Exploiter vs Scout (Dual X-Axis)'  # 标题模板
    if problem_name:  # 追加问题名
        title = f'{problem_name} | {title}'  # 拼接标题
    ax1.set_title(title)
    
    # Combined Legend
    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    # Focus on stable part (zoom in Y-axis)
    # Filter out the initial high values by setting max Y to 30% above best
    y_limit = overall_best * 1.02
    ax1.set_ylim(bottom=overall_best * 0.9999, top=y_limit)
    
    ax1.grid(True, alpha=0.3)
    
    if output_file:
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"[Saved] {output_file}")
    return fig

def plot_diversity(df0, df1, output_file=None, problem_name=None, axes=None):
    """绘制多样性变化图 (Dual X-Axis)"""
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    else:
        fig = axes[0].get_figure()
    
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
    title1 = 'Diversity Metrics (Bond Distance)'  # 标题模板
    if problem_name:  # 追加问题名
        title1 = f'{problem_name} | {title1}'  # 拼接标题
    ax1.set_title(title1)
    
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
    
    if output_file:
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"[Saved] {output_file}")
    return fig

def plot_events(df0, df1, output_file=None, problem_name=None, axes=None):
    """绘制事件时间线 (RTR接受率 + 迁移事件) - Independent X Axes"""
    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
    else:
        fig = axes[0].get_figure()
    
    # === Subplot 1: RTR Accept Rate (Dual Scale) ===
    ax1 = axes[0]
    ax1b = ax1.twiny()
    
    window = 50
    # Use copy to avoid SettingWithCopyWarning
    df0_c = df0.copy()
    df1_c = df1.copy()
    df0_c['rtr_rate'] = df0_c['rtr_accepts'].rolling(window).mean()
    df1_c['rtr_rate'] = df1_c['rtr_accepts'].rolling(window).mean()
    
    # Exploiter
    l1, = ax1.plot(df0_c['gen'], df0_c['rtr_rate'], label='Exploiter RTR Rate', alpha=0.7, color='blue')
    # Scout
    l2, = ax1b.plot(df1_c['gen'], df1_c['rtr_rate'], label='Scout RTR Rate', alpha=0.7, color='green', linestyle='--')
    
    ax1.set_ylabel(f'RTR Accepts (rolling {window})')
    title1 = 'Selection Pressure (Exploiter vs Scout)'  # 标题模板
    if problem_name:  # 追加问题名
        title1 = f'{problem_name} | {title1}'  # 拼接标题
    ax1.set_title(title1)
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
    title2 = 'Trauma Center Activity (Red=In, Green=Out)'  # 标题模板
    if problem_name:  # 追加问题名
        title2 = f'{problem_name} | {title2}'  # 拼接标题
    ax2.set_title(title2)
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
    title3 = 'Exploiter Reception & Fitness Impact'  # 标题模板
    if problem_name:  # 追加问题名
        title3 = f'{problem_name} | {title3}'  # 拼接标题
    ax3.set_title(title3)
    
    # Zoom in Y-axis
    min_fit = df0['best_fit'].min()
    ax3.set_ylim(bottom=min_fit * 0.9999, top=min_fit * 1.05)
    
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    
    if output_file:
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"[Saved] {output_file}")
    return fig

def plot_stagnation(df0, df1, output_file=None, problem_name=None, ax=None):
    """绘制停滞计数曲线 (Dual X-Axis)"""
    if ax is None:
        fig, ax1 = plt.subplots(figsize=(12, 5))
    else:
        ax1 = ax
        fig = ax.get_figure()

    ax2 = ax1.twiny()
    
    # Exploiter
    l1, = ax1.plot(df0['gen'], df0['stagnation'], label='Exploiter Stagnation', alpha=0.8, color='blue')
    
    # Scout (Usually 0 for LNS)
    l2, = ax2.plot(df1['gen'], df1['stagnation'], label='Scout Stagnation', alpha=0.6, color='green', linestyle='--')
    
    ax1.set_xlabel('Exploiter Generation')
    ax2.set_xlabel('Scout Iteration')
    ax1.set_ylabel('Stagnation Counter')
    title = 'Stagnation Counter (Exploiter vs Scout)'  # 标题模板
    if problem_name:  # 追加问题名
        title = f'{problem_name} | {title}'  # 拼接标题
    ax1.set_title(title)
    
    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    if output_file:
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"[Saved] {output_file}")
    return fig


def _compute_stagnation_lengths(stag_series):  # 停滞段
    """计算连续停滞段长度"""
    lengths = []  # 结果列表
    current = 0  # 当前长度
    for v in stag_series:  # 遍历序列
        if v <= 0:  # 停滞重置
            if current > 0:  # 有段落
                lengths.append(current)  # 记录
                current = 0  # 清零
        else:  # 停滞继续
            current += 1  # 累加
    if current > 0:  # 收尾
        lengths.append(current)  # 记录
    return lengths  # 返回结果


def _compute_migration_gain(df, window):  # 迁移收益
    """计算迁移后改进幅度"""
    gains = []  # 改进列表
    positions = df.index[df['migration'] == 1].to_numpy()  # 迁移位置
    if positions.size == 0:  # 无事件
        return gains  # 直接返回
    for pos in positions:  # 遍历事件
        base = float(df.at[pos, 'best_fit'])  # 事件时基准
        end = min(pos + window, len(df) - 1)  # 窗口终点
        after_min = float(df.loc[pos:end, 'best_fit'].min())  # 窗口最优
        gains.append(base - after_min)  # 改进幅度
    return gains  # 返回结果


def _compute_spike_ratio(series, quantile=0.9):  # 尖峰比例
    """计算尖峰比例"""
    values = series[series > 0]  # 过滤无效
    if len(values) == 0:  # 无有效值
        return 0.0, 0.0, 0, 0  # 返回空值
    threshold = float(values.quantile(quantile))  # 尖峰阈值
    spike_count = int((values >= threshold).sum())  # 尖峰数量
    total_count = int(values.shape[0])  # 总数
    ratio = spike_count / total_count  # 比例
    return ratio, threshold, spike_count, total_count  # 返回结果


def plot_diagnostics(df0, df1, output_file=None, problem_name=None, axes=None):  # 诊断图
    """绘制额外诊断指标"""
    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)  # 画布
    else:
        fig = axes[0].get_figure()

    window = 50  # 平滑窗口
    
    best_delta = df0['best_fit'].diff()  # 最优差分
    delta_roll = best_delta.rolling(window).mean()  # 差分平滑
    improve_rate = (best_delta < 0).rolling(window).mean()  # 改进率
    gap = df0['mean_fit'] - df0['best_fit']  # 群体差距
    
    ax1 = axes[0]  # 子图1
    ax1.plot(df0['gen'], -delta_roll, color='blue', alpha=0.8)  # 改进幅度
    title1 = 'Improvement Magnitude (Rolling)'  # 标题模板
    if problem_name:  # 追加问题名
        title1 = f'{problem_name} | {title1}'  # 拼接标题
    ax1.set_title(title1)  # 标题
    ax1.set_ylabel('Avg Improvement')  # 纵轴
    ax1.set_xlabel('Exploiter Gen')  # 横轴
    ax1.grid(True, alpha=0.3)  # 网格
    
    ax2 = axes[1]  # 子图2
    ax2.plot(df0['gen'], gap, color='purple', alpha=0.8)  # 群体差距
    title2 = 'Population Gap (Mean - Best)'  # 标题模板
    if problem_name:  # 追加问题名
        title2 = f'{problem_name} | {title2}'  # 拼接标题
    ax2.set_title(title2)  # 标题
    ax2.set_ylabel('Gap')  # 纵轴
    ax2.set_xlabel('Exploiter Gen')  # 横轴
    ax2.grid(True, alpha=0.3)  # 网格
    
    ax3 = axes[2]  # 子图3
    ax3.plot(df0['gen'], improve_rate, color='green', alpha=0.8)  # 改进频率
    title3 = 'Improvement Rate (Rolling)'  # 标题模板
    if problem_name:  # 追加问题名
        title3 = f'{problem_name} | {title3}'  # 拼接标题
    ax3.set_title(title3)  # 标题
    ax3.set_ylabel('Rate')  # 纵轴
    ax3.set_xlabel('Exploiter Gen')  # 横轴
    ax3.set_ylim(0, 1)  # 固定范围
    ax3.grid(True, alpha=0.3)  # 网格
    
    if output_file:
        plt.tight_layout()  # 紧凑布局
        plt.savefig(output_file, dpi=150)  # 保存图片
        print(f"[Saved] {output_file}")  # 输出提示
    return fig  # 返回图像


def plot_combined(df0, df1, output_file, problem_name=None):
    """合并四个核心诊断图为一张大长图"""
    # 核心诊断图共有 1 (convergence) + 3 (diagnostics) + 2 (diversity) + 3 (events) = 9 个子图
    fig = plt.figure(figsize=(15, 45))
    gs = fig.add_gridspec(9, 1, height_ratios=[1.5, 1, 1, 1, 1, 1, 1, 0.8, 1.5], hspace=0.4)
    
    # 预创建 axes
    ax_conv = fig.add_subplot(gs[0])
    axes_diag = [fig.add_subplot(gs[1]), fig.add_subplot(gs[2]), fig.add_subplot(gs[3])]
    axes_div = [fig.add_subplot(gs[4]), fig.add_subplot(gs[5])]
    axes_events = [fig.add_subplot(gs[6]), fig.add_subplot(gs[7]), fig.add_subplot(gs[8])]
    
    # 调用原有的绘图逻辑 (不保存到文件)
    plot_convergence(df0, df1, problem_name=problem_name, ax=ax_conv)
    plot_diagnostics(df0, df1, problem_name=problem_name, axes=axes_diag)
    plot_diversity(df0, df1, problem_name=problem_name, axes=axes_div)
    plot_events(df0, df1, problem_name=problem_name, axes=axes_events)
    
    # 全局标题
    title = "TSP Island Model Full Diagnostic Report"
    if problem_name:
        title = f"{problem_name} | {title}"
    fig.suptitle(title, fontsize=24, y=0.99)
    
    # 显式调整间距
    plt.savefig(output_file, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[Saved Combined Report] {output_file}")


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
    
    best_delta = df0['best_fit'].diff()  # 最优差分
    improve_mask = best_delta < 0  # 改进标记
    improve_ratio = float(improve_mask.mean())  # 改进比例
    improve_vals = (-best_delta[improve_mask])  # 改进幅度
    improve_mean = float(improve_vals.mean()) if len(improve_vals) > 0 else 0.0  # 平均改进
    gap = df0['mean_fit'] - df0['best_fit']  # 群体差距
    gap_mean = float(gap.mean())  # 平均差距
    gap_last = float(gap.iloc[-1])  # 末尾差距
    
    tail_start = int(len(df0) * 0.6)  # 尾段起点
    tail_ratio = float((best_delta.iloc[tail_start:] < 0).mean()) if len(df0) > 1 else 0.0  # 尾段改进率
    
    stag_lengths = _compute_stagnation_lengths(df0['stagnation'])  # 停滞段
    stag_mean = float(np.mean(stag_lengths)) if len(stag_lengths) > 0 else 0.0  # 平均长度
    stag_max = int(np.max(stag_lengths)) if len(stag_lengths) > 0 else 0  # 最大长度
    
    div_ratio, div_thr, div_cnt, div_tot = _compute_spike_ratio(df0['diversity'])  # 多样性尖峰
    ent_ratio, ent_thr, ent_cnt, ent_tot = _compute_spike_ratio(df0['entropy'])  # 熵尖峰
    
    mig_gain_10 = _compute_migration_gain(df0, 10)  # 迁移收益10
    mig_gain_20 = _compute_migration_gain(df0, 20)  # 迁移收益20
    mig_mean_10 = float(np.mean(mig_gain_10)) if len(mig_gain_10) > 0 else 0.0  # 平均收益10
    mig_mean_20 = float(np.mean(mig_gain_20)) if len(mig_gain_20) > 0 else 0.0  # 平均收益20
    mig_pos_10 = float(np.mean(np.array(mig_gain_10) > 0)) if len(mig_gain_10) > 0 else 0.0  # 正收益率10
    mig_pos_20 = float(np.mean(np.array(mig_gain_20) > 0)) if len(mig_gain_20) > 0 else 0.0  # 正收益率20
    
    rtr_roll = df0['rtr_accepts'].rolling(50).mean().dropna()  # RTR平滑
    if len(rtr_roll) > 0:  # 有效数据
        head_len = max(1, int(len(rtr_roll) * 0.33))  # 前段长度
        tail_len = max(1, int(len(rtr_roll) * 0.33))  # 后段长度
        rtr_head = float(rtr_roll.iloc[:head_len].mean())  # 前段均值
        rtr_tail = float(rtr_roll.iloc[-tail_len:].mean())  # 后段均值
    else:  # 无数据
        rtr_head = 0.0  # 前段默认
        rtr_tail = 0.0  # 后段默认
    
    print(f"  Improve Rate: {improve_ratio:.2%} | Avg Gain: {improve_mean:.2f}")  # 改进统计
    print(f"  Gap Mean/Last: {gap_mean:.2f} / {gap_last:.2f}")  # 群体差距
    print(f"  Tail Improve Rate (last 40%): {tail_ratio:.2%}")  # 尾段改进
    print(f"  Stag Episode Len (mean/max): {stag_mean:.1f} / {stag_max}")  # 停滞段
    print(f"  Diversity Spikes: {div_cnt}/{div_tot} ({div_ratio:.1%}) Thr={div_thr:.1f}")  # 多样性尖峰
    print(f"  Entropy Spikes: {ent_cnt}/{ent_tot} ({ent_ratio:.1%}) Thr={ent_thr:.1f}")  # 熵尖峰
    print(f"  Migration Gain(10/20): {mig_mean_10:.2f}/{mig_mean_20:.2f} | PosRate {mig_pos_10:.1%}/{mig_pos_20:.1%}")  # 迁移收益
    print(f"  RTR Trend (head->tail): {rtr_head:.2f} -> {rtr_tail:.2f}")  # 选择压力
    
    print(f"\n[Island 1 - Scout (The Scout)]")
    print(f"  Generations: {df1['gen'].max()}")
    print(f"  Best Fitness (Local): {df1['best_fit'].min():.4f}")
    # Scout 不接收迁移，所以 migration 通常为 0
    # 我们更关心它的重启次数
    stag_diff = df1['stagnation'].diff()
    restarts = (stag_diff < -10).sum()
    print(f"  Resets/Sends Triggered: {restarts}")
    print(f"  Avg RTR Acceptance: {df1['rtr_accepts'].mean():.2f}")
    print(f"  Admissions/Discharges: {df1['migration'].sum()} / {df1['repulsion'].sum()}")  # 收治与出院
    
    overall_best = min(df0['best_fit'].min(), df1['best_fit'].min())
    print(f"\n[Overall Performance]")
    print(f"  Global Best Fitness: {overall_best:.4f}")
    print("=" * 60)

def main():
    # ==============================================================================
    # 用户配置区域：只需要填入日志文件夹路径
    # ==============================================================================
    
    LOG_FOLDER = "logs_20251220_093101"  # 填入日志文件夹名称
    
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
        plot_convergence(df0, df1, f"{prefix}convergence.png", problem)  # 收敛图
        plot_diversity(df0, df1, f"{prefix}diversity.png", problem)  # 多样性图
        plot_events(df0, df1, f"{prefix}events.png", problem)  # 事件图
        plot_stagnation(df0, df1, f"{prefix}stagnation.png", problem)  # 停滞图
        plot_diagnostics(df0, df1, f"{prefix}diagnostics.png", problem)  # 诊断图
        
        # 合并图表
        plot_combined(df0, df1, f"{prefix}combined_report.png", problem)
        
        # 打印摘要
        print_summary(df0, df1)
    
    print(f"\n所有图表已保存到: {LOG_FOLDER}/")

if __name__ == "__main__":
    main()
