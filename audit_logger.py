"""
审计日志模块：事件驱动诊断系统

统一的日志格式：[TAG] event_name | field1=value1 | field2=value2 | ...

Tags:
    [CHK]   - 标尺一致性审计（D vs D_ls）
    [BEST]  - 最优更新审计
    [RTR]   - RTR 接纳行为画像
    [POP]   - 多样性分位数
    [XOV]   - 交叉贡献审计
    [LS]    - 局部搜索收益画像
    [LKH]   - LKH 差异拓扑趋势
    [GLS]   - GLS 状态审计
    [SCOUT] - Scout 贡献审计
    [RST]   - 重启审计
    [TIME]  - 时间预算画像
"""

import os
from datetime import datetime
import numpy as np

# 延迟导入 tour_length_jit（避免循环导入）
def _get_tour_length_jit():
    from r0927480 import tour_length_jit
    return tour_length_jit

def tour_length_jit(tour, D):
    """包装函数，延迟导入"""
    return _get_tour_length_jit()(tour, D)

class AuditLogger:
    """事件驱动审计日志器"""
    
    def __init__(self, csv_filename: str):
        """
        初始化日志器
        
        Args:
            csv_filename: 输入的 CSV 文件名（如 'tour250.csv'）
        """
        self.csv_basename = os.path.splitext(os.path.basename(csv_filename))[0]
        self.start_time = datetime.now()
        
        # 创建 logs 目录
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 生成日志文件名
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.log_filename = os.path.join(self.log_dir, f"audit_{timestamp}_{self.csv_basename}.txt")
        
        # 打开日志文件
        self.file = open(self.log_filename, 'w', encoding='utf-8')
        self._log(f"[INFO] audit_start | csv={self.csv_basename} | time={self.start_time.isoformat()}")
        
        # 统计变量（用于周期性统计）
        self.rtr_replaced_count = 0
        self.rtr_total_count = 0
        self.rtr_delta_sum = 0.0
        self.rtr_deltas = []  # 记录每次替换的 delta
        self.ls_count = 0
        self.ls_gains = []
        
        # P1 新增：跟踪上一次 best 的 LKH 对比信息（用于 delta_decomposition）
        self.prev_shared_with_lkh = 0
        self.prev_missing_count = 0
        self.prev_extra_count = 0
        
        # P3 新增：管道审计统计
        self.pipe_audit_samples = []  # 每 50 代收集的 child 审计数据
        
        # 新增：RTR target 质量统计
        self.rtr_target_fits = []  # 被挑战的 target 的 fitness
        self.rtr_replaced_target_fits = []  # 被替换的 target 的 fitness
        self.rtr_rejected_target_fits = []  # 被拒收时 target 的 fitness
        
        # 新增：父母池统计
        self.parent_fits = []  # 本代被选作父母的 fitness
        
        # 新增：HGreX 分层统计
        self.hgrex_parent_edge = 0  # 选自父代边的次数
        self.hgrex_knn_fallback = 0  # KNN 补漏次数
        self.hgrex_random_fallback = 0  # 随机探针次数
        self.hgrex_fullscan_fallback = 0  # 全图扫描次数
        self.hgrex_total_steps = 0  # 总步数
        
        # 新增：offspring 流水线统计
        self.pipe_generated = {'hgrex': 0, 'ox': 0, 'mutate': 0}
        self.pipe_feasible = 0  # 通过可行性检查
        self.pipe_tamed = 0  # 进入 boot camp
        self.pipe_submitted = 0  # 提交 RTR
        self.pipe_accepted = 0  # RTR 接收
        
        # 时间统计
        self.time_xov = 0.0
        self.time_ls = 0.0
        self.time_eval = 0.0
        self.time_scout = 0.0
        self.time_last_report = self.start_time
        
        print(f"📝 审计日志: {self.log_filename}")
    
    def _log(self, msg: str):
        """内部写日志"""
        print(msg)
        self.file.write(msg + "\n")
        self.file.flush()
    
    def close(self):
        """关闭日志文件"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self._log(f"[INFO] audit_end | elapsed={elapsed:.1f}s")
        self.file.close()
    
    # =========================================================================
    # P1: 标尺一致性审计 [CHK]
    # =========================================================================
    
    def chk_objective_audit(self, gen: int, gls_active: bool, 
                            tour, D, D_ls, fitness_array):
        """
        标尺一致性审计：检查 D 与 D_ls 是否混用
        
        Args:
            tour: 用于 spot-check 的 tour
            D: 真实距离矩阵
            D_ls: GLS 惩罚矩阵（可能 == D）
            fitness_array: 种群 fitness 数组
        """
        from r0927480 import tour_length_jit
        
        len_D = tour_length_jit(tour, D)
        len_Dls = tour_length_jit(tour, D_ls) if D_ls is not D else len_D
        
        # Spot check：取前3个个体检查
        fit_min = fitness_array.min()
        
        # 判断是否一致
        is_consistent = True
        if gls_active and D_ls is not D:
            # 如果 fitness 最小值 ≈ len_Dls 而不是 len_D，说明目标被污染
            # 这里简化检查
            pass
        
        self._log(f"[CHK] objective_audit | gen={gen} | gls={gls_active} | "
                  f"tour_D={len_D:.2f} | tour_Dls={len_Dls:.2f} | fit_min={fit_min:.2f}")
    
    # =========================================================================
    # P2: 最优更新审计 [BEST]
    # =========================================================================
    
    def best_update_event(self, gen: int, source: str, 
                          old_len: float, new_len: float, tour,
                          lkh_tour=None, D=None):
        """
        最优更新事件审计 + delta_decomposition
        
        输出改进是否真正"往 LKH 靠近"
        """
        delta = old_len - new_len
        tour_hash = self._tour_hash(tour)
        
        # LKH 对比
        shared_count = 0
        missing_count = len(tour)
        extra_count = len(tour)
        sim_lkh = 0.0
        bond = len(tour)
        
        if lkh_tour is not None:
            from diagnose_gap import edge_similarity, bond_distance, get_edges_set
            shared_count, sim_lkh = edge_similarity(tour, lkh_tour)
            bond = bond_distance(tour, lkh_tour)
            
            # 计算 missing 和 extra
            _, my_edges = get_edges_set(tour)
            _, lkh_edges = get_edges_set(lkh_tour)
            missing_count = len(lkh_edges - my_edges)
            extra_count = len(my_edges - lkh_edges)
        
        # 计算与上一次 best 的差值（delta_decomposition）
        d_shared = shared_count - self.prev_shared_with_lkh
        d_missing = missing_count - self.prev_missing_count
        d_extra = extra_count - self.prev_extra_count
        
        # 更新跟踪变量
        self.prev_shared_with_lkh = shared_count
        self.prev_missing_count = missing_count
        self.prev_extra_count = extra_count
        
        self._log(f"[BEST] update | gen={gen} | src={source} | "
                  f"old={old_len:.2f} | new={new_len:.2f} | delta={delta:.2f} | "
                  f"sim_lkh={sim_lkh:.1%} | bond={bond} | hash={tour_hash}")
        
        # P1 核心：delta_decomposition
        self._log(f"[BEST] delta_decomposition | gen={gen} | "
                  f"Δshared={d_shared:+d} | Δmissing={d_missing:+d} | Δextra={d_extra:+d} | "
                  f"shared={shared_count} | missing={missing_count} | extra={extra_count}")
        
        # 顺便触发 P6 拓扑分析
        if lkh_tour is not None and D is not None:
            self.lkh_missing_topology(gen, tour, lkh_tour, D)
    
    def _tour_hash(self, tour) -> str:
        """生成 tour 的简短 hash（用于追踪）"""
        if len(tour) >= 20:
            key = tuple(tour[:10]) + tuple(tour[-10:])
        else:
            key = tuple(tour)
        return f"{hash(key) & 0xFFFFFFFF:08x}"
    
    # =========================================================================
    # P3: RTR 接纳行为画像 [RTR]
    # =========================================================================
    
    def rtr_record(self, replaced: bool, delta: float, child_len: float = None, target_len: float = None, target_idx: int = None):
        """记录单次 RTR 结果（增强版：支持采样审计）"""
        self.rtr_total_count += 1
        if replaced:
            self.rtr_replaced_count += 1
            self.rtr_delta_sum += delta
            self.rtr_deltas.append(delta)
        
        # RTR-SAMPLE：记录单次决策详情
        if child_len is not None and target_len is not None:
            if not hasattr(self, 'rtr_samples'):
                self.rtr_samples = []
            self.rtr_samples.append({
                'child_len': child_len,
                'target_len': target_len,
                'target_idx': target_idx,
                'replaced': replaced,
                'should_replace': child_len < target_len  # 理论上应该替换？
            })
    
    def rtr_acceptance_profile(self, gen: int, lam: int):
        """
        输出增强版 RTR 门控报告 + RTR-SAMPLE
        """
        if self.rtr_total_count == 0:
            return
        
        rate = self.rtr_replaced_count / self.rtr_total_count
        avg_delta = self.rtr_delta_sum / max(1, self.rtr_replaced_count)
        
        # 计算 delta 分位数（过滤 inf/nan）
        delta_p10, delta_p50, delta_p90 = 0.0, 0.0, 0.0
        if self.rtr_deltas:
            valid_deltas = [d for d in self.rtr_deltas if np.isfinite(d)]
            if valid_deltas:
                deltas = np.array(valid_deltas)
                delta_p10 = np.percentile(deltas, 10)
                delta_p50 = np.percentile(deltas, 50)
                delta_p90 = np.percentile(deltas, 90)
        
        self._log(f"[RTR] gate_report | gen={gen} | children={self.rtr_total_count} | "
                  f"replaced={self.rtr_replaced_count} | rate={rate:.1%} | "
                  f"delta_P10={delta_p10:.2f} | delta_P50={delta_p50:.2f} | delta_P90={delta_p90:.2f}")
        
        # RTR-SAMPLE：输出采样详情（最多 3 个）
        if hasattr(self, 'rtr_samples') and self.rtr_samples:
            samples = self.rtr_samples[-3:]  # 取最后 3 个
            for i, s in enumerate(samples):
                mismatch = "⚠️MISMATCH" if s['replaced'] != s['should_replace'] else ""
                self._log(f"[RTR-SAMPLE] #{i+1} | child={s['child_len']:.2f} | target={s['target_len']:.2f} | "
                          f"child<target={s['should_replace']} | replaced={s['replaced']} {mismatch}")
        
        # 重置计数器
        self.rtr_total_count = 0
        self.rtr_replaced_count = 0
        self.rtr_delta_sum = 0.0
        self.rtr_deltas = []
        self.rtr_samples = []
    
    # =========================================================================
    # P4: 多样性分位数 [POP]
    # =========================================================================
    
    def pop_diversity_quantiles(self, gen: int, population, best_tour):
        """
        种群多样性分位数
        
        计算所有个体与 best_tour 的 bond distance 的 P10/P50/P90
        """
        from diagnose_gap import bond_distance
        
        n = len(best_tour)
        lam = len(population)
        
        # 采样计算 bond distance（全算太慢，采样 min(lam, 30) 个）
        sample_size = min(lam, 30)
        indices = np.random.choice(lam, sample_size, replace=False)
        
        bonds = []
        for idx in indices:
            bd = bond_distance(population[idx], best_tour)
            bonds.append(bd)
        
        bonds = np.array(bonds)
        p10 = int(np.percentile(bonds, 10))
        p50 = int(np.percentile(bonds, 50))
        p90 = int(np.percentile(bonds, 90))
        
        # 统计 distinct
        seen = set()
        for tour in population:
            key = tuple(tour[:10]) + tuple(tour[-10:]) if len(tour) >= 20 else tuple(tour)
            seen.add(key)
        distinct = len(seen)
        
        self._log(f"[POP] diversity | gen={gen} | bond_P10={p10} | bond_P50={p50} | "
                  f"bond_P90={p90} | distinct={distinct}/{lam}")
    
    def pop_quality_profile(self, gen: int, fitness_array, best_fitness: float):
        """
        种群质量画像：检查 median 是否被垃圾解拖垮
        """
        # 过滤有效值
        valid_fitness = [f for f in fitness_array if np.isfinite(f)]
        invalid_count = len(fitness_array) - len(valid_fitness)
        
        if not valid_fitness:
            self._log(f"[POP] quality_profile | gen={gen} | ALL_INVALID")
            return
        
        fits = np.array(valid_fitness)
        fit_min = fits.min()
        fit_median = np.median(fits)
        fit_max = fits.max()
        best_gap_to_median = fit_median - best_fitness
        
        self._log(f"[POP] quality_profile | gen={gen} | "
                  f"min={fit_min:.2f} | median={fit_median:.2f} | max={fit_max:.2f} | "
                  f"invalid={invalid_count} | best_gap_to_median={best_gap_to_median:.2f}")
    
    # =========================================================================
    # P5: LS 收益画像 [LS]
    # =========================================================================
    
    def ls_record(self, gain: float, before_len: float = None, after_len: float = None, 
                  passes: int = None, improvements: int = None):
        """记录单次 LS 收益（增强版：检测非法值 + VND 步数）"""
        self.ls_count += 1
        
        # 检测非法值
        if not np.isfinite(gain):
            self.ls_invalid_count = getattr(self, 'ls_invalid_count', 0) + 1
        else:
            self.ls_gains.append(gain)
        
        # 存储采样数据（用于 LS-SAMPLE + LS-STEP）
        if before_len is not None and after_len is not None:
            if not hasattr(self, 'ls_samples'):
                self.ls_samples = []
            self.ls_samples.append({
                'before': before_len,
                'after': after_len,
                'gain': gain,
                'before_finite': np.isfinite(before_len),
                'after_finite': np.isfinite(after_len),
                'passes': passes if passes is not None else 0,
                'improvements': improvements if improvements is not None else 0
            })
    
    def ls_gain_profile(self, gen: int):
        """输出 LS 收益画像（修复版：过滤 inf + LS-STEP）"""
        if self.ls_count == 0:
            return
        
        # 过滤非 finite 值
        valid_gains = [g for g in self.ls_gains if np.isfinite(g)]
        invalid_count = getattr(self, 'ls_invalid_count', 0)
        
        if valid_gains:
            gains = np.array(valid_gains)
            avg_gain = gains.mean()
            p90_gain = np.percentile(gains, 90)
        else:
            avg_gain = 0.0
            p90_gain = 0.0
        
        self._log(f"[LS] gain_profile | gen={gen} | count={self.ls_count} | "
                  f"valid={len(valid_gains)} | invalid={invalid_count} | "
                  f"avg={avg_gain:.2f} | p90={p90_gain:.2f}")
        
        # LS-STEP：输出 VND 步数统计（最多 3 个）
        if hasattr(self, 'ls_samples') and self.ls_samples:
            samples = self.ls_samples[-3:]  # 取最后 3 个
            for i, s in enumerate(samples):
                passes = s.get('passes', 0)
                imps = s.get('improvements', 0)
                self._log(f"[LS-STEP] #{i+1} | before={s['before']:.2f} | after={s['after']:.2f} | "
                          f"delta={s['gain']:+.2f} | passes={passes} | improvements={imps}")
        
        # 重置
        self.ls_count = 0
        self.ls_gains = []
        self.ls_invalid_count = 0
        self.ls_samples = []
    
    # =========================================================================
    # P6: LKH 差异拓扑趋势 [LKH]
    # =========================================================================
    
    def lkh_missing_topology(self, gen: int, my_tour, lkh_tour, D):
        """
        分析与 LKH 的差异拓扑
        """
        from diagnose_gap import get_edges_set
        
        n = len(my_tour)
        _, my_edges = get_edges_set(my_tour)
        _, lkh_edges = get_edges_set(lkh_tour)
        
        missing = lkh_edges - my_edges
        extra = my_edges - lkh_edges
        missing_count = len(missing)
        extra_count = len(extra)
        
        # 分析 missing 的拓扑结构
        chains, cycles, complex_knots = self._analyze_topology(list(missing))
        
        self._log(f"[LKH] topology | gen={gen} | missing={missing_count} | "
                  f"chains={chains} | cycles={cycles} | complex={complex_knots} | extra={extra_count}")
    
    def _analyze_topology(self, missing_edges):
        """分析缺失边的拓扑结构"""
        if not missing_edges:
            return 0, 0, 0
        
        # 构建邻接表
        adj = {}
        nodes = set()
        for edge in missing_edges:
            edge_list = list(edge)
            if len(edge_list) < 2:
                continue
            u, v = edge_list[0], edge_list[1]
            if u not in adj: adj[u] = []
            if v not in adj: adj[v] = []
            adj[u].append(v)
            adj[v].append(u)
            nodes.add(u)
            nodes.add(v)
        
        # 找连通分量
        visited = set()
        chains, cycles, complex_knots = 0, 0, 0
        
        for node in nodes:
            if node in visited:
                continue
            
            # BFS
            component_nodes = []
            stack = [node]
            visited.add(node)
            while stack:
                curr = stack.pop()
                component_nodes.append(curr)
                for neighbor in adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            
            # 计算边数
            comp_edges = sum(len(adj.get(u, [])) for u in component_nodes) // 2
            size = len(component_nodes)
            
            if size == comp_edges:
                cycles += 1
            elif size == comp_edges + 1:
                chains += 1
            else:
                complex_knots += 1
        
        return chains, cycles, complex_knots
    
    # =========================================================================
    # P8: 时间预算画像 [TIME]
    # =========================================================================
    
    def time_stage_budget(self, gen: int):
        """输出时间预算画像"""
        total = self.time_xov + self.time_ls + self.time_eval + self.time_scout
        if total < 0.001:
            return
        
        self._log(f"[TIME] budget | gen={gen} | "
                  f"xov={self.time_xov/total:.1%} | ls={self.time_ls/total:.1%} | "
                  f"eval={self.time_eval/total:.1%} | scout={self.time_scout/total:.1%}")
        
        # 重置
        self.time_xov = 0.0
        self.time_ls = 0.0
        self.time_eval = 0.0
        self.time_scout = 0.0
    
    # =========================================================================
    # 其他事件
    # =========================================================================
    
    def gls_state_change(self, gen: int, activated: bool, stagnation: int):
        """GLS 状态变化"""
        state = "activated" if activated else "deactivated"
        self._log(f"[GLS] {state} | gen={gen} | stagnation={stagnation}")
    
    def scout_event(self, gen: int, event_type: str, scout_fit: float, 
                    best_fit: float, accepted: bool):
        """Scout 事件"""
        self._log(f"[SCOUT] {event_type} | gen={gen} | scout_fit={scout_fit:.2f} | "
                  f"best_fit={best_fit:.2f} | accepted={accepted}")
    
    def restart_event(self, gen: int, old_best: float, reason: str):
        """重启事件"""
        self._log(f"[RST] restart | gen={gen} | old_best={old_best:.2f} | reason={reason}")
    
    # =========================================================================
    # P3 新增: 管道审计 [PIPE]
    # =========================================================================
    
    def pipe_child_audit(self, gen: int, 
                         shared_p1: int, shared_p2: int, 
                         shared_lkh_before: int, shared_lkh_after: int,
                         delta_repair: float, delta_ls: float,
                         accepted: bool):
        """
        P3: 单个 child 的管道审计
        
        追踪交叉/repair/LS 各阶段对边结构的影响
        """
        self._log(f"[PIPE] child_audit | gen={gen} | "
                  f"shared_p1={shared_p1} | shared_p2={shared_p2} | "
                  f"lkh_before={shared_lkh_before} | lkh_after={shared_lkh_after} | "
                  f"Δrepair={delta_repair:+.2f} | Δls={delta_ls:+.2f} | accepted={accepted}")
    
    def pipe_sample_summary(self, gen: int, samples: list):
        """
        P3: 多个 child 的管道审计汇总
        
        samples: list of dict {shared_p1, shared_p2, lkh_before, lkh_after, delta_repair, delta_ls, accepted}
        """
        if not samples:
            return
        
        n = len(samples)
        avg_shared_p1 = sum(s['shared_p1'] for s in samples) / n
        avg_shared_p2 = sum(s['shared_p2'] for s in samples) / n
        avg_lkh_before = sum(s['shared_lkh_before'] for s in samples) / n
        avg_lkh_after = sum(s['shared_lkh_after'] for s in samples) / n
        avg_delta_repair = sum(s['delta_repair'] for s in samples) / n
        avg_delta_ls = sum(s['delta_ls'] for s in samples) / n
        accepted_count = sum(1 for s in samples if s['accepted'])
        
        # 关键指标：LS 后与 LKH 的共享边是增加还是减少
        lkh_change = avg_lkh_after - avg_lkh_before
        
        self._log(f"[PIPE] summary | gen={gen} | samples={n} | "
                  f"avg_shared_p1={avg_shared_p1:.1f} | avg_shared_p2={avg_shared_p2:.1f} | "
                  f"lkh_before={avg_lkh_before:.1f} | lkh_after={avg_lkh_after:.1f} | "
                  f"lkh_change={lkh_change:+.1f} | "
                  f"Δrepair={avg_delta_repair:+.2f} | Δls={avg_delta_ls:+.2f} | accepted={accepted_count}/{n}")
    
    # =========================================================================
    # P4 新增: 候选边利用率 [CAND]
    # =========================================================================
    
    def cand_usage_report(self, gen: int, best_tour, lkh_tour, knn_idx, D):
        """
        P4: 候选边利用率报告
        
        检查 best_tour 和 lkh_tour 的边有多少在候选集中
        以及 missing edges 有多少其实在候选集中但没被用上
        """
        from diagnose_gap import get_edges_set
        
        n = len(best_tour)
        K = knn_idx.shape[1] if knn_idx is not None else 0
        
        # 构建候选集快速查找结构
        candidate_sets = [set(knn_idx[i]) - {-1} for i in range(n)] if knn_idx is not None else [set() for _ in range(n)]
        
        # best_tour 的边中有多少在候选集内
        _, best_edges = get_edges_set(best_tour)
        best_in_cand = 0
        for edge in best_edges:
            edge_list = list(edge)
            if len(edge_list) >= 2:
                u, v = edge_list[0], edge_list[1]
                if v in candidate_sets[u] or u in candidate_sets[v]:
                    best_in_cand += 1
        best_cand_ratio = best_in_cand / len(best_edges) if best_edges else 0
        
        # LKH tour 的边中有多少在候选集内
        lkh_in_cand = 0
        if lkh_tour is not None:
            _, lkh_edges = get_edges_set(lkh_tour)
            for edge in lkh_edges:
                edge_list = list(edge)
                if len(edge_list) >= 2:
                    u, v = edge_list[0], edge_list[1]
                    if v in candidate_sets[u] or u in candidate_sets[v]:
                        lkh_in_cand += 1
            lkh_cand_ratio = lkh_in_cand / len(lkh_edges) if lkh_edges else 0
            
            # missing edges 中有多少在候选集内
            missing = lkh_edges - best_edges
            missing_in_cand = 0
            for edge in missing:
                edge_list = list(edge)
                if len(edge_list) >= 2:
                    u, v = edge_list[0], edge_list[1]
                    if v in candidate_sets[u] or u in candidate_sets[v]:
                        missing_in_cand += 1
            missing_cand_ratio = missing_in_cand / len(missing) if missing else 1.0
        else:
            lkh_cand_ratio = 0
            missing_in_cand = 0
            missing_cand_ratio = 0
        
        self._log(f"[CAND] usage_report | gen={gen} | "
                  f"best_in_cand={best_cand_ratio:.1%} ({best_in_cand}/{n}) | "
                  f"lkh_in_cand={lkh_cand_ratio:.1%} | "
                  f"missing_in_cand={missing_cand_ratio:.1%} ({missing_in_cand}/{len(missing) if lkh_tour is not None else 0})")
    
    # =========================================================================
    # OX/Repair 诊断
    # =========================================================================
    
    def ox_repair_audit(self, gen: int, c_pop, population, fitness, D, best_tour):
        """
        抽样审计：repair 后 child 是否变成 parent 复制品？长度是否爆炸？
        
        抽样 5 个 child 检查：
        1. post_repair_identity: 是否与父代/best 相同
        2. repair_damage_report: 长度变化
        """
        from diagnose_gap import bond_distance
        
        lam = c_pop.shape[0]
        sample_indices = np.random.choice(lam, min(5, lam), replace=False)
        
        same_as_parent = 0
        same_as_best = 0
        len_exploded = 0  # 长度 > best * 3
        
        best_len = tour_length_jit(best_tour, D)
        
        for idx in sample_indices:
            child = c_pop[idx]
            child_len = tour_length_jit(child, D)
            
            # 检查是否与 best 相同
            if bond_distance(child, best_tour) == 0:
                same_as_best += 1
            
            # 检查是否长度爆炸
            if child_len > best_len * 3:
                len_exploded += 1
        
        n_samples = len(sample_indices)
        self._log(f"[OX] repair_audit | gen={gen} | samples={n_samples} | "
                  f"same_as_best={same_as_best}/{n_samples} | len_exploded={len_exploded}/{n_samples}")
    
    def ls_overwrite_audit(self, gen: int, tour_hash_before: str, tour_hash_after_ls: str, 
                           tour_hash_after_writeback: str):
        """
        LS 覆盖审计：检测 LS 的改动是否被后续回滚/同步覆盖
        
        如果 (after_ls != before) 但 (after_writeback == before)，说明 LS 被覆盖
        """
        ls_changed = (tour_hash_after_ls != tour_hash_before)
        overwritten = (tour_hash_after_writeback == tour_hash_before) and ls_changed
        
        status = "⚠️OVERWRITTEN" if overwritten else ("IMPROVED" if ls_changed else "NO_CHANGE")
        self._log(f"[LS] overwrite_audit | gen={gen} | before={tour_hash_before} | "
                  f"after_ls={tour_hash_after_ls} | writeback={tour_hash_after_writeback} | {status}")

    # =========================================================================
    # 决定性诊断 (4 类新日志)
    # =========================================================================
    
    def rtr_target_record(self, target_fit: float, replaced: bool):
        """记录 RTR 被挑战 target 的信息"""
        self.rtr_target_fits.append(target_fit)
        if replaced:
            self.rtr_replaced_target_fits.append(target_fit)
        else:
            self.rtr_rejected_target_fits.append(target_fit)
    
    def rtr_target_quality_report(self, gen: int, best_fit: float):
        """
        [RTR] target_quality_report
        统计被挑战的 target 质量分布
        """
        if not self.rtr_target_fits:
            return
        
        fits = np.array(self.rtr_target_fits)
        p10 = np.percentile(fits, 10)
        p50 = np.percentile(fits, 50)
        p90 = np.percentile(fits, 90)
        
        # 被替换的 target
        replaced_med = np.median(self.rtr_replaced_target_fits) if self.rtr_replaced_target_fits else 0
        
        # 被拒收时 target 是否在 top 区（< best * 1.2）
        top_rejected = sum(1 for f in self.rtr_rejected_target_fits if f < best_fit * 1.2)
        total_rejected = len(self.rtr_rejected_target_fits)
        
        self._log(f"[RTR] target_quality | gen={gen} | n={len(fits)} | "
                  f"P10={p10:.0f} | P50={p50:.0f} | P90={p90:.0f} | "
                  f"replaced_med={replaced_med:.0f} | top_rejected={top_rejected}/{total_rejected}")
        
        # 重置
        self.rtr_target_fits = []
        self.rtr_replaced_target_fits = []
        self.rtr_rejected_target_fits = []
    
    def mate_parent_record(self, p1_fit: float, p2_fit: float):
        """记录父母 fitness"""
        self.parent_fits.append(p1_fit)
        self.parent_fits.append(p2_fit)
    
    def mate_parent_pool_report(self, gen: int, best_fit: float, pop_median: float):
        """
        [MATE] parent_pool_report
        统计被选作父母的个体质量
        """
        if not self.parent_fits:
            return
        
        fits = np.array(self.parent_fits)
        p10 = np.percentile(fits, 10)
        p50 = np.percentile(fits, 50)
        p90 = np.percentile(fits, 90)
        
        # 父母来自 top 50% 的比例
        top_half_threshold = pop_median
        top_half = sum(1 for f in fits if f < top_half_threshold)
        top_ratio = top_half / len(fits)
        
        # elite+garbage 配对比例（fitness 比率 > 2x）
        # 需要成对看
        mismatch_count = 0
        for i in range(0, len(self.parent_fits) - 1, 2):
            f1, f2 = self.parent_fits[i], self.parent_fits[i+1]
            ratio = max(f1, f2) / min(f1, f2) if min(f1, f2) > 0 else 1
            if ratio > 2.0:
                mismatch_count += 1
        total_pairs = len(self.parent_fits) // 2
        
        self._log(f"[MATE] parent_pool | gen={gen} | n={len(fits)} | "
                  f"P10={p10:.0f} | P50={p50:.0f} | P90={p90:.0f} | "
                  f"top_50%={top_ratio:.1%} | mismatch_pairs={mismatch_count}/{total_pairs}")
        
        # 重置
        self.parent_fits = []
    
    def hgrex_step_record(self, source: str):
        """
        记录 HGreX 每一步的候选来源
        source: 'parent' / 'knn' / 'random' / 'fullscan'
        """
        self.hgrex_total_steps += 1
        if source == 'parent':
            self.hgrex_parent_edge += 1
        elif source == 'knn':
            self.hgrex_knn_fallback += 1
        elif source == 'random':
            self.hgrex_random_fallback += 1
        elif source == 'fullscan':
            self.hgrex_fullscan_fallback += 1
    
    def hgrex_fallback_breakdown(self, gen: int):
        """
        [XOV] hgrex_fallback_breakdown
        HGreX 分层候选详细统计
        """
        total = max(1, self.hgrex_total_steps)
        
        self._log(f"[XOV] hgrex_breakdown | gen={gen} | total_steps={total} | "
                  f"parent={self.hgrex_parent_edge} ({self.hgrex_parent_edge/total:.1%}) | "
                  f"knn={self.hgrex_knn_fallback} ({self.hgrex_knn_fallback/total:.1%}) | "
                  f"random={self.hgrex_random_fallback} ({self.hgrex_random_fallback/total:.1%}) | "
                  f"fullscan={self.hgrex_fullscan_fallback} ({self.hgrex_fullscan_fallback/total:.1%})")
        
        # 重置
        self.hgrex_parent_edge = 0
        self.hgrex_knn_fallback = 0
        self.hgrex_random_fallback = 0
        self.hgrex_fullscan_fallback = 0
        self.hgrex_total_steps = 0
    
    def pipe_record(self, stage: str, count: int = 1, op_type: str = None):
        """
        记录流水线各阶段
        stage: 'generated' / 'feasible' / 'tamed' / 'submitted' / 'accepted'
        """
        if stage == 'generated' and op_type:
            if op_type not in self.pipe_generated:
                self.pipe_generated[op_type] = 0
            self.pipe_generated[op_type] += count
        elif stage == 'feasible':
            self.pipe_feasible += count
        elif stage == 'tamed':
            self.pipe_tamed += count
        elif stage == 'submitted':
            self.pipe_submitted += count
        elif stage == 'accepted':
            self.pipe_accepted += count
    
    def pipe_offspring_flow_report(self, gen: int):
        """
        [PIPE] offspring_flow_report
        从生成到写回的完整流水线统计
        """
        total_gen = sum(self.pipe_generated.values())
        
        self._log(f"[PIPE] offspring_flow | gen={gen} | "
                  f"generated={total_gen} (HGreX={self.pipe_generated.get('hgrex', 0)}, "
                  f"OX={self.pipe_generated.get('ox', 0)}, Mut={self.pipe_generated.get('mutate', 0)}) | "
                  f"feasible={self.pipe_feasible} | tamed={self.pipe_tamed} | "
                  f"submitted={self.pipe_submitted} | accepted={self.pipe_accepted}")
        
        # 重置
        self.pipe_generated = {'hgrex': 0, 'ox': 0, 'mutate': 0}
        self.pipe_feasible = 0
        self.pipe_tamed = 0
        self.pipe_submitted = 0
        self.pipe_accepted = 0


