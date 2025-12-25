"""
诊断脚本：对比 LKH3 最佳路径与你的算法输出

使用方法：
    1. 先用 get_optimal_reference.py 生成 best_route_tour750.txt
    2. 在 r0927480.py 中 import 本模块并调用诊断函数

诊断维度：
    - 边相似度 (Edge Similarity): 你的解有多少边与 LKH3 相同
    - 候选边覆盖率 (Candidate Coverage): LKH3 使用的边有多少在你的候选集中
    - Bond Distance: 解与解之间的结构距离
"""

import numpy as np

def load_lkh_route(filename):
    """
    加载 LKH3 最佳路径（每行一个节点索引）
    
    注意：LKH 输出的格式可能是 "起点 -> ... -> 起点"，
    会自动去除重复的起点节点。
    """
    route = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                route.append(int(line))
    
    # 如果最后一个节点等于第一个节点（回到起点），则去除
    if len(route) > 1 and route[-1] == route[0]:
        route = route[:-1]
    
    return np.array(route, dtype=np.int32)


def get_edges_set(tour):
    """将 tour 转换为边集合（无向边用 frozenset，有向边用 tuple）"""
    n = len(tour)
    edges_directed = set()
    edges_undirected = set()
    for i in range(n):
        u, v = tour[i], tour[(i + 1) % n]
        edges_directed.add((u, v))
        edges_undirected.add(frozenset([u, v]))
    return edges_directed, edges_undirected


def edge_similarity(tour1, tour2, directed=False):
    """
    计算两个 tour 的边相似度
    
    Args:
        tour1: 第一个 tour
        tour2: 第二个 tour (通常是 LKH3 最佳路径)
        directed: 是否考虑边的方向（非对称 TSP 设为 True）
    
    Returns:
        shared_count: 共享边数量
        ratio: 重合率 (shared_count / n)
    """
    d1, u1 = get_edges_set(tour1)
    d2, u2 = get_edges_set(tour2)
    
    if directed:
        shared = d1 & d2
    else:
        shared = u1 & u2
    
    n = len(tour1)
    return len(shared), len(shared) / n


def bond_distance(tour1, tour2):
    """
    计算 Bond Distance（结构距离）
    
    Bond Distance = n - 共享边数量
    数值越小，表示两个 tour 结构越相似
    """
    n = len(tour1)
    _, u1 = get_edges_set(tour1)
    _, u2 = get_edges_set(tour2)
    shared = len(u1 & u2)
    return n - shared


def find_missing_edges(my_tour, lkh_tour, D=None):
    """
    找出 LKH3 用了但你没用的边
    
    Returns:
        missing_edges: 列表 [(u, v, distance), ...]
    """
    _, my_edges = get_edges_set(my_tour)
    _, lkh_edges = get_edges_set(lkh_tour)
    
    missing = lkh_edges - my_edges
    
    result = []
    for edge in missing:
        edge_list = list(edge)
        # 处理单元素 frozenset（理论上不应该存在，但以防万一）
        if len(edge_list) < 2:
            continue
        u, v = edge_list[0], edge_list[1]
        dist = D[u, v] if D is not None else None
        result.append((u, v, dist))
    
    # 按距离排序（如果有距离信息）
    if D is not None:
        result.sort(key=lambda x: x[2])
    
    return result


def candidate_coverage(lkh_tour, knn_idx):
    """
    检查 LKH3 使用的边有多少被 knn_idx 候选覆盖
    
    Args:
        lkh_tour: LKH3 最佳路径
        knn_idx: 你的 KNN 候选边 (n, K)
    
    Returns:
        covered_count: 被覆盖的边数量
        ratio: 覆盖率
        uncovered_edges: 未被覆盖的边列表 [(u, v), ...]
    """
    n = len(lkh_tour)
    K = knn_idx.shape[1]
    
    # 将 knn_idx 转为快速查找结构
    candidate_sets = [set(knn_idx[i]) - {-1} for i in range(n)]
    
    covered = 0
    uncovered_edges = []
    
    for i in range(n):
        u, v = lkh_tour[i], lkh_tour[(i + 1) % n]
        
        # 检查 u->v 或 v->u 是否在候选中
        if v in candidate_sets[u] or u in candidate_sets[v]:
            covered += 1
        else:
            uncovered_edges.append((u, v))
    
    return covered, covered / n, uncovered_edges


def diagnose_full(my_tour, lkh_tour, D, knn_idx=None, label=""):
    """
    完整诊断输出
    
    Args:
        my_tour: 你的算法当前最优解
        lkh_tour: LKH3 最佳路径
        D: 距离矩阵
        knn_idx: 可选，你的 KNN 候选边
        label: 诊断标签（如 "Gen 100"）
    """
    n = len(my_tour)
    
    # 1. 计算路径长度
    my_length = sum(D[my_tour[i], my_tour[(i + 1) % n]] for i in range(n))
    lkh_length = sum(D[lkh_tour[i], lkh_tour[(i + 1) % n]] for i in range(n))
    gap = (my_length - lkh_length) / lkh_length * 100
    
    # 2. 边相似度
    shared, ratio = edge_similarity(my_tour, lkh_tour, directed=False)
    bond_dist = bond_distance(my_tour, lkh_tour)
    
    # 3. 找缺失边（LKH 用了你没用）
    missing = find_missing_edges(my_tour, lkh_tour, D)
    
    print(f"\n{'='*60}")
    print(f"🔬 诊断报告 {label}")
    print(f"{'='*60}")
    print(f"  📏 你的路径长度: {my_length:.2f}")
    print(f"  🏆 LKH3 最佳长度: {lkh_length:.2f}")
    print(f"  📊 差距 (Gap): {gap:.4f}%")
    print(f"\n  🔗 边相似度: {shared}/{n} ({ratio*100:.2f}%)")
    print(f"  📐 Bond Distance: {bond_dist}")
    
    # 4. 候选覆盖率
    if knn_idx is not None:
        cov_count, cov_ratio, uncovered = candidate_coverage(lkh_tour, knn_idx)
        print(f"\n  📋 候选覆盖率: {cov_count}/{n} ({cov_ratio*100:.2f}%)")
        if uncovered:
            print(f"  ⚠️  LKH3 使用但你的候选未覆盖的边 (前5个):")
            for u, v in uncovered[:5]:
                print(f"      - ({u}, {v}), 距离: {D[u, v]:.2f}")
    
    # # 5. 显示 LKH3 用了你没用的边中距离最短的几个
    # if missing:
    #     print(f"\n  🔍 LKH3 用了但你没用的边 (按距离排序，前10个):")
    #     for u, v, dist in missing[:10]:
    #         print(f"      - ({u}, {v}), 距离: {dist:.2f}")
    
    print(f"{'='*60}\n")
    
    return {
        "my_length": my_length,
        "lkh_length": lkh_length,
        "gap_pct": gap,
        "edge_similarity": ratio,
        "bond_distance": bond_dist,
        "missing_edges": missing[:20]
    }


# =============================================================================
# 可以在 r0927480.py 中调用的辅助函数
# =============================================================================

_LKH_ROUTE = None  # 缓存 LKH 最佳路径

def init_lkh_reference(filename):
    """初始化 LKH 参考路径（只需调用一次）"""
    global _LKH_ROUTE
    try:
        _LKH_ROUTE = load_lkh_route(filename)
        print(f"✅ 加载 LKH 参考路径: {filename} (n={len(_LKH_ROUTE)})")
    except FileNotFoundError:
        print(f"⚠️ 未找到 LKH 参考路径: {filename}")
        _LKH_ROUTE = None


def quick_diagnose(my_tour, D, knn_idx=None, label=""):
    """快速诊断（在算法运行中周期性调用）"""
    global _LKH_ROUTE
    if _LKH_ROUTE is None:
        return None
    
    return diagnose_full(my_tour, _LKH_ROUTE, D, knn_idx, label)


# =============================================================================
# 高级诊断函数：种群多样性、Scout 效能、GLS 状态等
# =============================================================================

def calc_pop_diversity(population, sample_pairs=10):
    """
    计算种群多样性（平均 Bond Distance）
    
    Args:
        population: 种群 (lam, n)
        sample_pairs: 采样对数
    
    Returns:
        avg_bond_dist: 平均 Bond Distance
        diversity_ratio: 多样性比率 (avg_bond_dist / n)
    """
    lam, n = population.shape
    if lam < 2:
        return 0.0, 0.0
    
    total_dist = 0.0
    pairs = min(sample_pairs, lam * (lam - 1) // 2)
    
    for _ in range(pairs):
        i = np.random.randint(0, lam)
        j = np.random.randint(0, lam - 1)
        if j >= i:
            j += 1
        total_dist += bond_distance(population[i], population[j])
    
    avg_dist = total_dist / pairs if pairs > 0 else 0.0
    return avg_dist, avg_dist / n


def count_distinct_tours(population):
    """
    统计种群中不同解的数量（用于检测早熟）
    
    Returns:
        distinct_count: 不同解的数量
    """
    seen = set()
    for tour in population:
        # 用 tour 的哈希作为唯一标识（简化版：用前10个和后10个元素）
        key = tuple(tour[:10]) + tuple(tour[-10:])
        seen.add(key)
    return len(seen)


def analyze_error_edges(my_tour, lkh_tour, D):
    """
    分析错误边的特征
    
    Returns:
        my_avg_edge_len: 我的解的平均边长度
        lkh_avg_edge_len: LKH 解的平均边长度
        missing_avg_len: 缺失边的平均长度
        extra_avg_len: 多余边的平均长度
    """
    n = len(my_tour)
    
    # 计算边集合
    _, my_edges = get_edges_set(my_tour)
    _, lkh_edges = get_edges_set(lkh_tour)
    
    # 共享边、缺失边、多余边
    shared = my_edges & lkh_edges
    missing = lkh_edges - my_edges  # LKH 有但我没有
    extra = my_edges - lkh_edges    # 我有但 LKH 没有
    
    # 计算平均边长度
    def avg_edge_length(edge_set, D):
        if not edge_set:
            return 0.0
        total = 0.0
        count = 0
        for edge in edge_set:
            edge_list = list(edge)
            if len(edge_list) >= 2:
                u, v = edge_list[0], edge_list[1]
                if np.isfinite(D[u, v]):
                    total += D[u, v]
                    count += 1
        return total / count if count > 0 else 0.0
    
    my_avg = avg_edge_length(my_edges, D)
    lkh_avg = avg_edge_length(lkh_edges, D)
    missing_avg = avg_edge_length(missing, D)
    extra_avg = avg_edge_length(extra, D)
    
    return {
        "my_avg_edge_len": my_avg,
        "lkh_avg_edge_len": lkh_avg,
        "missing_avg_len": missing_avg,
        "extra_avg_len": extra_avg,
        "shared_count": len(shared),
        "missing_count": len(missing),
        "extra_count": len(extra)
    }


def check_gls_penalty_quality(tour, D, gls_penalties, lkh_tour):
    """
    检查 GLS 惩罚的质量：是否正确惩罚了错误边
    
    Returns:
        correct_penalty_ratio: 正确边中被惩罚的比例（应该低）
        wrong_penalty_ratio: 错误边中被惩罚的比例（应该高）
    """
    n = len(tour)
    _, my_edges = get_edges_set(tour)
    _, lkh_edges = get_edges_set(lkh_tour)
    
    # 正确边 = 共享边，错误边 = 我有但 LKH 没有
    shared = my_edges & lkh_edges
    extra = my_edges - lkh_edges
    
    def get_penalty_ratio(edge_set):
        if not edge_set:
            return 0.0
        penalized = 0
        for edge in edge_set:
            edge_list = list(edge)
            if len(edge_list) >= 2:
                u, v = edge_list[0], edge_list[1]
                if gls_penalties[u, v] > 0 or gls_penalties[v, u] > 0:
                    penalized += 1
        return penalized / len(edge_set)
    
    return {
        "correct_edge_penalty_ratio": get_penalty_ratio(shared),
        "wrong_edge_penalty_ratio": get_penalty_ratio(extra),
        "max_penalty": int(gls_penalties.max()),
        "nonzero_count": int(np.count_nonzero(gls_penalties))
    }


def advanced_diagnose(my_tour, D, population=None, gls_penalties=None, 
                      scout_accepted=0, scout_total=0, label=""):
    """
    高级诊断输出（在 quick_diagnose 基础上增加更多指标）
    """
    global _LKH_ROUTE
    if _LKH_ROUTE is None:
        return None
    
    n = len(my_tour)
    lkh_tour = _LKH_ROUTE
    
    # 基础指标
    my_length = sum(D[my_tour[i], my_tour[(i + 1) % n]] for i in range(n))
    lkh_length = sum(D[lkh_tour[i], lkh_tour[(i + 1) % n]] for i in range(n))
    gap = (my_length - lkh_length) / lkh_length * 100
    shared, ratio = edge_similarity(my_tour, lkh_tour, directed=False)
    
    # 错误边分析
    error_analysis = analyze_error_edges(my_tour, lkh_tour, D)
    
    print(f"\n{'='*70}")
    print(f"🔬 高级诊断 {label}")
    print(f"{'='*70}")
    print(f"  📊 Gap: {gap:.4f}% | 边相似度: {shared}/{n} ({ratio*100:.1f}%)")
    
    # 错误边特征
    print(f"  📏 边长度对比:")
    print(f"      我的平均边长: {error_analysis['my_avg_edge_len']:.2f}")
    print(f"      LKH 平均边长: {error_analysis['lkh_avg_edge_len']:.2f}")
    print(f"      缺失边平均长: {error_analysis['missing_avg_len']:.2f} (应该用这些)")
    print(f"      多余边平均长: {error_analysis['extra_avg_len']:.2f} (不应该用这些)")
    
    greedy_indicator = error_analysis['extra_avg_len'] < error_analysis['missing_avg_len']
    if greedy_indicator:
        print(f"      ⚠️ 诊断: 算法过于贪婪，选择了更短但非最优的边！")
    else:
        print(f"      ℹ️ 诊断: 算法可能优化力度不够")
    
    # 种群多样性
    if population is not None:
        avg_dist, div_ratio = calc_pop_diversity(population, 15)
        distinct = count_distinct_tours(population)
        print(f"  👥 种群多样性:")
        print(f"      平均 Bond Distance: {avg_dist:.1f} ({div_ratio*100:.1f}% of n)")
        print(f"      不同解数量: {distinct}/{len(population)}")
        if div_ratio < 0.05:
            print(f"      ⚠️ 警告: 种群严重早熟！")
    
    # GLS 状态 (Vanilla GLS: high penalty rate is NORMAL!)
    if gls_penalties is not None:
        gls_info = check_gls_penalty_quality(my_tour, D, gls_penalties, lkh_tour)
        print(f"  🎯 GLS 惩罚状态 (Vanilla模式):")
        print(f"      Max Penalty: {gls_info['max_penalty']} | 非零数量: {gls_info['nonzero_count']}")
        print(f"      正确边被惩罚: {gls_info['correct_edge_penalty_ratio']*100:.1f}% (Vanilla: 高是正常的!)")
        print(f"      错误边被惩罚: {gls_info['wrong_edge_penalty_ratio']*100:.1f}%")
    
    # Scout 效能
    if scout_total > 0:
        acc_rate = scout_accepted / scout_total * 100
        print(f"  🦅 Scout 效能: {scout_accepted}/{scout_total} ({acc_rate:.1f}%)")
    
    print(f"{'='*70}\n")
    
    return {
        "gap": gap,
        "similarity": ratio,
        "error_analysis": error_analysis
    }


# =============================================================================
# 示例用法
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # 示例：加载并对比
    if len(sys.argv) < 3:
        print("用法: python diagnose_gap.py <tour_csv> <lkh_route.txt>")
        print("示例: python diagnose_gap.py tour750.csv best_route_tour750.txt")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    lkh_file = sys.argv[2]
    
    # 加载距离矩阵
    D = np.loadtxt(csv_file, delimiter=',')
    n = D.shape[0]
    
    # 加载 LKH 最佳路径
    lkh_tour = load_lkh_route(lkh_file)
    
    # 生成一个简单的贪心解作为对比
    from get_optimal_reference import method_elkai
    _, my_tour = method_elkai(D, precision=1, runs=1)  # 只跑 1 次作为快速测试
    
    if my_tour is not None:
        my_tour = np.array(my_tour, dtype=np.int32)
        
        # 构建 KNN 候选
        finite_mask = np.isfinite(D)
        K = 32
        knn_idx = np.full((n, K), -1, np.int32)
        for i in range(n):
            row = D[i]
            valid = np.where(finite_mask[i])[0]
            if len(valid) > 0:
                order = np.argsort(row[valid])
                m = min(K, len(valid))
                knn_idx[i, :m] = valid[order[:m]]
        
        diagnose_full(my_tour, lkh_tour, D, knn_idx, label="快速测试")

# =============================================================================
# God Mode Debugging Tools
# =============================================================================

def create_golden_individual(D, lkh_tour, ruin_percent=0.3):
    """
    Generate 'Golden Individual': preserve (1-ruin_percent) of LKH optimal,
    destroy and repair the rest with greedy strategy.
    Used to test: will the algorithm 'optimize' this near-perfect solution badly?
    """
    n = len(lkh_tour)
    n_remove = int(n * ruin_percent)
    
    # 1. Copy LKH genes
    tour = lkh_tour.copy()
    
    # 2. Randomly destroy a contiguous region (Sequence Ruin)
    start = np.random.randint(0, n)
    mask = np.zeros(n, dtype=np.bool_)
    for i in range(n_remove):
        mask[tour[(start + i) % n]] = True
        
    # 3. Extract kept and removed cities
    kept = []
    removed = []
    for city in tour:
        if mask[city]: removed.append(city)
        else: kept.append(city)
    
    # Shuffle removed and reinsert with greedy
    current_tour = list(kept)
    np.random.shuffle(removed)
    
    # Cheapest Insertion
    for city in removed:
        best_delta = 1e20
        best_pos = -1
        m = len(current_tour)
        for i in range(m):
            u, v = current_tour[i], current_tour[(i + 1) % m]
            delta = D[u, city] + D[city, v] - D[u, v]
            if delta < best_delta:
                best_delta = delta
                best_pos = i
        current_tour.insert(best_pos + 1, city)
        
    return np.array(current_tour, dtype=np.int32)

def analyze_missing_topology(my_tour, lkh_tour):
    """
    Analyze missing edges (dead knots) topology
    """
    n = len(my_tour)
    _, my_edges = get_edges_set(my_tour)
    _, lkh_edges = get_edges_set(lkh_tour)
    
    # Find edges LKH has but I don't
    missing = list(lkh_edges - my_edges)
    missing_count = len(missing)
    
    print(f"\n🔍 拓扑死结分析:")
    print(f"   缺失边数量: {missing_count} (这些边构成了你无法跨越的墙)")
    
    if missing_count == 0:
        print("   ✅ 没有缺失边，已达到最优解！")
        return

    # Build adjacency for connectivity analysis
    adj = {}
    nodes = set()
    for edge in missing:
        u, v = list(edge)
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append(v)
        adj[v].append(u)
        nodes.add(u)
        nodes.add(v)
        
    # Find connected components
    visited = set()
    chains = 0
    cycles = 0
    complex_knots = 0
    
    for node in nodes:
        if node not in visited:
            # BFS for connected component
            component_nodes = []
            stack = [node]
            visited.add(node)
            while stack:
                curr = stack.pop()
                component_nodes.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            
            # Analyze component
            comp_edges = 0
            for u in component_nodes:
                comp_edges += len(adj[u])
            comp_edges //= 2
            
            size = len(component_nodes)
            
            if size == comp_edges:
                cycles += 1
                knot_type = "🔒 闭环 (Cycle)"
            elif size == comp_edges + 1:
                chains += 1
                knot_type = "🔗 链条 (Chain)"
            else:
                complex_knots += 1
                knot_type = "🕸️ 复杂纠缠 (Complex)"
            
            print(f"   - 组件: {size} 节点, {comp_edges} 边 -> {knot_type}")

    print(f"   📊 总结: {chains} 条链, {cycles} 个环, {complex_knots} 个复杂纠缠")
    
    if cycles > 0 or complex_knots > 0:
        print("   🚨 结论: 存在闭环或复杂纠缠。2-opt/3-opt 无法解开。")
        print("      需要 Double Bridge (4-opt) 或 Ejection Chains。")
    else:
        print("   ✅ 结论: 错误边比较分散，GLS 应该能解决。")
