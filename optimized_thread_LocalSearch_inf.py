# Write the implemented EA into the provided template file.
# 注：本脚本实现了基于遗传算法（Genetic Algorithm）的旅行商问题（TSP）求解器
# 核心特性：边缘重组交叉（ERX）、混合变异、K-最近邻（KNN）初始种群、并行加速评估、局部搜索

import Reporter  # 导入课程提供的结果上报器，用于提交结果
import numpy as np  # 导入NumPy库用于高效数值计算
from typing import List, Optional  # 导入类型提示
import os  # 导入操作系统接口
import time
# -------- 环境变量配置 (Environment Configuration) --------
# 限制底层数学库的线程数，防止与上层并行冲突
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# -------- Numba JIT 配置 (Numba Configuration) --------
try:
    from numba import njit, prange, set_num_threads
    # 强制要求 Numba 环境，否则报错并退出
    print("Numba imported successfully. JIT compilation enabled.")
except ImportError:
    print("Error: Numba is required for this script. Please install it with: pip install numba")
    exit(1)

# 设置 Numba 并行线程数（根据硬件调整，通常2-4个足够）
# ⭐ 智能检测：如果环境变量已限制线程数，则遵循该限制
_numba_threads_env = os.environ.get("NUMBA_NUM_THREADS", None)
if _numba_threads_env is not None:
    _max_threads = int(_numba_threads_env)
else:
    _max_threads = 4  # 默认使用 4 个线程
set_num_threads(_max_threads)


# ==============================================================================
# Part 1: JIT Accelerated Helper Functions (JIT加速辅助函数)
# ==============================================================================

@njit(cache=True, fastmath=True)
def _erx_jit(p1, p2):
    """
    边缘重组交叉 (Edge Recombination Crossover, ERX) 的 JIT 实现。
    旨在保留父代的邻接关系（边），适合 TSP 问题。
    """
    n = p1.size  # 获取城市数量（基因长度）
    child = np.empty(n, np.int32)  # 初始化子代数组
    used = np.zeros(n, np.uint8)  # 标记数组，记录城市是否已加入子代
    
    # 构建邻接表：每个城市最多4个邻居（来自两个父代的左右邻居）
    # neighbors: (n, 4) 存储邻居索引，-1 表示空位
    neighbors = np.full((n, 4), -1, np.int32)
    deg = np.zeros(n, np.int32)  # 记录每个城市的当前邻居数量

    # 内部函数：向 u 的邻接表中添加 v
    def add_edge(u, v):
        if v == u: return  # 忽略自环
        for k in range(deg[u]):  # 检查是否已存在该边
            if neighbors[u, k] == v:
                return
        if deg[u] < 4:  # 若邻居未满，则添加
            neighbors[u, deg[u]] = v
            deg[u] += 1

    # 遍历两个父代，构建完整的邻接图
    for parent in (p1, p2):
        for i in range(n):
            c = parent[i]  # 当前城市
            add_edge(c, parent[(i - 1) % n])  # 添加左邻居
            add_edge(c, parent[(i + 1) % n])  # 添加右邻居

    # ERX 构建过程
    cur = p1[0]  # 从父代1的第一个城市开始
    next_scan = 0  # 线性扫描指针，用于回退策略

    for t in range(n):
        child[t] = cur  # 将当前城市加入子代
        used[cur] = 1   # 标记为已使用

        # 选择下一个城市：优先选择拥有最少可用邻居的邻居
        best = -1
        best_score = 1_000_000  # 最小剩余邻居数，初始化为大数
        
        # 遍历当前城市的所有邻居
        for k in range(deg[cur]):
            nb = neighbors[cur, k]
            if nb == -1 or used[nb] == 1:  # 跳过无效或已使用的邻居
                continue
            
            # 计算邻居 nb 还有多少未使用的邻居
            cnt = 0
            for j in range(deg[nb]):
                x = neighbors[nb, j]
                if x != -1 and used[x] == 0:
                    cnt += 1
            
            # 贪婪选择：选择剩余邻居最少的点（减少死胡同概率）
            if cnt < best_score:
                best_score = cnt
                best = nb
        
        if best != -1:
            cur = best  # 找到合适邻居，移动到该城市
        else:
            # 死胡同处理：邻居都已使用，线性扫描找一个未使用的城市
            while next_scan < n and used[next_scan] == 1:
                next_scan += 1
            if next_scan < n:
                cur = next_scan
            else:
                # 最后的兜底（理论上不应进入这里，除非逻辑有误）
                for r in range(n):
                    if used[r] == 0:
                        cur = r
                        break
    return child  # 返回生成的子代


@njit(cache=True, fastmath=True)
def _ox_jit(p1, p2):
    """
    P2: Order Crossover (OX) - 保留父代的连续子路径结构。
    比 ERX 更简单，与 Or-Opt 配合更好。
    """
    n = p1.size
    child = np.full(n, -1, np.int32)
    
    # 随机选择切点 [cut1, cut2)
    cut1 = np.random.randint(0, n - 1)
    cut2 = np.random.randint(cut1 + 1, n)
    
    # 复制 p1[cut1:cut2] 到 child
    for i in range(cut1, cut2):
        child[i] = p1[i]
    
    # 构建已使用城市集合
    used = np.zeros(n, np.uint8)
    for i in range(cut1, cut2):
        used[p1[i]] = 1
    
    # 从 cut2 开始，按 p2 顺序填充剩余城市
    idx = cut2
    for i in range(n):
        city = p2[(cut2 + i) % n]
        if used[city] == 0:
            child[idx % n] = city
            idx += 1
            used[city] = 1
    
    return child


@njit(cache=True, fastmath=True)
def tour_length_jit(tour, D):
    """
    计算单条路径的总长度。
    """
    n = tour.shape[0]
    s = 0.0
    for i in range(n - 1):
        s += D[tour[i], tour[i+1]]  # 累加相邻距离
    s += D[tour[n-1], tour[0]]      # 累加回到起点的距离
    return s


@njit(cache=True, fastmath=True, parallel=True)
def batch_lengths_jit(pop2d, D, out):
    """
    并行批量计算种群中所有个体的适应度（路径长度）。
    """
    m, n = pop2d.shape  # m: 个体数, n: 城市数
    for r in prange(m):  # 并行循环
        s = 0.0
        row = pop2d[r]
        for i in range(n - 1):
            s += D[row[i], row[i+1]]
        s += D[row[n-1], row[0]]
        out[r] = s  # 存入结果数组


@njit(cache=True, fastmath=True, parallel=True)
def build_knn_idx(D, finite_mask, K):
    """
    预计算每个城市的 K 个最近可行邻居 (KNN)。
    这对 RCL-NN 初始化和启发式修复非常重要。
    """
    n = D.shape[0]
    knn = np.full((n, K), -1, np.int32)  # 初始化 KNN 表
    
    for i in prange(n):  # 并行处理每个城市
        # 1. 统计可行邻居数量
        cnt = 0
        for j in range(n):
            if finite_mask[i, j]:
                cnt += 1
        if cnt == 0: continue
        
        # 2. 收集所有可行邻居
        cand_idx = np.empty(cnt, np.int32)
        cand_dis = np.empty(cnt, np.float64)
        c = 0
        for j in range(n):
            if finite_mask[i, j]:
                cand_idx[c] = j
                cand_dis[c] = D[i, j]
                c += 1
                
        # 3. 排序并取前 K 个
        order = np.argsort(cand_dis)
        m = K if K < cnt else cnt
        for t in range(m):
            knn[i, t] = cand_idx[order[t]]
            
    return knn


@njit(cache=True, fastmath=True)
def _two_opt_once_jit_safe(tour, D):
    """
    尝试执行一次随机 2-opt 交换。
    如果能缩短路径且保持边可行，则应用并返回 True。
    """
    n = tour.size
    best_delta = 0.0
    bi = -1; bj = -1
    
    # 限制采样次数，避免在大规模问题上耗时过长
    tries = min(2000, n * 20)
    
    for _ in range(tries):
        # 随机选择两个切断点 i 和 j
        i = np.random.randint(0, n - 3)
        j = np.random.randint(i + 2, n - 1)
        
        # 涉及的四个城市：A-B ... C-D
        a = tour[i]; b = tour[(i + 1) % n]
        c = tour[j]; d = tour[(j + 1) % n]
        
        # 检查新边 A-C 和 B-D 是否存在（距离有限）
        if not np.isfinite(D[a, c]) or not np.isfinite(D[b, d]):
            continue
            
        # 计算距离增量：(AC + BD) - (AB + CD)
        delta = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
        
        if delta < best_delta:  # 只有改进才记录
            best_delta = delta
            bi = i
            bj = j
            
    if best_delta < 0.0:  # 找到改进
        # 执行反转操作 tour[i+1...j]
        l = bi + 1
        r = bj
        while l < r:
            tmp = tour[l]; tour[l] = tour[r]; tour[r] = tmp
            l += 1; r -= 1
        return True
    return False


@njit(cache=True, fastmath=True)
def _tour_feasible_jit(tour, finite_mask):
    """
    检查路径是否可行（所有相邻边均连通）。
    """
    n = tour.size
    for i in range(n):
        a = tour[i]; b = tour[(i + 1) % n]
        if not finite_mask[a, b]:  # 发现断路
            return False
    return True


@njit(cache=True, fastmath=True)
def _repair_jit(tour, D, finite_mask, max_tries=50):
    """
    使用 2-opt 动作尝试快速修复不可行的路径。
    """
    for _ in range(max_tries):
        if _tour_feasible_jit(tour, finite_mask):
            return True
        # 利用 2-opt 随机改变结构，期望碰巧连通
        if not _two_opt_once_jit_safe(tour, D):
            break
    return _tour_feasible_jit(tour, finite_mask)


@njit(cache=True, fastmath=True)
def _rand_perm_jit(n):
    """
    生成随机排列 (Fisher-Yates Shuffle)。
    """
    arr = np.arange(n, dtype=np.int32)
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp
    return arr


@njit(cache=True, fastmath=True)
def _or_opt_once_jit(tour, D, block_size):
    """
    尝试一次 Or-Opt (Block Shift) 移动。
    移动一个长度为 block_size 的片段到新位置。
    JIT 优化版。
    """
    n = tour.shape[0]
    
    # 随机化 block 起点
    start_node = np.random.randint(0, n)
    
    for i_offset in range(n):
        i = (start_node + i_offset) % n
        
        # 只处理不跨越数组边界的 block (简化逻辑，对随机起点足够)
        if i + block_size >= n: continue
        
        # 定义 block: [i ... i+BS-1]
        # 原始连接: A(i-1) -> B(i) 和 C(i+BS-1) -> D(i+BS)
        
        prev_idx = i - 1
        if prev_idx < 0: prev_idx = n - 1
        
        post_idx = i + block_size
        if post_idx >= n: post_idx = 0
        
        A = tour[prev_idx]
        B = tour[i]
        C = tour[i + block_size - 1]
        D_node = tour[post_idx] # D keyword conflict
        
        # 移除代价: -AB -CD +AD
        if not (np.isfinite(D[A, B]) and np.isfinite(D[C, D_node]) and np.isfinite(D[A, D_node])):
            continue
            
        remove_gain = D[A, B] + D[C, D_node] - D[A, D_node]
        
        # 扫描插入位置 j
        # j 不能在 block 内部，也不能是 prev_idx
        for j in range(n):
            if j >= i - 1 and j < i + block_size:
                continue
                
            # 插入到 j 和 j+1 之间
            # X(j) -> Y(j+1)
            # 变为: X -> B...C -> Y
            
            j_next = j + 1
            if j_next >= n: j_next = 0
            
            X = tour[j]
            Y = tour[j_next]
            
            if not (np.isfinite(D[X, Y]) and np.isfinite(D[X, B]) and np.isfinite(D[C, Y])):
                continue
                
            # gain = remove_gain + XY - XB - CY
            gain = remove_gain + D[X, Y] - D[X, B] - D[C, Y]
            
            if gain > 1e-6:
                # 执行移动 (非 inplace，使用 buffer)
                # 构造新 tour
                new_tour = np.empty_like(tour)
                idx = 0
                
                # 复制 0..j
                # 注意 j 可能在 i 前或后，逻辑要通用
                # 我们遍历原 tour 的所有非 block 元素
                
                # 优化: 手动 loop
                for k in range(n):
                    # 跳过 block
                    if k >= i and k < i + block_size:
                        continue
                        
                    new_tour[idx] = tour[k]
                    idx += 1
                    
                    if k == j: # 在 j 后面插入 block
                        # 插入 block
                        for b_idx in range(i, i + block_size):
                            new_tour[idx] = tour[b_idx]
                            idx += 1
                            
                tour[:] = new_tour[:]
                return True
                
    return False


@njit(cache=True, fastmath=True)
def double_bridge_move(tour):
    """
    Double Bridge (4-Opt) "Kick".
    将路径切成 4 段 (A, B, C, D) 并重组为 (A, D, C, B)。
    注意：这是 ATSP 安全的操作，因为它不反转任何片段的方向。
    """
    n = len(tour)
    if n < 4: return tour
    
    # 随机选 4 个切点 (需排序)
    # 使用 np.random.choice 不支持 replace=False 在 numba 中有时会有问题，改用简单的采样
    # 简单策略: 均匀切分加扰动
    q = n // 4
    p1 = 1 + np.random.randint(0, q)
    p2 = p1 + 1 + np.random.randint(0, q)
    p3 = p2 + 1 + np.random.randint(0, q)
    
    if p3 >= n: p3 = n - 1
    
    # 构造新路径: [0...p1] + [p3...end] + [p2...p3] + [p1...p2]
    # 原序: A(0-p1) B(p1-p2) C(p2-p3) D(p3-end)
    # 新序: A D C B
    
    # 手动拼接以兼容 Numba
    new_tour = np.empty(n, dtype=tour.dtype)
    idx = 0
    
    # A
    for i in range(0, p1):
        new_tour[idx] = tour[i]; idx += 1
    # D
    for i in range(p3, n):
        new_tour[idx] = tour[i]; idx += 1
    # C
    for i in range(p2, p3):
        new_tour[idx] = tour[i]; idx += 1
    # B
    for i in range(p1, p2):
        new_tour[idx] = tour[i]; idx += 1
        
    return new_tour

@njit(cache=True, fastmath=True)
def _swap_segments_jit(tour, D):
    """
    尝试交换两个不相邻的片段 (Swap Segments)。
    这也是 3-Opt 的一种特例，且不反转方向 (ATSP Safe)。
    """
    n = tour.shape[0]
    # 随机选择两个片段 A 和 B
    # A: [i...i+L1], B: [j...j+L2]
    # 限制片段长度为 2-4，避免破坏太大
    L1 = np.random.randint(2, 5)
    L2 = np.random.randint(2, 5)
    
    # 随机起点
    i = np.random.randint(0, n - L1 - L2 - 2)
    j = np.random.randint(i + L1 + 1, n - L2)
    
    # 边界检查
    if i + L1 >= j: return False # 重叠
    
    # 原始连接:
    # ... -> [i-1] -> [i...i+L1-1] -> [i+L1] -> ... -> [j-1] -> [j...j+L2-1] -> [j+L2] -> ...
    #        PreA     BlockA          PostA          PreB     BlockB          PostB
    
    PreA = i - 1
    PostA = i + L1
    PreB = j - 1
    PostB = j + L2
    
    nodes = tour
    
    c_preA = nodes[PreA]; c_startA = nodes[i]; c_endA = nodes[i+L1-1]; c_postA = nodes[PostA]
    c_preB = nodes[PreB]; c_startB = nodes[j]; c_endB = nodes[j+L2-1]; c_postB = nodes[PostB]
    
    # 移除 4 条边: (PreA, StartA), (EndA, PostA), (PreB, StartB), (EndB, PostB)
    rm_cost = D[c_preA, c_startA] + D[c_endA, c_postA] + D[c_preB, c_startB] + D[c_endB, c_postB]
    
    # 增加 4 条边: (PreA, StartB), (EndB, PostA), (PreB, StartA), (EndA, PostB)
    add_cost = D[c_preA, c_startB] + D[c_endB, c_postA] + D[c_preB, c_startA] + D[c_endA, c_postB]
    
    if not np.isfinite(add_cost): return False
    
    if add_cost < rm_cost - 1e-6:
        # 执行交换
        # 构造新数组比较稳妥
        new_tour = np.empty_like(tour)
        idx = 0
        
        # 0 ... PreA
        for k in range(0, i):
            new_tour[idx] = tour[k]; idx += 1
            
        # Block B
        for k in range(j, j + L2):
            new_tour[idx] = tour[k]; idx += 1
            
        # PostA ... PreB
        for k in range(i + L1, j):
            new_tour[idx] = tour[k]; idx += 1
            
        # Block A
        for k in range(i, i + L1):
            new_tour[idx] = tour[k]; idx += 1
            
        # PostB ... End
        for k in range(j + L2, n):
            new_tour[idx] = tour[k]; idx += 1
            
        tour[:] = new_tour[:]
        return True
        
    return False

@njit(cache=True, fastmath=True)
def _candidate_or_opt_jit(tour, D, knn_idx, max_iters=100, dlb_mask=None):
    """
    P1: 候选列表驱动的 Or-Opt 局部搜索，集成 DLB 加速。
    
    Args:
        dlb_mask: Boolean array for Don't Look Bits. If None, it's ignored.
                  True means 'Don't Look' (Skip).
    """
    n = tour.shape[0]
    K = knn_idx.shape[1]
    
    # 构建位置索引
    pos = np.empty(n, np.int32)
    for i in range(n):
        pos[tour[i]] = i
    
    improved = False
    
    # 如果没传 DLB，就临时建一个全 False 的（全检查）
    use_dlb = (dlb_mask is not None)
    
    for _ in range(max_iters):
        found_in_try = False
        
        # 随机遍历顺序
        # 为了 DLB 有效，最好不要完全随机跳，而是有序遍历或随机偏移
        start = np.random.randint(0, n)
        
        for offset in range(n):
            u_idx = (start + offset) % n
            u = tour[u_idx]
            
            # DLB Check
            if use_dlb and dlb_mask[u]:
                continue
                
            prev_u = tour[(u_idx - 1) % n]
            next_u = tour[(u_idx + 1) % n]
            
            remove_cost = D[prev_u, u] + D[u, next_u]
            new_edge_cost = D[prev_u, next_u]
            
            if not np.isfinite(new_edge_cost): continue
            
            move_found = False
            
            # 搜索 KNN
            for k in range(K):
                target = knn_idx[u, k]
                if target == -1: break
                
                # 检查 target 是否在 DLB (Strict 模式: 如果 target 也没变过，也许不用插? 
                # 但通常只检查起点 u。这里不检查 target 的 DLB)

                t_idx = pos[target]
                next_t = tour[(t_idx + 1) % n]
                
                if target == u or next_t == u or target == prev_u: continue
                
                insert_cost = D[target, u] + D[u, next_t]
                old_edge_cost = D[target, next_t]
                
                if not np.isfinite(insert_cost): continue
                
                gain = (remove_cost - new_edge_cost) + (old_edge_cost - insert_cost)
                
                if gain > 1e-6:
                    # Apply Move
                    # ... (Move Logic same as before) ...
                    # 简化：因为 Python loop 还是慢，这里用切片逻辑如果能在 numba 里跑最好
                    # 手动移动:
                    
                    # 0. 备份受影响的点，用于解锁 DLB
                    affected = [prev_u, next_u, target, next_t, u]
                    
                    # 1. 删除 u
                    temp_tour = np.empty(n - 1, dtype=tour.dtype)
                    if u_idx > 0:
                        temp_tour[:u_idx] = tour[:u_idx]
                        temp_tour[u_idx:] = tour[u_idx+1:]
                    else:
                        temp_tour[:] = tour[1:]
                        
                    # 2. 找到 target 在 temp 中的位置
                    # 如果 target 在 u 后面，它的索引减了 1
                    t_idx_new = t_idx
                    if t_idx > u_idx: t_idx_new -= 1
                    
                    # 3. 在 target (t_idx_new) 后面插入 u
                    # new tour len = n
                    # [0 ... t_idx_new] + [u] + [t_idx_new+1 ... end]
                    
                    tour[:t_idx_new+1] = temp_tour[:t_idx_new+1]
                    tour[t_idx_new+1] = u
                    tour[t_idx_new+2:] = temp_tour[t_idx_new+1:]
                    
                    # Rebuild pos (Slow? incremental update is better but complex)
                    for i in range(n): pos[tour[i]] = i
                    
                    move_found = True
                    improved = True
                    
                    # Unlock DLB for affected cities
                    if use_dlb:
                        for ac in affected:
                            dlb_mask[ac] = False
                            
                    break # Break KNN loop
            
            if move_found:
                found_in_try = True
            else:
                # 只有当彻底搜索了所有邻居都没找到移动，才 Lock u
                if use_dlb:
                    dlb_mask[u] = True
                    
        if not found_in_try and use_dlb:
            # 如果一整圈都没动静，说明陷入极值，直接退出
            break
            
    return improved


@njit(cache=True, fastmath=True)
def _rcl_nn_tour_jit(D, finite_mask, knn_idx, r):
    """
    基于 RCL (Restricted Candidate List) 的最近邻构造初始化。
    """
    n = D.shape[0]
    tour = np.empty(n, np.int32)
    used = np.zeros(n, np.uint8)
    
    cur = np.random.randint(0, n)  # 随机起点
    tour[0] = cur; used[cur] = 1
    
    K = knn_idx.shape[1]
    
    for t in range(1, n):
        # 1. 从 KNN 中收集可行且未访问的邻居
        tmp_idx = np.empty(K, np.int32)
        tmp_dis = np.empty(K, np.float64)
        cnt = 0
        for k in range(K):
            j = knn_idx[cur, k]
            if j == -1: continue
            if used[j] == 1: continue
            if not finite_mask[cur, j]: continue
            
            tmp_idx[cnt] = j
            tmp_dis[cnt] = D[cur, j]
            cnt += 1
            
        # 2. 如果 KNN 中没有合适邻居，退化处理
        if cnt == 0:
            pool_cnt = 0
            for j in range(n):
                if used[j] == 0 and finite_mask[cur, j]:
                    pool_cnt += 1
            
            if pool_cnt == 0:  # 彻底死胡同，随机选一个未访问的点跳过去（不可行边）
                nxt = 0
                for j in range(n):
                    if used[j] == 0:
                        nxt = j; break
            else:
                # 从全局可行邻居中随机选一个
                pool = np.empty(pool_cnt, np.int32)
                c = 0
                for j in range(n):
                    if used[j] == 0 and finite_mask[cur, j]:
                        pool[c] = j; c += 1
                nxt = int(pool[np.random.randint(0, pool_cnt)])
        else:
            # 3. 构建 RCL 并随机选择
            order = np.argsort(tmp_dis[:cnt])
            rsize = r if r < cnt else cnt
            pick = order[np.random.randint(0, rsize)]
            nxt = int(tmp_idx[pick])
            
        tour[t] = nxt; used[nxt] = 1; cur = nxt
        
    # 尝试修补闭环
    if not finite_mask[tour[n - 1], tour[0]]:
        for _ in range(20):
            if _two_opt_once_jit_safe(tour, D):
                if finite_mask[tour[n - 1], tour[0]]:
                    break
            else:
                break
    return tour


@njit(cache=True, fastmath=True)
def _insertion_tour_jit(D, finite_mask, use_farthest):
    """
    插入法构造初始化。支持“最远插入”和“随机插入”。
    """
    n = D.shape[0]
    # 初始两点回路
    a = np.random.randint(0, n)
    b = a
    for _ in range(16):
        b = np.random.randint(0, n)
        if b != a and finite_mask[a, b] and finite_mask[b, a]:
            break
            
    tour = np.empty(2, np.int32); tour[0] = a; tour[1] = b
    used = np.zeros(n, np.uint8); used[a] = 1; used[b] = 1
    m = 2
    
    while m < n:
        # 1. 选择待插入城市
        if use_farthest:
            best_city = -1; best_score = -1.0
            for c in range(n):
                if used[c] == 1: continue
                # 寻找距离当前回路最近点的最大值（max-min）
                mind = 1e100
                for t in range(m):
                    if finite_mask[c, tour[t]] and D[c, tour[t]] < mind:
                        mind = D[c, tour[t]]
                if mind > best_score:
                    best_score = mind; best_city = c
            insert_city = best_city if best_city != -1 else np.random.randint(0, n)
        else:
            # 随机选择第 k 个未访问城市
            remain = n - m
            k = np.random.randint(0, remain)
            idx = -1
            for c in range(n):
                if used[c] == 0:
                    if k == 0: idx = c; break
                    k -= 1
            insert_city = idx
            
        # 2. 选择最佳插入位置（最小增加成本）
        best_pos = -1; best_cost = 1e100
        for i in range(m):
            prev = tour[i - 1] if i > 0 else tour[m - 1]
            curr = tour[i]
            if finite_mask[prev, insert_city] and finite_mask[insert_city, curr]:
                cost = D[prev, insert_city] + D[insert_city, curr] - D[prev, curr]
                if cost < best_cost:
                    best_cost = cost; best_pos = i
                    
        # 3. 执行插入
        newtour = np.empty(m + 1, np.int32)
        if best_pos == -1:
            pos = np.random.randint(0, m + 1)
            for i in range(pos): newtour[i] = tour[i]
            newtour[pos] = insert_city
            for i in range(pos, m): newtour[i + 1] = tour[i]
        else:
            for i in range(best_pos): newtour[i] = tour[i]
            newtour[best_pos] = insert_city
            for i in range(best_pos, m): newtour[i + 1] = tour[i]
            
        tour = newtour; m += 1; used[insert_city] = 1
        
    # 尝试修补闭环
    for _ in range(20):
        if finite_mask[tour[m - 1], tour[0]]: break
        if not _two_opt_once_jit_safe(tour, D): break
        
    return tour


@njit(cache=True, fastmath=True)
def bond_distance_jit(t1, t2):
    """
    计算两个路径之间的Bond Distance（差异边数量）。
    Bond Distance = N - |Edges(t1) ∩ Edges(t2)|
    用于 Deterministic Crowding 中的距离度量。
    """
    n = t1.shape[0]
    # 构建 t2 的边查找表 (或直接索引查找)
    # 为了 O(N) 效率，先构建 t2 的位置映射 pos2[city] = index
    pos2 = np.empty(n, np.int32)
    for i in range(n):
        pos2[t2[i]] = i
        
    shared_edges = 0
    for i in range(n):
        u = t1[i]
        v = t1[(i + 1) % n]
        
        # 在 t2 中检查是否存在边 (u, v)
        # u 在 t2 中的位置
        idx_u = pos2[u]
        
        # u 在 t2 中的左右邻居
        left = t2[(idx_u - 1) % n]
        right = t2[(idx_u + 1) % n]
        
        if v == left or v == right:
            shared_edges += 1
            
    return n - shared_edges


@njit(cache=True, fastmath=True, parallel=True)
def calc_diversity_metrics_jit(pop, best_tour):
    """
    计算多样性指标：
    1. avg_dist: 种群与当前最优解的平均 Bond Distance
    2. entropy: 种群的边熵 (Edge Entropy)
    """
    m, n = pop.shape
    
    # --- 1. Average Bond Distance to Best ---
    total_dist = 0.0
    for i in prange(m):
         # bond_distance_jit 计算的是“不一样的边数”
        total_dist += bond_distance_jit(pop[i], best_tour)
    avg_dist = total_dist / m
    
    # --- 2. Edge Entropy ---
    # counts[i, j] 记录边 i->j 出现的次数
    # 空间 O(N^2)
    counts = np.zeros((n, n), dtype=np.int32)
    
    # 填充计数矩阵 (并行化需小心，这里用串行或原子操作，为简单起见用简单的循环)
    # 为保证效率，我们在外层串行，内层遍历边
    for i in range(m):
        t = pop[i]
        for j in range(n):
            u = t[j]
            v = t[(j + 1) % n]
            counts[u, v] += 1
            counts[v, u] += 1 # 无向图视角
            
    # 计算熵
    entropy = 0.0
    for i in range(n):
        for j in range(i + 1, n): # 只算上三角
            c = counts[i, j]
            if c > 0:
                p = c / m
                entropy -= p * np.log(p)
                
    return avg_dist, entropy


@njit(cache=True, fastmath=True)
def rtr_challenge_jit(child, child_fit, pop, fit, W, rng_seed, best_idx):
    """
    RTR (Restricted Tournament Replacement) 带多样性豁免 + 精英保护
    
    新增逻辑:
    1. 如果 child 与 target 距离足够远，即使 fitness 稍差也允许替换
    2. 永远不允许替换种群中的最优个体 (best_idx 由外部预计算传入)
    """
    m = pop.shape[0]
    n = child.shape[0]  # 城市数量
    np.random.seed(rng_seed)
    
    # 1. 随机选择 W 个窗口个体
    window_indices = np.random.choice(m, size=W, replace=False)
    
    # 2. 找到窗口中最像 child 的那个 (Bond Distance 最小)
    closest_idx = -1
    min_dist = 99999999
    
    for idx in window_indices:
        dist = bond_distance_jit(child, pop[idx])
        if dist < min_dist:
            min_dist = dist
            closest_idx = idx
            
    # 3. 竞争 (带多样性豁免 + 精英保护)
    target_idx = closest_idx
    target_fit = fit[target_idx]
    
    # === 精英保护: best_idx 由外部传入，避免 O(m) 开销 ===
    if target_idx == best_idx:
        return False, target_idx  # 保护精英
    
    better = False
    
    # 逻辑 A: 硬实力更强 (直接胜出)
    if child_fit < target_fit:
        better = True
    
    # 逻辑 B: 多样性豁免 (Diversity Exemption)
    # 如果距离超过 15% (n * 0.15)，且 fitness 差距在 10% 以内，允许替换
    else:
        threshold_dist = n * 0.15  # 750 城市时，要求至少 112 条边不同
        relax_factor = 1.05        # 允许差 10%
        
        if min_dist > threshold_dist and child_fit < target_fit * relax_factor:
            better = True
        
    return better, target_idx


@njit(cache=True, nogil=True)
def _ruin_and_recreate_jit(tour: np.ndarray, D: np.ndarray, ruin_pct: float) -> np.ndarray:
    """
    Ruin & Recreate (LNS Operator) - ATSP Optimized
    1. Ruin: 随机移除一段连续路径 (Segment Removal)
    2. Recreate: 贪婪最佳插入 (Cheapest Insertion)
    """
    n = len(tour)
    n_remove = int(n * ruin_pct)
    if n_remove < 2: 
        return tour.copy()
    
    # --- 1. Ruin: Remove Segment ---
    # 随机选择起始点
    start_idx = np.random.randint(0, n)
    
    # 提取被移除的城市 (注意处理循环边界)
    removed_cities = np.empty(n_remove, dtype=np.int32)
    kept_cities = np.empty(n - n_remove, dtype=np.int32)
    
    r_ptr = 0
    k_ptr = 0
    
    # 简单的循环遍历来分离城市
    end_idx = (start_idx + n_remove) % n
    
    if start_idx < end_idx:
        # 移除的是中间一段 [start, end)
        kept_cities[:start_idx] = tour[:start_idx]
        kept_cities[start_idx:] = tour[end_idx:]
        removed_cities[:] = tour[start_idx:end_idx]
    else:
        # 移除的是跨越边界的一段
        kept_cities[:] = tour[end_idx:start_idx]
        k = n - start_idx
        removed_cities[:k] = tour[start_idx:]
        removed_cities[k:] = tour[:end_idx]

    # --- 2. Recreate: Cheapest Insertion ---
    current_tour = kept_cities
    np.random.shuffle(removed_cities) # Shuffle to randomize insertion order
    
    for city in removed_cities:
        best_delta = 1e20
        best_pos = -1
        m = len(current_tour)
        
        # 寻找最佳插入位置 i: 插入到 u(i) -> v(i+1) 之间
        # Delta = D[u, c] + D[c, v] - D[u, v]
        
        for i in range(m):
            u = current_tour[i]
            v = current_tour[(i + 1) % m]
            
            delta = D[u, city] + D[city, v] - D[u, v]
            if delta < best_delta:
                best_delta = delta
                best_pos = i
        
        # 执行插入 (Rebuild array)
        new_tour = np.empty(m + 1, dtype=np.int32)
        new_tour[:best_pos+1] = current_tour[:best_pos+1]
        new_tour[best_pos+1] = city
        new_tour[best_pos+2:] = current_tour[best_pos+1:]
        current_tour = new_tour

    return current_tour


@njit(cache=True, fastmath=True)
def _double_bridge_jit(tour):
    """
    Double Bridge 扰动操作 (4-Opt 变体)。
    将 tour 切成 4 段并重新拼接，产生 2-Opt 无法达到的结构变化。
    这是 LKH 算法的核心组件之一。
    """
    n = tour.shape[0]
    if n < 8:
        return  # 城市太少，不适用
    
    # 随机选择 4 个切点 p1 < p2 < p3 < p4
    # 确保每段至少有 1 个城市
    p1 = np.random.randint(1, n // 4)
    p2 = np.random.randint(p1 + 1, n // 2)
    p3 = np.random.randint(p2 + 1, 3 * n // 4)
    p4 = n  # 最后一个切点就是数组末尾
    
    # 原始 4 段: A=[0:p1], B=[p1:p2], C=[p2:p3], D=[p3:n]
    # Double Bridge 重组: A + C + B + D (交换 B 和 C)
    new_tour = np.empty(n, dtype=np.int32)
    idx = 0
    
    # 段 A
    for i in range(0, p1):
        new_tour[idx] = tour[i]
        idx += 1
    # 段 C
    for i in range(p2, p3):
        new_tour[idx] = tour[i]
        idx += 1
    # 段 B
    for i in range(p1, p2):
        new_tour[idx] = tour[i]
        idx += 1
    # 段 D
    for i in range(p3, n):
        new_tour[idx] = tour[i]
        idx += 1
        
    tour[:] = new_tour[:]


@njit(cache=True, parallel=True)
def init_population_jit(pop, D, finite_mask, knn_idx, strat_probs, seeds, rcl_r):
    """
    并行生成初始种群，混合多种构造策略。
    使用 prange 实现真正的并行初始化。
    """
    lam, n = pop.shape
    
    for i in prange(lam):
        np.random.seed(seeds[i])  # 设置线程局部随机种子
        u = np.random.rand()
        
        # 根据概率选择初始化策略
        if u < strat_probs[0]:
            # 策略1: RCL-NN (10%)
            tour = _rcl_nn_tour_jit(D, finite_mask, knn_idx, rcl_r)
        elif u < strat_probs[0] + strat_probs[1]:
            # 策略2: 插入法 (30%)
            tour = _insertion_tour_jit(D, finite_mask, use_farthest=(np.random.rand() < 0.5))
        else:
            # 策略3: 随机 + 修复 (60%)
            tour = _rand_perm_jit(n)
            ok = _repair_jit(tour, D, finite_mask, 50)
            if not ok:
                 # 修复失败则退化为 RCL-NN
                tour = _rcl_nn_tour_jit(D, finite_mask, knn_idx, rcl_r)
                
        pop[i] = tour


# ==============================================================================
# Part 2: Main Solver Class (主求解类)
# ==============================================================================

class r0123456:
    """
    遗传算法 TSP 求解器类。
    包含了主要的进化循环、选择、变异等高层逻辑。
    """

    def __init__(self,
                 N_RUNS: int = 500,           # 总迭代代数
                 lam: int = 100,              # 种群规模 (Lambda)
                 mu: int = 100,               # 子代规模 (Mu)
                 k_tournament: int = 5,       # 锦标赛选择的 K 值
                 mutation_rate: float = 0.3,  # 变异概率
                 rng_seed: Optional[int] = None, # 随机种子
                 local_rate: float = 0.2,     # 局部搜索概率
                 ls_max_steps: int = 30,      # 局部搜索最大步数
                 stagnation_limit: int = 150, # 停滞阈值（用于灾变）
                 log_file: Optional[str] = None):  # 诊断日志文件路径
        
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        
        # 保存超参数
        self.N_RUNS = int(N_RUNS)
        self.lam = int(lam)
        self.mu = int(mu)
        self.k_tournament = int(k_tournament)
        self.mutation_rate = float(mutation_rate)
        self.rng = np.random.default_rng(rng_seed)  # 主随机生成器
        self.local_rate = float(local_rate)
        self.ls_max_steps = int(ls_max_steps)
        self.stagnation_limit = int(stagnation_limit)
        
        # 停滞检测状态
        self.best_ever_fitness = np.inf
        self.stagnation_counter = 0
        
        # 诊断日志
        self.log_file = log_file
        self._log_handle = None
        # Symmetry flag (initialized in optimize)
        self._is_symmetric = True

    def _run_lns_worker(self, D, n, mig_queue, recv_queue, island_id):
        """
        [New] Trauma Center (LNS Worker) Logic
        Single-Trajectory Heterogeneous Scout.
        """
        # Init 32 nearest neighbors
        finite_mask = ~np.isinf(D)
        np.fill_diagonal(D, np.inf)
        knn_idx = np.argsort(D, axis=1)[:, :32].astype(np.int32)
        
        # 1. Init Solution (Greedy)
        current_tour = _rcl_nn_tour_jit(D, finite_mask, knn_idx, 3) # Strict greedy
        current_fit = tour_length_jit(current_tour, D)
        best_tour = current_tour.copy()
        best_fit = current_fit
        
        # DLB Mask
        dlb_mask = np.zeros(n, dtype=np.bool_)
        
        # Loop Vars
        iter_count = 0
        accepted_count = 0
        received_count = 0
        sent_count = 0
        
        print(f"[Scout LNS] Started. Initial Fit: {best_fit:.2f}")
        
        start_time = time.time()
        
        # Log handle setup
        log_h = None
        if self.log_file:
            log_h = open(self.log_file, 'w', encoding='utf-8')
            log_h.write("gen,best_fit,mean_fit,diversity,entropy,stagnation,migration,repulsion,rtr_accepts\n")
        
        while True:
            iter_count += 1
            migration_event = 0 # 1=Recv
            repulsion_event = 0 # 1=Sent
            
            # --- A. Check Incoming Patients (from Exploiter) ---
            if recv_queue is not None:
                try:
                    patient = recv_queue.get(block=False)
                    p_fit = tour_length_jit(patient, D)
                    current_tour[:] = patient[:]
                    current_fit = p_fit
                    dlb_mask[:] = False 
                    received_count += 1
                    migration_event = 1
                    if received_count % 50 == 0:
                         print(f"[Scout LNS] Received Patient #{received_count}. Fit: {p_fit:.2f}")
                except:
                    pass

            # --- B. Ruin & Recreate (Kick) ---
            # 20% Ruin Rate usually good for LNS
            candidate = _ruin_and_recreate_jit(current_tour, D, 0.20)
            
            # Reset DLB for candidate (major structural change)
            dlb_mask[:] = False 
            
            # --- C. Local Search (Rehabilitation) ---
            # Deep Polish with DLB
            _candidate_or_opt_jit(candidate, D, knn_idx, max_iters=500, dlb_mask=dlb_mask)
            cand_fit = tour_length_jit(candidate, D)
            
            # --- D. Acceptance & Update ---
            # 1. Global Best Update (Miracle Recovery)
            if cand_fit < self.best_ever_fitness:
                # Update Local Best
                self.best_ever_fitness = cand_fit
                best_tour[:] = candidate[:]
                best_fit = cand_fit
                
                # Report
                solution_to_report = self._rotate_to_start(best_tour, 0)
                timeLeft = self.reporter.report(best_fit, best_fit, solution_to_report)
                if timeLeft < 0: break
                
                # Send back to Exploiter immediately
                if mig_queue is not None:
                    try:
                        mig_queue.put(best_tour.copy(), block=False)
                        sent_count += 1
                        repulsion_event = 1
                        print(f"[Scout LNS] MIRACLE! Sent healed solution {best_fit:.2f} to Exploiter!")
                    except:
                        pass
            
            # 2. Local Acceptance (Current Solution Update)
            # Accept if better or equal (Side-step allowed)
            if cand_fit <= current_fit:
                current_tour[:] = candidate[:]
                current_fit = cand_fit
                accepted_count += 1
            else:
                pass

            if log_h and (iter_count % 500 == 0 or migration_event or repulsion_event):
                row = f"{iter_count},{best_fit:.4f},{current_fit:.4f},0,0,0,{migration_event},{repulsion_event},0\n"
                log_h.write(row)
                if iter_count % 5000 == 0: log_h.flush()

            # Time Check (Interval)
            if iter_count % 100 == 0:
                elapsed = time.time() - self.reporter.startTime
                if self.reporter.allowedTime > 0 and elapsed > self.reporter.allowedTime:
                    break
        
        if log_h: log_h.close()
        print(f"[Scout LNS] Finished. Recv: {received_count}, Sent: {sent_count}")
        return 0

    def optimize(self, filename: str, mig_queue=None, recv_queue=None, island_id=0):
        """
        主优化流程：读取文件 -> 初始化 -> 进化循环 -> 上报结果
        mig_queue, recv_queue: 用于岛屿模型的多进程通信 (Optional)
        """
        # 1. 读取数据
        with open(filename) as file:
            D = np.loadtxt(file, delimiter=",", dtype=np.float64, ndmin=2)
        n = D.shape[0]
        D = np.ascontiguousarray(D)  # 内存连续化优化

        # 1.5 检查距离矩阵是否对称 (P0: Asymmetry Check)
        is_symmetric = np.allclose(D, D.T, rtol=1e-5, atol=1e-8, equal_nan=True)
        self._is_symmetric = is_symmetric
        if not is_symmetric:
            print(f"[Island {island_id}] [Warning] Asymmetric TSP detected! Using Or-Opt/Insertion logic.")
            
        # Dispatch to LNS Worker if Island 1 (Configured as Scout/Doctor)
        # Note: This hardcodes Island 1 as the LNS worker. 
        # In a more flexible system we would pass a mode flag, but for this task this is sufficient.
        if island_id == 1:
            return self._run_lns_worker(D, n, mig_queue, recv_queue, island_id)
            
        # --- Normal GA Flow (Exploiter) ---
        # If not LNS, execution continues here...
        pass

        # 2. 预处理可行性掩码 (True 代表边存在/距离有限)
        finite_mask = np.isfinite(D)
        np.fill_diagonal(finite_mask, False)

        # 3. 初始化种群容器
        population = np.empty((self.lam, n), dtype=np.int32)
        
        # 4. 构建 KNN 索引 (用于加速初始化)
        K = 32
        print(f"[Island {island_id}] [Init] Building KNN (K={K}) for {n} cities...")
        knn_idx = build_knn_idx(D, finite_mask, K)
        self._knn_idx = knn_idx  # 保存引用
        print(f"[Island {island_id}] [Init] KNN ready.")
        
        # --- DLB Mask Initialization ---
        dlb_mask = np.zeros(n, dtype=np.bool_)

        # 5. 生成初始种群 (并行)
        strat_probs = np.array([0.1, 0.3, 0.6], dtype=np.float64) # RCL / Insert / Rand
        seeds = np.empty(self.lam, dtype=np.int64)
        for i in range(self.lam):
            seeds[i] = int(self.rng.integers(1 << 30))
        rcl_r = int(self.rng.integers(3, 11))
        
        print(f"[Init] Generating population (lambda={self.lam})...")
        init_population_jit(population, D, finite_mask, knn_idx, strat_probs, seeds, rcl_r)
        print("[Init] Population ready.")

        # 6. 初始适应度评估
        fitness = np.empty(self.lam, dtype=np.float64)
        batch_lengths_jit(population, D, fitness)

        # 7. RTR (Restricted Tournament Replacement) Setup
        print(f"[Info] Using Restricted Tournament Replacement (RTR). W=50.")
        
        # 预分配 offspring buffer (batch size = lam or smaller chunk)
        # 为方便与 JIT 配合并保持逻辑清晰，我们生成一整代子代，然后逐个进行 RTR 竞争
        # 注意：RTR 通常是稳态的 (steady-state)，即生成一个插入一个。
        # 但为了利用 JIT 并行繁殖，我们采用 "Batch RTR"：
        # 1. 并行生成 lambda 个子代
        # 2. 串行（或部分并行）将子代尝试插入种群
        
        # 子代缓冲区
        c_pop = np.empty((self.lam, n), dtype=np.int32)
        c_fit = np.empty(self.lam, dtype=np.float64)
        
        # 父代索引 (用于繁殖选择)
        indices = np.arange(self.lam, dtype=np.int32)
        
        # 窗口大小 (Updated to 50 for better diversity maintenance)
        W = min(self.lam, 50)
        
        # --- 诊断日志初始化 ---
        if self.log_file:
            self._log_handle = open(self.log_file, 'w', encoding='utf-8')
            # CSV Header
            self._log_handle.write("gen,best_fit,mean_fit,diversity,entropy,stagnation,migration,repulsion,rtr_accepts\n")
            self._log_handle.flush()
        
        # 运行时追踪变量
        _migration_this_gen = False
        _repulsion_this_gen = False
        _rtr_accepts = 0
        _last_send_time = time.time()  # 基于时间的发送间隔追踪
        
        print("Gen | Best Cost | Avg Bonds | Edge Ent", flush=True)

        # ==================== 进化循环 (Evolution Loop: RTR) ====================
        for gen in range(1, self.N_RUNS + 1):
            
            # --- A. 繁殖 (Reproduction) ---
            # 使用锦标赛选择父代 (Tournament Selection)以增加选择压力
            # RTR 的替换已有压力，但在繁殖端施加压力能加速优良基因的扩散
            
            for i in range(0, self.lam - 1, 2):
                idx1 = self._k_tournament_idx(fitness, self.k_tournament)
                idx2 = self._k_tournament_idx(fitness, self.k_tournament)
                
                # P2: 使用 OX 交叉 (保留连续子路径结构)
                c1 = _ox_jit(population[idx1], population[idx2])
                c2 = _ox_jit(population[idx2], population[idx1])
                
                # Mutation (Check Symmetry!)
                if self.rng.random() < self.mutation_rate:
                    self._hybrid_mutation_inplace(c1)
                if self.rng.random() < self.mutation_rate:
                    self._hybrid_mutation_inplace(c2)
                    
                # P3 改动：移除了此处的按概率 LS，改为后面的精英优先 LS
                    
                # Repair
                self._repair_inplace(c1, D, finite_mask)
                self._repair_inplace(c2, D, finite_mask)
                
                c_pop[i] = c1
                c_pop[i+1] = c2
            
            # Evaluation (Batch) - 先评估所有子代
            batch_lengths_jit(c_pop, D, c_fit)
            
            # --- P3: Elite-Only Local Search (精英优先 LS) ---
            # Exploiter: 强力 LS (20% 精英, 完整步数)
            # Explorer (Scout): 强力 LS (Scout 模式下也要保证质量)
            
            # 简化策略: 所有岛屿都对前 20% 精英做 LS (保证 Scout 也产出高质量解)
            # 区别在于 Scout 会频繁重启，Exploiter 持续深挖
            elite_ratio = 0.2
            elite_count = max(1, int(self.lam * elite_ratio))
            elite_indices = np.argsort(c_fit)[:elite_count]
            
            for idx in elite_indices:
                # 必须为每个个体分别重置 DLB mask，否则上一个体的 bits 会污染下一个
                dlb_mask[:] = False
                # 使用 DLB 加速 Or-Opt
                _candidate_or_opt_jit(c_pop[idx], D, knn_idx, max_iters=self.ls_max_steps, dlb_mask=dlb_mask)
            
            for idx in elite_indices:
                c_fit[idx] = tour_length_jit(c_pop[idx], D)
            
            # --- B. RTR Replacement ---
            # 逐个子代尝试替换
            # 由于需要修改 population，这里串行执行最为安全
            # 如果 bottleneck，可以将整个循环 JIT 化
            
            # 预计算 best_idx，避免在 RTR 中重复计算 O(m) 次
            current_best_idx = np.argmin(fitness)
            
            for i in range(self.lam):
                # 调用 JIT 函数寻找窗口中最像的个体并决定是否替换
                # 为了随机性，我们需要传入随机种子
                seed = int(self.rng.integers(0, 1<<30))
                
                better, target_idx = rtr_challenge_jit(
                    c_pop[i], c_fit[i], population, fitness, W, seed, current_best_idx
                )
                
                if better:
                    population[target_idx][:] = c_pop[i][:]
                    fitness[target_idx] = c_fit[i]
                    _rtr_accepts += 1  # 诊断日志：记录 RTR 接受数

            # --- C. Kick Strategy (Double Bridge) ---
            # 如果即将陷入停滞 (Stagnation > 50% Limit)，尝试对最差个体注入"踢过的精英"，试图打破僵局
            if self.stagnation_counter > self.stagnation_limit * 0.5:
                 if gen % 10 == 0: 
                     current_best = np.argmin(fitness)
                     kick_cand = population[current_best].copy()
                     kick_cand = double_bridge_move(kick_cand)
                     # 踢完简单修复
                     dlb_mask[:] = False
                     _candidate_or_opt_jit(kick_cand, D, knn_idx, max_iters=50, dlb_mask=dlb_mask)
                     k_fit = tour_length_jit(kick_cand, D)
                     
                     # 替换当前最差
                     worst_idx = np.argmax(fitness)
                     population[worst_idx][:] = kick_cand[:]
                     fitness[worst_idx] = k_fit

            # --- D. Migration (Scout Model) ---
            # Pre-calc best_idx for export
            best_idx = np.argmin(fitness)
            
            # --- D. Migration (Trauma Center Model) ---
            # Exploiter (Island 0) Logic (Scout Logic is in _run_lns_worker)
            
            current_time = time.time()
            if not hasattr(self, '_last_patient_sent'): self._last_patient_sent = 0.0

            if mig_queue is not None:
                # 1. Send "Patient" (Stagnant Best) to Scout
                # Trigger: Stagnation > Limit / 2 (Warning Phase)
                if self.stagnation_counter > (self.stagnation_limit // 2):
                    if current_time - self._last_patient_sent > 5.0: # 5 sec cooldown
                        self._last_patient_sent = current_time
                        best_idx = np.argmin(fitness) 
                        patient = population[best_idx].copy() 
                        try:
                            mig_queue.put(patient, block=False)
                        except:
                            pass

            if recv_queue is not None:
                # 2. Receive "Healed" Solution from Scout
                imported_count = 0
                while True:
                    try:
                        healed = recv_queue.get(block=False)
                        h_fit = tour_length_jit(healed, D)
                        
                        # Unconditional Acceptance: Replace Worst
                        worst_idx = np.argmax(fitness)
                        population[worst_idx][:] = healed[:]
                        fitness[worst_idx] = h_fit
                        imported_count += 1
                        
                        # Update global best if miracle
                        if h_fit < self.best_ever_fitness:
                            self.best_ever_fitness = h_fit
                            self.stagnation_counter = 0 # Reset stagnation
                            print(f"[Exploiter] MIRACLE! Received Healed Solution {h_fit:.2f} (New Best)")
                        else:
                            pass
                    except:
                        break # Queue empty
                
                if imported_count > 0:
                    _migration_this_gen = True
                    # print(f"[Island {island_id}] Imported {imported_count}.")

            # --- E. Reporting & Metrics ---
            # best_idx 已经在上面更新过
            best_idx = np.argmin(fitness)
            bestObjective = float(fitness[best_idx])
            meanObjective = float(fitness.mean())
            bestSolution = self._rotate_to_start(population[best_idx].copy(), 0)
            
            # 记录 best_ever
            if bestObjective < self.best_ever_fitness:
                self.best_ever_fitness = bestObjective
                self.stagnation_counter = 0
                
                # New Best! 计算多样性并打印
                div_dist, div_ent = calc_diversity_metrics_jit(population, population[best_idx])
                print(f"Gen {gen:4d} | Best: {bestObjective:.2f} | Div: {div_dist:.1f} | Ent: {div_ent:.3f} (NEW BEST)")
                _diversity_computed = True  # 标记已计算
                
            else:
                self.stagnation_counter += 1
                _diversity_computed = False
                
                # 定期日志 (每 50 代)
                if gen % 50 == 0:
                    div_dist, div_ent = calc_diversity_metrics_jit(population, population[best_idx])
                    print(f"Gen {gen:4d} | Best: {bestObjective:.2f} | Div: {div_dist:.1f} | Ent: {div_ent:.3f}")
                    _diversity_computed = True
            
            # --- Call Reporter (Time Check & CSV Log) ---
            # 必须每代调用，以检查时间并记录 CSV
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            
            # --- 诊断日志: 写入 CSV 行 ---
            if self._log_handle:
                # 只有在需要时才计算多样性 (每 50 代或 new best 时已算过)
                if not _diversity_computed:
                    # 非关键代只用 0 填充，避免昂贵的 entropy 计算
                    div_dist, div_ent = 0.0, 0.0
                row = f"{gen},{bestObjective:.4f},{meanObjective:.4f},{div_dist:.2f},{div_ent:.4f},{self.stagnation_counter},{int(_migration_this_gen)},{int(_repulsion_this_gen)},{_rtr_accepts}\n"
                self._log_handle.write(row)
                if gen % 50 == 0:  # 每 50 代 flush 一次
                    self._log_handle.flush()
                # 重置每代计数器
                _migration_this_gen = False
                _repulsion_this_gen = False
                _rtr_accepts = 0
            
            if timeLeft < 0:
                print(f"[Island {island_id}] Time limit reached. Stopping.")
                break

            # --- Stagnation & Restart ---
            if self.stagnation_counter >= self.stagnation_limit:
                print(f"!! STAGNATION ({self.stagnation_counter} gens) -> RESTART !!")
                
                # Scout (Island 1) 在重启前发送 Best 给 Exploiter
                if island_id == 1 and mig_queue is not None:
                    best_tour_now = population[best_idx].copy()
                    # Deep polish (Use DLB)
                    dlb_mask[:] = False
                    _candidate_or_opt_jit(best_tour_now, D, knn_idx, max_iters=500, dlb_mask=dlb_mask)
                    try:
                        mig_queue.put(best_tour_now, block=True, timeout=1.0)
                        print(f"[Scout] RESTART BEQUEST SENT.")
                    except:
                        pass

                best_tour_ever = population[best_idx].copy()
                
                seeds = np.random.randint(0, 1<<30, self.lam).astype(np.int64)
                rcl_r = int(self.rng.integers(3, 11))
                init_population_jit(population, D, finite_mask, knn_idx, strat_probs, seeds, rcl_r)
                
                population[0] = best_tour_ever
                batch_lengths_jit(population, D, fitness)
                
                self.stagnation_counter = 0
                self.best_ever_fitness = fitness.min()
                
                # 重启后强制打印一次
                print(f"-- Restart Complete. Best kept: {self.best_ever_fitness:.2f} --")

            # Time check done above

        
        # ==================== Final Stage ====================
        # Removed Polish (Double Bridge + 2-Opt) as requested.
        # Just report the final result one last time.
        
        best_idx = np.argmin(fitness)
        bestObjective = float(fitness[best_idx])
        bestSolution = self._rotate_to_start(population[best_idx].copy(), 0)
        
        print(f"[Island {island_id}] Finished. Final Best: {bestObjective:.2f}")
        self.reporter.report(bestObjective, bestObjective, bestSolution)

        # --- 关闭诊断日志 ---
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None
              
        return 0
    # -------- 辅助方法 (Helpers) --------

    def _k_tournament_idx(self, fitness: np.ndarray, k: int) -> int:
        """锦标赛选择，返回被选中个体的索引"""
        k = max(1, min(k, fitness.shape[0]))
        cand = self.rng.choice(fitness.shape[0], size=k, replace=False)
        best_local = np.argmin(fitness[cand])
        return cand[best_local]

    def _erx(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """调用 JIT 版 ERX 交叉"""
        return _erx_jit(p1, p2)

    def _hybrid_mutation_inplace(self, tour: np.ndarray) -> None:
        """混合变异：70% 反转 (Inversion), 30% 插入 (Insertion)"""
        n = tour.shape[0]
        
        # P0: Asymmetry Check
        use_inversion = True
        if not self._is_symmetric:
            use_inversion = False # Force Insertion
            
        if use_inversion and self.rng.random() < 0.7:
            # 反转变异 [i, j)
            i = int(self.rng.integers(0, n - 1))
            j = int(self.rng.integers(i + 1, n))
            # NumPy 切片反转
            tour[i:j] = tour[i:j][::-1]
        else:
            # 插入变异
            i = int(self.rng.integers(0, n)) # 移谁
            j = int(self.rng.integers(0, n - 1)) # 移到哪
            if j >= i: j += 1
            if i != j:
                city = tour[i]
                if j < i: # 移到前面
                    tour[j+1:i+1] = tour[j:i]
                    tour[j] = city
                else: # 移到后面
                    tour[i:j] = tour[i+1:j+1]
                    tour[j] = city

    def _light_two_opt_inplace(self, tour: np.ndarray, D: np.ndarray, finite_mask: np.ndarray, max_steps: int):
        """调用 JIT 版局部搜索 (混合 2-opt 和 Or-opt)"""
        # P0: If Asymmetric, DISABLE 2-Opt
        if not self._is_symmetric:
            bs = int(self.rng.integers(1, 4))
            _or_opt_once_jit(tour, D, bs)
            return

        steps = int(max_steps)
        for _ in range(steps):
             # 80% 概率做 2-opt, 20% 概率做 Or-opt
            if self.rng.random() < 0.8:
                if not _two_opt_once_jit_safe(tour, D):
                     # 如果 2-opt 失败，尝试 Or-opt 救一下
                    if not _or_opt_once_jit(tour, D, 3): # 尝试移动长度为 3 的块
                         break
            else:
                 # 随机选择块大小 1, 2, 3
                bs = int(self.rng.integers(1, 4))
                if not _or_opt_once_jit(tour, D, bs):
                    # 如果 Or-opt 失败，尝试 2-opt
                     if not _two_opt_once_jit_safe(tour, D):
                        break

    def _repair_inplace(self, tour: np.ndarray, D: np.ndarray, finite_mask: np.ndarray):
        """调用 JIT 版修复逻辑"""
        # P0: If Asymmetric, DISABLE 2-Opt based repair
        if not self._is_symmetric:
            return
            
        _repair_jit(tour, D, finite_mask)

    def _tour_feasible(self, tour: np.ndarray, finite_mask: np.ndarray) -> bool:
        """调用 JIT 版可行性检查"""
        return _tour_feasible_jit(tour, finite_mask)

    def _greedy_feasible_tour(self, D: np.ndarray, finite_mask: np.ndarray) -> np.ndarray:
        """生成一个可行解的兜底方法 (RCL-NN)"""
        try:
            r = 5
            return _rcl_nn_tour_jit(D, finite_mask, self._knn_idx, r)
        except Exception:
            # 极端情况 fallback
            n = D.shape[0]
            return _rand_perm_jit(n) 

    def _rotate_to_start(self, tour: np.ndarray, start_city: int) -> np.ndarray:
        """旋转路径使 start_city 位于首位 (用于标准化输出)"""
        pos = np.where(tour == start_city)[0]
        if pos.size > 0:
            idx = int(pos[0])
            if idx == 0: return tour
            return np.concatenate([tour[idx:], tour[:idx]])
        return tour


# ==============================================================================
# Part 3: Entry Point (脚本入口)
# ==============================================================================

if __name__ == "__main__":
    # 在此处配置超参数
    # csv 文件路径将在实例化后通过 .optimize() 方法传入
    
    # 示例配置 (对应 Csv250)
    config = {
        "N_RUNS": 10_000_000,    # 最大迭代代数
        "lam": 2000,           # 种群规模 (Lambda)
        "mu": 1500,             # 子代规模 (Mu)
        "k_tournament": 3,     # 锦标赛选择 K 值
        "mutation_rate": 0.5,   # 变异概率
        "local_rate": 0.2,      # 局部搜索概率
        "ls_max_steps": 30,     # 局部搜索最大步数
        "stagnation_limit": 200   # 停滞多少代触发重启 (RTR 需要耐心)
    }
    
    print("Running with config:", config)
    
    # 实例化求解器
    ea = r0123456(
        N_RUNS=config["N_RUNS"],
        lam=config["lam"],
        mu=config["mu"],
        k_tournament=config["k_tournament"],
        mutation_rate=config["mutation_rate"],
        local_rate=config["local_rate"],
        ls_max_steps=config["ls_max_steps"],
        stagnation_limit=config["stagnation_limit"]
    )
    
    # 运行优化 (请修改此处文件名以运行不同测试)
    target_csv = "tour500.csv" 
    if os.path.exists(target_csv):
        ea.optimize(target_csv)
    else:
        print(f"File {target_csv} not found. Please check the path.")
