import os

# Configure environment for strictly 1 thread per process (2 processes total)
# MUST BE SET BEFORE IMPORTING NUMBA/NUMPY to take effect reliably
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import Reporter
import numpy as np
import multiprocessing
import queue
import time
from numba import njit, prange, set_num_threads
from scipy.optimize import linear_sum_assignment

try:
    set_num_threads(1)
except:
    pass

# ==============================================================================
# JIT Accelerated Helper Functions (Aligned with Baseline Logic)
# ==============================================================================

@njit(cache=True, fastmath=True)
def _ox_jit_inplace(p1, p2, child):
    n = p1.size
    child[:] = -1
    cut1 = np.random.randint(0, n - 1)
    cut2 = np.random.randint(cut1 + 1, n)
    for i in range(cut1, cut2):
        child[i] = p1[i]
    used = np.zeros(n, np.uint8)
    for i in range(cut1, cut2):
        used[p1[i]] = 1
    idx = cut2
    for i in range(n):
        city = p2[(cut2 + i) % n]
        if used[city] == 0:
            child[idx % n] = city
            idx += 1
            used[city] = 1

@njit(cache=True, fastmath=True)
def tour_length_jit(tour, D):
    n = tour.shape[0]
    s = 0.0
    for i in range(n - 1):
        s += D[tour[i], tour[i+1]]
    s += D[tour[n-1], tour[0]]
    return s

@njit(cache=True, fastmath=True, parallel=True)
def batch_lengths_jit(pop2d, D, out):
    m, n = pop2d.shape
    for r in prange(m):
        s = 0.0
        row = pop2d[r]
        for i in range(n - 1):
            s += D[row[i], row[i+1]]
        s += D[row[n-1], row[0]]
        out[r] = s

@njit(cache=True, fastmath=True, parallel=True)
def build_knn_idx(D, finite_mask, K):
    n = D.shape[0]
    knn = np.full((n, K), -1, np.int32)
    for i in prange(n):
        row = D[i]
        valid_indices = np.where(finite_mask[i])[0]
        if valid_indices.size == 0: continue
        valid_distances = row[valid_indices]
        order = np.argsort(valid_distances)
        m = K if K < valid_indices.size else valid_indices.size
        for t in range(m):
            knn[i, t] = valid_indices[order[t]]
    return knn

def compute_alpha_candidates(D, finite_mask, K_alpha=30):
    """
    计算 α-nearness 候选边集（使用 Assignment Relaxation）
    
    α(i,j) = c_ij - u_i - v_j
    u_i, v_j 是对偶变量（势函数），近似使用行/列最小值
    
    输入:
        D: 距离矩阵 (n x n)
        finite_mask: 可行边掩码
        K_alpha: 每个点选 α 最小的 K_alpha 个候选
    
    输出:
        candidates: (n, K_alpha) 候选边数组
    """
    n = D.shape[0]
    
    # 处理 inf：用大值替代
    D_safe = D.copy()
    max_val = np.nanmax(D_safe[np.isfinite(D_safe)]) * 10 if np.any(np.isfinite(D_safe)) else 1e10
    D_safe[~np.isfinite(D_safe)] = max_val
    np.fill_diagonal(D_safe, max_val)  # 对角线设为大值
    
    # 计算势函数 u, v（简化：使用行/列最小值近似对偶变量）
    u = np.min(D_safe, axis=1)  # 每行最小
    v = np.min(D_safe, axis=0)  # 每列最小
    
    # 计算约化代价 α
    alpha = D_safe - u[:, None] - v[None, :]
    
    # 将 inf 位置的 α 设为大值
    alpha[~finite_mask] = max_val
    np.fill_diagonal(alpha, max_val)
    
    # 每个点选 α 最小的 K_alpha 个
    candidates = np.full((n, K_alpha), -1, dtype=np.int32)
    for i in range(n):
        order = np.argsort(alpha[i])
        count = 0
        for j in order:
            if count >= K_alpha:
                break
            if j != i and finite_mask[i, j]:
                candidates[i, count] = j
                count += 1
    
    return candidates

@njit(cache=True, fastmath=True)
def merge_candidates(knn, alpha_cand, D, max_size=50):
    """
    合并 KNN 候选和 α-nearness 候选，按距离排序
    
    输入:
        knn: (n, K1) KNN 候选
        alpha_cand: (n, K2) α-nearness 候选
        D: 距离矩阵
        max_size: 合并后每个点最多保留的候选数
    
    输出:
        merged: (n, max_size) 合并后的候选边（按距离排序）
    """
    n = knn.shape[0]
    K1 = knn.shape[1]
    K2 = alpha_cand.shape[1]
    
    merged = np.full((n, max_size), -1, dtype=np.int32)
    
    for i in range(n):
        # 收集所有候选（去重）
        seen = np.zeros(n, dtype=np.uint8)
        temp_cand = np.empty(K1 + K2, dtype=np.int32)
        temp_dist = np.empty(K1 + K2, dtype=np.float64)
        count = 0
        
        # 添加 KNN 候选
        for k in range(K1):
            j = knn[i, k]
            if j != -1 and seen[j] == 0:
                seen[j] = 1
                temp_cand[count] = j
                temp_dist[count] = D[i, j]
                count += 1
        
        # 添加 α-nearness 候选
        for k in range(K2):
            j = alpha_cand[i, k]
            if j != -1 and seen[j] == 0:
                seen[j] = 1
                temp_cand[count] = j
                temp_dist[count] = D[i, j]
                count += 1
        
        # 按距离排序
        if count > 0:
            order = np.argsort(temp_dist[:count])
            for t in range(min(count, max_size)):
                merged[i, t] = temp_cand[order[t]]
    
    return merged

# ==============================================================================
# LK/LKH 核心函数 (Lin-Kernighan 局部搜索)
# ==============================================================================

@njit(cache=True, fastmath=True)
def _tour_to_adjacency(tour, succ, pred, pos):
    """将排列 tour 转换为邻接表示 succ/pred/pos"""
    n = len(tour)
    for i in range(n):
        city = tour[i]
        pos[city] = i                    # 城市在 tour 中的位置
        succ[city] = tour[(i + 1) % n]   # 城市的后继
        pred[city] = tour[(i - 1) % n]   # 城市的前驱

@njit(cache=True, fastmath=True)
def _adjacency_to_tour(succ, tour, start=0):
    """将邻接表示转换回排列 tour"""
    n = len(tour)
    tour[0] = start
    for i in range(1, n):
        tour[i] = succ[tour[i-1]]

@njit(cache=True, fastmath=True)
def _flip_segment(succ, pred, a, b):
    """翻转从 a 到 b 的路径段（用于 2-opt 类型的 move）"""
    # 收集段中的节点
    curr = a
    count = 0
    while curr != b and count < 10000:
        next_node = succ[curr]
        # 交换 succ 和 pred
        succ[curr], pred[curr] = pred[curr], succ[curr]
        curr = next_node
        count += 1
    # 处理最后一个节点 b
    succ[b], pred[b] = pred[b], succ[b]

@njit(cache=True, fastmath=True)
def _lk_try_move(t1, succ, pred, pos, D, candidates, max_depth, branch_limit):
    """
    从 t1 出发尝试找一个改进的 LK move
    
    返回: (gain, 是否找到改进, 需要执行的 move 信息)
    """
    n = len(succ)
    K = candidates.shape[1]
    
    # t2 是 t1 的后继，第一条要断的边是 (t1, t2)
    t2 = succ[t1]
    x1_cost = D[t1, t2]  # 断边代价
    
    # 遍历候选边 (t2, t3) 作为第一条加边
    for k in range(min(K, branch_limit)):
        t3 = candidates[t2, k]
        if t3 == -1 or t3 == t1 or t3 == t2:
            continue
        if not np.isfinite(D[t2, t3]):
            continue
            
        y1_cost = D[t2, t3]  # 加边代价
        G1 = x1_cost - y1_cost  # 第一步增益
        
        if G1 <= 0:  # 必须有正增益潜力
            continue
        
        # t4 是 t3 的后继，第二条断边是 (t3, t4)
        t4 = succ[t3]
        if t4 == t1:  # 可以直接闭合
            # 闭合增益：不需要加闭合边（因为闭合边就是 (t4, t1) = (succ[t3], t1)）
            # 实际上闭合需要检查
            continue  # 两步太短，继续找更深的
        
        x2_cost = D[t3, t4]  # 第二条断边代价
        
        # 尝试闭合：加边 (t4, t1)
        close_cost = D[t4, t1]
        if np.isfinite(close_cost):
            total_gain = G1 + x2_cost - close_cost
            if total_gain > 1e-9:
                # 找到改进！返回 2-opt 类型的 move
                return total_gain, True, t1, t2, t3, t4
        
        # 如果不能闭合，继续扩展到更深层
        G2 = G1 + x2_cost
        
        # 第二层：选择 y2 = (t4, t5)
        for k2 in range(min(K, branch_limit // 2)):
            t5 = candidates[t4, k2]
            if t5 == -1 or t5 == t1 or t5 == t2 or t5 == t3 or t5 == t4:
                continue
            if not np.isfinite(D[t4, t5]):
                continue
            
            y2_cost = D[t4, t5]
            if G2 - y2_cost <= 0:
                continue
            
            t6 = succ[t5]
            x3_cost = D[t5, t6]
            
            # 尝试 3-opt 闭合
            close_cost = D[t6, t1]
            if np.isfinite(close_cost):
                total_gain = G2 - y2_cost + x3_cost - close_cost
                if total_gain > 1e-9:
                    return total_gain, True, t1, t2, t3, t4  # 简化：返回 2-opt 部分
    
    return 0.0, False, -1, -1, -1, -1

@njit(cache=True, fastmath=True)
def _apply_2opt_move(tour, pos, i, j):
    """应用 2-opt move：翻转 tour[i+1:j+1]"""
    n = len(tour)
    # 翻转区间 [i+1, j]
    left = i + 1
    right = j
    while left < right:
        # 交换
        tmp = tour[left]
        tour[left] = tour[right]
        tour[right] = tmp
        # 更新位置
        pos[tour[left]] = left
        pos[tour[right]] = right
        left += 1
        right -= 1
    # 如果 left == right（奇数个元素），更新中间元素的位置
    if left == right:
        pos[tour[left]] = left

@njit(cache=True, fastmath=True)
def _lk_search(tour, D, candidates, max_depth=5, branch_limit=8, max_iters=1000):
    """
    Lin-Kernighan 局部搜索（简化版，基于排列表示）
    
    输入:
        tour: 当前 tour（排列表示，会被原地修改）
        D: 距离矩阵
        candidates: 候选边集
        max_depth: 最大搜索深度
        branch_limit: 每层最大分支数
        max_iters: 最大迭代次数
    
    输出:
        total_gain: 累计改进量
    """
    n = len(tour)
    K = candidates.shape[1]
    
    # 预分配位置数组
    pos = np.empty(n, np.int32)
    for i in range(n):
        pos[tour[i]] = i
    
    total_gain = 0.0
    improved = True
    iteration = 0
    
    while improved and iteration < max_iters:
        improved = False
        iteration += 1
        
        # 尝试每个起点（随机顺序）
        start_offset = np.random.randint(0, n)
        
        for offset in range(n):
            i = (start_offset + offset) % n
            
            # 边 (tour[i], tour[i+1])
            u = tour[i]
            v = tour[(i + 1) % n]
            
            if not np.isfinite(D[u, v]):
                continue
            
            old_edge_cost = D[u, v]
            best_gain = 0.0
            best_j = -1
            
            # 从候选集中找改进的 2-opt move
            for k in range(K):
                cand = candidates[u, k]
                if cand == -1:
                    break
                
                j = pos[cand]  # cand 在 tour 中的位置
                
                # 2-opt: 断 (i, i+1) 和 (j, j+1)，连 (i, j) 和 (i+1, j+1)
                if j <= i + 1 or j >= n - 1:
                    continue
                
                c = tour[j]
                d = tour[(j + 1) % n]
                
                if not np.isfinite(D[c, d]) or not np.isfinite(D[u, c]) or not np.isfinite(D[v, d]):
                    continue
                
                # 计算增益
                gain = (D[u, v] + D[c, d]) - (D[u, c] + D[v, d])
                
                if gain > best_gain + 1e-9:
                    best_gain = gain
                    best_j = j
            
            if best_j > 0:
                # 应用 2-opt move
                _apply_2opt_move(tour, pos, i, best_j)
                total_gain += best_gain
                improved = True
                break  # First improvement
    
    return total_gain

@njit(cache=True, fastmath=True)
def _edge_hash(u, v, n):
    """计算边的哈希值（用于 Tabu 表索引）"""
    return (u * n + v) % 10007  # 素数模

@njit(cache=True, fastmath=True)
def _lk_search_tabu(tour, D, candidates, tabu_tenure=7, max_depth=5, branch_limit=8, max_iters=1000):
    """
    带 Tabu 短期记忆的 Lin-Kernighan 局部搜索
    
    Tabu 机制:
        - 当一条边被移除时，记录到禁忌表
        - 在 tabu_tenure 次迭代内，禁止将该边重新加入
        - 允许"渴望准则"：如果移动产生的增益超过阈值，可忽略禁忌
    
    输入:
        tour: 当前 tour（排列表示，会被原地修改）
        D: 距离矩阵（可含 GLS 惩罚）
        candidates: 候选边集
        tabu_tenure: 禁忌期限（迭代次数）
        max_depth, branch_limit, max_iters: LK 搜索参数
    
    输出:
        total_gain: 累计改进量
    """
    n = len(tour)
    K = candidates.shape[1]
    
    # 预分配位置数组
    pos = np.empty(n, np.int32)
    for i in range(n):
        pos[tour[i]] = i
    
    # Tabu 表：存储边被禁忌的迭代号（禁到第几轮）
    # 使用哈希表简化：tabu_until[hash(u,v)] = 禁到第几轮
    TABU_SIZE = 10007  # 素数大小
    tabu_until = np.zeros(TABU_SIZE, np.int32)
    
    total_gain = 0.0
    improved = True
    iteration = 0
    best_known = tour_length_jit(tour, D)  # 用于渴望准则
    aspiration_threshold = best_known * 0.001  # 0.1% 的改进可忽略禁忌
    
    while improved and iteration < max_iters:
        improved = False
        iteration += 1
        
        # 尝试每个起点（随机顺序）
        start_offset = np.random.randint(0, n)
        
        for offset in range(n):
            i = (start_offset + offset) % n
            
            # 边 (tour[i], tour[i+1])
            u = tour[i]
            v = tour[(i + 1) % n]
            
            if not np.isfinite(D[u, v]):
                continue
            
            best_gain = 0.0
            best_j = -1
            is_tabu_move = False
            
            # 从候选集中找改进的 2-opt move
            for k in range(K):
                cand = candidates[u, k]
                if cand == -1:
                    break
                
                j = pos[cand]
                
                # 2-opt 约束
                if j <= i + 1 or j >= n - 1:
                    continue
                
                c = tour[j]
                d = tour[(j + 1) % n]
                
                if not np.isfinite(D[c, d]) or not np.isfinite(D[u, c]) or not np.isfinite(D[v, d]):
                    continue
                
                # 检查新边是否被禁忌
                # 新边: (u, c) 和 (v, d)
                h1 = _edge_hash(u, c, n)
                h2 = _edge_hash(v, d, n)
                is_tabu = (tabu_until[h1] >= iteration) or (tabu_until[h2] >= iteration)
                
                # 计算增益
                gain = (D[u, v] + D[c, d]) - (D[u, c] + D[v, d])
                
                if gain > best_gain + 1e-9:
                    # 检查渴望准则：如果增益足够大，可忽略禁忌
                    if is_tabu and gain < aspiration_threshold:
                        continue  # 被禁忌且增益不够大，跳过
                    
                    best_gain = gain
                    best_j = j
                    is_tabu_move = is_tabu
            
            if best_j > 0:
                # 记录被移除的边到禁忌表
                old_c = tour[best_j]
                old_d = tour[(best_j + 1) % n]
                
                # 禁止 (u, v) 和 (c, d) 重新加入
                h_uv = _edge_hash(u, v, n)
                h_cd = _edge_hash(old_c, old_d, n)
                tabu_until[h_uv] = iteration + tabu_tenure
                tabu_until[h_cd] = iteration + tabu_tenure
                
                # 应用 2-opt move
                _apply_2opt_move(tour, pos, i, best_j)
                total_gain += best_gain
                improved = True
                
                # 更新渴望阈值
                current_len = best_known - total_gain
                aspiration_threshold = current_len * 0.001
                
                break  # First improvement
    
    return total_gain

@njit(cache=True, fastmath=True)
def _apply_3opt_move(tour, pos, i, j, k, move_type):
    """
    应用 3-opt move：断开 (i,i+1), (j,j+1), (k,k+1) 并重新连接
    
    move_type 决定重连方式：
        0: [0..i] + [j+1..k] + [i+1..j] + [k+1..n]  (段交换)
        1: [0..i] + [j+1..k]^r + [i+1..j] + [k+1..n]  (带反转)
        2: [0..i] + [i+1..j]^r + [j+1..k]^r + [k+1..n]  (双反转)
    
    输入:
        tour: 当前 tour
        pos: 位置数组
        i, j, k: 三个断点位置（i < j < k）
        move_type: 重连类型
    """
    n = len(tour)
    temp = np.empty(n, np.int32)
    ptr = 0
    
    if move_type == 0:
        # 段交换：[0..i] + [j+1..k] + [i+1..j] + [k+1..n]
        for t in range(0, i + 1):
            temp[ptr] = tour[t]; ptr += 1
        for t in range(j + 1, k + 1):
            temp[ptr] = tour[t]; ptr += 1
        for t in range(i + 1, j + 1):
            temp[ptr] = tour[t]; ptr += 1
        for t in range(k + 1, n):
            temp[ptr] = tour[t]; ptr += 1
    elif move_type == 1:
        # 带反转：[0..i] + [j+1..k]^r + [i+1..j] + [k+1..n]
        for t in range(0, i + 1):
            temp[ptr] = tour[t]; ptr += 1
        for t in range(k, j, -1):  # 反转 [j+1..k]
            temp[ptr] = tour[t]; ptr += 1
        for t in range(i + 1, j + 1):
            temp[ptr] = tour[t]; ptr += 1
        for t in range(k + 1, n):
            temp[ptr] = tour[t]; ptr += 1
    elif move_type == 2:
        # 双反转：[0..i] + [i+1..j]^r + [j+1..k]^r + [k+1..n]
        for t in range(0, i + 1):
            temp[ptr] = tour[t]; ptr += 1
        for t in range(j, i, -1):  # 反转 [i+1..j]
            temp[ptr] = tour[t]; ptr += 1
        for t in range(k, j, -1):  # 反转 [j+1..k]
            temp[ptr] = tour[t]; ptr += 1
        for t in range(k + 1, n):
            temp[ptr] = tour[t]; ptr += 1
    
    # 拷回并更新 pos
    for t in range(n):
        tour[t] = temp[t]
        pos[temp[t]] = t

@njit(cache=True, fastmath=True)
def _try_3opt_move(tour, pos, D, candidates, tabu_until, iteration, aspiration_threshold):
    """
    尝试找一个改进的 3-opt move（非顺序移动）
    
    返回: (gain, found, i, j, k, move_type)
    """
    n = len(tour)
    K = candidates.shape[1]
    
    best_gain = 0.0
    best_i, best_j, best_k, best_type = -1, -1, -1, -1
    
    # 随机选择起点
    start = np.random.randint(0, n)
    
    # 限制搜索范围（3-opt 很慢，只搜索部分）
    max_tries = min(n // 2, 100)
    
    for offset in range(max_tries):
        i = (start + offset) % (n - 5)
        if i < 0: continue
        
        # 第一条边 (tour[i], tour[i+1])
        a, b = tour[i], tour[i + 1]
        if not np.isfinite(D[a, b]):
            continue
        
        # 从候选集中选 j
        for k1 in range(min(K, 10)):
            cand1 = candidates[b, k1]
            if cand1 == -1:
                break
            
            j = pos[cand1]
            if j <= i + 1 or j >= n - 3:
                continue
            
            # 第二条边 (tour[j], tour[j+1])
            c, d = tour[j], tour[j + 1]
            if not np.isfinite(D[c, d]):
                continue
            
            # 从候选集中选 k
            for k2 in range(min(K, 8)):
                cand2 = candidates[d, k2]
                if cand2 == -1:
                    break
                
                k = pos[cand2]
                if k <= j + 1 or k >= n - 1:
                    continue
                
                # 第三条边 (tour[k], tour[k+1])
                e, f = tour[k], tour[(k + 1) % n]
                if not np.isfinite(D[e, f]):
                    continue
                
                # 计算原始代价
                old_cost = D[a, b] + D[c, d] + D[e, f]
                
                # 尝试不同的重连方式
                # move_type 0: 段交换
                # 新边: (a, d), (c, b), (e, f) 变为 (a, d), (e, b), (c, f)
                # 这里简化为只检查一种有效的 3-opt
                
                # 简化的 3-opt：交换中间两段
                # 新边: (a, d), (e, b), (c, f)
                if np.isfinite(D[a, d]) and np.isfinite(D[e, b]) and np.isfinite(D[c, f]):
                    new_cost = D[a, d] + D[e, b] + D[c, f]
                    gain = old_cost - new_cost
                    
                    if gain > best_gain + 1e-9:
                        # 检查禁忌
                        h1 = _edge_hash(a, d, n)
                        h2 = _edge_hash(e, b, n)
                        h3 = _edge_hash(c, f, n)
                        is_tabu = (tabu_until[h1] >= iteration) or \
                                  (tabu_until[h2] >= iteration) or \
                                  (tabu_until[h3] >= iteration)
                        
                        if is_tabu and gain < aspiration_threshold:
                            continue
                        
                        best_gain = gain
                        best_i, best_j, best_k, best_type = i, j, k, 0
    
    return best_gain, (best_i >= 0), best_i, best_j, best_k, best_type

@njit(cache=True, fastmath=True)
def _lk_search_enhanced(tour, D, candidates, tabu_tenure=7, max_depth=5, branch_limit=8, max_iters=1000, use_3opt=True):
    """
    增强版 LK 搜索：2-opt + Tabu + 可选 3-opt（非顺序移动）
    
    当 2-opt 找不到改进时，尝试 3-opt 非顺序移动
    """
    n = len(tour)
    K = candidates.shape[1]
    
    # 预分配位置数组
    pos = np.empty(n, np.int32)
    for i in range(n):
        pos[tour[i]] = i
    
    # Tabu 表
    TABU_SIZE = 10007
    tabu_until = np.zeros(TABU_SIZE, np.int32)
    
    total_gain = 0.0
    improved = True
    iteration = 0
    best_known = tour_length_jit(tour, D)
    aspiration_threshold = best_known * 0.001
    
    no_2opt_count = 0  # 连续无 2-opt 改进计数
    
    while improved and iteration < max_iters:
        improved = False
        iteration += 1
        
        # 尝试 2-opt
        start_offset = np.random.randint(0, n)
        found_2opt = False
        
        for offset in range(n):
            i = (start_offset + offset) % n
            
            u = tour[i]
            v = tour[(i + 1) % n]
            
            if not np.isfinite(D[u, v]):
                continue
            
            best_gain_2opt = 0.0
            best_j = -1
            
            for k in range(K):
                cand = candidates[u, k]
                if cand == -1:
                    break
                
                j = pos[cand]
                if j <= i + 1 or j >= n - 1:
                    continue
                
                c = tour[j]
                d = tour[(j + 1) % n]
                
                if not np.isfinite(D[c, d]) or not np.isfinite(D[u, c]) or not np.isfinite(D[v, d]):
                    continue
                
                h1 = _edge_hash(u, c, n)
                h2 = _edge_hash(v, d, n)
                is_tabu = (tabu_until[h1] >= iteration) or (tabu_until[h2] >= iteration)
                
                gain = (D[u, v] + D[c, d]) - (D[u, c] + D[v, d])
                
                if gain > best_gain_2opt + 1e-9:
                    if is_tabu and gain < aspiration_threshold:
                        continue
                    best_gain_2opt = gain
                    best_j = j
            
            if best_j > 0:
                old_c = tour[best_j]
                old_d = tour[(best_j + 1) % n]
                
                h_uv = _edge_hash(u, v, n)
                h_cd = _edge_hash(old_c, old_d, n)
                tabu_until[h_uv] = iteration + tabu_tenure
                tabu_until[h_cd] = iteration + tabu_tenure
                
                _apply_2opt_move(tour, pos, i, best_j)
                total_gain += best_gain_2opt
                improved = True
                found_2opt = True
                no_2opt_count = 0
                
                current_len = best_known - total_gain
                aspiration_threshold = current_len * 0.001
                break
        
        # 如果 2-opt 找不到改进，尝试 3-opt
        if not found_2opt and use_3opt:
            no_2opt_count += 1
            
            # 每隔几次尝试 3-opt
            if no_2opt_count >= 3:
                gain_3opt, found, i3, j3, k3, move_type = _try_3opt_move(
                    tour, pos, D, candidates, tabu_until, iteration, aspiration_threshold
                )
                
                if found and gain_3opt > 1e-9:
                    # 记录被移除的边到禁忌表
                    a, b = tour[i3], tour[i3 + 1]
                    c, d = tour[j3], tour[j3 + 1]
                    e, f = tour[k3], tour[(k3 + 1) % n]
                    
                    tabu_until[_edge_hash(a, b, n)] = iteration + tabu_tenure
                    tabu_until[_edge_hash(c, d, n)] = iteration + tabu_tenure
                    tabu_until[_edge_hash(e, f, n)] = iteration + tabu_tenure
                    
                    _apply_3opt_move(tour, pos, i3, j3, k3, move_type)
                    total_gain += gain_3opt
                    improved = True
                    no_2opt_count = 0
                    
                    current_len = best_known - total_gain
                    aspiration_threshold = current_len * 0.001
    
    return total_gain


@njit(cache=True, fastmath=True)
def _or_opt_move_fast(tour, pos, D, candidates, block_size=1):
    """快速 Or-opt：移动一个 block 到更好的位置"""
    n = len(tour)
    K = candidates.shape[1]
    
    best_gain = 0.0
    best_u_idx = -1
    best_t_idx = -1
    
    for u_idx in range(n - block_size):
        u = tour[u_idx]
        
        # block 的边界
        prev_idx = (u_idx - 1) % n
        post_idx = (u_idx + block_size) % n
        
        prev_u = tour[prev_idx]
        block_tail = tour[u_idx + block_size - 1]
        next_after = tour[post_idx]
        
        if not np.isfinite(D[prev_u, u]) or not np.isfinite(D[block_tail, next_after]):
            continue
        if not np.isfinite(D[prev_u, next_after]):
            continue
        
        remove_cost = D[prev_u, u] + D[block_tail, next_after]
        new_edge = D[prev_u, next_after]
        
        # 从候选集中找插入位置
        for k in range(K):
            target = candidates[u, k]
            if target == -1:
                break
            
            t_idx = pos[target]
            if t_idx >= u_idx and t_idx < u_idx + block_size:
                continue
            if t_idx == prev_idx:
                continue
            
            target_next_idx = (t_idx + 1) % n
            target_next = tour[target_next_idx]
            
            if not np.isfinite(D[target, target_next]):
                continue
            if not np.isfinite(D[target, u]) or not np.isfinite(D[block_tail, target_next]):
                continue
            
            insert_cost = D[target, u] + D[block_tail, target_next]
            old_edge = D[target, target_next]
            gain = (remove_cost - new_edge) + (old_edge - insert_cost)
            
            if gain > best_gain + 1e-9:
                best_gain = gain
                best_u_idx = u_idx
                best_t_idx = t_idx
    
    return best_gain, best_u_idx, best_t_idx

@njit(cache=True, fastmath=True)
def iterated_lk(tour, D, candidates, max_kicks=10, lk_depth=5, lk_branch=8, lk_iters=500, tabu_tenure=7, use_3opt=True):
    """
    Iterated Lin-Kernighan (ILK) + Tabu + 3-opt 非顺序移动
    
    循环:
        1. LK（带 Tabu + 可选 3-opt）跑到无改进
        2. kick（double-bridge）
        3. 再 LK 到无改进
        4. 记录 best，重复直到 max_kicks 次无提升
    
    输入:
        tour: 初始 tour（会被修改）
        D: 距离矩阵
        candidates: 候选边集
        max_kicks: 最大 kick 次数
        lk_depth: LK 搜索深度
        lk_branch: LK 分支限制
        lk_iters: 每轮 LK 最大迭代次数
        tabu_tenure: Tabu 禁忌期限
        use_3opt: 是否启用 3-opt 非顺序移动
    
    输出:
        best_tour: 找到的最优 tour
        best_length: 最优 tour 长度
    """
    n = len(tour)
    
    # 初始 LK 优化（使用增强版：2-opt + Tabu + 3-opt）
    _lk_search_enhanced(tour, D, candidates, tabu_tenure, lk_depth, lk_branch, lk_iters, use_3opt)
    
    best_length = tour_length_jit(tour, D)
    best_tour = tour.copy()
    
    no_improve_count = 0
    
    for kick_iter in range(max_kicks):
        # Kick: double-bridge 扰动
        kicked_tour = double_bridge_move(tour)
        
        # LK 优化扰动后的 tour（使用增强版）
        _lk_search_enhanced(kicked_tour, D, candidates, tabu_tenure, lk_depth, lk_branch, lk_iters, use_3opt)
        
        current_length = tour_length_jit(kicked_tour, D)
        
        if current_length < best_length - 1e-9:
            best_length = current_length
            best_tour[:] = kicked_tour[:]
            tour[:] = kicked_tour[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
            # 有概率接受较差解（避免陷入）
            if np.random.rand() < 0.3:
                tour[:] = kicked_tour[:]
            if no_improve_count >= 3:
                break
    
    return best_tour, best_length



@njit(cache=True, fastmath=True)
def _tour_feasible_jit(tour, finite_mask):
    """检查路径是否全部边都是可行的（没有 inf）"""
    n = tour.shape[0]
    for i in range(n):
        u, v = tour[i], tour[(i + 1) % n]
        if not finite_mask[u, v]:
            return False
    return True

@njit(cache=True, fastmath=True)
def _two_opt_once_jit_safe(tour, D):
    n = tour.size
    best_delta = 0.0
    bi = -1; bj = -1
    tries = min(2000, n * 20)
    for _ in range(tries):
        i = np.random.randint(0, n - 3)
        j = np.random.randint(i + 2, n - 1)
        a = tour[i]; b = tour[(i + 1) % n]
        c = tour[j]; d = tour[(j + 1) % n]
        if not np.isfinite(D[a, c]) or not np.isfinite(D[b, d]):
            continue
        delta = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
        if delta < best_delta:
            best_delta = delta; bi = i; bj = j
    if best_delta < 0.0:
        l, r = bi + 1, bj
        while l < r:
            tmp = tour[l]; tour[l] = tour[r]; tour[r] = tmp
            l += 1; r -= 1
        return True
    return False

@njit(cache=True, fastmath=True)
def _tour_feasible_jit(tour, finite_mask):
    n = tour.size
    for i in range(n):
        if not finite_mask[tour[i], tour[(i + 1) % n]]: return False
    return True

@njit(cache=True, fastmath=True)
def _repair_jit(tour, D, finite_mask, max_tries=50):
    for _ in range(max_tries):
        if _tour_feasible_jit(tour, finite_mask): return True
        if not _two_opt_once_jit_safe(tour, D): break
    return _tour_feasible_jit(tour, finite_mask)

@njit(cache=True, fastmath=True)
def _rand_perm_jit(n):
    arr = np.arange(n, dtype=np.int32)
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp
    return arr

@njit(cache=True, fastmath=True)
def double_bridge_move(tour):
    n = tour.shape[0]
    if n < 8: return tour.copy()
    p1 = np.random.randint(1, n // 4)
    p2 = np.random.randint(p1 + 1, n // 2)
    p3 = np.random.randint(p2 + 1, 3 * n // 4)
    new_tour = np.empty(n, dtype=tour.dtype)
    new_tour[:p1] = tour[:p1]
    new_tour[p1:p1+(p3-p2)] = tour[p2:p3]
    new_tour[p1+(p3-p2):p1+(p3-p2)+(p2-p1)] = tour[p1:p2]
    new_tour[p1+(p3-p1):] = tour[p3:]
    return new_tour

@njit(cache=True, fastmath=True)
def _make_move_opt(tour, pos, u_idx, block_size, t_idx_new, temp_buffer):
    """零内存分配的 Or-opt 移动"""
    n = len(tour)
    
    # 移除 block 后重建 tour，并在 t_idx_new 后插入 block
    ptr = 0
    block_start = u_idx
    block_end = u_idx + block_size
    
    # 遍历原 tour，跳过 block
    inserted = False
    for i in range(n):
        if i >= block_start and i < block_end:
            continue
        temp_buffer[ptr] = tour[i]
        # 计算在去除 block 后的索引
        adj_idx = i if i < block_start else i - block_size
        if not inserted and adj_idx == t_idx_new:
            ptr += 1
            # 插入 block
            for b in range(block_size):
                temp_buffer[ptr] = tour[block_start + b]
                ptr += 1
            inserted = True
        else:
            ptr += 1
    
    # 如果 t_idx_new 是最后一个位置
    if not inserted:
        for b in range(block_size):
            temp_buffer[ptr] = tour[block_start + b]
            ptr += 1
    
    # 拷回
    for i in range(n):
        tour[i] = temp_buffer[i]
        pos[temp_buffer[i]] = i

@njit(cache=True, fastmath=True)
def _candidate_or_opt_jit(tour, D, knn_idx, pos_buf, tour_buf, max_iters=100, dlb_mask=None, block_size=1):
    """内存优化版 Or-opt：接收外部 buffer，零内存分配"""
    n = tour.shape[0]
    K = knn_idx.shape[1]
    block_size = int(block_size)
    if block_size < 1: block_size = 1
    if block_size >= n: return False 
    
    # 使用传入的 buffer
    for i in range(n): pos_buf[tour[i]] = i
    
    improved = False 
    use_dlb = (dlb_mask is not None) 
    
    for _ in range(max_iters): 
        found_in_try = False 
        start = np.random.randint(0, n) 
        for offset in range(n): 
            u_idx = (start + offset) % n
            u = tour[u_idx] 
            if use_dlb and dlb_mask[u]: continue 
            
            if block_size > 1 and u_idx + block_size >= n: 
                if use_dlb: dlb_mask[u] = True
                continue 
                
            prev_idx = (u_idx - 1) if u_idx > 0 else (n - 1)
            post_idx = (u_idx + block_size) % n
            
            prev_u = tour[prev_idx] 
            block_head = u 
            block_tail = tour[u_idx + block_size - 1] 
            next_after = tour[post_idx] 
            
            if not np.isfinite(D[prev_u, block_head]): continue
            if not np.isfinite(D[block_tail, next_after]): continue
            if not np.isfinite(D[prev_u, next_after]): continue 
            
            remove_cost = D[prev_u, block_head] + D[block_tail, next_after]
            new_edge_cost = D[prev_u, next_after]
            
            move_found = False 
            for k in range(K): 
                target = knn_idx[block_head, k] 
                if target == -1: break 
                t_idx = pos_buf[target] 
                if t_idx == prev_idx: continue 
                if t_idx >= u_idx and t_idx < u_idx + block_size: continue 
                
                target_next_idx = (t_idx + 1) % n
                target_next = tour[target_next_idx] 
                
                if not np.isfinite(D[target, target_next]): continue
                if not np.isfinite(D[target, block_head]): continue 
                if not np.isfinite(D[block_tail, target_next]): continue 
                
                insert_cost = D[target, block_head] + D[block_tail, target_next]
                old_edge_cost = D[target, target_next]
                gain = (remove_cost - new_edge_cost) + (old_edge_cost - insert_cost)
                
                if gain > 1e-6: 
                    # 计算去除 block 后的 t_idx
                    t_idx_new = t_idx
                    if t_idx > u_idx: t_idx_new -= block_size
                    
                    # 使用 buffer 执行移动
                    _make_move_opt(tour, pos_buf, u_idx, block_size, t_idx_new, tour_buf)
                    
                    improved = True
                    move_found = True
                    found_in_try = True
                    if use_dlb: 
                        dlb_mask[prev_u] = False 
                        dlb_mask[next_after] = False 
                        dlb_mask[target] = False 
                        dlb_mask[target_next] = False
                        for b in range(block_size): dlb_mask[tour[pos_buf[tour[u_idx + b]]]] = False
                    break 
            if move_found: continue 
            else: 
                if use_dlb: dlb_mask[block_head] = True
        if not found_in_try and use_dlb: break
    return improved


@njit(cache=True, fastmath=True)
def _candidate_block_swap_jit(tour, D, knn_idx, pos_buf, tour_buf, max_iters=50, dlb_mask=None, block_size=2):
    """内存优化版 Block Swap：接收外部 buffer，零内存分配"""
    n = tour.shape[0]
    K = knn_idx.shape[1]
    block_size = int(block_size)
    if block_size < 1: block_size = 1
    if block_size * 2 >= n: return False 
    
    # 使用传入的 buffer
    for i in range(n): pos_buf[tour[i]] = i
    
    improved = False
    use_dlb = (dlb_mask is not None)
    
    for _ in range(max_iters):
        found_in_try = False
        start = np.random.randint(0, n)
        
        for offset in range(n):
            u_idx = (start + offset) % n 
            u = tour[u_idx] 
            if use_dlb and dlb_mask[u]: continue
            
            if u_idx + block_size >= n: 
                if use_dlb: dlb_mask[u] = True
                continue
            
            i = u_idx        
            a_idx = i - 1 if i > 0 else n - 1
            d_idx = i + block_size 
            a, b, c, d = tour[a_idx], tour[i], tour[i + block_size - 1], tour[d_idx]
            
            if not np.isfinite(D[a, b]) or not np.isfinite(D[c, d]): continue
            
            move_found = False
            for k in range(K):
                target = knn_idx[a, k] 
                if target == -1: break
                j = pos_buf[target]  # 使用 pos_buf
                if j <= i or j < i + block_size or j + block_size >= n: continue
                
                e_idx = j - 1
                e, f, g, h = tour[e_idx], tour[j], tour[j + block_size - 1], tour[j + block_size]
                
                if j == i + block_size:
                    if not (np.isfinite(D[a, f]) and np.isfinite(D[g, b]) and np.isfinite(D[c, h])): continue
                    old_cost = D[a, b] + D[c, f] + D[g, h]
                    new_cost = D[a, f] + D[g, b] + D[c, h]
                else:
                    if not (np.isfinite(D[a, f]) and np.isfinite(D[g, d]) and np.isfinite(D[e, b]) and np.isfinite(D[c, h])): continue
                    old_cost = D[a, b] + D[c, d] + D[e, f] + D[g, h]
                    new_cost = D[a, f] + D[g, d] + D[e, b] + D[c, h]
                
                if old_cost - new_cost > 1e-6: 
                    # 使用 buffer 执行交换
                    ptr = 0
                    for t in range(0, i): tour_buf[ptr] = tour[t]; ptr += 1
                    for t in range(j, j + block_size): tour_buf[ptr] = tour[t]; ptr += 1
                    for t in range(i + block_size, j): tour_buf[ptr] = tour[t]; ptr += 1
                    for t in range(i, i + block_size): tour_buf[ptr] = tour[t]; ptr += 1
                    for t in range(j + block_size, n): tour_buf[ptr] = tour[t]; ptr += 1
                    
                    # 拷回并更新 pos
                    for t in range(n):
                        tour[t] = tour_buf[t]
                        pos_buf[tour_buf[t]] = t
                    
                    improved = True; move_found = True; found_in_try = True
                    if use_dlb:
                        dlb_mask[a] = False; dlb_mask[b] = False
                        dlb_mask[c] = False; dlb_mask[d] = False
                        dlb_mask[e] = False; dlb_mask[f] = False
                        dlb_mask[g] = False; dlb_mask[h] = False
                    break 
            if move_found: continue
            else:
                if use_dlb: dlb_mask[b] = True 
        if not found_in_try and use_dlb: break
    return improved

@njit(cache=True, fastmath=True)
def _rcl_nn_tour_jit(D, finite_mask, knn_idx, r):
    n = D.shape[0]
    tour = np.empty(n, np.int32)
    used = np.zeros(n, np.uint8)
    cur = np.random.randint(0, n)  
    tour[0] = cur; used[cur] = 1
    K = knn_idx.shape[1]
    for t in range(1, n):
        tmp_idx = np.empty(K, np.int32)
        tmp_dis = np.empty(K, np.float64)
        cnt = 0
        for k in range(K):
            j = knn_idx[cur, k]
            if j == -1 or used[j] == 1 or not finite_mask[cur, j]: continue
            tmp_idx[cnt] = j; tmp_dis[cnt] = D[cur, j]; cnt += 1
        if cnt == 0:
            nxt = -1
            for j in range(n):
                if used[j] == 0 and finite_mask[cur, j]: nxt = j; break
            if nxt == -1:
                for j in range(n):
                    if used[j] == 0: nxt = j; break
        else:
            order = np.argsort(tmp_dis[:cnt])
            pick = order[np.random.randint(0, min(r, cnt))]
            nxt = int(tmp_idx[pick])
        tour[t] = nxt; used[nxt] = 1; cur = nxt
    if not finite_mask[tour[n - 1], tour[0]]:
        for _ in range(20):
            if _two_opt_once_jit_safe(tour, D) and finite_mask[tour[n - 1], tour[0]]: break
    return tour

@njit(cache=True, fastmath=True)
def _insertion_tour_jit(D, finite_mask, use_farthest):
    n = D.shape[0]
    a = np.random.randint(0, n)
    b = a
    for _ in range(16):
        b = np.random.randint(0, n)
        if b != a and finite_mask[a, b] and finite_mask[b, a]: break
    tour = np.array([a, b], dtype=np.int32)
    used = np.zeros(n, np.uint8); used[a] = 1; used[b] = 1
    m = 2
    while m < n:
        if use_farthest:
            best_city = -1; best_score = -1.0
            for c in range(n):
                if used[c] == 1: continue
                mind = 1e100
                for t in range(m):
                    if finite_mask[c, tour[t]] and D[c, tour[t]] < mind: mind = D[c, tour[t]]
                if mind > best_score: best_score = mind; best_city = c
            insert_city = best_city if best_city != -1 else np.random.randint(0, n)
        else:
            remain = n - m
            k = np.random.randint(0, remain)
            insert_city = -1
            for c in range(n):
                if used[c] == 0:
                    if k == 0: insert_city = c; break
                    k -= 1
        best_pos = -1; best_cost = 1e100
        for i in range(m):
            prev, curr = tour[i - 1], tour[i]
            if finite_mask[prev, insert_city] and finite_mask[insert_city, curr]:
                cost = D[prev, insert_city] + D[insert_city, curr] - D[prev, curr]
                if cost < best_cost: best_cost = cost; best_pos = i
        newtour = np.empty(m + 1, np.int32)
        if best_pos == -1:
            pos = np.random.randint(0, m + 1)
            newtour[:pos] = tour[:pos]
            newtour[pos] = insert_city
            newtour[pos+1:] = tour[pos:]
        else:
            newtour[:best_pos] = tour[:best_pos]
            newtour[best_pos] = insert_city
            newtour[best_pos+1:] = tour[best_pos:]
        tour = newtour; m += 1; used[insert_city] = 1
    for _ in range(20):
        if finite_mask[tour[n - 1], tour[0]]: break
        if not _two_opt_once_jit_safe(tour, D): break
    return tour

@njit(cache=True, fastmath=True)
def bond_distance_jit(t1, t2):
    n = t1.shape[0]
    pos2 = np.empty(n, np.int32)
    for i in range(n): pos2[t2[i]] = i
    shared_edges = 0
    for i in range(n):
        u, v = t1[i], t1[(i + 1) % n]
        idx_u = pos2[u]
        if v == t2[(idx_u - 1) % n] or v == t2[(idx_u + 1) % n]: shared_edges += 1
    return n - shared_edges

@njit(cache=True, fastmath=True)
def rtr_challenge_jit(child, child_fit, pop, fit, W, rng_seed, best_idx):
    m, n = pop.shape[0], child.shape[0]  
    np.random.seed(rng_seed)
    window_indices = np.random.choice(m, size=W, replace=False)
    closest_idx = -1; min_dist = 1e9
    for idx in window_indices:
        dist = bond_distance_jit(child, pop[idx])
        if dist < min_dist: min_dist = dist; closest_idx = idx
    if closest_idx == best_idx: return False, closest_idx  
    target_fit = fit[closest_idx]
    if child_fit < target_fit: return True, closest_idx
    if min_dist > n * 0.15 and child_fit < target_fit * 1.05: return True, closest_idx
    return False, closest_idx

@njit(cache=True, fastmath=True)
def _bfs_ruin_mask_jit(n, knn_idx, n_remove):
    removed_mask = np.zeros(n, dtype=np.bool_)
    if n_remove >= n: return np.ones(n, dtype=np.bool_)
    center = np.random.randint(0, n)
    queue = np.empty(n + 10, dtype=np.int32)
    q_head = 0; q_tail = 0
    queue[q_tail] = center; q_tail += 1
    removed_mask[center] = True
    count = 1; K = knn_idx.shape[1]
    while count < n_remove and q_head < q_tail:
        curr = queue[q_head]; q_head += 1
        for i in range(K):
            neighbor = knn_idx[curr, i]
            if neighbor == -1: break
            if not removed_mask[neighbor]:
                removed_mask[neighbor] = True; count += 1
                queue[q_tail] = neighbor; q_tail += 1
                if count >= n_remove: break
    return removed_mask

@njit(cache=True, fastmath=True)
def _ruin_worst_edges_stochastic_jit(tour, D, n_remove):
    """锦标赛选择法的最差边破坏 - 极速 O(K) 复杂度"""
    n = tour.shape[0]
    mask = np.zeros(n, dtype=np.bool_)
    
    tournament_size = 4  # 每次随机看多少条边
    count = 0
    max_attempts = n_remove * 5  # 防止死循环
    attempts = 0
    
    while count < n_remove and attempts < max_attempts:
        attempts += 1
        
        # 举办一次锦标赛：找到候选者中最差的那个
        best_candidate_idx = -1
        max_dist = -1.0
        
        for _ in range(tournament_size):
            idx = np.random.randint(0, n)
            u, v = tour[idx], tour[(idx + 1) % n]
            if mask[u] or mask[v]: continue  # 已被炸
            dist = D[u, v]
            if dist > max_dist:
                max_dist = dist
                best_candidate_idx = idx
        
        # 炸掉赢家（最长的那条边）
        if best_candidate_idx != -1:
            u_idx = best_candidate_idx
            v_idx = (best_candidate_idx + 1) % n
            if not mask[tour[u_idx]]:
                mask[tour[u_idx]] = True
                count += 1
            if count >= n_remove: break
            if not mask[tour[v_idx]]:
                mask[tour[v_idx]] = True
                count += 1
    
    # 如果没凑够，随机补几个
    if count < n_remove:
        for _ in range((n_remove - count) * 2):
            idx = np.random.randint(0, n)
            if not mask[tour[idx]]:
                mask[tour[idx]] = True
                count += 1
            if count >= n_remove: break
    
    return mask

@njit(cache=True, nogil=True)
def _hybrid_ruin_mask_jit(tour, D, knn_idx, n_remove, mode):
    """混合破坏策略: mode=0 BFS, mode=1 Sequence, mode=2 Worst Edge"""
    n = len(tour)
    mask = np.zeros(n, np.bool_)
    
    if mode == 0:
        # 策略 0: BFS 空间破坏（原有）
        mask = _bfs_ruin_mask_jit(n, knn_idx, n_remove)
        
    elif mode == 1:
        # 策略 1: 序列破坏 - 移除 tour 中连续的一段
        start = np.random.randint(0, n)
        for i in range(n_remove):
            mask[tour[(start + i) % n]] = True
            
    elif mode == 2:
        # 策略 2: 锦标赛选择最差边破坏（极速 O(K) 复杂度）
        mask = _ruin_worst_edges_stochastic_jit(tour, D, n_remove)
    
    return mask

@njit(cache=True, nogil=True)
def _hybrid_ruin_and_recreate_jit(tour, D, ruin_pct, knn_idx, mode):
    """混合破坏策略的 ruin and recreate（保留原版用于兼容）"""
    n = len(tour)
    n_remove = int(n * ruin_pct)
    if n_remove < 2: return tour.copy()
    
    mask = _hybrid_ruin_mask_jit(tour, D, knn_idx, n_remove, mode)
    
    removed_cities = np.empty(n_remove, np.int32)
    kept_cities = np.empty(n - n_remove, np.int32)
    kp, rp = 0, 0
    for i in range(n):
        if mask[tour[i]]:
            if rp < n_remove: removed_cities[rp] = tour[i]; rp += 1
        else:
            if kp < n - n_remove: kept_cities[kp] = tour[i]; kp += 1
    
    current_tour = kept_cities
    np.random.shuffle(removed_cities)
    
    for idx in range(rp):
        city = removed_cities[idx]
        best_delta, best_pos, m_curr = 1e20, -1, len(current_tour)
        for i in range(m_curr):
            u, v = current_tour[i], current_tour[(i + 1) % m_curr]
            delta = D[u, city] + D[city, v] - D[u, v]
            if delta < best_delta: best_delta, best_pos = delta, i
        new_t = np.empty(len(current_tour) + 1, np.int32)
        new_t[:best_pos+1] = current_tour[:best_pos+1]
        new_t[best_pos+1] = city
        new_t[best_pos+2:] = current_tour[best_pos+1:]
        current_tour = new_t
    
    return current_tour

@njit(cache=True, fastmath=True, nogil=True)
def _hybrid_ruin_and_recreate_inplace(tour, D, ruin_pct, knn_idx, mode, tour_buf, removed_buf):
    """内存优化版：完全零分配的 Ruin & Recreate"""
    n = tour.shape[0]
    n_remove = int(n * ruin_pct)
    if n_remove < 2: 
        tour_buf[:] = tour[:]
        return
    
    mask = _hybrid_ruin_mask_jit(tour, D, knn_idx, n_remove, mode)
    
    kp = 0
    rp = 0
    for i in range(n):
        city = tour[i]
        if mask[city]:
            if rp < n_remove:
                removed_buf[rp] = city
                rp += 1
        else:
            tour_buf[kp] = city
            kp += 1
    
    current_len = kp
    np.random.shuffle(removed_buf[:rp])
    
    # 贪婪插入重建 (In-Place Shift)
    for idx in range(rp):
        city = removed_buf[idx]
        best_delta = 1e20
        best_pos = -1
        
        # 检查插入到末尾（连接最后一个和第一个）
        u = tour_buf[current_len - 1]
        v = tour_buf[0]
        delta = D[u, city] + D[city, v] - D[u, v]
        if delta < best_delta:
            best_delta = delta
            best_pos = current_len - 1
        
        # 检查其他位置
        for i in range(current_len - 1):
            u = tour_buf[i]
            v = tour_buf[i + 1]
            delta = D[u, city] + D[city, v] - D[u, v]
            if delta < best_delta:
                best_delta = delta
                best_pos = i
        
        # In-Place Shift 插入
        if best_pos == current_len - 1:
            tour_buf[current_len] = city
        else:
            for k in range(current_len, best_pos + 1, -1):
                tour_buf[k] = tour_buf[k - 1]
            tour_buf[best_pos + 1] = city
        
        current_len += 1

@njit(cache=True, nogil=True)
def _ruin_and_recreate_regret_jit(tour, D, ruin_pct, knn_idx, regret_frac, regret_sample, regret_min_remove):
    n = len(tour)
    n_remove = int(n * ruin_pct)
    if n_remove < 2: return tour.copy()
    mask = _bfs_ruin_mask_jit(n, knn_idx, n_remove)
    kept_cities = np.empty(n - n_remove, dtype=np.int32)
    removed_cities = np.empty(n_remove, dtype=np.int32)
    k_ptr, r_ptr = 0, 0
    for i in range(n):
        c = tour[i]
        if mask[c]:
            if r_ptr < n_remove: removed_cities[r_ptr] = c; r_ptr += 1
        else:
            if k_ptr < n - n_remove: kept_cities[k_ptr] = c; k_ptr += 1
    current_tour = kept_cities
    np.random.shuffle(removed_cities)
    r_len = n_remove
    if n_remove >= regret_min_remove and regret_frac > 0:
        regret_steps = max(1, int(r_len * regret_frac))
        for _ in range(regret_steps):
            if r_len <= 0: break
            sample = min(regret_sample, r_len)
            best_regret, best_delta, best_pos, best_idx = -1.0, 1e20, -1, -1
            for _ in range(sample):
                pick_idx = np.random.randint(0, r_len)
                city = removed_cities[pick_idx]
                d1, d2, pos = 1e20, 1e20, -1
                m_curr = len(current_tour)
                for j in range(m_curr):
                    u, v = current_tour[j], current_tour[(j + 1) % m_curr]
                    delta = D[u, city] + D[city, v] - D[u, v]
                    if delta < d1: d2 = d1; d1 = delta; pos = j
                    elif delta < d2: d2 = delta
                regret = d2 - d1
                if regret > best_regret: best_regret, best_delta, best_pos, best_idx = regret, d1, pos, pick_idx
            if best_idx == -1: break
            city = removed_cities[best_idx]
            new_t = np.empty(len(current_tour) + 1, np.int32)
            new_t[:best_pos+1] = current_tour[:best_pos+1]; new_t[best_pos+1] = city; new_t[best_pos+2:] = current_tour[best_pos+1:]
            current_tour = new_t; r_len -= 1; removed_cities[best_idx] = removed_cities[r_len]
    for i in range(r_len):
        city = removed_cities[i]
        best_delta, best_pos, m_curr = 1e20, -1, len(current_tour)
        for j in range(m_curr):
            u, v = current_tour[j], current_tour[(j + 1) % m_curr]
            delta = D[u, city] + D[city, v] - D[u, v]
            if delta < best_delta: best_delta, best_pos = delta, j
        new_t = np.empty(len(current_tour) + 1, np.int32)
        new_t[:best_pos+1] = current_tour[:best_pos+1]; new_t[best_pos+1] = city; new_t[best_pos+2:] = current_tour[best_pos+1:]
        current_tour = new_t
    return current_tour

@njit(cache=True, nogil=True)
def _ruin_and_recreate_jit(tour, D, ruin_pct, knn_idx=None):
    n = len(tour)
    n_remove = int(n * ruin_pct)
    if n_remove < 2: return tour.copy()
    if knn_idx is not None: mask = _bfs_ruin_mask_jit(n, knn_idx, n_remove)
    else:
        mask = np.zeros(n, np.bool_)
        s = np.random.randint(0, n)
        for i in range(n_remove): mask[tour[(s + i) % n]] = True
    removed_cities = np.empty(n_remove, np.int32)
    kept_cities = np.empty(n - n_remove, np.int32)
    kp, rp = 0, 0
    for i in range(n):
        if mask[tour[i]]: removed_cities[rp] = tour[i]; rp += 1
        else: kept_cities[kp] = tour[i]; kp += 1
    current_tour = kept_cities
    np.random.shuffle(removed_cities)
    for city in removed_cities:
        best_delta, best_pos, m_curr = 1e20, -1, len(current_tour)
        for i in range(m_curr):
            u, v = current_tour[i], current_tour[(i + 1) % m_curr]
            delta = D[u, city] + D[city, v] - D[u, v]
            if delta < best_delta: best_delta, best_pos = delta, i
        new_t = np.empty(len(current_tour) + 1, np.int32)
        new_t[:best_pos+1] = current_tour[:best_pos+1]; new_t[best_pos+1] = city; new_t[best_pos+2:] = current_tour[best_pos+1:]
        current_tour = new_t
    return current_tour

@njit(cache=True, parallel=True)
def init_population_jit(pop, D, finite_mask, knn_idx, strat_probs, seeds, rcl_r):
    lam, n = pop.shape
    for i in prange(lam):
        np.random.seed(seeds[i])  
        u = np.random.rand()
        if u < strat_probs[0]: 
            tour = _rcl_nn_tour_jit(D, finite_mask, knn_idx, rcl_r)
        elif u < strat_probs[0] + strat_probs[1]: 
            tour = _insertion_tour_jit(D, finite_mask, use_farthest=(np.random.rand() < 0.5))
        else:
            tour = _rand_perm_jit(n)
            # 尝试修复，最多 50 次 2-opt
            _repair_jit(tour, D, finite_mask, 50)
        
        # 【关键】确保路径可行，否则回退到安全的贪心方法
        if not _tour_feasible_jit(tour, finite_mask):
            tour = _rcl_nn_tour_jit(D, finite_mask, knn_idx, rcl_r)
        pop[i] = tour

@njit(cache=True)
def evolve_population_jit(population, c_pop, fitness, D, finite_mask, exploit_mut, is_symmetric):
    lam, n = population.shape
    for i in range(0, lam, 2):
        cand1 = np.random.choice(lam, 5, replace=False)
        p1 = cand1[np.argmin(fitness[cand1])]
        cand2 = np.random.choice(lam, 5, replace=False)
        p2 = cand2[np.argmin(fitness[cand2])]
        
        c1 = c_pop[i]
        c2 = c_pop[i+1]
        
        _ox_jit_inplace(population[p1], population[p2], c1)
        _ox_jit_inplace(population[p2], population[p1], c2)
        
        # Mutation & Repair C1
        if np.random.random() < exploit_mut:
            if is_symmetric and np.random.random() < 0.7:
                u = np.random.randint(0, n - 1); v = np.random.randint(u + 1, n)
                l, r = u, v - 1
                while l < r: tmp = c1[l]; c1[l] = c1[r]; c1[r] = tmp; l += 1; r -= 1
            else:
                u, v = np.random.randint(0, n), np.random.randint(0, n - 1)
                if v >= u: v += 1
                if u != v:
                    city = c1[u]
                    if v < u:
                        for k in range(u, v, -1): c1[k] = c1[k-1]
                    else:
                        for k in range(u, v): c1[k] = c1[k+1]
                    c1[v] = city
        # 【修复与回滚逻辑】区分对称和非对称
        c1_ok = False
        if is_symmetric:
            if _repair_jit(c1, D, finite_mask): c1_ok = True
        else:
            if _tour_feasible_jit(c1, finite_mask): c1_ok = True
        if not c1_ok:
            c1[:] = population[p1][:]  # 回滚到父代
        
        # Mutation & Repair C2
        if np.random.random() < exploit_mut:
            if is_symmetric and np.random.random() < 0.7:
                u = np.random.randint(0, n - 1); v = np.random.randint(u + 1, n)
                l, r = u, v - 1
                while l < r: tmp = c2[l]; c2[l] = c2[r]; c2[r] = tmp; l += 1; r -= 1
            else:
                u, v = np.random.randint(0, n), np.random.randint(0, n - 1)
                if v >= u: v += 1
                if u != v:
                    city = c2[u]
                    if v < u:
                        for k in range(u, v, -1): c2[k] = c2[k-1]
                    else:
                        for k in range(u, v): c2[k] = c2[k+1]
                    c2[v] = city
        # 【修复与回滚逻辑】区分对称和非对称
        c2_ok = False
        if is_symmetric:
            if _repair_jit(c2, D, finite_mask): c2_ok = True
        else:
            if _tour_feasible_jit(c2, finite_mask): c2_ok = True
        if not c2_ok:
            c2[:] = population[p2][:]  # 回滚到父代

# ==============================================================================
# Subprocess Worker
# ==============================================================================

def scout_worker(D, q_in, q_out, is_symmetric):
    try:
        n = D.shape[0]
        finite_mask = np.isfinite(D)
        knn_idx = build_knn_idx(D, finite_mask, 32)
        current_tour = _rcl_nn_tour_jit(D, finite_mask, knn_idx, 3)
        current_fit = tour_length_jit(current_tour, D)
        best_known_bound, dlb_mask = current_fit, np.zeros(n, dtype=np.bool_)
        iter_count, last_improv_iter, last_send_iter, scout_stagnation = 0, 0, 0, 0
        ruin_gears = np.array([0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
        patient_entry_fit = float('inf')
        
        # 内存优化：预分配所有 buffer
        pos_buffer = np.empty(n, dtype=np.int32)
        tour_buffer = np.empty(n, dtype=np.int32)
        rr_tour_buffer = np.empty(n, dtype=np.int32)  # Ruin & Recreate 专用
        rr_removed_buffer = np.empty(n, dtype=np.int32)
        
        # =====================================================
        # ALNS 自适应权重机制
        # =====================================================
        # 3 个 destroy 操作: 0=BFS, 1=Sequence, 2=Worst Edge
        num_operators = 3
        op_weights = np.array([1.0, 1.0, 1.0])  # 初始权重相等
        op_scores = np.zeros(num_operators)      # 累计得分
        op_usage = np.zeros(num_operators)       # 使用次数
        
        # ALNS 参数
        score_new_best = 10.0      # 发现新最优
        score_improvement = 4.0   # 比当前解更好
        score_accepted = 1.0      # 被接受（但不是更好）
        decay_factor = 0.8        # 权重衰减因子
        min_weight = 0.1          # 最小权重（避免操作被完全忽略）
        update_period = 100       # 每隔多少次迭代更新权重
        
        while True:
            iter_count += 1
            try:
                latest_patient = q_in.get_nowait()
                p_fit = tour_length_jit(latest_patient, D)
                current_tour[:], current_fit, dlb_mask[:] = latest_patient[:], p_fit, False
                patient_entry_fit, last_improv_iter, scout_stagnation, best_known_bound = p_fit, iter_count, 0, p_fit
            except queue.Empty: pass
            
            ruin_pct = ruin_gears[int((iter_count - last_improv_iter) // 250) % 10]
            
            # ALNS: 根据自适应权重选择 destroy 操作
            total_weight = op_weights.sum()
            probs = op_weights / total_weight
            rand_val = np.random.rand()
            if rand_val < probs[0]:
                mode = 0  # BFS
            elif rand_val < probs[0] + probs[1]:
                mode = 1  # Sequence
            else:
                mode = 2  # Worst Edge
            
            op_usage[mode] += 1
            
            # 使用原地版 Ruin & Recreate
            _hybrid_ruin_and_recreate_inplace(current_tour, D, ruin_pct, knn_idx, mode, rr_tour_buffer, rr_removed_buffer)
            candidate = rr_tour_buffer  # candidate 现在指向 buffer
            
            # 使用快速 VND（Scout 需要高频迭代，LK 太慢）
            dlb_mask[:], improved, block_steps = False, True, 10
            while improved:
                improved = False; dlb_mask[:] = False
                if _candidate_or_opt_jit(candidate, D, knn_idx, pos_buffer, tour_buffer, 5000, dlb_mask, 1): improved = True; continue
                dlb_mask[:] = False
                if _candidate_block_swap_jit(candidate, D, knn_idx, pos_buffer, tour_buffer, block_steps, dlb_mask, 2): improved = True; continue
                dlb_mask[:] = False
                if _candidate_or_opt_jit(candidate, D, knn_idx, pos_buffer, tour_buffer, block_steps, dlb_mask, 2): improved = True; continue
                dlb_mask[:] = False
                if _candidate_or_opt_jit(candidate, D, knn_idx, pos_buffer, tour_buffer, block_steps, dlb_mask, 3): improved = True; continue
            
            cand_fit = tour_length_jit(candidate, D); scout_stagnation += 1
            
            # ALNS: 根据结果更新得分
            if cand_fit < best_known_bound:
                op_scores[mode] += score_new_best
                best_known_bound = cand_fit
            elif cand_fit < current_fit:
                op_scores[mode] += score_improvement
            elif cand_fit <= current_fit * 1.001:  # 接近当前解
                op_scores[mode] += score_accepted
            
            gap = (cand_fit - patient_entry_fit) / patient_entry_fit if patient_entry_fit > 0 else 0
            is_breakthrough, tolerance = cand_fit < patient_entry_fit, 0.0
            if scout_stagnation > 500: tolerance = 0.003
            if scout_stagnation > 2000: tolerance = 0.008
            if is_breakthrough or ((gap <= tolerance) and (gap > -1.0) and (iter_count - last_send_iter > 200)):
                try:
                    q_out.put_nowait(candidate.copy()); last_send_iter = iter_count
                    if is_breakthrough: patient_entry_fit, scout_stagnation, last_improv_iter = cand_fit, 0, iter_count
                except queue.Full: pass
            if cand_fit <= current_fit: current_tour[:], current_fit = candidate[:], cand_fit
            
            # ALNS: 定期更新权重
            if iter_count % update_period == 0:
                for op in range(num_operators):
                    if op_usage[op] > 0:
                        avg_score = op_scores[op] / op_usage[op]
                        # 权重 = 衰减后的旧权重 + (1 - 衰减) * 平均得分
                        op_weights[op] = max(min_weight, 
                                             decay_factor * op_weights[op] + (1 - decay_factor) * avg_score)
                # 重置计数器
                op_scores[:] = 0
                op_usage[:] = 0
    except Exception: pass

# ==============================================================================
# Solver Class
# ==============================================================================

class r0927480:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.rng = np.random.default_rng()

    def optimize(self, filename):
        with open(filename) as f: distanceMatrix = np.loadtxt(f, delimiter=",")
        n = distanceMatrix.shape[0]
        lam, stagnation_limit, exploit_mut, exploit_ls = 200, 200, 0.3, 30
        if n < 300: lam, stagnation_limit = 1000, 500
        elif n < 600: lam, stagnation_limit, exploit_mut, exploit_ls = 150, 150, 0.15, 30
        elif n < 850: lam, stagnation_limit, exploit_mut, exploit_ls = 100, 200, 0.1, 10
        else: lam, stagnation_limit, exploit_mut, exploit_ls = 100, 150, 0.1, 10
        
        D = np.ascontiguousarray(distanceMatrix)
        is_symmetric = np.allclose(D, D.T, rtol=1e-5, atol=1e-8, equal_nan=True)
        finite_mask = np.isfinite(D); np.fill_diagonal(finite_mask, False)
        
        q_to_scout, q_from_scout = multiprocessing.Queue(maxsize=5), multiprocessing.Queue(maxsize=5)
        scout_process = multiprocessing.Process(target=scout_worker, args=(distanceMatrix, q_to_scout, q_from_scout, is_symmetric))
        scout_process.start()

        try:
            # 基础 KNN 候选集
            knn_idx = build_knn_idx(D, finite_mask, 32)
            
            # 计算 α-nearness 候选边并合并（增强候选集）
            K_alpha = 30 if n < 600 else 25  # 根据规模调整
            alpha_cand = compute_alpha_candidates(D, finite_mask, K_alpha)
            enhanced_knn = merge_candidates(knn_idx, alpha_cand, D, max_size=50)  # 合并候选
            print(f"[Init] 候选边集: KNN(32) + Alpha({K_alpha}) -> 合并后(50)")
            
            gls_penalties, gls_active, D_gls = np.zeros((n, n), dtype=np.int32), False, None
            population = np.empty((lam, n), dtype=np.int32)
            strat_probs = np.array([0.1, 0.3, 0.6], dtype=np.float64)
            seeds = np.random.randint(0, 1<<30, lam).astype(np.int64)
            init_population_jit(population, D, finite_mask, knn_idx, strat_probs, seeds, int(self.rng.integers(3, 11)))
            fitness = np.empty(lam, dtype=np.float64); batch_lengths_jit(population, D, fitness)
            best_ever_fitness, stagnation_counter, gen = fitness.min(), 0, 0
            current_run_best = best_ever_fitness  # 本轮最优（用于判定停滞）
            best_tour_ever = population[np.argmin(fitness)].copy()  # 全局最优解（用于报告）
            c_pop, c_fit, dlb_mask = np.empty((lam, n), dtype=np.int32), np.empty(lam, dtype=np.float64), np.zeros(n, dtype=np.bool_)
            last_patient_sent_time = 0.0
            
            # Main 进程 buffer
            main_pos_buffer = np.empty(n, dtype=np.int32)
            main_tour_buffer = np.empty(n, dtype=np.int32)

            while True:
                gen += 1
                
                # Scout Check
                try:
                    healed = q_from_scout.get_nowait()
                    h_fit = tour_length_jit(healed, D)
                    
                    # Debug: 诊断 Scout 结果是否有效
                    pop_mean = fitness.mean() if np.isfinite(fitness).all() else np.nanmean(fitness)
                    pop_min = fitness.min()
                    worst_idx = np.argmax(fitness)
                    population[worst_idx][:], fitness[worst_idx] = healed[:], h_fit
                    if h_fit < best_ever_fitness:
                        print(f"[Gen {gen}] Scout 贡献! {best_ever_fitness:.2f} -> {h_fit:.2f}")
                        best_ever_fitness, stagnation_counter, gls_penalties[:], gls_active = h_fit, 0, 0, False
                except queue.Empty: pass

                D_ls = D_gls if (gls_active and D_gls is not None) else D
                evolve_population_jit(population, c_pop, fitness, D, finite_mask, exploit_mut, is_symmetric)
                
                batch_lengths_jit(c_pop, D, c_fit)
                
                # 分级局部搜索策略
                elite_count = max(1, int(lam * 0.2))
                elite_indices = np.argsort(c_fit)[:elite_count]
                
                # Top 3 精英使用 Iterated LK（强力深挖）
                top_elite_count = min(3, len(elite_indices))
                for i in range(top_elite_count):
                    idx = elite_indices[i]
                    # 根据问题规模选择参数
                    kicks = 5 if n < 400 else (3 if n < 700 else 2)
                    c_pop[idx], c_fit[idx] = iterated_lk(
                        c_pop[idx], D_ls, enhanced_knn,  # 使用增强候选集
                        max_kicks=kicks, lk_depth=5, lk_branch=8, lk_iters=300
                    )
                
                # 其余精英使用快速 VND
                for i in range(top_elite_count, len(elite_indices)):
                    idx = elite_indices[i]
                    dlb_mask[:] = False
                    self._vnd_or_opt_inplace(c_pop[idx], D_ls, knn_idx, dlb_mask, exploit_ls, 3, main_pos_buffer, main_tour_buffer)
                    c_fit[idx] = tour_length_jit(c_pop[idx], D)

                cur_best_idx = np.argmin(fitness)
                for i in range(lam):
                    better, tidx = rtr_challenge_jit(c_pop[i], c_fit[i], population, fitness, min(lam, 50), int(self.rng.integers(0, 1<<30)), cur_best_idx)
                    if better: population[tidx][:], fitness[tidx] = c_pop[i][:], c_fit[i]
                
                best_idx = np.argmin(fitness); bestObjective = float(fitness[best_idx])
                
                # 1. 判定是否打破"本轮"最优（只要本轮还在进步，就给机会继续挖）
                if bestObjective < current_run_best:
                    current_run_best = bestObjective
                    stagnation_counter = 0  # 只要本轮在进步，就清零！
                else:
                    stagnation_counter += 1  # 真的挖不动了才累加
                
                # 2. 判定是否打破"历史"最优（用于记录和报告）
                if bestObjective < best_ever_fitness:
                    print(f"[Gen {gen}] 新最优! {best_ever_fitness:.2f} -> {bestObjective:.2f} (提升 {best_ever_fitness - bestObjective:.2f})")
                    best_ever_fitness = bestObjective
                    best_tour_ever = population[best_idx].copy()
                    stagnation_counter = 0  # 打破历史记录当然也清零
                    gls_penalties[:], gls_active = 0, False
                
                if stagnation_counter > (stagnation_limit // 2) and (time.time() - last_patient_sent_time > 5.0):
                    try: q_to_scout.put_nowait(population[best_idx].copy()); last_patient_sent_time = time.time()
                    except queue.Full: pass
                
                if stagnation_counter >= max(30, int(stagnation_limit * 0.6)):
                    if not gls_active:
                        print(f"[Gen {gen}] GLS 激活 (停滞 {stagnation_counter})")
                    gls_active = True
                    if gen % 50 == 0:
                        self._gls_update_penalties(population[best_idx], D, gls_penalties)
                        D_gls = np.ascontiguousarray(D + (0.03 * (bestObjective / n)) * gls_penalties)
                else: gls_active = False
                
                if stagnation_counter >= stagnation_limit:
                    print(f"[Gen {gen}] 重启! 当前最优 {best_ever_fitness:.2f}")
                    # 1. 清空 Scout 发回的旧消息
                    while not q_from_scout.empty():
                        try: q_from_scout.get_nowait()
                        except queue.Empty: break
                    
                    # 2. 彻底的 GLS 遗忘
                    gls_penalties[:] = 0
                    gls_active = False
                    D_gls = None
                    
                    # 3. 【70/30 混合策略】
                    # Part A: 70% 完全重新生成（模拟手动重启 python verify_submission.py）
                    reset_count = int(lam * 0.7)
                    restart_strat_probs = np.array([0.05, 0.15, 0.8], dtype=np.float64)  # 80% 随机
                    init_population_jit(population[:reset_count], D, finite_mask, knn_idx, restart_strat_probs, 
                                        np.random.randint(0, 1<<30, reset_count).astype(np.int64), 
                                        int(self.rng.integers(15, 40)))  # rcl_r 很大
                    
                    # Part B: 30% 保留旧皇血脉（1-3次双桥变异）
                    for i in range(reset_count, lam):
                        mutated = best_tour_ever.copy()
                        kicks = int(self.rng.integers(1, 4))  # 1-3 次双桥
                        for _ in range(kicks):
                            mutated = double_bridge_move(mutated)
                        population[i] = mutated
                    
                    batch_lengths_jit(population, D, fitness)                    
                    # 4. 让 Scout 也同步到新区域
                    try:
                        q_to_scout.put_nowait(population[np.argmin(fitness)].copy())
                    except queue.Full:
                        try:
                            q_to_scout.get_nowait()
                            q_to_scout.put_nowait(population[np.argmin(fitness)].copy())
                        except: pass
                    
                    # 5. 重置本轮最优和停滞计数器
                    current_run_best = fitness.min()  # 让新种群从新起点开始计算
                    stagnation_counter = 0

                # 报告时使用全局最优解 best_tour_ever
                start_pos = np.where(best_tour_ever == 0)[0]
                bestSolution = np.concatenate((best_tour_ever[start_pos[0]:], best_tour_ever[:start_pos[0]])) if start_pos.size > 0 else best_tour_ever
                
                if self.reporter.report(float(fitness.mean()), best_ever_fitness, bestSolution) < 0: break
            return 0
        finally:
            if scout_process.is_alive(): scout_process.terminate(); scout_process.join()

    def _vnd_or_opt_inplace(self, tour, D, knn_idx, dlb_mask, max_iters, block_steps, pos_buf, tour_buf):
        improved = True
        while improved:
            improved = False; dlb_mask[:] = False
            if _candidate_or_opt_jit(tour, D, knn_idx, pos_buf, tour_buf, max_iters, dlb_mask, 1): improved = True; continue
            dlb_mask[:] = False
            if _candidate_block_swap_jit(tour, D, knn_idx, pos_buf, tour_buf, block_steps, dlb_mask, 2): improved = True; continue
            dlb_mask[:] = False
            if _candidate_or_opt_jit(tour, D, knn_idx, pos_buf, tour_buf, block_steps, dlb_mask, 2): improved = True; continue
            dlb_mask[:] = False
            if _candidate_or_opt_jit(tour, D, knn_idx, pos_buf, tour_buf, block_steps, dlb_mask, 3): improved = True; continue

    def _gls_update_penalties(self, tour, D, penalties):
        n, max_util = tour.shape[0], -1.0
        for i in range(n):
            a, b = tour[i], tour[(i + 1) % n]
            if np.isfinite(D[a, b]):
                util = D[a, b] / (1.0 + penalties[a, b])
                if util > max_util: max_util = util
        if max_util < 0.0: return
        for i in range(n):
            a, b = tour[i], tour[(i + 1) % n]
            if np.isfinite(D[a, b]) and (D[a, b] / (1.0 + penalties[a, b])) >= max_util - 1e-12: penalties[a, b] += 1

if __name__ == '__main__':
    multiprocessing.freeze_support()
