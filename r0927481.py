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

# 诊断模块：对比 LKH3 最佳路径
try:
    from diagnose_gap import init_lkh_reference, quick_diagnose, advanced_diagnose, diagnose_full
    DIAGNOSE_AVAILABLE = True
except ImportError:
    DIAGNOSE_AVAILABLE = False

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
            # ✅ 策略二：10% 概率使用核武器 (Double Bridge)
            if np.random.random() < 0.1:
                # Double Bridge 可以用外部函数，但 JIT 内改为内联版本
                # 断开 4 条边，交错重连
                seg_len = n // 4
                if seg_len > 1:
                    i1 = np.random.randint(1, seg_len)
                    i2 = i1 + np.random.randint(1, seg_len)
                    i3 = i2 + np.random.randint(1, seg_len)
                    # 重组: [0:i1] + [i3:] + [i2:i3] + [i1:i2] → 简化版本
                    temp = c1.copy()
                    ptr = 0
                    for t in range(0, i1): c1[ptr] = temp[t]; ptr += 1
                    for t in range(i3, n): c1[ptr] = temp[t]; ptr += 1
                    for t in range(i2, i3): c1[ptr] = temp[t]; ptr += 1
                    for t in range(i1, i2): c1[ptr] = temp[t]; ptr += 1
            elif is_symmetric and np.random.random() < 0.7:
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
        # ✅ 修复2：Scout "核爆模式" - 增加超高破坏档位
        ruin_gears = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.55, 0.6])
        patient_entry_fit = float('inf')
        
        # 内存优化：预分配所有 buffer
        pos_buffer = np.empty(n, dtype=np.int32)
        tour_buffer = np.empty(n, dtype=np.int32)
        rr_tour_buffer = np.empty(n, dtype=np.int32)  # Ruin & Recreate 专用
        rr_removed_buffer = np.empty(n, dtype=np.int32)
        
        while True:
            iter_count += 1
            try:
                latest_patient = q_in.get_nowait()
                p_fit = tour_length_jit(latest_patient, D)
                current_tour[:], current_fit, dlb_mask[:] = latest_patient[:], p_fit, False
                patient_entry_fit, last_improv_iter, scout_stagnation, best_known_bound = p_fit, iter_count, 0, p_fit
            except queue.Empty: pass
            
            ruin_pct = ruin_gears[int((iter_count - last_improv_iter) // 250) % 10]
            
            rand_val = np.random.rand()
            # ✅ 修复2：大幅提高 Worst Edge Ruin 的概率，定向爆破长边
            if rand_val < 0.3: 
                mode = 0  # BFS (降低到 30%)
            elif rand_val < 0.7: 
                mode = 2  # Worst Edge (提升到 40%!) 炸掉那些长边
            else: 
                mode = 1  # Sequence (30%)
            
            # ✅ 方案B：在 Ruin & Recreate 前先用 Double Bridge 打乱结构
            # 当停滞严重时，增加 Double Bridge 的概率
            if scout_stagnation > 500 and np.random.rand() < 0.5:
                # 应用 1-3 次 Double Bridge
                kicks = np.random.randint(1, 4)
                for _ in range(kicks):
                    current_tour[:] = double_bridge_move(current_tour)
                current_fit = tour_length_jit(current_tour, D)
            
            # 使用原地版 Ruin & Recreate
            _hybrid_ruin_and_recreate_inplace(current_tour, D, ruin_pct, knn_idx, mode, rr_tour_buffer, rr_removed_buffer)
            candidate = rr_tour_buffer  # candidate 现在指向 buffer
            
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
            if cand_fit < best_known_bound: best_known_bound = cand_fit
            gap = (cand_fit - patient_entry_fit) / patient_entry_fit if patient_entry_fit > 0 else 0
            is_breakthrough = cand_fit < patient_entry_fit
            
            # ✅ 速射模式：放宽 Scout 准入
            if scout_stagnation > 1000:
                tolerance = 0.002  # 仍然严格但不变态
            elif scout_stagnation > 500:
                tolerance = 0.005  # 放宽
            else:
                tolerance = 0.002  # 平时也稍微宽松
            
            # ✅ 修复3：防止 Scout "克隆" - 如果 gap 极其接近 0 且不是突破，忽略
            if abs(gap) < 1e-6 and not is_breakthrough:
                pass  # 它是当前最优解的克隆，且没有突破，忽略它
            elif is_breakthrough or ((gap <= tolerance) and (gap > -1.0) and (iter_count - last_send_iter > 200)):
                try:
                    q_out.put_nowait(candidate.copy()); last_send_iter = iter_count
                    if is_breakthrough: patient_entry_fit, scout_stagnation, last_improv_iter = cand_fit, 0, iter_count
                except queue.Full: pass
            if cand_fit <= current_fit: current_tour[:], current_fit = candidate[:], cand_fit
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
            knn_idx = build_knn_idx(D, finite_mask, 32)
            gls_penalties, gls_active, D_gls = np.zeros((n, n), dtype=np.int32), False, None
            
            # 【诊断】初始化 LKH 参考路径
            if DIAGNOSE_AVAILABLE:
                lkh_ref_file = f"best_route_{filename.replace('.csv', '')}.txt"
                init_lkh_reference(lkh_ref_file)
                diagnose_interval = 50  # 每 50 代诊断一次
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
            
            # Scout 统计变量
            scout_total = 0       # Scout 发送解的总次数
            scout_accepted = 0    # 主进程采纳的次数
            scout_breakthrough = 0  # Scout 打破全局最优的次数
            
            # Main 进程 buffer
            main_pos_buffer = np.empty(n, dtype=np.int32)
            main_tour_buffer = np.empty(n, dtype=np.int32)

            while True:
                gen += 1
                
                # Scout Check
                try:
                    healed = q_from_scout.get_nowait()
                    h_fit = tour_length_jit(healed, D)
                    scout_total += 1  # Scout 发送了一个解
                    
                    # Debug: 诊断 Scout 结果是否有效
                    pop_mean = fitness.mean() if np.isfinite(fitness).all() else np.nanmean(fitness)
                    pop_min = fitness.min()
                    
                    # 判断是否采纳：如果比最差个体好就接受
                    worst_idx = np.argmax(fitness)
                    if h_fit < fitness[worst_idx]:
                        scout_accepted += 1
                        population[worst_idx][:], fitness[worst_idx] = healed[:], h_fit
                    
                    if h_fit < best_ever_fitness:
                        scout_breakthrough += 1
                        print(f"[Gen {gen}] Scout 突破! {best_ever_fitness:.2f} -> {h_fit:.2f} (贡献 #{scout_breakthrough})")
                        best_ever_fitness, stagnation_counter, gls_penalties[:], gls_active = h_fit, 0, 0, False
                except queue.Empty: pass

                D_ls = D_gls if (gls_active and D_gls is not None) else D
                evolve_population_jit(population, c_pop, fitness, D, finite_mask, exploit_mut, is_symmetric)
                
                batch_lengths_jit(c_pop, D, c_fit)
                
                elite_count = max(1, int(lam * 0.2))
                elite_indices = np.argsort(c_fit)[:elite_count]
                for idx in elite_indices:
                    dlb_mask[:] = False
                    self._vnd_or_opt_inplace(c_pop[idx], D_ls, knn_idx, dlb_mask, exploit_ls, 3, main_pos_buffer, main_tour_buffer)
                    c_fit[idx] = tour_length_jit(c_pop[idx], D)

                cur_best_idx = np.argmin(fitness)
                for i in range(lam):
                    better, tidx = rtr_challenge_jit(c_pop[i], c_fit[i], population, fitness, min(lam, 50), int(self.rng.integers(0, 1<<30)), cur_best_idx)
                    if better: population[tidx][:], fitness[tidx] = c_pop[i][:], c_fit[i]
                
                best_idx = np.argmin(fitness); bestObjective = float(fitness[best_idx])
                
                # 1. 判定是否打破"本轮"最优（只有显著改进才重置计数器）
                improvement = (current_run_best - bestObjective) / current_run_best if current_run_best > 0 else 0
                
                if improvement > 0.0005:  # 只有超过 0.05% 的改进才算打破停滞
                    current_run_best = bestObjective
                    stagnation_counter = 0
                elif bestObjective < current_run_best:
                    # 微小的改进不重置停滞计数，让 GLS 有机会启动
                    current_run_best = bestObjective  # 更新最优值，但不重置计数
                    stagnation_counter += 1
                else:
                    stagnation_counter += 1  # 真的挖不动了才累加
                
                # 2. 判定是否打破"历史"最优（用于记录和报告）
                if bestObjective < best_ever_fitness:
                    best_ever_fitness = bestObjective
                    best_tour_ever = population[best_idx].copy()
                    stagnation_counter = 0  # 打破历史记录当然也清零
                    
                    # ✅ 速射模式：软衰减 0.5x（更快收敛）
                    gls_penalties[:] = (gls_penalties * 0.5).astype(np.int32)
                    print(f"[Gen {gen}] New best! GLS Decayed (0.5x). Max={gls_penalties.max()}, NonZero={np.count_nonzero(gls_penalties)}")
                
                if stagnation_counter > (stagnation_limit // 2) and (time.time() - last_patient_sent_time > 5.0):
                    try: q_to_scout.put_nowait(population[best_idx].copy()); last_patient_sent_time = time.time()
                    except queue.Full: pass
                
                # ✅ 修复1：GLS "重炮模式" - 严重停滞时每代更新 + 加大 lambda
                is_severe_stagnation = stagnation_counter >= max(50, int(stagnation_limit * 0.8))
                
                if gen > 1000 or stagnation_counter >= max(30, int(stagnation_limit * 0.6)):
                    gls_active = True
                    
                    # Frequency boost: severe stagnation updates every gen
                    update_freq = 1 if is_severe_stagnation else 5
                    
                    # ✅ 速射模式：加快 Boost 频率（30 代一次）
                    if gls_active and gen % 30 == 0 and stagnation_counter > 10:
                        mask = gls_penalties > 0
                        if np.any(mask):
                            old_max = gls_penalties.max()
                            gls_penalties[mask] += 5
                            print(f"[Gen {gen}] Penalty BOOST! +5. Max: {old_max} -> {gls_penalties.max()}")
                    
                    # ✅ 修正2：重正化 (Renormalization) - 防止焦土效应
                    current_max_penalty = gls_penalties.max()
                    if current_max_penalty > 30:
                        nonzero_mask = gls_penalties > 0
                        if np.any(nonzero_mask):
                            min_nonzero = gls_penalties[nonzero_mask].min()
                            if min_nonzero > 5:
                                decay_val = min_nonzero - 1
                                gls_penalties[:] = np.maximum(0, gls_penalties - decay_val)
                                print(f"[Gen {gen}] 💧 GLS Renormalized! -{decay_val}. Max: {current_max_penalty} -> {gls_penalties.max()}")
                    
                    if gen % update_freq == 0:
                        # Pass best_tour_ever for elite protection
                        self._gls_update_penalties(population[best_idx], D, gls_penalties, best_tour_ever)
                        
                        # Vanilla GLS: moderate lambda (no protection = need gentler pressure)
                        lambda_val = 0.05  # Balanced: not too aggressive, not too weak
                        D_gls = np.ascontiguousarray(D + (lambda_val * (bestObjective / n)) * gls_penalties)
                else: 
                    gls_active = False
                
                if stagnation_counter >= stagnation_limit:                    
                    # 1. 清空 Scout 发回的旧消息
                    while not q_from_scout.empty():
                        try: q_from_scout.get_nowait()
                        except queue.Empty: break
                    
                    # ✅ 方案A：保留 GLS 惩罚！不要清空！
                    # GLS 是跨越重启的“长期记忆”，必须保留
                    # 只有当 best_ever_fitness 被打破时，才应该清空 GLS
                    # gls_penalties[:] = 0  # ← 删除这行！
                    # gls_active = False    # ← 删除这行！
                    # D_gls = None          # ← 删除这行！
                    print(f"[Gen {gen}] 重启! GLS 保留 (Max={gls_penalties.max()}, 非零={np.count_nonzero(gls_penalties)})")
                    
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

                # 【诊断】周期性输出与 LKH3 最佳路径的对比报告
                if DIAGNOSE_AVAILABLE and gen % diagnose_interval == 0:
                    advanced_diagnose(
                        best_tour_ever, D, 
                        population=population, 
                        gls_penalties=gls_penalties,
                        scout_accepted=scout_accepted, 
                        scout_total=scout_total,
                        label=f"Gen {gen}"
                    )
                    # 同时输出 diagnose_full 的完整边分析
                    from diagnose_gap import _LKH_ROUTE
                    if _LKH_ROUTE is not None:
                        diagnose_full(best_tour_ever, _LKH_ROUTE, D, knn_idx, label=f"Gen {gen} [Full]")
                
                # 🧪 Golden DNA Injection Test (Gen 1000) - DISABLED
                # if gen == 1000 and DIAGNOSE_AVAILABLE:
                #     from diagnose_gap import _LKH_ROUTE, create_golden_individual
                #     if _LKH_ROUTE is not None:
                #         print(f"[Gen {gen}] 💉 注入黄金基因 (95% LKH)...")
                #         golden_tour = create_golden_individual(D, _LKH_ROUTE, ruin_percent=0.1)
                #         golden_fit = tour_length_jit(golden_tour, D)
                #         
                #         worst_idx = np.argmax(fitness)
                #         population[worst_idx][:] = golden_tour[:]
                #         fitness[worst_idx] = golden_fit
                #         
                #         lkh_opt = 102464.22
                #         print(f"    ↳ 注入个体 Fitness: {golden_fit:.2f} (Gap: {(golden_fit - lkh_opt)/lkh_opt*100:.2f}%)")
                #         
                #         gls_penalties[:] = 0
                #         gls_active = False
                #         print(f"    ↳ GLS 已清空，给黄金基因一个公平的起点")
                
                # 🧪 Topology Analysis (every 500 gens)
                # if DIAGNOSE_AVAILABLE and gen % 500 == 0:
                #     from diagnose_gap import _LKH_ROUTE, analyze_missing_topology
                #     if _LKH_ROUTE is not None:
                #         analyze_missing_topology(best_tour_ever, _LKH_ROUTE)
                
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

    def _gls_update_penalties(self, tour, D, penalties, best_tour_ever=None):
        """
        Vanilla GLS: No protection, pure mathematical elegance
        - High utility edges get punished
        - No elite/short protection - let the math do the work
        """
        n = tour.shape[0]
        
        # 1. Calculate max utility
        max_util = -1.0
        for i in range(n):
            u, v = tour[i], tour[(i + 1) % n]
            if np.isfinite(D[u, v]):
                util = D[u, v] / (1.0 + penalties[u, v])
                if util > max_util:
                    max_util = util
        
        if max_util <= 0:
            return
        
        # 2. Simple rule: high utility = punishment
        # 0.9 threshold for saturation bombing
        util_threshold = max_util * 0.9
        punished_cnt = 0
        
        for i in range(n):
            u, v = tour[i], tour[(i + 1) % n]
            if np.isfinite(D[u, v]):
                util = D[u, v] / (1.0 + penalties[u, v])
                if util >= util_threshold:
                    # NO protection logic - just punish!
                    penalties[u, v] += 1
                    punished_cnt += 1
        
        # Debug (sparse output)
        # if punished_cnt > 0:
        #     print(f"    [Vanilla GLS] Punished: {punished_cnt}, Max={penalties.max()}")

if __name__ == '__main__':
    multiprocessing.freeze_support()