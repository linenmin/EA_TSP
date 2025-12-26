import os

# Configure environment for strictly 1 thread per process (2 processes total)
# MUST BE SET BEFORE IMPORTING NUMBA/NUMPY to take effect reliably
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import Reporter
import numpy as np
import multiprocessing
import queue
import time
from numba import njit, prange, set_num_threads

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
def _scx_jit_inplace_ok(p1, p2, D, finite_mask, knn_idx, child, map1, map2, used):
    """Fail-Fast SCX: 增强版 - KNN不行时尝试全图搜索 + 闭环修补 + 反死路 Lookahead
    
    Returns:
        int: 0=成功, 1=中途死路, 2=闭环失败
    """
    n = p1.shape[0]

    # 1. 建立映射表
    for i in range(n - 1):
        map1[p1[i]] = p1[i + 1]
        map2[p2[i]] = p2[i + 1]
    map1[p1[n - 1]] = p1[0]
    map2[p2[n - 1]] = p2[0]

    used[:] = False

    # 2. 随机起点
    start = p1[np.random.randint(0, n)]
    cur = start
    child[0] = cur
    used[cur] = True

    K = knn_idx.shape[1]

    for i in range(1, n):
        n1 = map1[cur]
        n2 = map2[cur]
        chosen = -1      
        # A. 优先尝试继承父代 (Greedy)
        c1 = (not used[n1]) and finite_mask[cur, n1]
        c2 = (not used[n2]) and finite_mask[cur, n2]
        if c1 and c2:
            if D[cur, n1] < D[cur, n2]: chosen = n1
            elif D[cur, n2] < D[cur, n1]: chosen = n2
            else: chosen = n1 if (np.random.random() < 0.5) else n2
        elif c1: chosen = n1
        elif c2: chosen = n2        
        # B. 父代不行，尝试 KNN (快速)
        if chosen == -1:
            for k in range(K):
                nb = knn_idx[cur, k]
                if nb != -1 and (not used[nb]) and finite_mask[cur, nb]:
                    chosen = nb
                    break        
        # C. 【关键修正】KNN也不行，做最后的全图线性扫描 (O(N))
        if chosen == -1:
            best_backup = -1
            best_dist = 1e20
            # 扫描所有节点找到一个未访问且可达的
            for candidate in range(n):
                if not used[candidate] and finite_mask[cur, candidate]:
                    d = D[cur, candidate]
                    if d < best_dist:
                        best_dist = d
                        best_backup = candidate
            chosen = best_backup
        # D. 真的全图都无路可走了 (死胡同)
        if chosen == -1:
            return 1  # 返回错误码 1: 中途死路        
        if i < n - 1:  # 不是最后一步时才检查（最后一步只需闭环）
            has_future = False
            # 快速扫描 chosen 的 KNN
            for k in range(K):
                next_nb = knn_idx[chosen, k]
                if next_nb != -1 and (not used[next_nb]) and finite_mask[chosen, next_nb]:
                    has_future = True
                    break            
            # 如果 KNN 里没有未来，再尝试全图扫描一次（防止 K 太小）
            if not has_future:
                for candidate in range(n):
                    if not used[candidate] and finite_mask[chosen, candidate]:
                        has_future = True
                        break            
            # 如果 chosen 无未来，尝试换用第二候选
            if not has_future:
                # 尝试找一个有未来的替代候选
                alternative = -1
                # 先从 KNN 找
                for k in range(K):
                    alt_nb = knn_idx[cur, k]
                    if alt_nb != -1 and (not used[alt_nb]) and finite_mask[cur, alt_nb]:
                        # 检查这个候选是否有未来
                        alt_has_future = False
                        for k2 in range(K):
                            next_nb = knn_idx[alt_nb, k2]
                            if next_nb != -1 and (not used[next_nb]) and finite_mask[alt_nb, next_nb]:
                                alt_has_future = True
                                break
                        if alt_has_future:
                            alternative = alt_nb
                            break                
                # 如果找到了更好的候选，替换
                if alternative != -1:
                    chosen = alternative
        child[i] = chosen
        used[chosen] = True
        cur = chosen

    # 3. 闭环检查与简单修补
    last = child[n - 1]
    if not finite_mask[last, start]:
        if n > 3:
            prev = child[n - 2]
            pprev = child[n - 3]
            if (finite_mask[pprev, last] and 
                finite_mask[last, prev] and 
                finite_mask[prev, start]):
                # 执行交换
                child[n - 1] = prev
                child[n - 2] = last
                return 0  # 成功：闭环修补成功
        return 2  # 返回错误码 2: 闭环失败

    return 0  # 返回错误码 0: 成功

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
                    # 在move前保存block的城市ID（避免move后索引失效）
                    block_cities = np.empty(block_size, dtype=np.int32)
                    for b in range(block_size):
                        block_cities[b] = tour[u_idx + b]
                    
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
                        # 使用保存的block城市ID更新DLB
                        for b in range(block_size): 
                            dlb_mask[block_cities[b]] = False
                    break 
            if move_found: continue 
            else: 
                if use_dlb: dlb_mask[block_head] = True
        if not found_in_try and use_dlb: break
    return improved


@njit(cache=True, fastmath=True)
def _candidate_2opt_jit(tour, D, knn_idx, pos_buf, max_iters=100, dlb_mask=None):
    n = tour.shape[0]
    K = knn_idx.shape[1]
    eps = 1e-6
    
    # 建立位置映射
    for i in range(n):
        pos_buf[tour[i]] = i
    
    improved = False
    use_dlb = (dlb_mask is not None)
    
    for _ in range(max_iters):
        found_in_iter = False
        start = np.random.randint(0, n)
        
        for offset in range(n):
            i = (start + offset) % n
            a = tour[i]
            
            if use_dlb and dlb_mask[a]:
                continue
            
            b = tour[(i + 1) % n]
            
            # 检查边 (a, b) 可行性
            if not np.isfinite(D[a, b]):
                continue
            
            move_found = False
            
            # 遍历 a 的 KNN 候选
            for k in range(K):
                c = knn_idx[a, k]
                if c == -1:
                    break
                
                j = pos_buf[c]
                
                # 跳过相邻和无效位置
                # 2-opt 要求 i < j 且不相邻
                if j <= i or j == (i + 1) % n or j == i:
                    continue
                
                d = tour[(j + 1) % n]
                
                # 检查新边可行性
                if not np.isfinite(D[c, d]) or not np.isfinite(D[a, c]) or not np.isfinite(D[b, d]):
                    continue
            
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                
                if delta < -eps:
                    left = (i + 1) % n
                    right = j
                    
                    # 反转段 [left, right]
                    if left < right:
                        tour[left:right+1] = tour[left:right+1][::-1]
                    else:
                        continue
                    
                    # 更新位置映射（反转后的节点位置变化）
                    for idx in range(left, right + 1):
                        pos_buf[tour[idx]] = idx
                    
                    improved = True
                    move_found = True
                    found_in_iter = True
                    
                    # 更新 DLB
                    if use_dlb:
                        dlb_mask[a] = False
                        dlb_mask[b] = False
                        dlb_mask[c] = False
                        dlb_mask[d] = False
                        # 反转段内的所有节点都需要重新检查
                        for idx in range(left, right + 1):
                            dlb_mask[tour[idx]] = False                    
                    break            
            if move_found:
                continue
            else:
                if use_dlb:
                    dlb_mask[a] = True        
        if not found_in_iter and use_dlb:
            break    
    return improved


@njit(cache=True, fastmath=True)
def _candidate_blockswap3_jit(tour, D, knn_idx, pos_buf, max_iters=100, dlb_mask=None):
    n = tour.shape[0]
    K = knn_idx.shape[1]
    eps = 1e-6    
    # 建立位置映射
    for idx in range(n):
        pos_buf[tour[idx]] = idx    
    improved = False
    use_dlb = (dlb_mask is not None)    
    # 离散长度集合（segment B 的长度）
    segment_lengths = np.array([1, 2, 3, 5, 8], dtype=np.int32)    
    for _ in range(max_iters):
        found_in_iter = False
        start = np.random.randint(0, n)        
        for offset in range(n):
            i = (start + offset) % n
            a = tour[i]            
            if use_dlb and dlb_mask[a]:
                continue            
            # 限制：确保 k+1 不越界
            if i >= n - 3:  # 至少需要 3 个点
                continue
            
            b = tour[i + 1]  # segment A 的起点            
            # 检查边 a->b 可行性
            if not np.isfinite(D[a, b]):
                continue            
            move_found = False
            for k_idx in range(K):
                d = knn_idx[a, k_idx]
                if d == -1:
                    break
                if not np.isfinite(D[a, d]):
                    continue                
                j_pos = pos_buf[d]                
                j = j_pos - 1                
                # 确保 i < j < n-1
                if j <= i or j >= n - 1:
                    continue                
                c = tour[j]  # segment A 的终点
                if not np.isfinite(D[c, d]):
                    continue
                for seg_len in segment_lengths:
                    k = j + seg_len
                    if k >= n - 1:
                        break                    
                    e = tour[k]      # segment B 的终点
                    f = tour[k + 1]  # segment B 之后
                    if not np.isfinite(D[e, f]) or not np.isfinite(D[e, b]) or not np.isfinite(D[c, f]):
                        continue
                    old_cost = D[a, b] + D[c, d] + D[e, f]
                    new_cost = D[a, d] + D[e, b] + D[c, f]
                    delta = new_cost - old_cost                    
                    if delta < -eps:        
                        len_A = j - i
                        len_B = k - j                    
                        temp_A = np.empty(len_A, dtype=np.int32)
                        for idx in range(len_A):
                            temp_A[idx] = tour[i + 1 + idx]
                        for idx in range(len_B):
                            tour[i + 1 + idx] = tour[j + 1 + idx]
                        for idx in range(len_A):
                            tour[i + 1 + len_B + idx] = temp_A[idx]
                        for idx in range(i + 1, k + 1):
                            pos_buf[tour[idx]] = idx                        
                        improved = True
                        move_found = True
                        found_in_iter = True
                        
                        # 更新 DLB
                        if use_dlb:
                            dlb_mask[a] = False
                            dlb_mask[b] = False
                            dlb_mask[c] = False
                            dlb_mask[d] = False
                            dlb_mask[e] = False
                            dlb_mask[f] = False
                            # 受影响的所有节点
                            for idx in range(i + 1, k + 1):
                                dlb_mask[tour[idx]] = False                        
                        break  # 找到一个改进就跳出                
                if move_found:
                    break            
            if move_found:
                continue
            else:
                if use_dlb:
                    dlb_mask[a] = True        
        if not found_in_iter and use_dlb:
            break    
    return improved


@njit(cache=True, fastmath=True)
def _candidate_block_swap_jit(tour, D, knn_idx, pos_buf, tour_buf, max_iters=50, dlb_mask=None, block_size=2):
    """内存优化版 Block Swap：接收外部 buffer，零内存分配"""
    n = tour.shape[0]
    K = knn_idx.shape[1]
    block_size = int(block_size)
    if block_size < 1: block_size = 1
    if block_size * 2 >= n: return False 
    

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
    """强壮版 RCL：内存优化 + 保证 100% 返回可行解"""
    n = D.shape[0]
    tour = np.empty(n, np.int32)
    used = np.zeros(n, np.uint8)
    
    # 【优化】将临时数组分配移出循环
    K = knn_idx.shape[1]
    tmp_idx = np.empty(K, np.int32)
    tmp_dis = np.empty(K, np.float64)
    
    # 无限重试直到找到可行闭环
    while True:
        cur = np.random.randint(0, n)  
        tour[0] = cur
        used[:] = 0
        used[cur] = 1        
        valid_tour = True        
        for t in range(1, n):
            cnt = 0
            # 1. 收集可行 KNN
            for k in range(K):
                j = knn_idx[cur, k]
                if j == -1: break
                if used[j] == 0 and finite_mask[cur, j]:
                    tmp_idx[cnt] = j
                    tmp_dis[cnt] = D[cur, j]
                    cnt += 1
            
            nxt = -1
            if cnt == 0:
                # 2. 全局搜索兜底
                best_dist = 1e20
                for j in range(n):
                    if used[j] == 0 and finite_mask[cur, j]: 
                        if D[cur, j] < best_dist:
                            best_dist = D[cur, j]
                            nxt = j
                
                # 3. 绝望模式：随便找一个未访问的 (虽然会导致断裂，但在下一次 while True 会重置)
                if nxt == -1:
                    for j in range(n):
                        if used[j] == 0: nxt = j; break
            else:
                # 4. RCL 选择
                # 简单的选择排序找前 limit 个最小值 (比 full argsort 快)
                limit = min(r, cnt)
                for i in range(limit):
                    min_idx = i
                    for j in range(i + 1, cnt):
                        if tmp_dis[j] < tmp_dis[min_idx]:
                            min_idx = j
                    # Swap
                    tmp_dis[i], tmp_dis[min_idx] = tmp_dis[min_idx], tmp_dis[i]
                    tmp_idx[i], tmp_idx[min_idx] = tmp_idx[min_idx], tmp_idx[i]
                
                pick = np.random.randint(0, limit)
                nxt = int(tmp_idx[pick])
            
            # 检查连接性
            if nxt != -1 and not finite_mask[cur, nxt]:
                valid_tour = False
                break
                
            tour[t] = nxt; used[nxt] = 1; cur = nxt
        
        if valid_tour:
            # 检查闭环
            if finite_mask[tour[n - 1], tour[0]]:
                return tour
        
        # 失败则重试

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
    # ATSP: 不使用 2-opt 修复（反转会破坏方向），失败由外层重试处理
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
def rtr_challenge_jit(child, child_fit, pop, fit, W, best_idx):
    m, n = pop.shape[0], child.shape[0]
    # 直接使用randint生成窗口索引，避免重复seed
    window_indices = np.empty(W, dtype=np.int32)
    for i in range(W):
        window_indices[i] = np.random.randint(0, m)
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
def _gls_update_penalties_jit(tour, D, penalties):
    """GLS惩罚更新 - JIT加速版本"""
    n = tour.shape[0]
    max_util = -1.0
    # 第一遍：找到最大utility
    for i in range(n):
        a, b = tour[i], tour[(i + 1) % n]
        if np.isfinite(D[a, b]):
            util = D[a, b] / (1.0 + penalties[a, b])
            if util > max_util: 
                max_util = util
    
    if max_util < 0.0: 
        return
    
    # 第二遍：惩罚所有高utility的边
    for i in range(n):
        a, b = tour[i], tour[(i + 1) % n]
        if np.isfinite(D[a, b]):
            util = D[a, b] / (1.0 + penalties[a, b])
            if util >= max_util - 1e-12:
                penalties[a, b] += 1

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


@njit(cache=True, fastmath=True)
def double_bridge_move(tour):
    """
    Double Bridge (4-opt): 强力 Kick 算子，3切点版本（Numba稳定）
    使用 randint 抽样，避免 np.random.choice
    """
    n = len(tour)
    new_tour = np.empty(n, np.int32)
    
    if n < 8:
        new_tour[:] = tour[:]
        return new_tour
    
    # 3切点版本：分段采样保证间隔
    p1 = np.random.randint(1, n // 4 + 1)
    p2 = np.random.randint(p1 + 2, n // 2 + 1)
    p3 = np.random.randint(p2 + 2, 3 * n // 4 + 1)
    
    idx = 0
    for i in range(0, p1): new_tour[idx] = tour[i]; idx += 1
    for i in range(p2, p3): new_tour[idx] = tour[i]; idx += 1
    for i in range(p1, p2): new_tour[idx] = tour[i]; idx += 1
    for i in range(p3, n): new_tour[idx] = tour[i]; idx += 1
    
    return new_tour

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
def evolve_population_jit(population, c_pop, fitness, D, D_eval, finite_mask, knn_idx, exploit_mut, is_symmetric):
    lam, n = population.shape    
    # === Buffer Allocation (One-time) ===
    map1_buf = np.empty(n, dtype=np.int32)
    map2_buf = np.empty(n, dtype=np.int32)
    used_buf = np.empty(n, dtype=np.bool_)
    backup_buf = np.empty(n, dtype=np.int32)  # 变异回滚备份
        
    SCX_RETRY = 3  # ATSP 重试次数
    
    for i in range(0, lam, 2):
        # 锦标赛选择 (保留原有逻辑)
        cand1 = np.random.choice(lam, 5, replace=False)
        p1 = population[cand1[np.argmin(fitness[cand1])]]
        cand2 = np.random.choice(lam, 5, replace=False)
        p2 = population[cand2[np.argmin(fitness[cand2])]]
        
        c1 = c_pop[i]
        c2 = c_pop[i+1]
        
        # ==========================================
        # 分支 1: 对称 TSP (Symmetric) - 保持原样 (OX + Repair)
        # ==========================================
        if is_symmetric:
            _ox_jit_inplace(p1, p2, c1)
            _ox_jit_inplace(p2, p1, c2)
            
            # Mutation C1
            if np.random.random() < exploit_mut:
                if np.random.random() < 0.7:  # Reverse mutation for symmetric
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
            # Repair C1
            if not _tour_feasible_jit(c1, finite_mask):
                if not _repair_jit(c1, D, finite_mask, 50):
                    c1[:] = p1[:]  # Fallback to parent

            # Mutation C2
            if np.random.random() < exploit_mut:
                if np.random.random() < 0.7:
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
            # Repair C2
            if not _tour_feasible_jit(c2, finite_mask):
                if not _repair_jit(c2, D, finite_mask, 50):
                    c2[:] = p2[:]

        else:
            
            # --- Child 1 ---
            result1 = 0  # 0=成功, 1=中途死路, 2=闭环失败
            for _ in range(SCX_RETRY):
                result1 = _scx_jit_inplace_ok(p1, p2, D_eval, finite_mask, knn_idx, c1, map1_buf, map2_buf, used_buf)
                if result1 == 0:
                    break
            
            if result1 != 0:
                # 彻底失败，用 RCL 生成新血 (必然可行)
                c1[:] = _rcl_nn_tour_jit(D, finite_mask, knn_idx, 3)[:]
            
            # 【修正】统一备份，无论SCX成功还是RCL兜底
            backup_buf[:] = c1[:]
            
            # === 变异：10% Double Bridge + 90% Smart Shift ===
            if np.random.random() < exploit_mut:
                    if np.random.random() < 0.10:
                        mut_tour = double_bridge_move(c1)
                        if _tour_feasible_jit(mut_tour, finite_mask):
                            c1[:] = mut_tour[:]
                        else:
                            c1[:] = backup_buf[:]
                    
                    # 90% Smart Shift（改进版：O(1)定位+原地搬移）
                    else:
                        # 建立位置映射 O(n)
                        for pos in range(n):
                            map1_buf[c1[pos]] = pos
                        
                        found_improving = False
                        K = knn_idx.shape[1]
                        
                        # u 重试最多2次
                        for u_try in range(2):
                            if found_improving:
                                break
                            
                            # O(1) 选点：直接按位置
                            u_pos = np.random.randint(0, n)
                            u = c1[u_pos]
                            u_prev = c1[(u_pos - 1) % n]
                            u_next = c1[(u_pos + 1) % n]
                            
                            # 遍历 KNN 找可行且改进的插入点
                            for k in range(K):
                                v = knn_idx[u, k]
                                if v == -1 or v == u or v == u_prev or v == u_next:
                                    continue
                                
                                # O(1) 定位 v
                                v_pos = map1_buf[v]
                                v_next = c1[(v_pos + 1) % n]
                                
                                # 预判可行性
                                if (finite_mask[u_prev, u_next] and 
                                    finite_mask[v, u] and 
                                    finite_mask[u, v_next]):
                                    
                                    delta = -D_eval[u_prev, u] - D_eval[u, u_next] + D_eval[u_prev, u_next]
                                    delta += -D_eval[v, v_next] + D_eval[v, u] + D_eval[u, v_next]
                                    
                                    # 微放宽接受
                                    accept = False
                                    if delta <= 0.0:
                                        accept = True
                                    elif delta < 5.0 and np.random.random() < 0.005:
                                        accept = True
                                    
                                    if accept:
                                        # 原地搬移（remove u + insert after v）
                                        city = u
                                        if u_pos < v_pos:
                                            # u 在 v 前面：先移除 u，v_pos 会减1
                                            for i in range(u_pos, v_pos):
                                                c1[i] = c1[i + 1]
                                            c1[v_pos] = city
                                        elif u_pos > v_pos:
                                            # u 在 v 后面：先移除 u，然后插入到 v+1
                                            for i in range(u_pos, v_pos + 1, -1):
                                                c1[i] = c1[i - 1]
                                            c1[v_pos + 1] = city
                                        
                                        found_improving = True
                                        break
                        
                        if not found_improving:
                            # Feasible Kick 兜底（3% 只要求可行，O(1)定位）
                            if np.random.random() < 0.03:
                                u_pos = np.random.randint(0, n)
                                u = c1[u_pos]
                                u_prev = c1[(u_pos - 1) % n]
                                u_next = c1[(u_pos + 1) % n]
                                
                                K = knn_idx.shape[1]
                                for k in range(K):
                                    v = knn_idx[u, k]
                                    if v == -1 or v == u or v == u_prev or v == u_next:
                                        continue
                                    
                                    v_pos = map1_buf[v]
                                    v_next = c1[(v_pos + 1) % n]
                                    
                                    if (finite_mask[u_prev, u_next] and 
                                        finite_mask[v, u] and 
                                        finite_mask[u, v_next]):
                                        # 可行！原地搬移
                                        city = u
                                        if u_pos < v_pos:
                                            for i in range(u_pos, v_pos):
                                                c1[i] = c1[i + 1]
                                            c1[v_pos] = city
                                        elif u_pos > v_pos:
                                            for i in range(u_pos, v_pos + 1, -1):
                                                c1[i] = c1[i - 1]
                                            c1[v_pos + 1] = city
                                        break
                            elif np.random.random() < 0.05:
                                u, v = np.random.randint(0, n), np.random.randint(0, n - 1)
                                if v >= u: v += 1
                                if u != v:
                                    city = c1[u]
                                    if v < u:
                                        for k in range(u, v, -1): c1[k] = c1[k-1]
                                    else:
                                        for k in range(u, v): c1[k] = c1[k+1]
                                    c1[v] = city
                                    if not _tour_feasible_jit(c1, finite_mask):
                                        c1[:] = backup_buf[:]

            # --- Child 2 ---
            result2 = 0  # 0=成功, 1=中途死路, 2=闭环失败
            for _ in range(SCX_RETRY):
                result2 = _scx_jit_inplace_ok(p2, p1, D_eval, finite_mask, knn_idx, c2, map1_buf, map2_buf, used_buf)
                if result2 == 0:
                    break
            
            if result2 != 0:
                c2[:] = _rcl_nn_tour_jit(D, finite_mask, knn_idx, 3)[:]
            
            # 【修正】统一备份，无论SCX成功还是RCL兜底
            backup_buf[:] = c2[:]
            
            # === 变异：10% Double Bridge + 90% Smart Shift ===
            if np.random.random() < exploit_mut:
                    if np.random.random() < 0.10:
                        mut_tour = double_bridge_move(c2)
                        if _tour_feasible_jit(mut_tour, finite_mask):
                            c2[:] = mut_tour[:]
                        else:
                            c2[:] = backup_buf[:]
                    
                    # 90% Smart Shift（O(1)定位+原地搬移）
                    else:
                        # 建立位置映射
                        for pos in range(n):
                            map2_buf[c2[pos]] = pos
                        
                        found_improving = False
                        K = knn_idx.shape[1]
                        
                        for u_try in range(2):
                            if found_improving:
                                break
                            
                            # O(1) 选点
                            u_pos = np.random.randint(0, n)
                            u = c2[u_pos]
                            u_prev = c2[(u_pos - 1) % n]
                            u_next = c2[(u_pos + 1) % n]
                            
                            for k in range(K):
                                v = knn_idx[u, k]
                                if v == -1 or v == u or v == u_prev or v == u_next:
                                    continue
                                
                                # O(1) 定位 v
                                v_pos = map2_buf[v]
                                v_next = c2[(v_pos + 1) % n]
                                
                                if (finite_mask[u_prev, u_next] and 
                                    finite_mask[v, u] and 
                                    finite_mask[u, v_next]):
                                    
                                    delta = -D_eval[u_prev, u] - D_eval[u, u_next] + D_eval[u_prev, u_next]
                                    delta += -D_eval[v, v_next] + D_eval[v, u] + D_eval[u, v_next]
                                    
                                    accept = False
                                    if delta <= 0.0:
                                        accept = True
                                    elif delta < 5.0 and np.random.random() < 0.005:
                                        accept = True
                                    
                                    if accept:
                                        # 原地搬移
                                        city = u
                                        if u_pos < v_pos:
                                            for i in range(u_pos, v_pos):
                                                c2[i] = c2[i + 1]
                                            c2[v_pos] = city
                                        elif u_pos > v_pos:
                                            for i in range(u_pos, v_pos + 1, -1):
                                                c2[i] = c2[i - 1]
                                            c2[v_pos + 1] = city
                                        
                                        found_improving = True
                                        break
                        
                        if not found_improving:
                            # Feasible Kick 兜底（O(1)定位）
                            if np.random.random() < 0.03:
                                u_pos = np.random.randint(0, n)
                                u = c2[u_pos]
                                u_prev = c2[(u_pos - 1) % n]
                                u_next = c2[(u_pos + 1) % n]
                                
                                K = knn_idx.shape[1]
                                for k in range(K):
                                    v = knn_idx[u, k]
                                    if v == -1 or v == u or v == u_prev or v == u_next:
                                        continue
                                    
                                    v_pos = map2_buf[v]
                                    v_next = c2[(v_pos + 1) % n]
                                    
                                    if (finite_mask[u_prev, u_next] and 
                                        finite_mask[v, u] and 
                                        finite_mask[u, v_next]):
                                        city = u
                                        if u_pos < v_pos:
                                            for i in range(u_pos, v_pos):
                                                c2[i] = c2[i + 1]
                                            c2[v_pos] = city
                                        elif u_pos > v_pos:
                                            for i in range(u_pos, v_pos + 1, -1):
                                                c2[i] = c2[i - 1]
                                            c2[v_pos + 1] = city
                                        break
                            elif np.random.random() < 0.05:
                                u, v = np.random.randint(0, n), np.random.randint(0, n - 1)
                                if v >= u: v += 1
                                if u != v:
                                    city = c2[u]
                                    if v < u:
                                        for k in range(u, v, -1): c2[k] = c2[k-1]
                                    else:
                                        for k in range(u, v): c2[k] = c2[k+1]
                                    c2[v] = city
                                    if not _tour_feasible_jit(c2, finite_mask):
                                        c2[:] = backup_buf[:]


# ==============================================================================
# Subprocess Worker
# ==============================================================================

def scout_worker(D, q_in, q_out, is_symmetric):
    try:
        n = D.shape[0]
        finite_mask = np.isfinite(D)
        knn_idx = build_knn_idx(D, finite_mask, 64)
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
            if rand_val < 0.7: mode = 0
            elif rand_val < 0.9: mode = 2
            else: mode = 1
            
            # 使用原地版 Ruin & Recreate
            _hybrid_ruin_and_recreate_inplace(current_tour, D, ruin_pct, knn_idx, mode, rr_tour_buffer, rr_removed_buffer)
            candidate = rr_tour_buffer  # candidate 现在指向 buffer
            
            dlb_mask[:], improved, block_steps = False, True, 10
            while improved:
                improved = False; dlb_mask[:] = False
                # 1. Or-opt(1) - 最快，清理单点
                if _candidate_or_opt_jit(candidate, D, knn_idx, pos_buffer, tour_buffer, 5000, dlb_mask, 1): 
                    improved = True
                    continue
                dlb_mask[:] = False
                
                # 2. 对称TSP用2-opt，ATSP用Or-opt(2,3) - 清理局部交叉
                if is_symmetric:
                    if _candidate_2opt_jit(candidate, D, knn_idx, pos_buffer, 5000, dlb_mask):
                        improved = True
                        continue
                    dlb_mask[:] = False
                else:
                    # ATSP: Or-opt(2) 和 Or-opt(3)
                    if _candidate_or_opt_jit(candidate, D, knn_idx, pos_buffer, tour_buffer, block_steps, dlb_mask, 2): 
                        improved = True
                        continue
                    dlb_mask[:] = False
                    if _candidate_or_opt_jit(candidate, D, knn_idx, pos_buffer, tour_buffer, block_steps, dlb_mask, 3): 
                        improved = True
                        continue
                    dlb_mask[:] = False
                
                # 3. Block swap(2)
                if _candidate_block_swap_jit(candidate, D, knn_idx, pos_buffer, tour_buffer, block_steps, dlb_mask, 2): 
                    improved = True
                    continue
                dlb_mask[:] = False
                
                # 4. Directed 3-opt (仅ATSP) - 昂贵的强力邻域，最后才用
                if not is_symmetric:
                    if _candidate_blockswap3_jit(candidate, D, knn_idx, pos_buffer, 1000, dlb_mask):
                        improved = True
                        continue
                    dlb_mask[:] = False
            
            cand_fit = tour_length_jit(candidate, D); scout_stagnation += 1
            if cand_fit < best_known_bound: best_known_bound = cand_fit
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
    except Exception: pass

# ==============================================================================
# Solver Class
# ==============================================================================

class r0927482:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.rng = np.random.default_rng()

    def optimize(self, filename):
        with open(filename) as f: distanceMatrix = np.loadtxt(f, delimiter=",")
        n = distanceMatrix.shape[0]
        lam, stagnation_limit, exploit_mut, exploit_ls = 200, 200, 0.3, 30
        if n < 300: lam, stagnation_limit = 1000, 500
        elif n < 600: lam, stagnation_limit, exploit_mut, exploit_ls = 150, 150, 0.15, 30
        elif n < 850: lam, stagnation_limit, exploit_mut, exploit_ls = 100, 200, 0.25, 10
        else: lam, stagnation_limit, exploit_mut, exploit_ls = 60, 300, 0.25, 20
        
        D = np.ascontiguousarray(distanceMatrix)
        is_symmetric = np.allclose(D, D.T, rtol=1e-5, atol=1e-8, equal_nan=True)
        finite_mask = np.isfinite(D); np.fill_diagonal(finite_mask, False)
        
        # === 根据对称性调整初始化策略 ===
        if is_symmetric:
            strat_probs = np.array([0.1, 0.3, 0.6], dtype=np.float64)  # 对称 TSP 可以保留随机排列
        else:
            strat_probs = np.array([0.7, 0.3, 0.0], dtype=np.float64)  # ATSP 禁用随机排列，主打 RCL
            
        q_to_scout, q_from_scout = multiprocessing.Queue(maxsize=5), multiprocessing.Queue(maxsize=5)
        scout_process = multiprocessing.Process(target=scout_worker, args=(distanceMatrix, q_to_scout, q_from_scout, is_symmetric))
        scout_process.start()

        try:
            knn_idx = build_knn_idx(D, finite_mask, 64)
            gls_penalties, gls_active, D_gls = np.zeros((n, n), dtype=np.int32), False, None
            population = np.empty((lam, n), dtype=np.int32)
            seeds = np.random.randint(0, 1<<30, lam).astype(np.int64)
            init_population_jit(population, D, finite_mask, knn_idx, strat_probs, seeds, int(self.rng.integers(3, 11)))
            fitness = np.empty(lam, dtype=np.float64); batch_lengths_jit(population, D, fitness)
            best_ever_fitness, stagnation_counter, gen = fitness.min(), 0, 0
            current_run_best = best_ever_fitness  # 本轮最优（用于判定停滞）
            best_tour_ever = population[np.argmin(fitness)].copy()  # 全局最优解（用于报告）
            c_pop, c_fit, dlb_mask = np.empty((lam, n), dtype=np.int32), np.empty(lam, dtype=np.float64), np.zeros(n, dtype=np.bool_)
            last_patient_sent_time = 0.0
            
            main_pos_buffer = np.empty(n, dtype=np.int32)
            main_tour_buffer = np.empty(n, dtype=np.int32)

            while True:
                gen += 1                
                # Scout Check
                try:
                    healed = q_from_scout.get_nowait()
                    h_fit = tour_length_jit(healed, D)
                    
                    worst_idx = np.argmax(fitness)
                    population[worst_idx][:], fitness[worst_idx] = healed[:], h_fit
                    if h_fit < best_ever_fitness:
                        best_ever_fitness, stagnation_counter, gls_penalties[:], gls_active = h_fit, 0, 0, False
                except queue.Empty: pass

                
                D_ls = D_gls if (gls_active and D_gls is not None) else D
                evolve_population_jit(population, c_pop, fitness, D, D_ls, finite_mask, knn_idx, exploit_mut, is_symmetric)
                
                batch_lengths_jit(c_pop, D, c_fit)
            
                # === 动态调整精英比例，防止早熟收敛 ===
                base_elite_pct = 0.1
                if stagnation_counter > (stagnation_limit // 2):
                    base_elite_pct = 0.2  # 停滞时加大力度
                
                elite_count = max(1, int(lam * base_elite_pct))
                elite_indices = np.argsort(c_fit)[:elite_count]
                
                for idx in elite_indices:
                    # 不在这里重置DLB，让VND内部管理
                    self._vnd_or_opt_inplace(c_pop[idx], D_ls, knn_idx, dlb_mask, exploit_ls, 3, main_pos_buffer, main_tour_buffer, is_symmetric)
                    c_fit[idx] = tour_length_jit(c_pop[idx], D)
                
                
                cur_best_idx = np.argmin(fitness)
                for i in range(lam):
                    better, tidx = rtr_challenge_jit(c_pop[i], c_fit[i], population, fitness, min(lam, 50), cur_best_idx)
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
                    best_ever_fitness = bestObjective
                    best_tour_ever = population[best_idx].copy()
                    stagnation_counter = 0  # 打破历史记录当然也清零
                    gls_penalties[:], gls_active = 0, False
                
                if stagnation_counter > (stagnation_limit // 2) and (time.time() - last_patient_sent_time > 5.0):
                    try: q_to_scout.put_nowait(population[best_idx].copy()); last_patient_sent_time = time.time()
                    except queue.Full: pass
                
                if stagnation_counter >= max(30, int(stagnation_limit * 0.6)):
                    gls_active = True
                    if gen % 50 == 0:
                        _gls_update_penalties_jit(population[best_idx], D, gls_penalties)
                        D_gls = np.ascontiguousarray(D + (0.03 * (bestObjective / n)) * gls_penalties)
                else: gls_active = False
                
                if stagnation_counter >= stagnation_limit:                    
                    # 1. 清空 Scout 发回的旧消息
                    while not q_from_scout.empty():
                        try: q_from_scout.get_nowait()
                        except queue.Empty: break
                    
                    # 2. 彻底的 GLS 遗忘
                    gls_penalties[:] = 0
                    gls_active = False
                    D_gls = None
                    
                    # 3. 【70/30 混合策略 - ATSP专用优化】
                    # Part A: 70% 完全重新生成（根据对称性调整策略）
                    reset_count = int(lam * 0.7)
                    # ATSP 禁用 random，只用 RCL + insertion
                    if is_symmetric:
                        restart_strat_probs = np.array([0.05, 0.15, 0.8], dtype=np.float64)  # 对称TSP保留80%随机
                        restart_rcl_r = int(self.rng.integers(15, 40))  # 大 r 用于多样性
                    else:
                        restart_strat_probs = np.array([0.7, 0.3, 0.0], dtype=np.float64)  # ATSP禁用random
                        restart_rcl_r = int(self.rng.integers(3, 11))  # 较小 r (3-10)
                    
                    init_population_jit(population[:reset_count], D, finite_mask, knn_idx, restart_strat_probs, 
                                        np.random.randint(0, 1<<30, reset_count).astype(np.int64), 
                                        restart_rcl_r)
                    
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
                
                if self.reporter.report(float(fitness.mean()), best_ever_fitness, bestSolution) < 0:
                    break
            return 0
        finally:
            if scout_process.is_alive(): scout_process.terminate(); scout_process.join()

    def _vnd_or_opt_inplace(self, tour, D, knn_idx, dlb_mask, max_iters, block_steps, pos_buf, tour_buf, is_symmetric):
        # 只在VND开始时重置一次DLB
        dlb_mask[:] = False
        improved = True
        while improved:
            improved = False
            # 不在循环内重置DLB，让各算子自己管理DLB更新
            # 1. Or-opt(1) - 最快，清理单点
            if _candidate_or_opt_jit(tour, D, knn_idx, pos_buf, tour_buf, max_iters, dlb_mask, 1): 
                improved = True
                continue
            
            # 2. 对称TSP用2-opt，ATSP用Or-opt(2,3) - 清理局部交叉
            if is_symmetric:
                if _candidate_2opt_jit(tour, D, knn_idx, pos_buf, max_iters, dlb_mask):
                    improved = True
                    continue
            else:
                # ATSP: Or-opt(2) 和 Or-opt(3)
                if _candidate_or_opt_jit(tour, D, knn_idx, pos_buf, tour_buf, block_steps, dlb_mask, 2): 
                    improved = True
                    continue
                if _candidate_or_opt_jit(tour, D, knn_idx, pos_buf, tour_buf, block_steps, dlb_mask, 3): 
                    improved = True
                    continue
            
            # 3. Block swap(2)
            if _candidate_block_swap_jit(tour, D, knn_idx, pos_buf, tour_buf, block_steps, dlb_mask, 2): 
                improved = True
                continue
            
            # 4. Directed 3-opt (仅ATSP) - 昂贵的强力邻域，最后才用
            if not is_symmetric:
                if _candidate_blockswap3_jit(tour, D, knn_idx, pos_buf, 500, dlb_mask):
                    improved = True
                    continue

if __name__ == '__main__':
    multiprocessing.freeze_support()
