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
def _candidate_or_opt_jit(tour, D, knn_idx, max_iters=100, dlb_mask=None, block_size=1):
    n = tour.shape[0]
    K = knn_idx.shape[1]
    block_size = int(block_size)
    if block_size < 1: block_size = 1
    if block_size >= n: return False 
    
    pos = np.empty(n, np.int32)
    for i in range(n): pos[tour[i]] = i
    
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
                t_idx = pos[target] 
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
                    # Strictly align with Baseline copy logic
                    block = np.empty(block_size, dtype=tour.dtype)
                    for b in range(block_size): block[b] = tour[u_idx + b]
                    
                    temp_len = n - block_size
                    temp_tour = np.empty(temp_len, dtype=tour.dtype)
                    if u_idx > 0: temp_tour[:u_idx] = tour[:u_idx]
                    if u_idx + block_size < n: temp_tour[u_idx:] = tour[u_idx + block_size:]
                    
                    t_idx_new = t_idx
                    if t_idx > u_idx: t_idx_new -= block_size
                    
                    new_tour = np.empty(n, dtype=tour.dtype)
                    idx = 0
                    for i in range(temp_len):
                        new_tour[idx] = temp_tour[i]; idx += 1
                        if i == t_idx_new:
                            for b in range(block_size): new_tour[idx] = block[b]; idx += 1
                    
                    tour[:] = new_tour[:]
                    for i in range(n): pos[tour[i]] = i
                    improved = True
                    move_found = True
                    found_in_try = True
                    if use_dlb: 
                        dlb_mask[prev_u] = False 
                        dlb_mask[next_after] = False 
                        dlb_mask[target] = False 
                        dlb_mask[target_next] = False
                        for b in range(block_size): dlb_mask[block[b]] = False
                    break 
            if move_found: continue 
            else: 
                if use_dlb: dlb_mask[block_head] = True
        if not found_in_try and use_dlb: break
    return improved

@njit(cache=True, fastmath=True)
def _candidate_block_swap_jit(tour, D, knn_idx, max_iters=50, dlb_mask=None, block_size=2):
    n = tour.shape[0]
    K = knn_idx.shape[1]
    block_size = int(block_size)
    if block_size < 1: block_size = 1
    if block_size * 2 >= n: return False 
    
    pos = np.empty(n, np.int32)
    for i in range(n): pos[tour[i]] = i
    
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
                j = pos[target] 
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
                    new_tour = np.empty_like(tour)
                    idx = 0
                    for t in range(0, i): new_tour[idx] = tour[t]; idx += 1
                    for t in range(j, j + block_size): new_tour[idx] = tour[t]; idx += 1
                    for t in range(i + block_size, j): new_tour[idx] = tour[t]; idx += 1
                    for t in range(i, i + block_size): new_tour[idx] = tour[t]; idx += 1
                    for t in range(j + block_size, n): new_tour[idx] = tour[t]; idx += 1
                    tour[:] = new_tour[:]
                    for t in range(n): pos[tour[t]] = t
                    improved = True; move_found = True; found_in_try = True
                    if use_dlb:
                        for city in [a, b, c, d, e, f, g, h]: dlb_mask[city] = False
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
    """混合破坏策略的 ruin and recreate"""
    n = len(tour)
    n_remove = int(n * ruin_pct)
    if n_remove < 2: return tour.copy()
    
    mask = _hybrid_ruin_mask_jit(tour, D, knn_idx, n_remove, mode)
    
    # 分离 kept 和 removed
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
    
    # 贪婪插入重建
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
        if u < strat_probs[0]: tour = _rcl_nn_tour_jit(D, finite_mask, knn_idx, rcl_r)
        elif u < strat_probs[0] + strat_probs[1]: tour = _insertion_tour_jit(D, finite_mask, use_farthest=(np.random.rand() < 0.5))
        else:
            tour = _rand_perm_jit(n)
            if not _repair_jit(tour, D, finite_mask, 50): tour = _rcl_nn_tour_jit(D, finite_mask, knn_idx, rcl_r)
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
        if is_symmetric: _repair_jit(c1, D, finite_mask)
        
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
        if is_symmetric: _repair_jit(c2, D, finite_mask)

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
        
        while True:
            iter_count += 1
            if iter_count % 100 == 0:
                print(f"[Scout] iter={iter_count}, best={best_known_bound:.2f}, stagnation={scout_stagnation}")
            try:
                latest_patient = q_in.get_nowait()
                p_fit = tour_length_jit(latest_patient, D)
                print(f"[Scout] NEW PATIENT: iter={iter_count}, patient_fit={p_fit:.2f}")
                current_tour[:], current_fit, dlb_mask[:] = latest_patient[:], p_fit, False
                patient_entry_fit, last_improv_iter, scout_stagnation, best_known_bound = p_fit, iter_count, 0, p_fit
            except queue.Empty: pass
            
            ruin_pct = ruin_gears[int((iter_count - last_improv_iter) // 250) % 10]
            
            # 轮盘赌选择破坏策略: 70% BFS, 20% Worst Edge, 10% Sequence
            rand_val = np.random.rand()
            if rand_val < 0.7: mode = 0      # BFS 空间破坏
            elif rand_val < 0.9: mode = 2    # Worst Edge 最差边破坏
            else: mode = 1                   # Sequence 序列破坏
            
            candidate = _hybrid_ruin_and_recreate_jit(current_tour, D, ruin_pct, knn_idx, mode)
            
            dlb_mask[:], improved, block_steps = False, True, 10
            while improved:
                improved = False; dlb_mask[:] = False
                if _candidate_or_opt_jit(candidate, D, knn_idx, max_iters=5000, dlb_mask=dlb_mask, block_size=1): improved = True; continue
                dlb_mask[:] = False
                if _candidate_block_swap_jit(candidate, D, knn_idx, max_iters=block_steps, dlb_mask=dlb_mask, block_size=2): improved = True; continue
                dlb_mask[:] = False
                if _candidate_or_opt_jit(candidate, D, knn_idx, max_iters=block_steps, dlb_mask=dlb_mask, block_size=2): improved = True; continue
                dlb_mask[:] = False
                if _candidate_or_opt_jit(candidate, D, knn_idx, max_iters=block_steps, dlb_mask=dlb_mask, block_size=3): improved = True; continue
            
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

            while True:
                gen += 1
                
                # Scout Check
                try:
                    healed = q_from_scout.get_nowait()
                    h_fit = tour_length_jit(healed, D); worst_idx = np.argmax(fitness)
                    population[worst_idx][:], fitness[worst_idx] = healed[:], h_fit
                    if h_fit < best_ever_fitness:
                        best_ever_fitness, stagnation_counter, gls_penalties[:], gls_active = h_fit, 0, 0, False
                except queue.Empty: pass

                D_ls = D_gls if (gls_active and D_gls is not None) else D
                evolve_population_jit(population, c_pop, fitness, D, finite_mask, exploit_mut, is_symmetric)
                
                batch_lengths_jit(c_pop, D, c_fit)
                
                elite_count = max(1, int(lam * 0.2))
                elite_indices = np.argsort(c_fit)[:elite_count]
                for idx in elite_indices:
                    dlb_mask[:] = False
                    self._vnd_or_opt_inplace(c_pop[idx], D_ls, knn_idx, dlb_mask, exploit_ls, 3)
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
                        self._gls_update_penalties(population[best_idx], D, gls_penalties)
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

    def _vnd_or_opt_inplace(self, tour, D, knn_idx, dlb_mask, max_iters, block_steps):
        improved = True
        while improved:
            improved = False; dlb_mask[:] = False
            if _candidate_or_opt_jit(tour, D, knn_idx, max_iters=max_iters, dlb_mask=dlb_mask, block_size=1): improved = True; continue
            dlb_mask[:] = False
            if _candidate_block_swap_jit(tour, D, knn_idx, max_iters=block_steps, dlb_mask=dlb_mask, block_size=2): improved = True; continue
            dlb_mask[:] = False
            if _candidate_or_opt_jit(tour, D, knn_idx, max_iters=block_steps, dlb_mask=dlb_mask, block_size=2): improved = True; continue
            dlb_mask[:] = False
            if _candidate_or_opt_jit(tour, D, knn_idx, max_iters=block_steps, dlb_mask=dlb_mask, block_size=3): improved = True; continue

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
