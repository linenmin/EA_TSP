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
# JIT Accelerated Helper Functions
# ==============================================================================

@njit(cache=True, fastmath=True)
def _ox_jit(p1, p2):
    n = p1.size
    child = np.full(n, -1, np.int32)
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
    return child

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
        cnt = 0
        for j in range(n):
            if finite_mask[i, j]:
                cnt += 1
        if cnt == 0: continue
        cand_idx = np.empty(cnt, np.int32)
        cand_dis = np.empty(cnt, np.float64)
        c = 0
        for j in range(n):
            if finite_mask[i, j]:
                cand_idx[c] = j
                cand_dis[c] = D[i, j]
                c += 1
        order = np.argsort(cand_dis)
        m = K if K < cnt else cnt
        for t in range(m):
            knn[i, t] = cand_idx[order[t]]
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
            best_delta = delta
            bi = i
            bj = j
    if best_delta < 0.0:
        l = bi + 1
        r = bj
        while l < r:
            tmp = tour[l]; tour[l] = tour[r]; tour[r] = tmp
            l += 1; r -= 1
        return True
    return False

@njit(cache=True, fastmath=True)
def _tour_feasible_jit(tour, finite_mask):
    n = tour.size
    for i in range(n):
        a = tour[i]; b = tour[(i + 1) % n]
        if not finite_mask[a, b]:
            return False
    return True

@njit(cache=True, fastmath=True)
def _repair_jit(tour, D, finite_mask, max_tries=50):
    for _ in range(max_tries):
        if _tour_feasible_jit(tour, finite_mask):
            return True
        if not _two_opt_once_jit_safe(tour, D):
            break
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
    idx = 0
    for i in range(0, p1):
        new_tour[idx] = tour[i]; idx += 1
    for i in range(p2, p3):
        new_tour[idx] = tour[i]; idx += 1
    for i in range(p1, p2):
        new_tour[idx] = tour[i]; idx += 1
    for i in range(p3, n):
        new_tour[idx] = tour[i]; idx += 1
    return new_tour

@njit(cache=True, fastmath=True)
def _candidate_or_opt_jit(tour, D, knn_idx, max_iters=100, dlb_mask=None, block_size=1):
    n = tour.shape[0]
    K = knn_idx.shape[1]
    block_size = int(block_size)
    if block_size < 1: block_size = 1
    if block_size >= n: return False 
    pos = np.empty(n, np.int32)
    for i in range(n):
        pos[tour[i]] = i
    improved = False 
    use_dlb = (dlb_mask is not None) 
    for _ in range(max_iters): 
        found_in_try = False 
        start = np.random.randint(0, n) 
        for offset in range(n): 
            u_idx = (start + offset) % n
            u = tour[u_idx] 
            if use_dlb and dlb_mask[u]: 
                continue 
            if block_size > 1 and u_idx + block_size >= n: 
                if use_dlb: dlb_mask[u] = True
                continue 
            prev_idx = u_idx - 1 
            if prev_idx < 0: prev_idx = n - 1
            post_idx = u_idx + block_size 
            if post_idx >= n: post_idx = 0
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
                target_next = tour[(t_idx + 1) % n] 
                if not np.isfinite(D[target, target_next]): continue
                if not np.isfinite(D[target, block_head]): continue 
                if not np.isfinite(D[block_tail, target_next]): continue 
                insert_cost = D[target, block_head] + D[block_tail, target_next]
                old_edge_cost = D[target, target_next]
                gain = (remove_cost - new_edge_cost) + (old_edge_cost - insert_cost)
                if gain > 1e-6: 
                    block = np.empty(block_size, dtype=tour.dtype)
                    for b in range(block_size): 
                        block[b] = tour[u_idx + b]
                    temp_len = n - block_size
                    temp_tour = np.empty(temp_len, dtype=tour.dtype)
                    if u_idx > 0: 
                        temp_tour[:u_idx] = tour[:u_idx]
                    if u_idx + block_size < n: 
                        temp_tour[u_idx:] = tour[u_idx + block_size:]
                    t_idx_new = t_idx 
                    if t_idx > u_idx: 
                        t_idx_new -= block_size
                    new_tour = np.empty(n, dtype=tour.dtype)
                    idx = 0
                    for i in range(temp_len):
                        new_tour[idx] = temp_tour[i]
                        idx += 1
                        if i == t_idx_new: 
                            for b in range(block_size):
                                new_tour[idx] = block[b]
                                idx += 1
                    tour[:] = new_tour[:]
                    for i in range(n): 
                        pos[tour[i]] = i
                    improved = True
                    move_found = True
                    found_in_try = True
                    if use_dlb: 
                        dlb_mask[prev_u] = False 
                        dlb_mask[next_after] = False 
                        dlb_mask[target] = False 
                        dlb_mask[target_next] = False
                        for b in range(block_size): 
                            dlb_mask[block[b]] = False
                    break 
            if move_found: 
                continue 
            else: 
                if use_dlb: 
                    dlb_mask[block_head] = True
        if not found_in_try and use_dlb: 
            break
    return improved

@njit(cache=True, fastmath=True)
def _candidate_block_swap_jit(tour, D, knn_idx, max_iters=50, dlb_mask=None, block_size=2):
    n = tour.shape[0]
    K = knn_idx.shape[1]
    block_size = int(block_size)
    max_iters = int(max_iters)
    if block_size < 1: block_size = 1
    if block_size * 2 >= n: return False 
    pos = np.empty(n, np.int32)
    for i in range(n):
        pos[tour[i]] = i
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
            a_idx = i - 1    
            if a_idx < 0: a_idx = n - 1
            d_idx = i + block_size 
            a = tour[a_idx]
            b = tour[i]
            c = tour[i + block_size - 1]
            d = tour[d_idx]
            if not np.isfinite(D[a, b]): continue
            if not np.isfinite(D[c, d]): continue
            move_found = False
            for k in range(K):
                target = knn_idx[a, k] 
                if target == -1: break
                j = pos[target] 
                if j <= i: continue
                if j < i + block_size: continue
                if j + block_size >= n: continue
                e_idx = j - 1
                e = tour[e_idx]
                f = tour[j]           
                g = tour[j + block_size - 1]
                h = tour[j + block_size]
                if j == i + block_size:
                    if not np.isfinite(D[a, f]): continue
                    if not np.isfinite(D[g, b]): continue
                    if not np.isfinite(D[c, h]): continue
                    old_cost = D[a, b] + D[c, f] + D[g, h]
                    new_cost = D[a, f] + D[g, b] + D[c, h]
                else:
                    if not np.isfinite(D[a, f]): continue
                    if not np.isfinite(D[g, d]): continue
                    if not np.isfinite(D[e, b]): continue
                    if not np.isfinite(D[c, h]): continue
                    old_cost = D[a, b] + D[c, d] + D[e, f] + D[g, h]
                    new_cost = D[a, f] + D[g, d] + D[e, b] + D[c, h]
                gain = old_cost - new_cost
                if gain > 1e-6: 
                    new_tour = np.empty_like(tour)
                    idx = 0
                    for t in range(0, i):
                        new_tour[idx] = tour[t]
                        idx += 1
                    for t in range(j, j + block_size):
                        new_tour[idx] = tour[t]
                        idx += 1
                    for t in range(i + block_size, j):
                        new_tour[idx] = tour[t]
                        idx += 1
                    for t in range(i, i + block_size):
                        new_tour[idx] = tour[t]
                        idx += 1
                    for t in range(j + block_size, n):
                        new_tour[idx] = tour[t]
                        idx += 1
                    tour[:] = new_tour[:]
                    for t in range(n): pos[tour[t]] = t
                    improved = True
                    move_found = True
                    found_in_try = True
                    if use_dlb:
                        dlb_mask[a] = False
                        dlb_mask[b] = False
                        dlb_mask[c] = False
                        dlb_mask[d] = False
                        dlb_mask[e] = False
                        dlb_mask[f] = False
                        dlb_mask[g] = False
                        dlb_mask[h] = False
                    break 
            if move_found:
                continue
            else:
                if use_dlb: dlb_mask[b] = True 
        if not found_in_try and use_dlb:
            break
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
            if j == -1: continue
            if used[j] == 1: continue
            if not finite_mask[cur, j]: continue
            tmp_idx[cnt] = j
            tmp_dis[cnt] = D[cur, j]
            cnt += 1
        if cnt == 0:
            pool_cnt = 0
            for j in range(n):
                if used[j] == 0 and finite_mask[cur, j]:
                    pool_cnt += 1
            if pool_cnt == 0:  
                nxt = 0
                for j in range(n):
                    if used[j] == 0:
                        nxt = j; break
            else:
                pool = np.empty(pool_cnt, np.int32)
                c = 0
                for j in range(n):
                    if used[j] == 0 and finite_mask[cur, j]:
                        pool[c] = j; c += 1
                nxt = int(pool[np.random.randint(0, pool_cnt)])
        else:
            order = np.argsort(tmp_dis[:cnt])
            rsize = r if r < cnt else cnt
            pick = order[np.random.randint(0, rsize)]
            nxt = int(tmp_idx[pick])
        tour[t] = nxt; used[nxt] = 1; cur = nxt
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
    n = D.shape[0]
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
        if use_farthest:
            best_city = -1; best_score = -1.0
            for c in range(n):
                if used[c] == 1: continue
                mind = 1e100
                for t in range(m):
                    if finite_mask[c, tour[t]] and D[c, tour[t]] < mind:
                        mind = D[c, tour[t]]
                if mind > best_score:
                    best_score = mind; best_city = c
            insert_city = best_city if best_city != -1 else np.random.randint(0, n)
        else:
            remain = n - m
            k = np.random.randint(0, remain)
            idx = -1
            for c in range(n):
                if used[c] == 0:
                    if k == 0: idx = c; break
                    k -= 1
            insert_city = idx
        best_pos = -1; best_cost = 1e100
        for i in range(m):
            prev = tour[i - 1] if i > 0 else tour[m - 1]
            curr = tour[i]
            if finite_mask[prev, insert_city] and finite_mask[insert_city, curr]:
                cost = D[prev, insert_city] + D[insert_city, curr] - D[prev, curr]
                if cost < best_cost:
                    best_cost = cost; best_pos = i
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
    for _ in range(20):
        if finite_mask[tour[m - 1], tour[0]]: break
        if not _two_opt_once_jit_safe(tour, D): break
    return tour

@njit(cache=True, fastmath=True)
def bond_distance_jit(t1, t2):
    n = t1.shape[0]
    pos2 = np.empty(n, np.int32)
    for i in range(n):
        pos2[t2[i]] = i
    shared_edges = 0
    for i in range(n):
        u = t1[i]
        v = t1[(i + 1) % n]
        idx_u = pos2[u]
        left = t2[(idx_u - 1) % n]
        right = t2[(idx_u + 1) % n]
        if v == left or v == right:
            shared_edges += 1
    return n - shared_edges

@njit(cache=True, fastmath=True)
def rtr_challenge_jit(child, child_fit, pop, fit, W, rng_seed, best_idx):
    m = pop.shape[0]
    n = child.shape[0]  
    np.random.seed(rng_seed)
    window_indices = np.random.choice(m, size=W, replace=False)
    closest_idx = -1
    min_dist = 99999999
    for idx in window_indices:
        dist = bond_distance_jit(child, pop[idx])
        if dist < min_dist:
            min_dist = dist
            closest_idx = idx
    target_idx = closest_idx
    target_fit = fit[target_idx]
    if target_idx == best_idx:
        return False, target_idx  
    better = False
    if child_fit < target_fit:
        better = True
    else:
        threshold_dist = n * 0.15  
        relax_factor = 1.05        
        if min_dist > threshold_dist and child_fit < target_fit * relax_factor:
            better = True
    return better, target_idx

@njit(cache=True, fastmath=True)
def _bfs_ruin_mask_jit(n, knn_idx, n_remove):
    removed_mask = np.zeros(n, dtype=np.bool_)
    if n_remove >= n:
        removed_mask[:] = True
        return removed_mask
    count = 0
    center = np.random.randint(0, n)
    queue = np.empty(n_remove + 100, dtype=np.int32)
    q_head = 0
    q_tail = 0
    queue[q_tail] = center; q_tail += 1
    removed_mask[center] = True
    count += 1
    K = knn_idx.shape[1]
    while count < n_remove and q_head < q_tail:
        curr = queue[q_head]; q_head += 1
        for i in range(K):
            neighbor = knn_idx[curr, i]
            if neighbor == -1: break
            if not removed_mask[neighbor]:
                removed_mask[neighbor] = True
                count += 1
                if q_tail < queue.size:
                    queue[q_tail] = neighbor
                    q_tail += 1
                if count >= n_remove:
                    break
    return removed_mask

@njit(cache=True, nogil=True)
def _insert_city_jit(current_tour, city, pos):
    m = len(current_tour)  
    new_tour = np.empty(m + 1, dtype=np.int32)  
    new_tour[:pos+1] = current_tour[:pos+1]  
    new_tour[pos+1] = city  
    new_tour[pos+2:] = current_tour[pos+1:]  
    return new_tour  

@njit(cache=True, nogil=True)
def _best_two_positions_jit(city, current_tour, D):
    best_delta = 1e20  
    second_delta = 1e20  
    best_pos = -1  
    m = len(current_tour)  
    for i in range(m):  
        u = current_tour[i]  
        v = current_tour[(i + 1) % m]  
        delta = D[u, city] + D[city, v] - D[u, v]  
        if delta < best_delta:  
            second_delta = best_delta  
            best_delta = delta  
            best_pos = i  
        elif delta < second_delta:  
            second_delta = delta  
    return best_delta, second_delta, best_pos  

@njit(cache=True, nogil=True)
def _ruin_and_recreate_regret_jit(tour: np.ndarray, D: np.ndarray, ruin_pct: float,
                                  knn_idx: np.ndarray, regret_frac: float,
                                  regret_sample: int, regret_min_remove: int) -> np.ndarray:
    n = len(tour)  
    n_remove = int(n * ruin_pct)  
    if n_remove < 2:  
        return tour.copy()
    kept_cities = np.empty(n - n_remove, dtype=np.int32)  
    removed_cities = np.empty(n_remove, dtype=np.int32)  
    mask = _bfs_ruin_mask_jit(n, knn_idx, n_remove)  
    k_ptr = 0  
    r_ptr = 0  
    for i in range(n):  
        c = tour[i]  
        if mask[c]:  
            if r_ptr < n_remove:  
                removed_cities[r_ptr] = c  
                r_ptr += 1  
        else:  
            if k_ptr < n - n_remove:  
                kept_cities[k_ptr] = c  
                k_ptr += 1  
    current_tour = kept_cities  
    np.random.shuffle(removed_cities)  
    r_len = n_remove  
    use_regret = (n_remove >= regret_min_remove) and (regret_frac > 0.0) and (regret_sample > 0)  
    if use_regret:  
        regret_steps = int(r_len * regret_frac)  
        if regret_steps < 1:  
            regret_steps = 1  
        for _ in range(regret_steps):  
            if r_len <= 0:  
                break  
            sample = regret_sample if regret_sample < r_len else r_len  
            best_regret = -1.0  
            best_delta = 1e20  
            best_pos = -1  
            best_idx = -1  
            for _ in range(sample):  
                pick_idx = np.random.randint(0, r_len)  
                city = removed_cities[pick_idx]  
                d1, d2, pos = _best_two_positions_jit(city, current_tour, D)  
                regret = d2 - d1  
                if regret > best_regret:  
                    best_regret = regret  
                    best_delta = d1  
                    best_pos = pos  
                    best_idx = pick_idx  
                elif regret == best_regret and d1 < best_delta:  
                    best_delta = d1  
                    best_pos = pos  
                    best_idx = pick_idx  
            if best_idx == -1:  
                break  
            city = removed_cities[best_idx]  
            current_tour = _insert_city_jit(current_tour, city, best_pos)  
            r_len -= 1  
            removed_cities[best_idx] = removed_cities[r_len]  
    for i in range(r_len):  
        city = removed_cities[i]  
        best_delta = 1e20  
        best_pos = -1  
        m = len(current_tour)  
        for j in range(m):  
            u = current_tour[j]  
            v = current_tour[(j + 1) % m]  
            delta = D[u, city] + D[city, v] - D[u, v]  
            if delta < best_delta:  
                best_delta = delta  
                best_pos = j  
        current_tour = _insert_city_jit(current_tour, city, best_pos)  
    return current_tour  

@njit(cache=True, nogil=True)
def _ruin_and_recreate_jit(tour: np.ndarray, D: np.ndarray, ruin_pct: float, knn_idx: np.ndarray = None) -> np.ndarray:
    n = len(tour)
    n_remove = int(n * ruin_pct)
    if n_remove < 2: 
        return tour.copy()
    kept_cities = np.empty(n - n_remove, dtype=np.int32)
    removed_cities = np.empty(n_remove, dtype=np.int32)
    if knn_idx is not None:
        mask = _bfs_ruin_mask_jit(n, knn_idx, n_remove)
        k_ptr = 0
        r_ptr = 0
        for i in range(n):
            c = tour[i]
            if mask[c]:
                if r_ptr < n_remove:
                    removed_cities[r_ptr] = c
                    r_ptr += 1
            else:
                if k_ptr < n - n_remove:
                    kept_cities[k_ptr] = c
                    k_ptr += 1
    else:
        start_idx = np.random.randint(0, n)
        end_idx = (start_idx + n_remove) % n
        if start_idx < end_idx:
            kept_cities[:start_idx] = tour[:start_idx]
            kept_cities[start_idx:] = tour[end_idx:]
            removed_cities[:] = tour[start_idx:end_idx]
        else:
            kept_cities[:] = tour[end_idx:start_idx]
            k = n - start_idx
            removed_cities[:k] = tour[start_idx:]
            removed_cities[k:] = tour[:end_idx]
    current_tour = kept_cities
    np.random.shuffle(removed_cities) 
    for city in removed_cities:
        best_delta = 1e20
        best_pos = -1
        m = len(current_tour)
        for i in range(m):
            u = current_tour[i]
            v = current_tour[(i + 1) % m]
            delta = D[u, city] + D[city, v] - D[u, v]
            if delta < best_delta:
                best_delta = delta
                best_pos = i
        new_tour = np.empty(m + 1, dtype=np.int32)
        new_tour[:best_pos+1] = current_tour[:best_pos+1]
        new_tour[best_pos+1] = city
        new_tour[best_pos+2:] = current_tour[best_pos+1:]
        current_tour = new_tour
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
            ok = _repair_jit(tour, D, finite_mask, 50)
            if not ok:
                tour = _rcl_nn_tour_jit(D, finite_mask, knn_idx, rcl_r)
        pop[i] = tour

@njit(cache=True, parallel=True)
def _resync_population_jit(pop, fit, best_tour, D, seed, cutoff_ratio=0.3):
    lam, n = pop.shape
    cutoff = int(lam * cutoff_ratio) 
    order = np.argsort(fit)[::-1]
    targets = order[:cutoff]
    for i in prange(cutoff):
        idx = targets[i]
        np.random.seed(seed + idx) 
        pop[idx] = best_tour[:]
        num_kicks = 1 if (i % 2 == 0) else 4
        valid_kick = False
        for attempt in range(10): 
            temp_tour = pop[idx].copy()
            for _ in range(num_kicks):
                temp_tour = double_bridge_move(temp_tour)
            l = tour_length_jit(temp_tour, D)
            if l < 1e20:
                pop[idx] = temp_tour
                fit[idx] = l
                valid_kick = True
                break
        if not valid_kick:
            pop[idx] = best_tour[:]
            n_ = n
            for _ in range(5):
                p1 = np.random.randint(0, n_)
                p2 = np.random.randint(0, n_)
                if p1 != p2:
                    city = pop[idx][p1]
                    if p1 < p2:
                        for k in range(p1, p2):
                            pop[idx][k] = pop[idx][k+1]
                        pop[idx][p2] = city
                    else:
                        for k in range(p1, p2, -1):
                            pop[idx][k] = pop[idx][k-1]
                        pop[idx][p2] = city
            fit[idx] = tour_length_jit(pop[idx], D)

# ==============================================================================
# Top-level Subprocess Worker (Memory Efficient, No Pickle Issues)
# ==============================================================================

def scout_worker(D, q_in, q_out, config):
    """
    Subprocess: Scout / Trauma Center (LNS)
    """
    try:
        n = D.shape[0]
        finite_mask = np.isfinite(D)
        
        # Build independent KNN in subprocess to save pickle overhead/issues
        knn_idx = np.empty((n, 32), dtype=np.int32)
        for i in range(n):
            row = D[i]
            # Simple top-k sorting for KNN (single thread here)
            # Mask diagonal and infinites
            masked = np.copy(row)
            masked[i] = np.inf
            masked[~finite_mask[i]] = np.inf
            # Argsort
            idx = np.argsort(masked)[:32]
            knn_idx[i] = idx.astype(np.int32)

        # Init (Greedy)
        current_tour = _rcl_nn_tour_jit(D, finite_mask, knn_idx, 3)
        current_fit = tour_length_jit(current_tour, D)
        best_known_bound = current_fit
        
        dlb_mask = np.zeros(n, dtype=np.bool_)
        iter_count = 0
        last_improv_iter = 0
        last_send_iter = 0
        scout_stagnation = 0
        ruin_gears = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
        
        # State
        patient_entry_fit = float('inf')
        
        while True:
            iter_count += 1
            
            # 1. Check Incoming Patient (Non-blocking)
            try:
                latest_patient = q_in.get_nowait()
                p_fit = tour_length_jit(latest_patient, D)
                current_tour[:] = latest_patient[:]
                current_fit = p_fit
                dlb_mask[:] = False
                patient_entry_fit = p_fit
                last_improv_iter = iter_count
                scout_stagnation = 0
                best_known_bound = patient_entry_fit
            except queue.Empty:
                pass
            
            # 2. Ruin & Recreate
            gear_idx = int((iter_count - last_improv_iter) // 250) % 6
            ruin_pct = ruin_gears[gear_idx]
            
            # Regret Logic
            if scout_stagnation >= 80:
                candidate = _ruin_and_recreate_regret_jit(
                    current_tour, D, ruin_pct, knn_idx,
                    0.25, 8, 30 
                )
            else:
                candidate = _ruin_and_recreate_jit(current_tour, D, ruin_pct, knn_idx)
            
            dlb_mask[:] = False
            
            # 3. Local Search
            # Reimplement VND loop locally or call function?
            # Since helper methods are in class, we must call the JIT functions directly here
            # Copy _vnd_or_opt_inplace logic here
            block_steps = 3
            improved = True
            while improved:
                improved = False
                dlb_mask[:] = False
                if _candidate_or_opt_jit(candidate, D, knn_idx, max_iters=500, dlb_mask=dlb_mask, block_size=1):
                    improved = True; continue
                dlb_mask[:] = False
                if _candidate_block_swap_jit(candidate, D, knn_idx, max_iters=block_steps, dlb_mask=dlb_mask, block_size=2):
                    improved = True; continue
                dlb_mask[:] = False
                if _candidate_or_opt_jit(candidate, D, knn_idx, max_iters=block_steps, dlb_mask=dlb_mask, block_size=2):
                    improved = True; continue
                dlb_mask[:] = False
                if _candidate_or_opt_jit(candidate, D, knn_idx, max_iters=block_steps, dlb_mask=dlb_mask, block_size=3):
                    improved = True; continue
            
            cand_fit = tour_length_jit(candidate, D)
            
            # 4. Acceptance & Discharge
            scout_stagnation += 1
            if cand_fit < best_known_bound:
                best_known_bound = cand_fit
            
            gap = (cand_fit - patient_entry_fit) / patient_entry_fit if patient_entry_fit > 0 else 0
            is_breakthrough = cand_fit < patient_entry_fit
            
            tolerance = 0.0
            if scout_stagnation > 50: tolerance = 0.005
            if scout_stagnation > 200: tolerance = 0.01
            is_trojan = (gap <= tolerance) and (gap > -1.0)
            
            should_discharge = False
            if is_breakthrough:
                should_discharge = True
            elif is_trojan:
                if (iter_count - last_send_iter > 200):
                    should_discharge = True
            
            if should_discharge:
                try:
                    q_out.put_nowait(candidate.copy())
                    last_send_iter = iter_count
                    if is_breakthrough:
                        patient_entry_fit = cand_fit
                        scout_stagnation = 0
                        last_improv_iter = iter_count
                except queue.Full:
                    pass
            
            # 5. Local Hill Climbing
            if cand_fit <= current_fit:
                current_tour[:] = candidate[:]
                current_fit = cand_fit
            
            # Note: No time check here, relies on Process.terminate() from main
            
    except Exception:
        pass # Silent fail per strict rules

# Modify the class name to match your student number.
class r0927480:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)
		self.rng = np.random.default_rng()

	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		# Your code here.
		n = distanceMatrix.shape[0]
		
		# --- Adaptive Configuration ---
		# Default configs
		N_RUNS = 10_000_000
		lam = 200
		stagnation_limit = 200
		exploit_mut = 0.3
		exploit_ls = 30
		regret_enabled = True
		
		if n < 300: # Small (e.g. 200, 250)
			lam = 500
			stagnation_limit = 400
			exploit_mut = 0.3
			exploit_ls = 30
		elif n < 600: # Medium (e.g. 400, 500)
			lam = 300
			stagnation_limit = 300
			exploit_mut = 0.1
			exploit_ls = 40
		elif n < 850: # Large (e.g. 600, 750, 800)
			lam = 100
			stagnation_limit = 200
			exploit_mut = 0.1
			exploit_ls = 20
		else: # Huge (e.g. 1000)
			lam = 100
			stagnation_limit = 150
			exploit_mut = 0.1
			exploit_ls = 30

		# Setup IPC
		q_to_scout = multiprocessing.Queue(maxsize=5)
		q_from_scout = multiprocessing.Queue(maxsize=5)
		
		scout_process = multiprocessing.Process(
			target=scout_worker, 
			args=(distanceMatrix, q_to_scout, q_from_scout, None)
		)
		scout_process.start()

		try:
			# --- Preprocessing ---
			D = np.ascontiguousarray(distanceMatrix)
			is_symmetric = np.allclose(D, D.T, rtol=1e-5, atol=1e-8, equal_nan=True)
			finite_mask = np.isfinite(D)
			np.fill_diagonal(finite_mask, False)

			# KNN
			K = 32
			knn_idx = build_knn_idx(D, finite_mask, K)
			
			# GLS Init
			gls_penalties = np.zeros((n, n), dtype=np.int32)
			D_gls = None
			gls_active = False

			# --- Init Population ---
			population = np.empty((lam, n), dtype=np.int32)
			strat_probs = np.array([0.1, 0.3, 0.6], dtype=np.float64)
			seeds = np.random.randint(0, 1<<30, lam).astype(np.int64)
			rcl_r = int(self.rng.integers(3, 11))
			init_population_jit(population, D, finite_mask, knn_idx, strat_probs, seeds, rcl_r)
			
			fitness = np.empty(lam, dtype=np.float64)
			batch_lengths_jit(population, D, fitness)
			
			best_idx = np.argmin(fitness)
			best_ever_fitness = fitness[best_idx]
			stagnation_counter = 0
			
			# Buffers
			c_pop = np.empty((lam, n), dtype=np.int32)
			c_fit = np.empty(lam, dtype=np.float64)
			W = min(lam, 50)
			dlb_mask = np.zeros(n, dtype=np.bool_)
			
			last_patient_sent_time = 0.0

			yourConvergenceTestsHere = True
			gen = 0
			while( yourConvergenceTestsHere ):
				gen += 1
				
				# 1. Check from Scout (Non-blocking)
				try:
					healed = q_from_scout.get_nowait()
					h_fit = tour_length_jit(healed, D)
					worst_idx = np.argmax(fitness)
					population[worst_idx][:] = healed[:]
					fitness[worst_idx] = h_fit
					if h_fit < best_ever_fitness:
						best_ever_fitness = h_fit
						stagnation_counter = 0
						gls_penalties[:] = 0
						gls_active = False
						D_gls = None
					# # Report from Scout
					if gen % 10 == 0:
						print(f"Gen {gen}: New Best (Scout) {best_ever_fitness:.2f}")
				except queue.Empty:
					pass

				# 2. Reproduction (Exploiter Logic)
				D_ls = D_gls if (gls_active and D_gls is not None) else D
				
				k_tournament = 5
				for i in range(0, lam - 1, 2):
					# Binary Tournament
					cand1 = self.rng.choice(lam, size=k_tournament, replace=False)
					p1 = cand1[np.argmin(fitness[cand1])]
					cand2 = self.rng.choice(lam, size=k_tournament, replace=False)
					p2 = cand2[np.argmin(fitness[cand2])]

					c1 = _ox_jit(population[p1], population[p2])
					c2 = _ox_jit(population[p2], population[p1])
					
					if self.rng.random() < exploit_mut:
						self._hybrid_mutation_inplace(c1, is_symmetric)
					if self.rng.random() < exploit_mut:
						self._hybrid_mutation_inplace(c2, is_symmetric)
						
					if not is_symmetric:
						# Simple repair only
						pass
					else:
						_repair_jit(c1, D, finite_mask)
						_repair_jit(c2, D, finite_mask)
					
					c_pop[i] = c1
					c_pop[i+1] = c2
				
				# 3. Evaluation
				batch_lengths_jit(c_pop, D, c_fit)
				
				# 4. Elite LS
				elite_ratio = 0.2
				elite_count = max(1, int(lam * elite_ratio))
				elite_indices = np.argsort(c_fit)[:elite_count]
				for idx in elite_indices:
					dlb_mask[:] = False
					self._vnd_or_opt_inplace(c_pop[idx], D_ls, knn_idx, dlb_mask, max_iters=exploit_ls, block_steps=3)
				
				for idx in elite_indices:
					c_fit[idx] = tour_length_jit(c_pop[idx], D)

				# 5. RTR Replacement
				current_best_idx = np.argmin(fitness)
				for i in range(lam):
					seed = int(self.rng.integers(0, 1<<30))
					better, target_idx = rtr_challenge_jit(
						c_pop[i], c_fit[i], population, fitness, W, seed, current_best_idx
					)
					if better:
						population[target_idx][:] = c_pop[i][:]
						fitness[target_idx] = c_fit[i]
				
				# 6. Updates & Reporting
				best_idx = np.argmin(fitness)
				bestObjective = float(fitness[best_idx])
				meanObjective = float(fitness.mean())
				
				if bestObjective < best_ever_fitness:
					best_ever_fitness = bestObjective
					# Report from Local
					print(f"Gen {gen}: New Best (Local) {best_ever_fitness:.2f}")
					stagnation_counter = 0
					gls_penalties[:] = 0
					gls_active = False
				else:
					stagnation_counter += 1
				
				# 7. Scout Communication (Send Patient)
				if stagnation_counter > (stagnation_limit // 2):
					cur_time = time.time()
					if cur_time - last_patient_sent_time > 5.0:
						try:
							patient = population[best_idx].copy()
							q_to_scout.put_nowait(patient)
							last_patient_sent_time = cur_time
						except queue.Full:
							pass
				
				# 8. GLS
				gls_trigger = max(30, int(stagnation_limit * 0.6))
				if stagnation_counter >= gls_trigger:
					gls_active = True
					if gen % 50 == 0:
						self._gls_update_penalties(population[best_idx], D, gls_penalties)
						gls_lambda = 0.03 * (bestObjective / n)
						D_gls = D + gls_lambda * gls_penalties
						D_gls = np.ascontiguousarray(D_gls)
				else:
					gls_active = False
				
				# 9. Restart
				if stagnation_counter >= stagnation_limit:
					best_tour_ever = population[best_idx].copy()
					seeds = np.random.randint(0, 1<<30, lam).astype(np.int64)
					rcl_r = int(self.rng.integers(3, 11))
					init_population_jit(population, D, finite_mask, knn_idx, strat_probs, seeds, rcl_r)
					population[0] = best_tour_ever
					batch_lengths_jit(population, D, fitness)
					stagnation_counter = 0
					best_ever_fitness = fitness.min()
					gls_penalties[:] = 0
					gls_active = False
					D_gls = None

				# Prepare bestSolution
				bestSolution = self._rotate_to_start(population[np.argmin(fitness)], 0)

				# Call the reporter with:
				#  - the mean objective function value of the population
				#  - the best objective function value of the population
				#  - a 1D numpy array in the cycle notation containing the best solution 
				#    with city numbering starting from 0
				timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
				if timeLeft < 0:
					break
			
			# Your code here.
			return 0
			
		finally:
			if scout_process.is_alive():
				scout_process.terminate()
				scout_process.join()

	def _hybrid_mutation_inplace(self, tour, is_symmetric):
		n = tour.shape[0]
		use_inversion = is_symmetric
		if use_inversion and self.rng.random() < 0.7:
			i = int(self.rng.integers(0, n - 1))
			j = int(self.rng.integers(i + 1, n))
			tour[i:j] = tour[i:j][::-1]
		else:
			i = int(self.rng.integers(0, n))
			j = int(self.rng.integers(0, n - 1))
			if j >= i: j += 1
			if i != j:
				city = tour[i]
				if j < i:
					tour[j+1:i+1] = tour[j:i]
					tour[j] = city
				else:
					tour[i:j] = tour[i+1:j+1]
					tour[j] = city

	def _vnd_or_opt_inplace(self, tour, D, knn_idx, dlb_mask, max_iters, block_steps):
		max_iters = int(max_iters)
		block_steps = int(block_steps)
		if block_steps < 1: block_steps = 1
		improved = True
		while improved:
			improved = False
			dlb_mask[:] = False
			if _candidate_or_opt_jit(tour, D, knn_idx, max_iters=max_iters, dlb_mask=dlb_mask, block_size=1):
				improved = True; continue
			dlb_mask[:] = False
			if _candidate_block_swap_jit(tour, D, knn_idx, max_iters=block_steps, dlb_mask=dlb_mask, block_size=2):
				improved = True; continue
			dlb_mask[:] = False
			if _candidate_or_opt_jit(tour, D, knn_idx, max_iters=block_steps, dlb_mask=dlb_mask, block_size=2):
				improved = True; continue
			dlb_mask[:] = False
			if _candidate_or_opt_jit(tour, D, knn_idx, max_iters=block_steps, dlb_mask=dlb_mask, block_size=3):
				improved = True; continue

	def _gls_update_penalties(self, tour, D, penalties):
		n = tour.shape[0]
		max_util = -1.0
		for i in range(n):
			a = tour[i]
			b = tour[(i + 1) % n]
			if not np.isfinite(D[a, b]): continue
			util = D[a, b] / (1.0 + penalties[a, b])
			if util > max_util: max_util = util
		if max_util < 0.0: return
		for i in range(n):
			a = tour[i]
			b = tour[(i + 1) % n]
			if not np.isfinite(D[a, b]): continue
			util = D[a, b] / (1.0 + penalties[a, b])
			if util >= max_util - 1e-12: penalties[a, b] += 1

	def _rotate_to_start(self, tour, start_city):
		pos = np.where(tour == start_city)[0]
		if pos.size > 0:
			idx = int(pos[0])
			if idx == 0: return tour.copy()
			return np.ascontiguousarray(np.concatenate([tour[idx:], tour[:idx]]))
		return tour.copy()

if __name__ == '__main__':
	multiprocessing.freeze_support() # Essential for Windows spawn
	# Write any testing code here.
	pass
