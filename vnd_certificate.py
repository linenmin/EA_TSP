@njit(cache=True, fastmath=True)
def compute_knn_best_gain(tour, D, knn_idx):
    """
    VND 证书验证：计算全 KNN 邻域的最佳 Gain
    
    遍历所有点 u 和它的 KNN v，计算：
    - Or-opt(1): 移动单个点
    - Or-opt(2): 移动2个连续点
    - Or-opt(3): 移动3个连续点
    - Swap(2): 交换两个点
    
    Returns:
        best_gain: 最佳改进值（负数表示有改进）
        move_type: 0=无改进, 1=Or-opt(1), 2=Or-opt(2), 3=Or-opt(3), 4=Swap(2)
    """
    n = len(tour)
    K = knn_idx.shape[1]
    
    # 建立位置索引
    pos = np.empty(n, np.int32)
    for i in range(n):
        pos[tour[i]] = i
    
    best_gain = 0.0
    move_type = 0
    
    # 遍历所有点及其 KNN
    for u in range(n):
        u_pos = pos[u]
        u_prev = tour[(u_pos - 1) % n]
        u_next = tour[(u_pos + 1) % n]
        
        for k in range(K):
            v = knn_idx[u, k]
            if v == -1 or v == u:
                continue
            
            v_pos = pos[v]
            v_prev = tour[(v_pos - 1) % n]
            v_next = tour[(v_pos + 1) % n]
            
            # === Or-opt(1): 移动单个点 u 到 v 后面 ===
            if abs(u_pos - v_pos) > 2:  # 避免平凡移动
                # 移除 u: u_prev -> u_next
                # 插入 u: v -> u -> v_next
                delta = -D[u_prev, u] - D[u, u_next] + D[u_prev, u_next]
                delta += -D[v, v_next] + D[v, u] + D[u, v_next]
                if delta < best_gain:
                    best_gain = delta
                    move_type = 1
            
            # === Or-opt(2): 移动2个连续点 (u, u_next) 到 v 后 ===
            if u_next != v and abs(u_pos - v_pos) > 3:
                u_next2 = tour[(u_pos + 2) % n]
                # 移除 (u, u_next): u_prev -> u_next2
                # 插入 (u, u_next): v -> u -> u_next -> v_next
                delta = -D[u_prev, u] - D[u_next, u_next2] + D[u_prev, u_next2]
                delta += -D[v, v_next] + D[v, u] + D[u_next, v_next]
                if delta < best_gain:
                    best_gain = delta
                    move_type = 2
            
            # === Or-opt(3): 移动3个连续点 ===
            if abs(u_pos - v_pos) > 4:
                u_next2 = tour[(u_pos + 2) % n]
                u_next3 = tour[(u_pos + 3) % n]
                if u_next3 != v:
                    delta = -D[u_prev, u] - D[u_next2, u_next3] + D[u_prev, u_next3]
                    delta += -D[v, v_next] + D[v, u] + D[u_next2, v_next]
                    if delta < best_gain:
                        best_gain = delta
                        move_type = 3
            
            # === Swap(2): 交换 u 和 v ===
            if abs(u_pos - v_pos) > 2:
                # u_prev -> v -> u_next 和 v_prev -> u -> v_next
                delta = -D[u_prev, u] - D[u, u_next] - D[v_prev, v] - D[v, v_next]
                delta += D[u_prev, v] + D[v, u_next] + D[v_prev, u] + D[u, v_next]
                if delta < best_gain:
                    best_gain = delta
                    move_type = 4
    
    return best_gain, move_type


