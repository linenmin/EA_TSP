# 完整的 Child 2 变异逻辑替换代码
# 找到 Child 2 的 "else:" (SCX成功后)，替换整个变异部分

            else:
                # SCX 成功，统一备份用于回滚
                backup_buf[:] = c2[:]
                
                # === 变异：10% Double Bridge + 90% Smart Shift ===
                if np.random.random() < exploit_mut:
                    # 10% Double Bridge
                    if np.random.random() < 0.10:
                        db_attempt += 1
                        mut_tour = double_bridge_move(c2)
                        if _tour_feasible_jit(mut_tour, finite_mask):
                            c2[:] = mut_tour[:]
                            db_success += 1
                        else:
                            c2[:] = backup_buf[:]
                    
                    # 90% Smart Shift
                    else:
                        found_improving = False
                        K = knn_idx.shape[1]
                        
                        for u_try in range(2):
                            if found_improving:
                                break
                            
                            u = np.random.randint(0, n)
                            u_prev = -1
                            u_next = -1
                            for pos in range(n):
                                if c2[pos] == u:
                                    u_prev = c2[(pos - 1) % n]
                                    u_next = c2[(pos + 1) % n]
                                    break
                            
                            for k in range(K):
                                v = knn_idx[u, k]
                                if v == -1 or v == u or v == u_prev or v == u_next:
                                    continue
                                
                                v_next = -1
                                for pos in range(n):
                                    if c2[pos] == v:
                                        v_next = c2[(pos + 1) % n]
                                        break
                                
                                if (finite_mask[u_prev, u_next] and 
                                    finite_mask[v, u] and 
                                    finite_mask[u, v_next]):
                                    
                                    delta = -D[u_prev, u] - D[u, u_next] + D[u_prev, u_next]
                                    delta += -D[v, v_next] + D[v, u] + D[u, v_next]
                                    
                                    if delta <= 0.0:
                                        new_tour = np.empty(n, np.int32)
                                        new_idx = 0
                                        for pos in range(n):
                                            if c2[pos] == u:
                                                continue
                                            new_tour[new_idx] = c2[pos]
                                            if c2[pos] == v:
                                                new_idx += 1
                                                new_tour[new_idx] = u
                                            new_idx += 1
                                        c2[:] = new_tour[:]
                                        
                                        smart_shift_success += 1
                                        found_improving = True
                                        break
                        
                        if not found_improving:
                            smart_shift_fail += 1
                            if np.random.random() < 0.05:
                                rand_shift_attempt += 1
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
    
    # 返回诊断统计（更新返回值）
    return scx_fail_count, scx_deadend_count, scx_closure_fail_count, rcl_fallback_count, mut_infeasible_count, mut_rollback_count, smart_shift_success, smart_shift_fail, db_attempt, db_success, rand_shift_attempt, total_offspring
