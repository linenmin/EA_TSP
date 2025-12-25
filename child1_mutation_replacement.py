# 完整的 Child 1 变异逻辑替换代码
# 替换位置：Line 1022-1089 (从 "else:" 到 Child 2 开始前)

            else:
                # SCX 成功，统一备份用于回滚
                backup_buf[:] = c1[:]
                
                # === 变异：10% Double Bridge + 90% Smart Shift ===
                if np.random.random() < exploit_mut:
                    # 10% Double Bridge（强力 Kick）
                    if np.random.random() < 0.10:
                        db_attempt += 1
                        mut_tour = double_bridge_move(c1)
                        if _tour_feasible_jit(mut_tour, finite_mask):
                            c1[:] = mut_tour[:]
                            db_success += 1
                        else:
                            c1[:] = backup_buf[:]
                    
                    # 90% Smart Shift（改进版：delta<=0，u重试2次）
                    else:
                        found_improving = False
                        K = knn_idx.shape[1]
                        
                        # u 重试最多2次
                        for u_try in range(2):
                            if found_improving:
                                break
                            
                            u = np.random.randint(0, n)
                            u_prev = -1
                            u_next = -1
                            for pos in range(n):
                                if c1[pos] == u:
                                    u_prev = c1[(pos - 1) % n]
                                    u_next = c1[(pos + 1) % n]
                                    break
                            
                            # 遍历 KNN 找可行且改进的插入点
                            for k in range(K):
                                v = knn_idx[u, k]
                                if v == -1 or v == u or v == u_prev or v == u_next:
                                    continue
                                
                                v_next = -1
                                for pos in range(n):
                                    if c1[pos] == v:
                                        v_next = c1[(pos + 1) % n]
                                        break
                                
                                # 预判可行性
                                if (finite_mask[u_prev, u_next] and 
                                    finite_mask[v, u] and 
                                    finite_mask[u, v_next]):
                                    
                                    # Delta 计算（放宽到 <= 0）
                                    delta = -D[u_prev, u] - D[u, u_next] + D[u_prev, u_next]
                                    delta += -D[v, v_next] + D[v, u] + D[u, v_next]
                                    
                                    if delta <= 0.0:  # 接受 delta == 0
                                        # 执行 Shift
                                        new_tour = np.empty(n, np.int32)
                                        new_idx = 0
                                        for pos in range(n):
                                            if c1[pos] == u:
                                                continue
                                            new_tour[new_idx] = c1[pos]
                                            if c1[pos] == v:
                                                new_idx += 1
                                                new_tour[new_idx] = u
                                            new_idx += 1
                                        c1[:] = new_tour[:]
                                        
                                        smart_shift_success += 1
                                        found_improving = True
                                        break
                        
                        if not found_improving:
                            smart_shift_fail += 1
                            # 5% 兜底：随机 shift
                            if np.random.random() < 0.05:
                                rand_shift_attempt += 1
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

            # --- Child 2 --- (从这里开始是 Child 2 的代码)
