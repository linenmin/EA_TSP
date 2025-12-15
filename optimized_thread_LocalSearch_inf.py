# Write the implemented EA into the provided template file.
# 注：本脚本实现了基于遗传算法（Genetic Algorithm）的旅行商问题（TSP）求解器
# 核心特性：边缘重组交叉（ERX）、混合变异、K-最近邻（KNN）初始种群、并行加速评估、局部搜索

import Reporter  # 导入课程提供的结果上报器，用于提交结果
import numpy as np  # 导入NumPy库用于高效数值计算
from typing import List, Optional  # 导入类型提示
import os  # 导入操作系统接口

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
set_num_threads(4)


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


@njit(cache=True, fastmath=True, parallel=True)
def init_population_jit(pop, D, finite_mask, knn_idx, strat_probs, seeds, rcl_r):
    """
    并行生成初始种群，混合多种构造策略。
    """
    lam, n = pop.shape
    step = max(1, lam // 10)
    
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
        if i % step == 0:
            # 注意：在JIT并发中 print 可能无序，仅作基本进度提示
            # print("[init] generated", i, "/", lam) 
            pass


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
                 stagnation_limit: int = 150):# 停滞阈值（用于灾变）
        
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

    def optimize(self, filename: str):
        """
        主优化流程：读取文件 -> 初始化 -> 进化循环 -> 上报结果
        """
        # 1. 读取数据
        with open(filename) as file:
            D = np.loadtxt(file, delimiter=",", dtype=np.float64, ndmin=2)
        n = D.shape[0]
        D = np.ascontiguousarray(D)  # 内存连续化优化

        # 2. 预处理可行性掩码 (True 代表边存在/距离有限)
        finite_mask = np.isfinite(D)
        np.fill_diagonal(finite_mask, False)

        # 3. 初始化种群容器
        population = np.empty((self.lam, n), dtype=np.int32)
        
        # 4. 构建 KNN 索引 (用于加速初始化)
        K = 32
        print(f"[Init] Building KNN (K={K}) for {n} cities...")
        knn_idx = build_knn_idx(D, finite_mask, K)
        self._knn_idx = knn_idx  # 保存引用
        print("[Init] KNN ready.")

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

        # 7. 预分配子代缓冲区
        off_count = self.mu
        offspring = np.empty((off_count, n), dtype=np.int32)
        offspring_f = np.empty(off_count, dtype=np.float64)
        
        # 8. 预分配合并缓冲区 (用于截断选择)
        union_size = self.lam + self.mu
        union_fit = np.empty(union_size, dtype=np.float64)

        print("generation,best_cost", flush=True)

        # ==================== 进化循环 (Evolution Loop) ====================
        for gen in range(1, self.N_RUNS + 1):
            
            # --- A. 繁殖阶段 (Reproduction) ---
            o = 0
            while o < off_count:
                # 锦标赛选择父代
                p1 = population[self._k_tournament_idx(fitness, self.k_tournament)]
                p2 = population[self._k_tournament_idx(fitness, self.k_tournament)]
                
                # 交叉 (ERX)
                c1 = self._erx(p1, p2)
                c2 = self._erx(p2, p1)
                
                # 变异 (Inversion / Insertion)
                if self.rng.random() < self.mutation_rate:
                    self._hybrid_mutation_inplace(c1)
                if self.rng.random() < self.mutation_rate:
                    self._hybrid_mutation_inplace(c2)
                    
                # 局部搜索 (Light 2-opt)
                if self.rng.random() < self.local_rate:
                    self._light_two_opt_inplace(c1, D, finite_mask, self.ls_max_steps)
                if self.rng.random() < self.local_rate:
                    self._light_two_opt_inplace(c2, D, finite_mask, self.ls_max_steps)
                    
                # 简单的可行性修复尝试
                self._repair_inplace(c1, D, finite_mask)
                self._repair_inplace(c2, D, finite_mask)
                
                offspring[o] = c1; o += 1
                if o < off_count:
                    offspring[o] = c2; o += 1

            # --- B. 评估阶段 (Evaluation) ---
            batch_lengths_jit(offspring, D, offspring_f)

            # --- C. 选择阶段 (Selection: Lambda + Mu) ---
            # 合并父代与子代适应度
            union_fit[:self.lam] = fitness
            union_fit[self.lam:] = offspring_f
            
            # 策略：优先保留可行解，再按距离排序
            is_feasible = np.isfinite(union_fit)
            idx_all = np.arange(union_size)
            idx_feas = idx_all[is_feasible]
            idx_infe = idx_all[~is_feasible]
            
            keep = np.empty(self.lam, dtype=np.int64)
            
            if idx_feas.size >= self.lam:
                # 可行解充足，取前 lambda 个最好的
                # 使用 argpartition 避免全排序，提升效率
                best_feas_indices = np.argpartition(union_fit[idx_feas], self.lam - 1)[:self.lam]
                keep = idx_feas[best_feas_indices]
                # 对保留的这部分再排个序（可选，为了让population有序）
                keep = keep[np.argsort(union_fit[keep])]
            else:
                # 可行解不足，先全拿，剩下用不可行解填补
                if idx_feas.size > 0:
                     keep[:idx_feas.size] = idx_feas[np.argsort(union_fit[idx_feas])]
                
                needed = self.lam - idx_feas.size
                if needed > 0:
                    # 补充不可行解
                    fill = idx_infe[:needed] 
                    if idx_infe.size >= needed:
                         fill = idx_infe[:needed]
                    else:
                         # 极其罕见：总数不够（逻辑上不可能，因为 union_size > lam）
                         raise ValueError("Population underflow error.")
                    keep[idx_feas.size:] = fill

            # 构造下一代种群
            next_pop = np.empty_like(population)
            next_fit = np.empty_like(fitness)
            
            p_i = 0
            for idx in keep:
                if idx < self.lam:
                    # 来自父代
                    cand = population[idx].copy()
                else:
                    # 来自子代
                    cand = offspring[idx - self.lam].copy()
                
                # 最终兜底：如果选入的个体仍不可行，尝试强制修复或重生
                if not self._tour_feasible(cand, finite_mask):
                    self._repair_inplace(cand, D, finite_mask)
                    if not self._tour_feasible(cand, finite_mask):
                        # 重生成一个新的可行解
                        cand = self._greedy_feasible_tour(D, finite_mask)
                
                next_pop[p_i] = cand
                next_fit[p_i] = tour_length_jit(cand, D) # 重新计算确保准确
                p_i += 1
                
            population, fitness = next_pop, next_fit

            # --- D. 上报与统计 (Reporting) ---
            best_idx = np.argmin(fitness)
            bestObjective = float(fitness[best_idx])
            bestSolution = self._rotate_to_start(population[best_idx].copy(), 0)
            meanObjective = float(fitness.mean())

            # 打印日志
            print(f"{gen},{bestObjective:.6f}", flush=True)

            # 调用 Reporter
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            
            # --- E. 停滞检测与灾变 (Stagnation & Cataclysm) ---
            if bestObjective < self.best_ever_fitness:
                self.best_ever_fitness = bestObjective
                self.stagnation_counter = 0
                print(f"    -> [Gen {gen}] !! NEW BEST: {bestObjective:.6f} !!")
            else:
                self.stagnation_counter += 1
                
            if self.stagnation_counter >= self.stagnation_limit:
                print(f"    -> [Gen {gen}] !! STAGNATION ({self.stagnation_counter} gens) !!")
                print(f"    -> !! RESTARTING POPULATION (Cataclysm) !!")
                
                # 1. 保存当前历史最优
                best_tour_ever = population[best_idx].copy()
                
                # 2. 重新初始化种群
                seeds = np.random.randint(0, 1<<30, self.lam).astype(np.int64)
                rcl_r = int(self.rng.integers(3, 11))
                init_population_jit(population, D, finite_mask, knn_idx, strat_probs, seeds, rcl_r)
                
                # 3. 注入历史最优
                population[0] = best_tour_ever
                
                # 4. 重新评估
                batch_lengths_jit(population, D, fitness)
                
                # 5. 重置状态
                self.stagnation_counter = 0
                self.best_ever_fitness = fitness.min() # 确保一致
                
                print(f"    -> [Gen {gen}] !! RESTART COMPLETE !!")

            if timeLeft < 0:
                print("Time limit reached.")
                break

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
        if self.rng.random() < 0.7:
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
        """调用 JIT 版局部搜索"""
        steps = int(max_steps)
        for _ in range(steps):
             # 若没有改进则提前退出
            if not _two_opt_once_jit_safe(tour, D):
                break

    def _repair_inplace(self, tour: np.ndarray, D: np.ndarray, finite_mask: np.ndarray):
        """调用 JIT 版修复逻辑"""
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
        "k_tournament": 30,     # 锦标赛选择 K 值
        "mutation_rate": 0.3,   # 变异概率
        "local_rate": 0.2,      # 局部搜索概率
        "ls_max_steps": 30,     # 局部搜索最大步数
        "stagnation_limit": 8   # 停滞多少代触发重启
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
