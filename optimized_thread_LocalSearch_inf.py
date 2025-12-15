# Write the implemented EA into the provided template file.  # 注：在提供的模板文件中实现进化算法

import Reporter  # 导入课程提供的结果上报器
import numpy as np  # 导入NumPy用于数值计算
from typing import List  # 类型提示：列表
import os  # 操作系统相关（环境变量等）
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # 限制OpenBLAS线程数，避免过度并行
os.environ.setdefault("MKL_NUM_THREADS", "1")  # 限制MKL线程数
os.environ.setdefault("OMP_NUM_THREADS", "1")  # 限制OpenMP线程数
from numba import njit, prange, set_num_threads  # 直接使用Numba（若不可用则脚本不运行）
set_num_threads(2)  # 设置Numba内部并行线程数
try:
    from numba import njit, prange, set_num_threads  # 直接使用Numba（若不可用则脚本不运行）
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False
    print("Numba not found, will use pure Python implementation.")
    exit()

@njit(cache=True, fastmath=True)
def _erx_jit(p1, p2):  # JIT版本的ERX交叉算子
    n = p1.size  # 个体长度（城市数）
    child = np.empty(n, np.int32)  # 子代数组
    used = np.zeros(n, np.uint8)  # 标记城市是否已使用
    neighbors = np.full((n, 4), -1, np.int32)  # 邻接表（最多4个邻居）
    deg = np.zeros(n, np.int32)  # 每个城市的邻接度

    def add_edge(u, v):  # 向邻接表添加一条边
        if v == u:  # 忽略自环
            return
        for k in range(deg[u]):  # 若已存在边则跳过
            if neighbors[u, k] == v:
                return
        if deg[u] < 4:  # 邻居数未超上限时添加
            neighbors[u, deg[u]] = v
            deg[u] += 1

    for parent in (p1, p2):  # 从两个父代构建邻接表
        for i in range(n):
            c = parent[i]  # 当前城市
            add_edge(c, parent[(i - 1) % n])  # 添加左邻
            add_edge(c, parent[(i + 1) % n])  # 添加右邻

    cur = p1[0]  # 起始城市
    next_scan = 0  # 下次线性扫描起点

    for t in range(n):  # 逐步填充子代
        child[t] = cur  # 记录当前城市
        used[cur] = 1  # 标记已使用

        best = -1  # 备选下一城市
        best_score = 1_000_000  # 初始化最优度量
        for k in range(deg[cur]):  # 遍历当前城市的邻居
            nb = neighbors[cur, k]
            if nb == -1 or used[nb] == 1:  # 忽略无效或已用邻居
                continue
            cnt = 0  # 统计邻居的未用邻居数
            for j in range(deg[nb]):
                x = neighbors[nb, j]
                if x != -1 and used[x] == 0:
                    cnt += 1
            if cnt < best_score:  # 选择未用邻居更少的城市
                best_score = cnt
                best = nb

        if best != -1:  # 若找到候选则前往
            cur = best
        else:
            while next_scan < n and used[next_scan] == 1:  # 线性找未用城市
                next_scan += 1
            if next_scan < n:
                cur = next_scan  # 使用下一个未用城市
            else:
                for r in range(n):  # 兜底：再找一个未用城市
                    if used[r] == 0:
                        cur = r
                        break
    return child  # 返回子代

@njit(cache=True, fastmath=True)
def tour_length_jit(tour, D):  # 计算单条路径长度
    n = tour.shape[0]  # 城市数
    s = 0.0  # 累计长度
    for i in range(n - 1):  # 相邻边求和
        s += D[tour[i], tour[i+1]]
    s += D[tour[n-1], tour[0]]  # 回到起点
    return s  # 返回路径长度

@njit(cache=True, fastmath=True, parallel=True)
def batch_lengths_jit(pop2d, D, out):  # 并行批量计算多条路径长度
    m, n = pop2d.shape  # m个个体，长度n
    for r in prange(m):  # 并行遍历个体
        s = 0.0  # 当前个体长度
        row = pop2d[r]  # 第r个个体
        for i in range(n - 1):  # 累计相邻边
            s += D[row[i], row[i+1]]
        s += D[row[n-1], row[0]]  # 回到起点
        out[r] = s  # 写入结果

# -------- KNN 索引预计算（JIT并行） --------  # 为每个城市预选K个最近可行邻居
@njit(cache=True, fastmath=True, parallel=True)
def build_knn_idx(D, finite_mask, K):  # 生成 knn_idx: (n,K)
    n = D.shape[0]  # 城市数量
    knn = np.full((n, K), -1, np.int32)  # 初始化为-1
    for i in prange(n):  # 并行处理每个城市
        # 统计可行候选数量
        cnt = 0
        for j in range(n):
            if finite_mask[i, j]:  # 仅保留有限边
                cnt += 1
        if cnt == 0:  # 无可行边
            continue
        cand_idx = np.empty(cnt, np.int32)  # 候选索引
        cand_dis = np.empty(cnt, np.float64)  # 候选距离
        c = 0
        for j in range(n):
            if finite_mask[i, j]:
                cand_idx[c] = j
                cand_dis[c] = D[i, j]
                c += 1
        order = np.argsort(cand_dis)  # 对候选距离排序
        m = K if K < cnt else cnt  # 取前K个
        for t in range(m):
            knn[i, t] = cand_idx[order[t]]
    return knn  # 返回KNN索引


def tour_length_np(tour: np.ndarray, D: np.ndarray) -> float:  # 纯NumPy计算路径长度（用于个别场景）
    idx_from = tour  # 起点索引数组
    idx_to = np.roll(tour, -1)  # 终点索引数组（右移一位并首尾相连）
    return float(np.sum(D[idx_from, idx_to]))  # 累加对应距离并转为float

# -------- local search (2-opt light, JIT only) --------  # 局部搜索：轻量2-opt，仅JIT
@njit(cache=True, fastmath=True)
def _two_opt_once_jit_safe(tour, D):  # 对tour尝试一次改进（带inf检查）
    n = tour.size  # 个体长度
    best_delta = 0.0  # 最优改变量
    bi = -1; bj = -1  # 最优区间
    tries = min(2000, n * 20)  # 采样次数上限
    for _ in range(tries):  # 随机采样(i,j)
        i = np.random.randint(0, n - 3)  # 左端点
        j = np.random.randint(i + 2, n - 1)  # 右端点
        a = tour[i]; b = tour[(i + 1) % n]  # 边(a,b)
        c = tour[j]; d = tour[(j + 1) % n]  # 边(c,d)
        # 若新边存在inf，则跳过该候选
        if not np.isfinite(D[a, c]) or not np.isfinite(D[b, d]):
            continue
        delta = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])  # 改变量
        if delta < best_delta:  # 更优则记录
            best_delta = delta; bi = i; bj = j
    if best_delta < 0.0:  # 有改进则反转
        l = bi + 1; r = bj  # 反转区间
        while l < r:
            tmp = tour[l]; tour[l] = tour[r]; tour[r] = tmp  # 交换两端
            l += 1; r -= 1
        return True  # 成功改进
    return False  # 未改进

# -------- 可行性与随机工具（JIT） --------
@njit(cache=True, fastmath=True)
def _tour_feasible_jit(tour, finite_mask):  # 检查整环是否可行
    n = tour.size
    for i in range(n):
        a = tour[i]; b = tour[(i + 1) % n]
        if not finite_mask[a, b]:
            return False
    return True

@njit(cache=True, fastmath=True)
def _repair_jit(tour, D, finite_mask, max_tries=50):  # 基于2-opt的快速修复
    for _ in range(max_tries):
        if _tour_feasible_jit(tour, finite_mask):
            return True
        if not _two_opt_once_jit_safe(tour, D):  # 若无法改进则退出
            break
    return _tour_feasible_jit(tour, finite_mask)

@njit(cache=True, fastmath=True)
def _rand_perm_jit(n):  # 生成随机排列（Fisher–Yates）
    arr = np.arange(n, dtype=np.int32)
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp
    return arr

# -------- RCL-NN（基于KNN加速） --------
@njit(cache=True, fastmath=True)
def _rcl_nn_tour_jit(D, finite_mask, knn_idx, r):  # 仅在KNN中选，形成RCL随机挑选
    n = D.shape[0]
    tour = np.empty(n, np.int32)
    used = np.zeros(n, np.uint8)
    cur = np.random.randint(0, n)  # 随机起点
    tour[0] = cur; used[cur] = 1
    K = knn_idx.shape[1]
    for t in range(1, n):
        # 从KNN列表收集候选
        tmp_idx = np.empty(K, np.int32)
        tmp_dis = np.empty(K, np.float64)
        cnt = 0
        for k in range(K):
            j = knn_idx[cur, k]
            if j == -1:
                continue
            if used[j] == 1:
                continue
            if not finite_mask[cur, j]:
                continue
            tmp_idx[cnt] = j
            tmp_dis[cnt] = D[cur, j]
            cnt += 1
        if cnt == 0:  # 回退：线性找一个可行未用城市
            pool_cnt = 0
            for j in range(n):
                if used[j] == 0 and finite_mask[cur, j]:
                    pool_cnt += 1
            if pool_cnt == 0:  # 再退：找任意未用
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
            order = np.argsort(tmp_dis[:cnt])  # 仅对<=K个元素排序
            rsize = r if r < cnt else cnt
            pick = order[np.random.randint(0, rsize)]
            nxt = int(tmp_idx[pick])
        tour[t] = nxt; used[nxt] = 1; cur = nxt
    # 闭环修补
    if not finite_mask[tour[n - 1], tour[0]]:
        for _ in range(20):
            if _two_opt_once_jit_safe(tour, D):
                if finite_mask[tour[n - 1], tour[0]]:
                    break
            else:
                break
    return tour

# -------- 插入法（JIT，随机/最远） --------
@njit(cache=True, fastmath=True)
def _insertion_tour_jit(D, finite_mask, use_farthest):  # 近似实现，适配JIT
    n = D.shape[0]
    a = np.random.randint(0, n)
    b = a
    for _ in range(16):  # 尝试找到可行的第二个点
        b = np.random.randint(0, n)
        if b != a and finite_mask[a, b] and finite_mask[b, a]:
            break
    tour = np.empty(2, np.int32); tour[0] = a; tour[1] = b
    used = np.zeros(n, np.uint8); used[a] = 1; used[b] = 1
    m = 2
    while m < n:
        # 选择待插入城市
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
            # 随机从未用城市挑一个
            remain = n - m; k = np.random.randint(0, remain)
            idx = -1
            for c in range(n):
                if used[c] == 0:
                    if k == 0: idx = c; break
                    k -= 1
            insert_city = idx
        # 找最小增量成本位置
        best_pos = -1; best_cost = 1e100
        for i in range(m):
            prev = tour[i - 1] if i > 0 else tour[m - 1]
            curr = tour[i]
            if finite_mask[prev, insert_city] and finite_mask[insert_city, curr]:
                cost = D[prev, insert_city] + D[insert_city, curr] - D[prev, curr]
                if cost < best_cost:
                    best_cost = cost; best_pos = i
        # 构建新tour
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
    # 闭环修补
    for _ in range(20):
        if finite_mask[tour[m - 1], tour[0]]:
            break
        if not _two_opt_once_jit_safe(tour, D):
            break
    return tour

# -------- 并行初始化（JIT） --------
@njit(cache=True, fastmath=True, parallel=True)
def init_population_jit(pop, D, finite_mask, knn_idx, strat_probs, seeds, rcl_r):  # 并行生成整个人口
    lam, n = pop.shape
    step = max(1, lam // 10)  # 打印步长（每10%）
    for i in prange(lam):  # 并行每个个体
        np.random.seed(seeds[i])  # 线程私有seed
        u = np.random.rand()  # 策略抽样
        if u < strat_probs[0]:  # RCL-NN
            tour = _rcl_nn_tour_jit(D, finite_mask, knn_idx, rcl_r)
        elif u < strat_probs[0] + strat_probs[1]:  # 插入法
            tour = _insertion_tour_jit(D, finite_mask, use_farthest=(np.random.rand() < 0.5))
        else:  # 随机+修复
            tour = _rand_perm_jit(n)
            ok = _repair_jit(tour, D, finite_mask, 50)
            if not ok:
                tour = _rcl_nn_tour_jit(D, finite_mask, knn_idx, rcl_r)
        pop[i] = tour  # 写入种群
        if i % step == 0:  # 关键阶段打印进度
            print("[init] generated", i, "/", lam)

class r0123456:  # 主求解类（类名应改为学号）

    def __init__(self,
                 N_RUNS: int = 500,  # 迭代代数上限
                 lam: int = 100,  # 种群规模λ
                 mu: int = 100,  # 子代规模μ
                 k_tournament: int = 5,  # 锦标赛选择的k
                 mutation_rate: float = 0.3,  # 变异概率
                 rng_seed: int | None = None,  # 随机种子
                 local_rate: float = 0.2,  # 局部搜索触发概率
                 ls_max_steps: int = 30,  # 每个个体2-opt步数
                 stagnation_limit: int = 150):  # 停滞代数阈值（触发灾变重启）
        self.reporter = Reporter.Reporter(self.__class__.__name__)  # 初始化上报器
        self.N_RUNS = int(N_RUNS)  # 保存参数：代数
        self.lam = int(lam)  # 保存参数：λ
        self.mu = int(mu)  # 保存参数：μ
        self.k_tournament = int(k_tournament)  # 保存参数：k
        self.mutation_rate = float(mutation_rate)  # 保存参数：变异率
        self.rng = np.random.default_rng(rng_seed)  # NumPy随机数生成器
        self.local_rate = float(local_rate)  # 保存参数：局部搜索概率
        self.ls_max_steps = int(ls_max_steps)  # 保存参数：最大步数
        self.stagnation_limit = int(stagnation_limit)  # 保存参数：停滞阈值
        # -------- 停滞检测与灾变重启 --------
        self.best_ever_fitness = np.inf  # 记录全局最优适应度
        self.stagnation_counter = 0  # 停滞计数器（连续无改进代数）

    # ---- Core EA ----
    def optimize(self, filename: str):  # 主优化入口
        with open(filename) as file:  # 打开CSV距离矩阵文件
            D = np.loadtxt(file, delimiter=",", dtype=np.float64, ndmin=2)  # 读取为二维浮点数组
        n = D.shape[0]  # 城市数量
        D = np.ascontiguousarray(D)  # 确保内存连续以便JIT高效访问

        # ---------- 第一层防线（可行性基线） ----------
        finite_mask = np.isfinite(D)  # True表示边可用
        np.fill_diagonal(finite_mask, False)  # 禁止自环

        # population as 2D array for fast batch fitness  # 使用2D数组存储种群，便于批量评估
        population = np.empty((self.lam, n), dtype=np.int32)  # 分配种群数组
        # ---------- 预计算KNN并并行初始化 ----------
        K = 32  # KNN候选数量
        print("[init] building KNN with K=", K)
        knn_idx = build_knn_idx(D, finite_mask, K)  # JIT构建KNN
        self._knn_idx = knn_idx  # 保存到实例，供后续修复/重生使用
        print("[init] KNN ready.")
        # 策略概率：RCL-NN / 插入 / 随机修复
        strat_probs = np.array([0.1, 0.3, 0.6], dtype=np.float64)
        # 每个线程的随机种子
        seeds = np.empty(self.lam, dtype=np.int64)
        for i in range(self.lam):
            seeds[i] = int(self.rng.integers(1 << 30))
        rcl_r = int(self.rng.integers(3, 11))  # RCL大小（3..10）
        print("[init] generating population in parallel... r=", rcl_r)
        init_population_jit(population, D, finite_mask, knn_idx, strat_probs, seeds, rcl_r)  # 并行初始化
        print("[init] population ready.")

        fitness = np.empty(self.lam, dtype=np.float64)  # 分配适应度数组
        self._eval_batch(population, D, fitness)  # 计算初始适应度

        # prealloc buffers  # 预分配子代和适应度缓冲
        off_count = self.mu  # 子代数量
        offspring = np.empty((off_count, n), dtype=np.int32)  # 子代数组
        offspring_f = np.empty(off_count, dtype=np.float64)  # 子代适应度

        # union buffers for truncation  # 合并选择时的适应度缓冲
        union_size = self.lam + self.mu  # 合并后大小
        union_fit = np.empty(union_size, dtype=np.float64)  # 合并适应度视图

        print("generation,best_cost", flush=True)  # 实时打印表头

        for gen in range(1, self.N_RUNS + 1):  # 主迭代循环
            # --- reproduction ---  # 复制阶段：选择、交叉、变异
            o = 0  # 子代写入游标
            while o < off_count:  # 生成μ个子代
                p1 = population[self._k_tournament_idx(fitness, self.k_tournament)]  # 选择父1
                p2 = population[self._k_tournament_idx(fitness, self.k_tournament)]  # 选择父2
                # 仅使用ERX交叉（JIT）
                c1 = self._erx(p1, p2)  # ERX子1
                c2 = self._erx(p2, p1)  # ERX子2
                if self.rng.random() < self.mutation_rate:  # 以概率对c1变异
                    self._hybrid_mutation_inplace(c1)  # 混合变异（反转/插入）
                if self.rng.random() < self.mutation_rate:  # 以概率对c2变异
                    self._hybrid_mutation_inplace(c2)  # 混合变异
                if self.rng.random() < self.local_rate:  # 以概率对c1做局部搜索
                    self._light_two_opt_inplace(c1, D, finite_mask, self.ls_max_steps)  # 轻量2-opt（带可行性）
                if self.rng.random() < self.local_rate:  # 以概率对c2做局部搜索
                    self._light_two_opt_inplace(c2, D, finite_mask, self.ls_max_steps)  # 轻量2-opt（带可行性）
                # 新增：子代可行性修复
                self._repair_inplace(c1, D, finite_mask)
                self._repair_inplace(c2, D, finite_mask)
                offspring[o] = c1; o += 1  # 记录子1
                if o < off_count:  # 若仍需子代
                    offspring[o] = c2; o += 1  # 记录子2

            # --- fitness ---  # 计算子代适应度
            self._eval_batch(offspring, D, offspring_f)  # 批量评估子代

            # --- (λ+μ) truncation with argpartition (no full sort) ---  # 使用argpartition保留前λ
            # build virtual union view  # 构建合并适应度视图
            union_fit[:self.lam] = fitness  # 前半：父代适应度
            union_fit[self.lam:] = offspring_f  # 后半：子代适应度

            # indices into virtual union: 0..lam-1 are parents, lam..lam+mu-1 are offspring  # 索引解释
            # ---------- 第二层防线（可行优先截断 + 兜底） ----------
            is_feasible = np.isfinite(union_fit)  # 可行掩码
            idx_all = np.arange(union_fit.shape[0])
            idx_feas = idx_all[is_feasible]
            idx_infe = idx_all[~is_feasible]
            if idx_feas.size >= self.lam:  # 可行充足
                feas_sorted = idx_feas[np.argsort(union_fit[idx_feas])]
                keep = feas_sorted[:self.lam]
            else:  # 可行不足
                keep = np.empty(self.lam, dtype=idx_all.dtype)
                if idx_feas.size > 0:
                    feas_sorted = idx_feas[np.argsort(union_fit[idx_feas])]
                    keep[:idx_feas.size] = feas_sorted
                need = self.lam - idx_feas.size
                if need > 0:
                    fill = idx_infe[:need] if idx_infe.size >= need else np.concatenate([idx_infe, idx_feas[:max(0, need - idx_infe.size)]])
                    keep[idx_feas.size:] = fill
            # materialize next population  # 实体化下一代种群
            next_pop = np.empty_like(population)  # 新种群
            next_fit = np.empty_like(fitness)  # 新适应度
            p_i = 0  # 写入游标
            for idx in keep:  # 遍历保留索引
                if idx < self.lam:  # 来自父代
                    cand = population[idx].copy()  # 复制个体
                    # 若该个体不可行则尝试修复，否则就地重生
                    if not self._tour_feasible(cand, finite_mask):
                        self._repair_inplace(cand, D, finite_mask)
                        if not self._tour_feasible(cand, finite_mask):
                            cand = self._greedy_feasible_tour(D, finite_mask)
                    next_pop[p_i] = cand
                    next_fit[p_i] = tour_length_np(cand, D)
                else:  # 来自子代
                    cand = offspring[idx - self.lam].copy()
                    if not self._tour_feasible(cand, finite_mask):
                        self._repair_inplace(cand, D, finite_mask)
                        if not self._tour_feasible(cand, finite_mask):
                            cand = self._greedy_feasible_tour(D, finite_mask)
                    next_pop[p_i] = cand
                    next_fit[p_i] = tour_length_np(cand, D)
                p_i += 1  # 前进游标
            population, fitness = next_pop, next_fit  # 替换为下一代

            # --- report ---  # 统计与上报
            best_idx = int(np.argmin(fitness))  # 最优个体索引
            bestObjective = float(fitness[best_idx])  # 最优适应度
            bestSolution = self._rotate_to_start(population[best_idx].copy(), 0)  # 旋转使0在起点
            meanObjective = float(fitness.mean())  # 平均适应度

            print(f"{gen},{bestObjective}", flush=True)  # 打印当前代与最佳值

            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)  # 向报告器上报
            
            # -------- 停滞检测与灾变重启 --------
            if bestObjective < self.best_ever_fitness:  # 发现新最优
                self.best_ever_fitness = bestObjective  # 更新全局最优
                self.stagnation_counter = 0  # 计数器清零
                print(f"    -> [Gen {gen}] !! NEW BEST: {bestObjective:.6f} !!")
            else:  # 无改进
                self.stagnation_counter += 1  # 计数器+1
            
            if self.stagnation_counter >= self.stagnation_limit:  # 达到停滞阈值
                print(f"    -> [Gen {gen}] !! STAGNATION ({self.stagnation_counter} gens) !!")
                print(f"    -> !! RESTARTING POPULATION (Cataclysm) !!")
                
                # 1. 保存当前最优个体
                best_tour_ever = population[best_idx].copy()
                
                # 2. 重新生成随机种子
                seeds = np.empty(self.lam, dtype=np.int64)
                for i in range(self.lam):
                    seeds[i] = int(self.rng.integers(1 << 30))
                rcl_r = int(self.rng.integers(3, 11))  # 新的RCL大小
                
                # 3. 并行重新初始化整个种群
                init_population_jit(population, D, finite_mask, knn_idx, strat_probs, seeds, rcl_r)
                
                # 4. 将历史最优解注入新种群首位
                population[0] = best_tour_ever
                
                # 5. 重新评估新种群适应度
                self._eval_batch(population, D, fitness)
                
                # 6. 重置计数器并更新全局最优
                self.stagnation_counter = 0
                self.best_ever_fitness = float(fitness[0])  # 保证一致性
                print(f"    -> [Gen {gen}] !! RESTART COMPLETE, BEST PRESERVED !!")
            
            if timeLeft < 0:
               break  # 可选：时间用尽则提前停止

        return 0  # 返回0表示正常结束

    # ---- Helpers ----
    def _eval_batch(self, pop2d: np.ndarray, D: np.ndarray, out: np.ndarray) -> None:  # 批量评估适应度
        if NUMBA_OK:  # 若Numba可用
            batch_lengths_jit(pop2d, D, out)  # 使用JIT并行计算
        else:
            # vectorized fallback  # 退化：逐个体用NumPy计算
            for i in range(pop2d.shape[0]):
                out[i] = tour_length_np(pop2d[i], D)

    def _random_permutation(self, n: int) -> np.ndarray:  # 生成随机排列（个体）
        return self.rng.permutation(n).astype(np.int32, copy=False)  # 返回int32类型排列

    def _k_tournament_idx(self, fitness: np.ndarray, k: int) -> int:  # 锦标赛选择返回索引
        k = 1 if k < 1 else k  # 至少为1
        k = min(k, fitness.shape[0])  # 不超过种群规模
        cand = self.rng.choice(fitness.shape[0], size=k, replace=False)  # 随机选k个候选
        # return index of minimal fitness among candidates  # 返回候选中适应度最小的索引
        best_local = np.argmin(fitness[cand])  # 找到局部最优下标
        return int(cand[best_local])  # 映射回全局索引

    def _light_two_opt_inplace(self, tour: np.ndarray, D: np.ndarray, finite_mask: np.ndarray, max_steps: int) -> None:  # 轻量2-opt局部搜索
        steps = int(max_steps)  # 步数上限
        for _ in range(steps):  # 多步尝试
            if not _two_opt_once_jit_safe(tour, D):  # 无改进则停止
                break

    def _hybrid_mutation_inplace(self, tour: np.ndarray) -> None:
        n = tour.shape[0]
        if self.rng.random() < 0.7:
            # inversion
            i = int(self.rng.integers(0, n - 1))
            j = int(self.rng.integers(i + 1, n))
            tour[i:j] = tour[i:j][::-1]
        else:
            # insertion (in-place)
            i = int(self.rng.integers(0, n))
            j = int(self.rng.integers(0, n - 1))
            if j >= i: j += 1
            if i == j: 
                return
            city = int(tour[i])
            if j < i:
                # 右移 j..i-1
                tour[j+1:i+1] = tour[j:i]
                tour[j] = city
            else:
                # 左移 i+1..j
                tour[i:j] = tour[i+1:j+1]
                tour[j] = city


    def _inversion_mutation_inplace(self, tour: np.ndarray) -> None:  # 反转变异（就地）
        """
        Inversion mutation for TSP:
        Select two indices i < j, then reverse the subsequence between them.
        """  # 中文：选定i<j，反转区间[i,j)
        n = tour.shape[0]  # 个体长度
        i = int(self.rng.integers(0, n - 1))  # 随机起点
        j = int(self.rng.integers(i + 1, n))  # 随机终点
        tour[i:j] = np.flip(tour[i:j])  # 执行反转

    def _swap_mutation_inplace(self, tour: np.ndarray) -> None:  # 交换变异（就地）
        n = tour.shape[0]  # 个体长度
        i = int(self.rng.integers(n))  # 随机位置i
        j = int(self.rng.integers(n - 1))  # 随机位置j（先少一格）
        if j >= i:  # 若j越过i则右移一位
            j += 1
        tour[i], tour[j] = tour[j], tour[i]  # 交换两位置

    def _tour_feasible(self, tour: np.ndarray, finite_mask: np.ndarray) -> bool:  # 判断整条环是否可行
        n = tour.shape[0]
        for i in range(n):
            a = int(tour[i]); b = int(tour[(i + 1) % n])
            if not finite_mask[a, b]:
                return False
        return True

    def _greedy_feasible_tour(self, D: np.ndarray, finite_mask: np.ndarray) -> np.ndarray:  # 兼容用：基于KNN的可行重生
        n = D.shape[0]  # 城市数
        try:
            knn_idx = self._knn_idx  # 取已缓存的KNN
            r = 5  # RCL大小
            tour = _rcl_nn_tour_jit(D, finite_mask, knn_idx, r)  # 走JIT路径
            return tour
        except Exception:
            # 兜底：线性随机可行扩展
            start = int(self.rng.integers(n))
            tour = np.full(n, -1, dtype=np.int32)
            used = np.zeros(n, dtype=np.bool_)
            cur = start
            tour[0] = cur; used[cur] = True
            for t in range(1, n):
                candidates = np.where((finite_mask[cur]) & (~used))[0]
                if candidates.size == 0:
                    remain = np.where(~used)[0]
                    ok = [int(c) for c in remain if finite_mask[cur, int(c)]]
                    nxt = int(self.rng.choice(ok)) if len(ok) > 0 else int(self.rng.choice(remain))
                else:
                    nxt = int(self.rng.choice(candidates))
                tour[t] = nxt; used[nxt] = True; cur = nxt
            return tour

    def _rotate_to_start(self, tour: np.ndarray, start_city: int) -> np.ndarray:  # 将指定城市旋转到首位
        pos = int(np.where(tour == start_city)[0][0])  # 找到起点位置
        if pos == 0:  # 已在首位则直接返回
            return tour
        return np.concatenate([tour[pos:], tour[:pos]])  # 拼接形成旋转后的排列

    def _rcl_nn_tour(self, D: np.ndarray, finite_mask: np.ndarray, r: int = 5) -> np.ndarray:  # RCL-NN初始化（限制候选列表）
        n = D.shape[0]  # 城市数量
        start = int(self.rng.integers(n))  # 随机起点
        tour = np.full(n, -1, dtype=np.int32)  # 初始化线路
        used = np.zeros(n, dtype=np.bool_)  # 使用标记
        cur = start  # 当前城市
        tour[0] = cur; used[cur] = True  # 放入起点
        for t in range(1, n):  # 逐步扩展
            candidates = np.where((finite_mask[cur]) & (~used))[0]  # 可行且未用
            if candidates.size == 0:  # 僵局
                remain = np.where(~used)[0]  # 剩余城市
                ok = [int(c) for c in remain if finite_mask[cur, int(c)]]  # 与cur可达
                nxt = int(self.rng.choice(ok)) if len(ok) > 0 else int(self.rng.choice(remain))  # 退而求其次
            else:
                distances = D[cur, candidates]  # 提取距离
                sorted_idx = np.argsort(distances)  # 排序索引
                rcl_size = min(r, candidates.size)  # RCL大小（限制为r或候选数）
                rcl = candidates[sorted_idx[:rcl_size]]  # 取前r个最近邻作为RCL
                nxt = int(self.rng.choice(rcl))  # 从RCL中随机选
            tour[t] = nxt; used[nxt] = True; cur = nxt  # 前进
        # 闭环不可行则尝试一次2-opt式修补
        if not finite_mask[tour[-1], tour[0]]:  # 末尾回起点不可达
            for i in range(1, n - 1):  # 枚举一个断点
                a, b = int(tour[-1]), int(tour[0])  # (a->b)
                c, d = int(tour[i - 1]), int(tour[i])  # (c->d)
                if finite_mask[a, d] and finite_mask[c, b]:  # 替换为(a->d)和(c->b)可行
                    tour[0:i] = tour[0:i][::-1]  # 反转修补
                    break
        return tour  # 返回可行初解

    def _insertion_tour(self, D: np.ndarray, finite_mask: np.ndarray, use_farthest: bool = False) -> np.ndarray:  # 插入法初始化（随机/最远插入）
        n = D.shape[0]  # 城市数量
        # 选择种子点（初始子路径）
        seed_size = max(2, min(5, n // 10))  # 种子数量
        remaining = np.arange(n, dtype=np.int32)  # 剩余城市
        selected = self.rng.choice(n, size=seed_size, replace=False)  # 随机选种子
        tour = selected.copy().astype(np.int32)  # 初始子路径
        remaining = np.setdiff1d(remaining, selected)  # 移除已选
        # 迭代插入剩余城市
        while remaining.size > 0:  # 还有未插入城市
            if use_farthest:  # 最远插入
                # 找到距离已选路径最远的城市
                max_dist = -1.0; farthest = -1
                for c in remaining:  # 遍历剩余城市
                    min_to_tour = np.inf  # 到已选路径的最小距离
                    for t in tour:  # 检查到每个已选城市的距离
                        if finite_mask[c, t] and D[c, t] < min_to_tour:
                            min_to_tour = D[c, t]
                    if min_to_tour > max_dist:  # 更新最远
                        max_dist = min_to_tour; farthest = c
                insert_city = farthest if farthest >= 0 else int(self.rng.choice(remaining))  # 最远城市
            else:  # 随机插入
                insert_city = int(self.rng.choice(remaining))  # 随机选城市
            # 找到插入位置（最小增量成本）
            best_pos = -1; best_cost = np.inf
            for i in range(len(tour)):  # 枚举插入位置
                prev = int(tour[i - 1]) if i > 0 else int(tour[-1])  # 前驱
                curr = int(tour[i])  # 当前位置
                # 可行性检查
                if finite_mask[prev, insert_city] and finite_mask[insert_city, curr]:
                    cost = D[prev, insert_city] + D[insert_city, curr] - D[prev, curr]  # 增量成本
                    if cost < best_cost:  # 更优则记录
                        best_cost = cost; best_pos = i
            if best_pos >= 0:  # 找到可行位置
                tour = np.insert(tour, best_pos, insert_city).astype(np.int32)  # 插入
            else:  # 无可行位置，随机插入（兜底）
                tour = np.insert(tour, self.rng.integers(len(tour)), insert_city).astype(np.int32)
            remaining = np.setdiff1d(remaining, [insert_city])  # 移除已插入
        # 闭环不可行则尝试一次2-opt式修补
        if not finite_mask[tour[-1], tour[0]]:  # 末尾回起点不可达
            for i in range(1, len(tour) - 1):  # 枚举一个断点
                a, b = int(tour[-1]), int(tour[0])  # (a->b)
                c, d = int(tour[i - 1]), int(tour[i])  # (c->d)
                if finite_mask[a, d] and finite_mask[c, b]:  # 替换为(a->d)和(c->b)可行
                    tour[0:i] = tour[0:i][::-1]  # 反转修补
                    break
        return tour  # 返回完整路径

    def _random_repair_tour(self, D: np.ndarray, finite_mask: np.ndarray, max_attempts: int = 10) -> np.ndarray:  # 纯随机+修复初始化
        n = D.shape[0]  # 城市数量
        for _ in range(max_attempts):  # 最多尝试max_attempts次
            tour = self.rng.permutation(n).astype(np.int32)  # 纯随机排列
            tour_copy = tour.copy()  # 复制以便修复
            self._repair_inplace(tour_copy, D, finite_mask)  # 尝试修复
            if self._tour_feasible(tour_copy, finite_mask):  # 修复成功
                return tour_copy  # 返回可行解
        # 若修复失败，回退到RCL-NN
        return self._rcl_nn_tour(D, finite_mask, r=5)  # 兜底：使用RCL-NN

    def _repair_inplace(self, tour: np.ndarray, D: np.ndarray, finite_mask: np.ndarray, max_tries: int = 20) -> None:  # 不可行修复
        n = tour.shape[0]  # 城市数
        tries = 0  # 尝试计数
        while tries < max_tries:  # 限制总尝试
            fixed = True  # 标记是否已修复
            for i in range(n):  # 扫描每条边
                a = int(tour[i]); b = int(tour[(i + 1) % n])  # 边(a->b)
                if not finite_mask[a, b]:  # 遇到不可达
                    fixed = False  # 需要修复
                    for _ in range(30):  # 随机挑j尝试2-opt修补
                        j = int(self.rng.integers(0, n - 1))  # 位置j
                        if j == i or j == (i + 1) % n:  # 跳过相邻
                            continue
                        c = int(tour[j]); d = int(tour[(j + 1) % n])  # 边(c->d)
                        if finite_mask[a, c] and finite_mask[b, d]:  # 新边可行
                            l = (i + 1) % n; r = j  # 反转区间
                            if l <= r:
                                tour[l:r + 1] = tour[l:r + 1][::-1]  # 直接反转
                            else:
                                seg = np.concatenate([tour[l:], tour[:r + 1]])  # 跨尾拼接
                                seg = seg[::-1]  # 反转
                                k = n - l  # 切回两段
                                tour[l:] = seg[:k]; tour[:r + 1] = seg[k:]
                            break  # 本次修补完成
                    break  # 跳出重新扫描
            if fixed:  # 无不可达边
                return  # 修复完成
            tries += 1  # 增加次数
        # 超过尝试次数则保留现状（后续淘汰/重生处理）

    # ---- ERX ----
    def _erx(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:  # ERX交叉（仅JIT路径）
        return _erx_jit(p1.astype(np.int32, copy=False),  # 转为int32并调用JIT
                        p2.astype(np.int32, copy=False))

    # CSCX 相关代码已删除（精简为 ERX-only）


if __name__ == "__main__":  # 脚本入口
    ea = r0123456(N_RUNS=10000000, lam=200, mutation_rate=0.7, k_tournament=30, mu=150, local_rate=0.2, ls_max_steps=30, stagnation_limit=8)  # 停滞150代后触发灾变重启
    ea.optimize("tour1000.csv")  # 运行优化并输出进度
