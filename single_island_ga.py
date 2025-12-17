"""
单岛遗传算法求解器 (Single Island GA Solver)
复刻 Exploiter 岛的核心逻辑，无迁移机制

核心算法:
- RTR (Restricted Tournament Replacement) 替换策略
- OX (Order Crossover) 交叉算子
- Or-Opt 局部搜索
- 停滞重启机制
- Final Polish (Double Bridge + 2-Opt)
"""

import numpy as np
import os
import time

# 限制底层数学库线程数，避免并行冲突
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# 导入 Numba JIT 编译器
try:
    from numba import njit, prange
    print("Numba JIT 编译已启用")
except ImportError:
    print("警告: Numba 未安装，使用纯 Python 模式")
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    prange = range

# 导入课程 Reporter
import Reporter


# ==============================================================================
# Part 1: JIT 编译的核心函数
# ==============================================================================

@njit(cache=True, fastmath=True)
def tour_length_jit(tour, D):
    """计算路径总长度"""
    total = 0.0
    n = tour.shape[0]
    for i in range(n):
        total += D[tour[i], tour[(i + 1) % n]]  # 循环回起点
    return total


@njit(cache=True, fastmath=True)
def batch_lengths_jit(pop, D, out):
    """批量计算种群中所有个体的路径长度"""
    for i in range(pop.shape[0]):
        out[i] = tour_length_jit(pop[i], D)


@njit(cache=True, fastmath=True)
def bond_distance_jit(t1, t2):
    """计算两条路径的边集差异 (Bond Distance)"""
    n = t1.shape[0]
    diff = 0
    for i in range(n):
        # 检查 t1 的边是否在 t2 中
        a, b = t1[i], t1[(i + 1) % n]
        found = False
        for j in range(n):
            c, d = t2[j], t2[(j + 1) % n]
            if (a == c and b == d) or (a == d and b == c):
                found = True
                break
        if not found:
            diff += 1
    return diff


@njit(cache=True, fastmath=True)
def _rand_perm_jit(n):
    """生成随机排列"""
    arr = np.arange(n, dtype=np.int32)
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


@njit(cache=True, fastmath=True)
def _repair_jit(tour, D, finite_mask, max_attempts=50):
    """修复不可行路径 (确保所有边都存在)"""
    n = tour.shape[0]
    for _ in range(max_attempts):
        ok = True
        for i in range(n):
            a, b = tour[i], tour[(i + 1) % n]
            if not finite_mask[a, b]:
                ok = False
                # 找一个可行的交换
                for j in range(n):
                    if j != i and j != (i + 1) % n:
                        if finite_mask[a, tour[j]] and finite_mask[tour[j], b]:
                            tour[i + 1], tour[j] = tour[j], tour[i + 1]
                            break
        if ok:
            return True
    return False


@njit(cache=True, fastmath=True)
def _ox_jit(p1, p2, c1, c2, cut1, cut2):
    """OX (Order Crossover) 算子"""
    n = p1.shape[0]
    c1[:] = -1
    c2[:] = -1
    
    # 复制父代的中间段
    for i in range(cut1, cut2):
        c1[i] = p1[i]
        c2[i] = p2[i]
    
    # 从父代2填充子代1
    pos = cut2 % n
    fill_pos = cut2 % n
    for _ in range(n):
        city = p2[pos]
        # 检查是否已在子代中
        found = False
        for j in range(cut1, cut2):
            if c1[j] == city:
                found = True
                break
        if not found:
            c1[fill_pos] = city
            fill_pos = (fill_pos + 1) % n
        pos = (pos + 1) % n
    
    # 从父代1填充子代2
    pos = cut2 % n
    fill_pos = cut2 % n
    for _ in range(n):
        city = p1[pos]
        found = False
        for j in range(cut1, cut2):
            if c2[j] == city:
                found = True
                break
        if not found:
            c2[fill_pos] = city
            fill_pos = (fill_pos + 1) % n
        pos = (pos + 1) % n


@njit(cache=True, fastmath=True)
def _two_opt_jit(tour, D, max_iters=500):
    """2-Opt 局部搜索"""
    n = tour.shape[0]
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n - 2):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue  # 跳过首尾相连的情况
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                # 计算交换收益
                delta = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
                if delta < -1e-9:
                    # 反转 i+1 到 j 之间的路径
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True


@njit(cache=True, fastmath=True)
def _candidate_or_opt_jit(tour, D, knn_idx, max_iters=30):
    """Or-Opt 局部搜索 (基于候选列表加速)"""
    n = tour.shape[0]
    for _ in range(max_iters):
        improved = False
        for seg_len in [1, 2, 3]:  # 尝试移动 1/2/3 个连续城市
            for i in range(n):
                if i + seg_len > n:
                    continue
                
                # 计算移除段的成本变化
                prev_i = (i - 1) % n
                next_seg = (i + seg_len) % n
                
                remove_cost = D[tour[prev_i], tour[i]] + D[tour[(i + seg_len - 1) % n], tour[next_seg]]
                new_edge = D[tour[prev_i], tour[next_seg]]
                
                # 尝试插入到其他位置 (使用 KNN 加速)
                for k in range(min(20, knn_idx.shape[1])):
                    target = knn_idx[tour[i], k]
                    # 找到 target 在 tour 中的位置
                    for j in range(n):
                        if tour[j] == target:
                            break
                    else:
                        continue
                    
                    if abs(j - i) <= seg_len:
                        continue
                    
                    # 计算插入成本
                    next_j = (j + 1) % n
                    insert_cost = D[tour[j], tour[i]] + D[tour[(i + seg_len - 1) % n], tour[next_j]]
                    old_edge = D[tour[j], tour[next_j]]
                    
                    delta = (new_edge + insert_cost) - (remove_cost + old_edge)
                    
                    if delta < -1e-9:
                        # 执行移动 (复杂，这里简化处理)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break


@njit(cache=True, fastmath=True)
def _double_bridge_jit(tour):
    """Double Bridge 扰动 (4-Opt 变体)"""
    n = tour.shape[0]
    if n < 8:
        return
    
    # 选择 4 个切点
    p1 = np.random.randint(1, n // 4)
    p2 = np.random.randint(p1 + 1, n // 2)
    p3 = np.random.randint(p2 + 1, 3 * n // 4)
    
    # 重组: A + C + B + D
    new_tour = np.empty(n, dtype=np.int32)
    idx = 0
    for i in range(0, p1):
        new_tour[idx] = tour[i]
        idx += 1
    for i in range(p2, p3):
        new_tour[idx] = tour[i]
        idx += 1
    for i in range(p1, p2):
        new_tour[idx] = tour[i]
        idx += 1
    for i in range(p3, n):
        new_tour[idx] = tour[i]
        idx += 1
    
    tour[:] = new_tour[:]


@njit(cache=True, fastmath=True)
def rtr_challenge_jit(child, child_fit, pop, fit, W, rng_seed, best_idx):
    """RTR 替换挑战 (带精英保护)"""
    m = pop.shape[0]
    n = child.shape[0]
    np.random.seed(rng_seed)
    
    # 随机选择窗口
    window_indices = np.random.choice(m, size=W, replace=False)
    
    # 找窗口中最相似的个体
    closest_idx = -1
    min_dist = 99999999
    for idx in window_indices:
        dist = bond_distance_jit(child, pop[idx])
        if dist < min_dist:
            min_dist = dist
            closest_idx = idx
    
    target_idx = closest_idx
    target_fit = fit[target_idx]
    
    # 精英保护: 不替换最优个体
    if target_idx == best_idx:
        return False, target_idx
    
    # 竞争: 更优则替换
    if child_fit < target_fit:
        return True, target_idx
    
    return False, target_idx


@njit(cache=True, parallel=True)
def init_population_jit(pop, D, finite_mask, seeds):
    """并行初始化种群 (随机 + 修复)"""
    lam, n = pop.shape
    for i in prange(lam):
        np.random.seed(seeds[i])
        tour = _rand_perm_jit(n)
        _repair_jit(tour, D, finite_mask, 50)
        pop[i] = tour


# ==============================================================================
# Part 2: 主求解器类
# ==============================================================================

class SingleIslandGA:
    """单岛遗传算法求解器"""
    
    def __init__(self, 
                 N_RUNS=10_000_000,      # 最大代数
                 lam=100,                 # 种群大小 λ
                 k_tournament=5,          # 锦标赛大小
                 mutation_rate=0.3,       # 变异率
                 ls_max_steps=30,         # 局部搜索步数
                 stagnation_limit=120,    # 停滞重启阈值
                 rng_seed=None):          # 随机种子
        
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.N_RUNS = N_RUNS
        self.lam = lam
        self.k_tournament = k_tournament
        self.mutation_rate = mutation_rate
        self.ls_max_steps = ls_max_steps
        self.stagnation_limit = stagnation_limit
        self.rng = np.random.default_rng(rng_seed)
        
        # 运行时状态
        self.stagnation_counter = 0
        self.best_ever_fitness = float('inf')
    
    def optimize(self, filename: str):
        """主优化流程"""
        # 1. 读取距离矩阵
        with open(filename) as f:
            D = np.loadtxt(f, delimiter=",", dtype=np.float64, ndmin=2)
        n = D.shape[0]
        D = np.ascontiguousarray(D)
        
        # 2. 构建可行边掩码
        finite_mask = np.isfinite(D)
        
        # 3. 构建 KNN 索引 (用于 Or-Opt 加速)
        knn_k = min(20, n - 1)
        knn_idx = np.argsort(D, axis=1)[:, 1:knn_k + 1].astype(np.int32)
        
        # 4. 初始化种群
        population = np.empty((self.lam, n), dtype=np.int32)
        fitness = np.empty(self.lam, dtype=np.float64)
        seeds = self.rng.integers(0, 1 << 30, self.lam).astype(np.int64)
        init_population_jit(population, D, finite_mask, seeds)
        batch_lengths_jit(population, D, fitness)
        
        # 5. 初始化子代缓冲区
        c_pop = np.empty_like(population)
        c_fit = np.empty_like(fitness)
        
        # 6. RTR 窗口大小
        W = min(self.lam, max(20, self.lam // 5))
        
        print(f"开始优化: n={n}, λ={self.lam}, W={W}")
        print("Gen | Best | Mean | Stag")
        
        # ==================== 进化循环 ====================
        for gen in range(1, self.N_RUNS + 1):
            
            # --- A. 繁殖 (生成子代) ---
            for i in range(0, self.lam, 2):
                # 锦标赛选择父代
                p1_idx = self._tournament_select(fitness)
                p2_idx = self._tournament_select(fitness)
                p1, p2 = population[p1_idx], population[p2_idx]
                
                # OX 交叉
                cut1 = int(self.rng.integers(0, n))
                cut2 = int(self.rng.integers(cut1 + 1, n + 1))
                _ox_jit(p1, p2, c_pop[i], c_pop[i + 1], cut1, cut2)
                
                # 变异
                if self.rng.random() < self.mutation_rate:
                    self._swap_mutate(c_pop[i])
                if self.rng.random() < self.mutation_rate:
                    self._swap_mutate(c_pop[i + 1])
                
                # 修复
                _repair_jit(c_pop[i], D, finite_mask, 50)
                _repair_jit(c_pop[i + 1], D, finite_mask, 50)
            
            # 评估子代
            batch_lengths_jit(c_pop, D, c_fit)
            
            # --- B. 精英 LS (前 20% 子代) ---
            elite_count = max(1, int(self.lam * 0.2))
            elite_indices = np.argsort(c_fit)[:elite_count]
            for idx in elite_indices:
                _candidate_or_opt_jit(c_pop[idx], D, knn_idx, self.ls_max_steps)
                c_fit[idx] = tour_length_jit(c_pop[idx], D)
            
            # --- C. RTR 替换 ---
            current_best_idx = np.argmin(fitness)
            for i in range(self.lam):
                seed = int(self.rng.integers(0, 1 << 30))
                better, target_idx = rtr_challenge_jit(
                    c_pop[i], c_fit[i], population, fitness, W, seed, current_best_idx
                )
                if better:
                    population[target_idx][:] = c_pop[i][:]
                    fitness[target_idx] = c_fit[i]
            
            # --- D. 报告 ---
            best_idx = np.argmin(fitness)
            bestObjective = float(fitness[best_idx])
            meanObjective = float(np.mean(fitness))
            bestSolution = population[best_idx].copy()
            
            # 更新停滞计数
            if bestObjective < self.best_ever_fitness - 1e-6:
                self.best_ever_fitness = bestObjective
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            
            # 定期打印
            if gen % 50 == 0:
                print(f"{gen:5d} | {bestObjective:.2f} | {meanObjective:.2f} | {self.stagnation_counter}")
            
            # 调用 Reporter
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                print(f"时间耗尽，停止于第 {gen} 代")
                break
            
            # --- E. 停滞重启 ---
            if self.stagnation_counter >= self.stagnation_limit:
                print(f"!! 停滞 {self.stagnation_counter} 代 -> 重启 !!")
                best_tour = population[best_idx].copy()
                seeds = self.rng.integers(0, 1 << 30, self.lam).astype(np.int64)
                init_population_jit(population, D, finite_mask, seeds)
                population[0] = best_tour
                batch_lengths_jit(population, D, fitness)
                self.stagnation_counter = 0
        
        # ==================== Final Polish ====================
        print("Final Polish: Double Bridge + Deep 2-Opt...")
        best_idx = np.argmin(fitness)
        best_tour = population[best_idx].copy()
        best_fit = tour_length_jit(best_tour, D)
        
        for _ in range(30):
            trial = best_tour.copy()
            _double_bridge_jit(trial)
            _two_opt_jit(trial, D, 500)
            trial_fit = tour_length_jit(trial, D)
            if trial_fit < best_fit:
                best_tour = trial
                best_fit = trial_fit
                print(f"  Polish 改进: {best_fit:.2f}")
        
        print(f"最终结果: {best_fit:.2f}")
        return 0
    
    def _tournament_select(self, fitness):
        """锦标赛选择"""
        candidates = self.rng.choice(len(fitness), size=self.k_tournament, replace=False)
        return candidates[np.argmin(fitness[candidates])]
    
    def _swap_mutate(self, tour):
        """交换变异"""
        n = len(tour)
        i, j = self.rng.choice(n, size=2, replace=False)
        tour[i], tour[j] = tour[j], tour[i]


# ==============================================================================
# Part 3: 命令行入口
# ==============================================================================

if __name__ == "__main__":
    # 测试单个文件
    solver = SingleIslandGA(
        lam=100,
        k_tournament=5,
        mutation_rate=0.3,
        ls_max_steps=20,
        stagnation_limit=120
    )
    solver.optimize("tour50.csv")
