# TSP Memetic Algorithm 技术文档

## 概述

本项目实现了一个**Memetic Algorithm (MA)** 求解旅行商问题 (TSP)。Memetic Algorithm 是遗传算法 (GA) 与局部搜索 (LS) 的结合体，利用 GA 的全局探索能力和 LS 的局部优化能力来高效求解组合优化问题。

### 核心文件

| 文件 | 功能 |
|------|------|
| `optimized_thread_LocalSearch_inf.py` | 核心求解器，包含所有算法组件 |
| `run_island_model.py` | 并行岛屿模型启动器，管理多进程通信 |

---

## 算法架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                         Island Model                            │
│  ┌─────────────────────┐       ┌─────────────────────┐          │
│  │   Island 0          │       │   Island 1          │          │
│  │   (Exploiter)       │◄─────►│   (Explorer)        │          │
│  │   高选择压力         │  移民  │   低选择压力         │          │
│  │   低变异率           │       │   高变异率           │          │
│  └─────────────────────┘       └─────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Evolution Loop (每代)                         │
│                                                                  │
│  1. Tournament Selection (亲本选择)                              │
│  2. OX Crossover (交叉)                                          │
│  3. Hybrid Mutation (变异)                                       │
│  4. Repair (修复不可行解)                                        │
│  5. Batch Evaluation (适应度评估)                                │
│  6. Elite-Only Candidate Or-Opt (精英局部搜索)                   │
│  7. RTR Replacement (受限锦标赛替换)                             │
│  8. Migration & Repulsion (移民与排斥)                           │
│  9. Stagnation Restart (停滞重启)                                │
│  10. Final Polish (Double Bridge + 2-Opt)                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. 交叉算子 (Crossover)

### 当前使用: Order Crossover (OX)

**函数**: `_ox_jit(p1, p2)` (lines 118-149)

**工作原理**:
1. 随机选择两个切点 `[cut1, cut2)`
2. 将父代1在切点间的片段**原封不动**复制到子代
3. 从切点2开始，按父代2的顺序填充剩余城市

**代码片段**:
```python
cut1 = np.random.randint(0, n - 1)
cut2 = np.random.randint(cut1 + 1, n)
for i in range(cut1, cut2):
    child[i] = p1[i]  # 保留 p1 的子路径
# 按 p2 顺序填充其余
```

### 为什么选择 OX 而非 ERX?

| 特性 | OX | ERX |
|------|-----|-----|
| 保留结构 | **连续子路径** | 边 (Edge) |
| 与 Or-Opt 配合 | ⭐ 完美 | ❌ 差 |
| 代码复杂度 | 34 行 | 77 行 |
| 收敛速度 | ⭐ 快 | 慢 |

**关键洞察**: 
- ERX 保留"边"，会打碎已经被局部搜索优化好的连续子路径。
- OX 保留"子路径"，与 Or-Opt 形成完美互补：OX 负责宏观组合，Or-Opt 负责微观调整。

### 其他未采用的交叉算子

| 算子 | 未采用原因 |
|------|-----------|
| **PMX** | 与 OX 类似但更复杂，对 TSP 无额外优势 |
| **Cycle Crossover** | 保留位置而非子路径，不适合 TSP |
| **EAX** | 最强但实现极复杂，需 AB-cycle 分解 |

---

## 2. 变异算子 (Mutation)

### 当前使用: Hybrid Mutation

**函数**: `_hybrid_mutation_inplace(tour)` (未直接显示，在 r0123456 类中)

**策略**:
- 70% 概率: **Inversion (反转变异)** — 随机选择一段，反转其顺序
- 30% 概率: **Insertion (插入变异)** — 将一个城市移动到另一个位置

### 为什么选择这个组合?

- **Inversion**: 大扰动，改变路径的"宏观形状"，帮助逃离局部最优。
- **Insertion**: 小扰动，微调城市位置，与 Or-Opt 类似但更轻量。

### 未采用的变异算子

| 算子 | 未采用原因 |
|------|-----------|
| **Swap** | 太弱，同时破坏两处结构 |
| **Scramble** | 太激进，破坏已优化结构 |

---

## 3. 局部搜索 (Local Search)

### 当前使用: Candidate-List Driven Or-Opt

**函数**: `_candidate_or_opt_jit(tour, D, knn_idx, max_iters)` (lines 391-484)

**核心创新 (P1)**:
- 使用预计算的 **KNN 索引** (K=32) 驱动搜索
- 只尝试将城市移动到其 K 个最近邻附近
- 避免随机采样的"盲人摸象"问题

**代码片段**:
```python
for k in range(K):
    target = knn_idx[u, k]  # 只考虑 K 个最近邻
    # 尝试将 u 插入到 target 后面
    delta = (new_edge_cost - remove_cost) + (insert_cost - old_edge_cost)
    if delta < -1e-6:  # 找到改进
        # 执行移动
```

### 为什么选择 Or-Opt 而非 2-Opt?

**关键发现**: 您的 TSP 实例是**非对称的** (`D[i,j] ≠ D[j,i]`)!

| 算子 | 对称 TSP | 非对称 TSP |
|------|----------|------------|
| **2-Opt** | ✅ 正确 | ❌ Delta 计算错误 |
| **Or-Opt** | ✅ 正确 | ✅ 正确 |

**原因**: 2-Opt 反转一段路径，会改变内部所有边的方向。在非对称 TSP 中，`D[A,B] ≠ D[B,A]`，所以反转后的路径长度不能用简单的 Delta 公式计算。

Or-Opt (块移动) 只改变**切点处的 3 条边**，不改变内部边的方向，因此在非对称 TSP 上是正确的。

### 为什么使用 KNN 候选列表?

**问题**: 随机采样的 2-Opt/Or-Opt 只能覆盖 ~0.4% 的候选对，效率极低。

**解决**: 使用 KNN 候选列表将搜索聚焦到最可能产生改进的邻域。

| 方法 | 每次尝试数 | 找到改进概率 |
|------|-----------|-------------|
| 随机采样 | ~2000 随机 | 低 (盲目) |
| **KNN 候选列表** | n × K 有序 | 高 (有目标) |

### 精英优先策略 (P3)

**改动**: 只对适应度前 20% 的子代进行局部搜索。

**原因**:
1. **防止同质化**: 如果对所有子代都做 LS，会把整个种群拉到同一个局部最优。
2. **节省时间**: LS 是最耗时的操作，只对精英做可以跑更多代数。
3. **保留多样性**: 未经 LS 的子代保留了交叉/变异产生的结构差异。

---

## 4. 选择与替换机制

### 亲本选择: Tournament Selection

**函数**: `_k_tournament_idx(fitness, k)` (lines 1076-1081)

**参数**:
- **Exploiter (Island 0)**: k=5 (高压力)
- **Explorer (Island 1)**: k=2 (低压力)

### 替换机制: RTR (Restricted Tournament Replacement)

**函数**: `rtr_challenge_jit(child, child_fit, pop, fit, W, rng_seed)` (lines 707-740)

**工作原理**:
1. 随机选择 W=20 个个体作为"窗口"
2. 在窗口中找到与子代**最相似**的个体 (Bond Distance 最小)
3. 如果子代更优，则替换该个体

**为什么选择 RTR?**

| 替换策略 | 多样性保持 | 选择压力 |
|----------|-----------|----------|
| Generational | ❌ 差 | 高 |
| Elitism | ❌ 较差 | 极高 |
| **RTR** | ⭐ 优秀 | 中等 |

RTR 强制子代与"相似"的个体竞争，这意味着在搜索空间的不同区域可以同时存在多个"优秀家族"，避免过早收敛。

### 未采用的替换策略

| 策略 | 未采用原因 |
|------|-----------|
| **Deterministic Crowding** | RTR 是其增强版 |
| **(μ+λ)** | 没有多样性保护机制 |
| **Fitness Sharing** | 计算开销大 |

---

## 5. 岛屿模型 (Island Model)

### 架构

```
Island 0 (Exploiter)  ◄────────►  Island 1 (Explorer)
   高选择压力 (k=5)                 低选择压力 (k=2)
   低变异率 (0.3)                   高变异率 (0.8)
   深度搜索 (ls=30)                 浅搜索 (ls=15)
```

**通信**: 每 50 代交换最优个体。

### 为什么使用异构岛屿?

**问题**: 同构岛屿往往快速收敛到相同区域，失去并行优势。

**解决**: 
- **Exploiter**: 负责精细优化当前最佳区域
- **Explorer**: 负责探索新区域，避免"把所有鸡蛋放在一个篮子里"

### 排斥机制 (Repulsion)

当两岛收敛到相同解时 (Bond Distance < 5% × n)，Explorer 岛触发**排斥重启**，强制逃离当前区域。

**代码片段**:
```python
if dist < repulsion_threshold:
    print("[Island 1] REPULSION TRIGGERED! Scrambling...")
    # 重新初始化种群
    init_population_jit(population, ...)
```

---

## 6. 初始化策略

### 混合初始化

**函数**: `init_population_jit(pop, D, finite_mask, knn_idx, strat_probs, seeds, rcl_r)` (lines 786-816)

**策略组合**:
- 10%: **RCL-NN** (贪婪最近邻 with 随机候选列表) — 高质量种子
- 30%: **Insertion Heuristic** (插入法) — 中等质量
- 60%: **Random + Repair** (随机 + 修复) — 保证多样性

### 为什么不用纯贪婪?

纯贪婪初始化 (如最近邻) 会导致种群起始点过于集中，降低多样性。混合策略确保种群覆盖搜索空间的不同区域。

---

## 7. 停滞检测与重启

### 停滞检测

如果连续 `stagnation_limit` 代没有发现更好的解，触发重启。

**参数** (tour750.csv):
- Exploiter: 80 代
- Explorer: 150 代

### 重启策略

1. 保留当前最优个体
2. 重新生成其余种群
3. 重置停滞计数器

---

## 8. Final Polish

### Double Bridge + Deep 2-Opt

**时机**: 在时间耗尽前的最后阶段。

**函数**: `_double_bridge_jit(tour)` (lines 743-783)

**流程**:
1. 取当前最优解
2. 循环 30 次:
   a. Double Bridge 扰动 (4-Opt 变体，打破局部结构)
   b. 500 步深度 2-Opt 优化
   c. 如果比之前更好，保留

**为什么需要 Double Bridge?**

2-Opt/Or-Opt 只能做"局部改进"，无法跨越某些结构障碍。Double Bridge 通过切断并重组路径的 4 个位置，产生 2-Opt 无法达到的全新结构。

---

## 超参数配置

### 问题规模自适应 (反比策略)

| 问题 | 种群大小 | 原因 |
|------|----------|------|
| tour50 | 10000 | 小问题计算快，用大种群保证多样性 |
| tour250 | 1000 | 平衡 |
| tour500 | 300 | 过渡区 |
| tour750 | 200 | 大问题计算慢，用小种群换代数 |
| tour1000 | 100 | 极简种群，专注 Memetic |

---

## 性能演进记录

| 版本 | tour750 成绩 | 关键改动 |
|------|-------------|----------|
| 基线 (ERX + 随机 2-Opt) | ~130000 | - |
| + P0 (非对称检测 + Or-Opt) | ~130000 | 修复 Delta 计算错误 |
| + P3 (精英优先 LS) | ~130000 | 保留多样性 |
| + P1 (KNN 候选列表 Or-Opt) | ~129451 | LS 效率提升 10x |
| + P2 (OX 交叉) | **~119208** | 保留子路径结构 |

---

## 依赖项

- **NumPy**: 数值计算
- **Numba**: JIT 编译加速热点函数
- **multiprocessing**: 岛屿模型并行
- **Reporter**: 课程提供的结果上报模块

---

## 运行方式

```bash
# 单次运行 (双岛屿模型)
python run_island_model.py

# 修改目标文件
# 编辑 run_island_model.py 中的 TARGET_CSV 变量
```
