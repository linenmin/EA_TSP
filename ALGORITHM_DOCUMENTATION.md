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
│  │   (Exploiter)       │◄───── │   (Scout)           │          │
│  │   高选择压力         │ Best  │   小种群, 强LS      │          │
│  │   低变异率           │ Only  │   频繁重启           │          │
│  └─────────────────────┘       └─────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Evolution Loop (每代)                         │
│                                                                  │
│  1. Tournament Selection (亲本选择)                              │
│  2. OX Crossover (交叉)                                          │
│  3. Hybrid Mutation (Scramble/Insert if Asym)                    │
│  4. Repair (修复不可行解) - Valid Ops Only                       │
│  5. Batch Evaluation (适应度评估)                                │
│  6. Elite-Only Candidate Or-Opt (精英局部搜索)                   │
│  7. RTR Replacement (受限锦标赛替换)                             │
│  8. Scout Migration (单向: Scout -> Exploiter)                   │
│  9. Stagnation Restart (停滞重启)                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. 核心修正：非对称 TSP (Asymmetric TSP) 处理

**重要更新**: 代码增加了对非对称 TSP (`D[i,j] != D[j,i]`) 的智能检测与处理。

| 组件 | 对称 TSP (Symmetric) | 非对称 TSP (Asymmetric) |
|------|----------------------|-------------------------|
| **Crossover** | OX 或 EAX | **OX (Order Crossover)** (保留方向) |
| **Mutation** | Inversion (反转) | **Insertion/Scramble** (非反转) |
| **2-Opt** | ✅ 有效 | ❌ **禁用** (Delta 计算错误) |
| **Or-Opt** | ✅ 有效 | ✅ **有效** (不改变边方向) |

**原因**: 2-Opt 和 Inversion 涉及对路径片段的**反转 (Reverse)**。在非对称图中，反转路径会改变边的方向，导致成本剧烈变化，而标准的 Delta 公式无法捕捉这一变化，导致算法接受劣解。

---

## 2. 岛屿模型：Scout & Exploiter

我们将原有的 Explorer (低压随机探索) 重构为 **Scout (快速侦察兵)**。

### Island 0: Exploiter (主力)
- **目标**: 深度挖掘当前最优解。
- **配置**: 大种群 (300+), 强 LS, 严格停滞阈值.
- **接收**: 贪婪接收 Scout 发来的 Best 解 (如果比当前更优则直接采纳).
- **Repulsion**: **禁用** (允许 Exploiter 专注收敛).

### Island 1: Scout (侦察兵)
- **目标**: 快速扫描搜索空间的不同**吸引盆 (Basins of Attraction)**, 寻找潜在的高质量起点.
- **配置**:
    - **小种群**: 150 (平衡速度与多样性)
    - **强 LS**: 50 步 (保证局部极值质量)
    - **动态重启**: `max(100, BaseStags * 0.4)`. 确保在大规模问题上有足够的收敛时间 (避免"幼儿期夭折")。
- **发送**: 仅发送当前运行周期内的 **Best Solution**.

**协作模式**: Scout 像一个并在运行的 "Iterated Local Search" 或 "Random Restart" 引擎，不断向 Exploiter 输送高质量的种子，Exploiter 负责将这些种子深挖到底。

---

## 3. 交叉与变异

### Crossover: Order Crossover (OX)
我们坚持使用 OX，因为它天然适合非对称 TSP。
- **OX**: 复制父代一段连续子路径 -> 保留了边的方向性。
- **EAX (未采用)**: 标准 EAX 是基于无向图 AB-Cycle 的，在非对称问题上会失效。

### Mutation: Adaptive
- 如果是对称图: 70% Inversion (大步跳跃).
- 如果是非对称图: 强制使用 **Insertion/Scramble** (Or-Opt like), 避免非法反转。

---

## 4. 局部搜索 (Local Search)

### Candidate-List Driven Or-Opt
针对非对称 TSP，我们完全依赖 **Or-Opt** (Block Shift)。
- **操作**: 移动一段连续城市 (Block) 到新位置。
- **优势**: 不改变 Block 内部的边方向，也不改变 Block 之外的边方向，仅改变 3 条连接边。
- **候选列表**: 使用 KNN (K=32) 加速，只尝试移动到最近邻附近。


### VND Or-Opt (k=1/2/3) [NEW]
- **k=1**: 候选列表 Or-Opt + DLB，精细改进。
- **k=2/3**: 小步数块移动（不反转，ATSP 安全），补全中等规模结构调整。
- **使用节点**: 精英 LS / Trauma Center 康复 / Kick 修补 / 重启前抛光。
- **目的**: 避免只靠单点插入导致的早期锁死。

### 4. 搜索加速 (Search Acceleration) [NEW]
为了提升大规模问题 (ATSP/TSP) 的搜索效率，引入了以下关键技术：

1.  **Don't Look Bits (DLB)**
    *   **原理**: 为每个城市维护一个比特位 (Bit)，标记其是否在上一轮局搜中被改进。如果某城市的邻域未发生变化，则下一轮跳过检查。
    *   **应用**: 集成于 `Or-Opt` 和局部搜索主循环。
    *   **效果**: 在收敛后期显著减少无效计算，提升搜索速度 3-5 倍。

2.  **Kick Strategy (Double Bridge)**
    *   **Double Bridge (4-Opt Kick)**: 将路径切为 4 段并重组 (A-B-C-D -> A-D-C-B)。此操作对 ATSP 安全（不反转方向），且能跨越 2-Opt 无法逃离的局部最优。
    *   **触发机制**: 当停滞代数超过阈值的 50% 时，对精英个体执行 Kick 并尝试替换最差个体，从死局中“踢”出来。

3.  **Elite-Only Local Search**
    *   **策略**: 只对前 20% 的精英子代执行 VND Or-Opt (k=1/2/3)。
    *   **目的**: 集中算力于高潜力个体，避免对劣质解浪费计算资源。

4.  **Guided Local Search (GLS)**
    *   **触发**: stagnation >= 0.6 * limit，每 50 代更新一次惩罚。
    *   **代价**: D_gls = D + lambda * P，lambda = 0.03 * (best / n)。
    *   **复位**: new best / restart / MIRACLE 时清空惩罚。
    *   **候选**: GLS 激活时 K 扩大至 64，破解 KNN 锁死。
    *   **说明**: 只影响 LS 决策，适应度仍用原始 D。

---


## 5. 初始种群与重启

- **混合初始化**: RCL-NN (贪婪), Insertion (插入法), Random (随机).
- **死前遗言 (Deathbed Bequest)**: Scout 在重启前，会将当前种群的最优解发送给 Exploiter，确保"侦察成果"不丢失。

---

### 6. 当前性能诊断 (Status Diagnosis 2025-12-17)

### 现状分析
1.  **Exploiter (Island 0)**:
    - **表现**: 引入 DLB 和 Kick 策略后，搜索效率大幅提升。Tour1000 最佳解达到 **58628** (远优于历史)。
    - **瓶颈**: 随着代数增加，多样性下降，陷入局部最优后难以逃离。

2.  **Scout (Island 1)**:
    - **表现**: 严重落后。Tour500 中，Exploiter 已达 99k，Scout 重启多次后仍停留在 100k+。
    - **问题**:
        - **搜索浅层化**: 由于频繁重启 (Stagnation Limit ~100-200)，Scout 实际上是在反复进行"浅层爬山"，从未真正进入深层 Basin。
        - **无效迁移**: 也是因为质量太差，发送给 Exploiter 的解 (Bequest) 几乎从未被接受 (Acceptance Rate < 2%)。
    - **结论**: 当前的 "Mini-GA Scout" 模式对于大规模 Rugged Landscape 无效。

## 7. Trauma Center Model (Phase 4 Implementation)
针对上述问题，我们实施了 "Trauma Center" (创伤中心) 模型，彻底改造了 Island 1 (Scout) 的角色。

- **Scout (Trauma Center)**:
    - **模式**: **Single-Trajectory Iterated Local Search (ILS)** (不再是 GA)。
    - **工作流 (Trauma Protocol)**:
        1. **收治 (Admission)**: 接收 Exploiter 发来的停滞解 ("重症患者")，立即覆盖当前工作解。
        2. **手术 (Surgery)**: 使用 **Ruin & Recreate** (Segment Removal + Cheapest Insertion) 破坏 20% 的结构。
        3. **康复 (Rehab)**: 使用深度 **Or-Opt (with DLB)** 进行局搜。
        4. **出院 (Discharge)**: 一旦发现更优解 (Global Best Improvement)，立即发回 Exploiter。
    - **优势**: 无种群开销，迭代极快；Ruin 操作能打破 Exploiter 无法逃离的深坑。

- **Exploiter (Society)**:
    - **送诊**: 当停滞超过阈值的一半时，将当前最优解发送给 Scout。
    - **接收**: 无条件接收 Scout 发回的 "Healed Solution"，替换种群中的最差个体。

此架构实现了 **异构搜索 (Heterogeneous Search)**，结合了 GA 的群体挖掘能力和 ILS 的单点突破能力。
---

## 8. Trauma Center 优化日志 (Refinements 2025-12-18)

针对 Trauam Center 初期的 "滞后 (Lag)" 和 "多样性丧失 (Starvation)" 问题，进行了以下关键升级：

### 8.1 实时分诊 (Real-Time Triage)
- **问题**: Scout 之前按顺序处理 "挂号队列"，导致在 Exploiter 快速进化时，Scout 还在处理过时的旧解 (Lag)。
- **解决方案**: **Queue Flushing**。Scout 每次只从队列中取出**最新**的一个病人，丢弃所有积压的旧病人。确保 Scout 永远工作在 Exploiter 的最前沿。
- **状态同步**: 一旦收到更优病人，立即更新 Scout 内部的 `Best Record`，消除视觉上的性能Gap。

### 8.2 自适应手术 (Adaptive Ruin Gears)
- **问题**: 固定的 20% Ruin Rate 在后期难以逃离 Exploiter 已陷入的深层局部最优。
- **解决方案**: **Gear Shifting (换挡机制)**。
    - 根据停滞时间动态调整破坏强度。
    - **档位**: `[15%, 20%, 25%, 30%, 40%, 50%]`.
    - **逻辑**: 初始精密手术 (15%) -> 若无效则加大剂量 -> 最高截肢 (50%) -> 一旦好转立即重置回 15%。

### 8.3 多样性回流 (Diversity Injection)
### 8.3 多样性回流 (The Trojan Horse Protocol)
- **问题**: 早期版本只允许 "更好 (Strict Improvement)" 的解出院，导致 Scout 经常沉默，Exploiter 缺乏外部基因流入。
- **解决方案**: **Relaxed Discharge (特洛伊木马)**.
    - **逻辑**: 允许 "结构迥异" 且 "质量相近 (Tolerance)" 的解回流。
    - **动态宽容 (Dynamic Tolerance)**: 
        - Scout 根据自身的 **Stagnation** 程度动态调整 Tolerance。
        - 停滞 < 50: 0% (Strict Mode).
        - 停滞 > 50: 2%.
        - 停滞 > 200: 5% (Desperate Mode).
    - **Stagnation Persistence**: 当发送 Trojan (非最优但符合 Tolerance) 时，Scout **不重置** 自身的停滞计数器。只有这样，Scout 才能保持在高 Tolerance 状态，持续向 Exploiter 输送多样性，直到真正实现全局突破。

### 8.4 拓扑径向破坏 (Topological Radial Ruin - BFS)
- **问题**: 常规 Random Ruin 在 TSP 中效率低下。Spatial Radial Ruin 需要坐标，但我们只有距离矩阵。
- **解决方案**: **KNN-BFS Ruin**.
    - 利用预计算的 **KNN 列表** (K=32)。
    - 从随机中心点开始，进行 **广度优先搜索 (BFS)**，直到抓取到 N_Remove (e.g., 200) 个城市。
    - **效果**: 保证了被移除的城市在拓扑空间（距离图）上是一个紧密连接的 **Cluster (连通块)**，这在无坐标的情况下完美实现了 Radial Ruin 的效果。
    - **优势**: 迫使算法重建局部高密区域的连接，极大提升了跳出局部最优的能力。

---

## 9. 最新结果与分析 (Performance Update 2025-12-19)

### 9.1 总体进步
- **整体最优**: 不同规模实例的 Overall Best 进入 ~104.2k / ~98.8k / ~56.8k 区间，相比此前 105k+ / 99k+ / 57k+ 明显下探。
- **收敛节奏**: 早期快速下降 + 中后期缓慢爬坡的形态更稳定，尾段仍持续产生小改进，说明上限被抬高。

### 9.2 行为指标解读
- **选择压力**: RTR rolling accept rate 长期保持在稳定区间，没有提前归零，说明 Exploiter 仍有可接受的改进空间。
- **多样性**: Bond Distance / Edge Entropy 在中后期仍有周期性尖峰，显示 Kick + GLS 能持续注入结构扰动，避免完全冻结。
- **Trauma Center 贡献**: 出院事件达到几十次级别，已从“几乎无用”转为“稳定输出”，但主导改进仍在 Exploiter。

### 9.3 结论与下一步
- **结论**: VND Or-Opt + GLS 的组合显著抬高了上限，并延长了有效爬坡时间。
- **风险点**: 若 GLS 触发过频或 K 扩大过早，可能导致局搜成本过高、收益变小。
- **建议**: 维持当前策略结构，优先用日志监控 GLS 触发频率与平均改进量，再考虑更重的算子。

---

## 10. 近期结果与问题 (Performance Update 2025-12-19)

### 10.1 观察
- **tour500**: 改进幅度在 ~120 代后快速归零，改进率长期接近 0；Population Gap 在重启后出现大幅回弹，但未带来新的 Best；出院事件极多（300+）但对 Best 曲线影响很弱。
- **tour750**: 早期改进明显，中后期改进率逐步归零，RTR 接受率降到 ~1 附近；停滞段变长，尾段趋于冻结。
- **tour1000**: 改进率仍保持在中等水平（~0.2 左右），Gap 缓慢下降，停滞较轻，仍有爬坡空间。

### 10.2 诊断
- **M4 (Regret-lite 自适应强度)** 并未显著改善 tour500/tour750 的后期停滞。
- 当前瓶颈不是“探索次数不足”，而是**Scout 重建结果与 Exploiter 接收目标不匹配**：多样性和出院数量上升，但转化为 Best 改进的比例极低。

### 10.3 下一步方向 (待验证)
- **质量优先的重建/出院标准**: 以 “可带来可测提升” 为目标，而不是单纯输出大量候选。
- **结构对齐**: 重建过程应更贴近 Exploiter 的局部搜索偏好，减少“噪声注入”。
- **后期突破机制**: 对停滞后期设计更强的结构性扰动，但避免过早消耗预算。

---

## 11. 最新结果与分析 (Performance Update 2025-12-19)

### 11.1 关键改动
- VND 的 k=2/3 从随机块扫描改为候选驱动 Block Or-Opt（KNN 限制），降低无效扫描。
- 继续沿用 M4（Regret-lite + 自适应强度）与 GLS/DLB/Kick 结构。

### 11.2 结果摘要
- **tour500**: Overall Best ~98889；改进幅度在早期快速归零，改进率长期接近 0；Population Gap 呈周期性回弹（重启带来多样性但未转化为新 Best）；Trauma 出院次数 1000+，Best 贡献偏弱。
- **tour750**: Overall Best ~104134；中后期改进率接近 0，RTR 接受率下降到低位；Gap 多次“锯齿”，说明重启在复原多样性但缺少结构性突破；出院事件 200+。
- **tour1000**: Overall Best ~52574；改进率仍持续有小幅输出，Gap 继续下降，停滞相对轻；说明候选驱动 k=2/3 对大规模更有效。

### 11.3 诊断
- k=2/3 的改造主要改善大规模，500/750 的瓶颈是**缺少中等尺度结构交换**，而不是局搜深度不足。
- Scout 的高频出院多为“相似结构 + 轻微扰动”，质量门槛不足，导致接收多、收益少。


### 11.4 下一步方向 (优先级)
- **优先 1**: 引入 KNN 限制的 Segment Swap / Block Exchange（不反转，ATSP 安全），放在 k=1 Or-Opt 后。
- **优先 2**: Scout 出院质量门槛：出院前跑一轮 VND（k=1/2/3），并设定“改进/多样性”双阈值。
- **优先 3**: 小规模实例（500/750）中后期动态扩大 K 或启用更强 Kick，但仅在长期停滞后触发。

## 12. Segment Swap 失败原因分析 (Diagnosis 2025-12-19)

### 12.1 现象回顾
在将 `Segment Swap` (Block Exchange) 设为优先 1 (放在 Or-Opt k=1 后) 后，`tour500` 结果显著恶化 (98889 -> 99241)。表现为改进幅度快速归零，中后期无效计算增加。

### 12.2 代码逻辑审计
经检查 `optimized_thread_LocalSearch_inf.py` 中的 `_candidate_block_swap_jit` 实现，发现 **KNN 启发式搜索逻辑存在严重的错位 (Misalignment)**：

- **当前实现**:
    - 对于 Block 1 (Head `b`)，在 KNN(`b`) 中搜索目标 `target`。
    - 设定 Block 2 Head `f = target`。
    - 试图交换 Block 1 (`b...c`) 和 Block 2 (`f...g`)。
    - **问题**: 交换后产生的新边是 `(e, b)` 和 `(a, f)` (其中 `e` 是 `f` 的前驱)。
    - **逻辑错误**: code 使用 KNN(`b`) 找到了 `f`，这意味着 `D[b, f]` 很小。但交换后的路径**并不包含**边 `(b, f)`。
    - **后果**: 算法花费算力去搜索 "头靠头" (Head-to-Head) 距离近的块，但这对于构造新连接毫无意义。这导致 Segment Swap 实际上退化为一种**只有 1/K 命中率的随机交换** (只有当 `e` 恰好也是 `b` 的邻居时才有效)，不仅效率极低，还因高优先级运行而挤占了正规 Or-Opt (k=2/3) 的优化机会，破坏了良性结构。

### 12.3 修正方案 (Correct Heuristic)
为了形成高质量的新边 `(e, b)`，我们应该：
1.  对于 Block 1 (Head `b`)，在 KNN(`b`) 中搜索 `target`。
2.  设定 `target = e` (Block 2 的前驱)。
3.  设定 Block 2 Head `f = next(target)`.
4.  **原理**: 这样确保了 `D[e, b]` (即 `D[target, b]`) 是最小的，直接优化了新产生的连接边。

### 12.4 下一步行动 (Action Plan)
1.  **修复 Bug**: 修改 `optimized_thread_LocalSearch_inf.py` 中 `_candidate_block_swap_jit` 的索引逻辑，使 `target` 指向 Block 2 的前驱 (`e`) 而非 Head (`f`).
2.  **保持优先级**: 修复后，维持其在 VND 中的位置 (k=1 之后)，因为它是填补 Or-Opt 盲区的重要算子。
### 13. Segment Swap 修复后结果分析 (Analysis 2025-12-22 Post-Fix)

### 13.1 结果对比
- **tour500**: 99234.88 (停滞，甚至略有退步).
- **tour750**: 103725.2 (显著进步，此前 ~105k).
- **tour1000**: 52152.7 (显著进步，此前 ~57k).

### 13.2 结论
1.  **Segment Swap Fix 有效**: 大规模实例 (750/1000) 的显著提升验证了 `KNN(a)->f` 修正的正确性。结构性算子开始发挥作用，填补了 Or-Opt 的盲区。
2.  **tour500 特殊性**: 仅仅修复算子未能解决 `tour500` 的问题。原因在于其**种群健康度 (Population Health)** 极差。

### 13.3 tour500 深度诊断: 种群离散 (Population Divergence)
查看 `tour500` 的日志和图表发现异常现象：
- **Gap (Mean - Best) 极大**: 均值与最优值的差距达到 ~400,000 (400%)。这意味着虽然 Elite 在 99k，绝大多数个体仍在 500k 的水平（可能是半随机或损坏的解）。
- **后果**:
    - **Crossover 失效**: 精英 (99k) 与 垃圾 (500k) 交叉，只会产生 300k 的平庸后代，无法产生有竞争力的子代。
    - **RTR Accept 虚高**: Avg RTR Acceptance ~29 (非常高)。这说明子代虽然平庸，但仍比种群中更差的个体要好。Scout 只是在“清理垃圾”，而不是在“挑战精英”。
    - **单打独斗**: 整个系统退化为 "Elite + Local Search" 的轨迹，GA 的群体智慧完全失效。

### 13.4 下一步策略: 种群重同步 (Population Resync)
针对 `tour500` 这种“精英孤跑，大众掉队”的现象，需要强制拉回种群。

- **Prior 1: Population Resync (Culling)**
    - **机制**: 当 `Gap > Threshold` (ea. 50%) 时，强制执行 "Culling"：将种群中后 50% 的个体替换为 Elite 的变异克隆 (Mutated Clones)。
    - **目的**: 重新将搜索集中在 Elite 附近的 Basin，恢复 Crossover 的有效性。

- **Prior 2: Reduce Mutation Rate for Diverged Pop**
    - 如果种群方差过大，降低 mutation rate，防止好不容易聚合的种群再次被打散。

**Next Step**: 实现 `check_population_health` 和 `resync_population` 逻辑。

### 14. Population Health Crisis & The "Great Purge" (2025-12-19)

#### 14.1 现象诊断: 种群僵尸�?(Zombie Population)
�?2025-12-19 的一轮深入分析中，我们在 `tour500`, `tour750`, `tour1000` 上发现了致命的种群健康问题：
- **`inf` Mean Fitness**: `tour750` �?`tour1000` �?`mean_fit` 长期显示�?`inf`。这表明种群中有大量的无效解（断路）�?
- **Divergence (离散)**: `tour500` �?Gap (Mean - Best) 超过 **300%**。Best �?99k，�?Mean �?340k+�?
- **失效后果**: 整个 Genetic Algorithm 失效。Crossover 在这种环境下毫无意义（任�?Elite 与垃圾交叉都是垃圾）。系统本质上退化为 Single-Trajectory Search�?

#### 14.2 解决方案: Population Resync (The Great Purge)
为了挽救崩溃的种群，我们引入�?**Population Resync (种群重同�?** 机制，类似于“重启”或“大清洗”，但保留了 Elite�?

**Resync Protocol**:
1.  **INF Culling**: 每代检查种群。如果发现任�?`inf` fitness，视为“极度危险”�?
    - **Action**: 立即执行 `_resync_population`，清除所�?`inf` 个体以及额外的最差个体（Cutoff Ratio = `inf_ratio + 10%`）�?
2.  **Gap Control**: 如果没有 `inf`，但 `(Mean - Best) / Best > 50%`�?
    - **Action**: 触发 Resync，清除最差的 **50%** 个体�?
3.  **Refilling Strategy**: 被清除的空位�?Elite �?**Clone + Mutation** 填补�?
    - **50%** slots: Double Bridge (微扰，拉�?Elite).
    - **50%** slots: 4x Double Bridge (强扰，保持局部多样�?.

#### 14.3 效果验证 & Pacification
- **Before Fix**: `tour500` Gap > 300%, Mean = 340k.
- **After Fix**: `tour500` Gap ~6%, Mean = 105k. `inf` 完全消失�?
- **副作�?*: 引入了剧烈的 "Pulse Effect" (Gap 爆炸 -> Resync -> 恢复 -> Gap 爆炸)。这表明基础 Mutation Rate 依然过高�?
- **调整 (Pacification)**:
    - **Mutation Rate**: �?Exploiter �?`mutation_rate` �?`0.2` 下调�?`0.1` (tour500/750/1000)，减少人为引入的 Chaos�?
    - **Resync Threshold**: �?Gap 阈值从 `0.5` 放宽�?`1.0`，减�?Resync 频率，允许更自然的进化�?
