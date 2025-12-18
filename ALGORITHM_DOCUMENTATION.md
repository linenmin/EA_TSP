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


### 4. 搜索加速 (Search Acceleration) [NEW]
为了提升大规模问题 (ATSP/TSP) 的搜索效率，引入了以下关键技术：

1.  **Don't Look Bits (DLB)**
    *   **原理**: 为每个城市维护一个比特位 (Bit)，标记其是否在上一轮局搜中被改进。如果某城市的邻域未发生变化，则下一轮跳过检查。
    *   **应用**: 集成于 `Or-Opt` 和局部搜索主循环。
    *   **效果**: 在收敛后期显著减少无效计算，提升搜索速度 3-5 倍。

2.  **Kick Strategy (Double Bridge + Swap)**
    *   **Double Bridge (4-Opt Kick)**: 将路径切为 4 段并重组 (A-B-C-D -> A-D-C-B)。此操作对 ATSP 安全（不反转方向），且能跨越 2-Opt 无法逃离的局部最优。
    *   **Swap Segments (3-Opt Variant)**: 随机交换两个不相邻的片段。同样是 ATSP 安全的扰动。
    *   **触发机制**: 当停滞代数超过阈值的 50% 时，对精英个体执行 Kick 并尝试替换最差个体，从死局中“踢”出来。

3.  **Elite-Only Local Search**
    *   **策略**: 只对前 20% 的精英子代执行深度局部搜索 (DLB-Or-Opt)。
    *   **目的**: 集中算力于高潜力个体，避免对劣质解浪费计算资源。

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
- **问题**: 早期版本只允许 "更好 (Strict Improvement)" 的解出院，导致 Scout 经常沉默，Exploiter 缺乏外部基因流入。
- **解决方案**: **Relaxed Discharge**.
    - 允许 **相近 (Equal)** 或 **更优 (Better)** 的解出院。
    - **节流阀 (Throttle)**: 对 "Equal" 解施加冷却时间 (200 iters)，防止刷屏。
    - **效果**: 持续向 Exploiter 输送结构不同但质量相当的 "康复变体"，有效维持了种群多样性。

---
