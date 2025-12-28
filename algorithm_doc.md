# TSP/ATSP Hybrid Genetic Algorithm - Algorithm Documentation

## 1. Algorithm Overview

This is a **Hybrid Genetic Algorithm (HGA)** designed for both symmetric TSP and asymmetric TSP (ATSP):

- **Dual-process architecture**: Main process for genetic evolution, Scout process for deep exploration
- **Multiple crossover operators**: EAX-lite (ATSP), SCX (ATSP backup), OX (symmetric TSP)
- **Multiple local search**: VND framework with Or-opt, 2-opt, Block Swap
- **Adaptive mechanisms**: GLS penalties, adaptive ruin ratio, adaptive VND intensity

---

## 2. Thread Control and Initialization

### 2.1 Environment Setup (Lines 1-16)

All numerical libraries are limited to single thread to avoid multiprocessing conflicts.

---

## 3. Core Data Structures

### 3.1 K-Nearest Neighbor Index (build_knn_idx, Lines 422-435)

- Input: Distance matrix D (nxn), finite_mask, K=64
- Output: knn[i, k] = k-th nearest valid neighbor of node i
- Diagonal finite_mask[i,i] = False excludes self

---

## 4. Initial Population Generation

### 4.1 Population Initialization Strategy

| Strategy  | Symmetric TSP | ATSP |
| --------- | ------------- | ---- |
| RCL-NN    | 10%           | 70%  |
| Insertion | 30%           | 30%  |
| Random    | 60%           | 0%   |

### 4.2 RCL-NN Heuristic (_rcl_nn_tour_jit, Lines 838-889)

Restricted Candidate List greedy nearest neighbor:
1. Random start
2. For each position: collect feasible unvisited neighbors from KNN, sort by distance, pick from top r
3. Check loop feasibility, retry up to 5000 times

### 4.3 Insertion Heuristic (_insertion_tour_jit, Lines 891-961)

Cheapest Insertion: sample 32 unvisited cities, find minimum cost insertion.

---

## 5. Crossover Operators

### 5.1 EAX-lite (_eax_lite_atsp_inplace, Lines 163-270)

Edge Assembly Crossover lite for ATSP:

**AB-Cycle Construction**:
- Build adjacency tables: succ[u]=v, pred[v]=u
- Find start point using KNN heuristic
- Traverse alternating A-edges and B-edges
- visited_mask check prevents 8-shaped non-simple cycles

**Parameters**:
- MAX_M = 50 (attempts)
- MIN_CYCLE_LEN = 4
- K0 = 16, K_merge = 16

**Subtour Merging**: When AB-cycle creates multiple subtours, merge using KNN-guided reconnection.

**Permutation Validity Check**: Final defense - verify child is valid permutation.

### 5.2 SCX (_scx_jit_inplace_ok, Lines 292-400)

Sequential Constructive Crossover (ATSP backup):
1. Build successor maps from both parents
2. Random start, greedily select next city from parents or KNN
3. Full graph scan O(n) as fallback
4. Check if chosen has future (can continue)
5. Loop closure check with swap repair

Error codes: 0=success, 1=dead end, 2=loop failure

### 5.3 OX (_ox_jit_inplace, Lines 273-290)

Order Crossover for symmetric TSP:
1. Random cut points cut1 < cut2
2. Copy p1[cut1:cut2] to child
3. Fill remaining from p2 circularly

---

## 6. Mutation Operators

### 6.1 Double Bridge (double_bridge_move, Lines 1171-1191)

Classic 4-opt mutation (3 cut points):
```
new_tour = tour[0:p1] + tour[p2:p3] + tour[p1:p2] + tour[p3:n]
```

### 6.2 Smart Shift (Lines 1347-1449, ATSP)

O(1) position lookup city relocation:
1. Build position map
2. Random city u, search KNN for target v
3. Check feasibility and delta
4. Accept if delta <= 0 or (delta < 5 and rand < 0.005)

Mutation probability: 10% Double Bridge, 90% Smart Shift

---

## 7. Local Search Operators

### 7.1 KNN-based Or-opt (_candidate_or_opt_jit, Lines 525-590)

Move block_size consecutive cities to better position using KNN candidates and DLB acceleration.

### 7.2 KNN-based 2-opt (_candidate_2opt_jit, Lines 593-665)

Classic 2-opt for symmetric TSP only.

### 7.3 Block Swap (_candidate_block_swap_jit, Lines 759-836)

Swap two non-adjacent blocks.

### 7.4 Directed 3-opt (_candidate_blockswap3_jit, Lines 667-756)

Three-segment swap for ATSP. Segment B lengths: [1, 2, 3, 5, 8].

### 7.5 VND Framework (_vnd_or_opt_inplace, Lines 1872-1909)

| Level        | Operators                           |
| ------------ | ----------------------------------- |
| 0 (Light)    | Or-opt(1) + 2-opt/Or-opt(2)         |
| 1 (Standard) | Level 0 + Or-opt(3) + Block Swap(2) |
| 2 (Heavy)    | Level 1 + Directed 3-opt            |

Adaptive switching based on stagnation.

---

## 8. Ruin and Recreate

### 8.1 Destroy Strategies

| Mode | Strategy   | Probability |
| ---- | ---------- | ----------- |
| 0    | BFS        | 70%         |
| 1    | Sequence   | 10%         |
| 2    | Worst Edge | 20%         |

### 8.2 Greedy Rebuild

Shuffle removed cities, insert each at minimum cost position.

### 8.3 Adaptive Ruin Ratio

```python
ruin_gears = [0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
ruin_pct = ruin_gears[(iter - last_improv) // 250 % 10]
```

---

## 9. Population Management

### 9.1 Tournament Selection

- Symmetric TSP: tournament_size = 5
- ATSP: tournament_size = 3 (protect diversity)

### 9.2 RTR Replacement (rtr_challenge_jit, Lines 976-993)

Restricted Tournament Replacement:
1. Sample W=min(50, lambda) individuals
2. Find closest by bond distance
3. Replace if child is better (with diversity protection)

### 9.3 Bond Distance

```
bond_distance = n - shared_edges
```

---

## 10. Guided Local Search (GLS)

### 10.1 Penalty Update

Activation: stagnation >= max(30, 20% * stagnation_limit)

```
utility[a,b] = D[a,b] / (1 + penalties[a,b])
Penalize edges with utility >= max_utility
```

### 10.2 Penalty Distance Matrix

```python
D_gls = D + 0.03 * (bestObjective / n) * penalties
```

Update every 5 generations.

---

## 11. Scout Process

### 11.1 Initialization (Lines 1592-1606)

- Independent finite_mask (diagonal cleared)
- Independent knn_idx (K=64)
- Pre-allocated buffers (zero allocation runtime)

### 11.2 Main Loop

1. Check patient from main process
2. Select ruin strategy and ratio
3. Execute Ruin and Recreate
4. Execute adaptive VND
5. Evaluate and send back if breakthrough or within tolerance

### 11.3 Adaptive VND Level

| Stagnation | VND Level |
| ---------- | --------- |
| < 300      | 0         |
| < 1000     | 1         |
| >= 1000    | 2         |

### 11.4 Return Strategy

- Breakthrough: blocking send
- Tolerance (0.3%-0.8% based on stagnation): non-blocking

---

## 12. Main Process Evolution Loop

### 12.1 Parameter Configuration

| Size   | lambda | stagnation_limit | exploit_mut | exploit_ls |
| ------ | ------ | ---------------- | ----------- | ---------- |
| n<300  | 1000   | 500              | 0.30        | 30         |
| n<600  | 250    | 150              | 0.15        | 30         |
| n<850  | 200    | 250              | 0.25        | 20         |
| n>=850 | 80     | 100              | 0.25        | 20         |

### 12.2 Per-Generation Flow

1. Check Scout returns -> replace worst
2. Evolve (crossover + mutation)
3. Batch evaluate children
4. Elite VND (top 10-20%)
5. RTR replacement
6. Update best and stagnation
7. Send patient to Scout if stagnant
8. GLS penalty activation/update
9. Population restart if needed
10. Report progress

### 12.3 Population Restart

Trigger: stagnation >= stagnation_limit

- 70% reinitialize completely
- 30% preserve best lineage (1-3 Double Bridge mutations)

---

## 13. Memory Optimization

### 13.1 Pre-allocated Buffers (Lines 1729-1740)

18 buffers pre-allocated: map, used, backup, adjacency, cycle, nodes, RTR buffers.

Effect: Zero allocation in evolution loop.

### 13.2 Zero-allocation Move (_make_move_opt)

Uses temp_buffer for in-place Or-opt moves.

---

## 14. Key Hyperparameters Summary

| Parameter       | Value           | Purpose                     |
| --------------- | --------------- | --------------------------- |
| K (KNN)         | 64              | Neighbor candidates         |
| K0 (AB-cycle)   | 16              | Cycle start check range     |
| K_merge         | 16              | Subtour merge KNN range     |
| MAX_M           | 50              | AB-cycle attempts           |
| MIN_CYCLE_LEN   | 4               | Minimum cycle length        |
| SCX_RETRY       | 3               | SCX failure retries         |
| p_eax           | 0.80            | EAX-lite probability (ATSP) |
| tournament_size | 3/5             | Tournament size (ATSP/TSP)  |
| RTR W           | min(50, lambda) | RTR window size             |
| GLS lambda      | 0.03            | Penalty coefficient         |

---

## 15. ATSP vs Symmetric TSP Differences

| Feature           | ATSP                  | Symmetric TSP     |
| ----------------- | --------------------- | ----------------- |
| Main crossover    | EAX-lite (80%) + SCX  | OX                |
| VND Level 0       | Or-opt(1) + Or-opt(2) | Or-opt(1) + 2-opt |
| VND Level 1       | + Or-opt(3)           | -                 |
| VND Level 2       | + Directed 3-opt      | -                 |
| Tournament size   | 3                     | 5                 |
| Init random ratio | 0%                    | 60%               |

---

## 16. Error Handling

| Scenario                   | Handling                  |
| -------------------------- | ------------------------- |
| EAX-lite fail              | fallback to SCX           |
| SCX fail                   | fallback to Double Bridge |
| Double Bridge infeasible   | rollback to parent        |
| Smart Shift no improvement | Feasible Kick fallback    |
| RCL-NN 5000 fails          | return random permutation |
| Initial tour infeasible    | RCL-NN rescue             |

---

## 17. Performance Optimization Techniques

1. **Numba JIT**: All core functions use @njit(cache=True, fastmath=True)
2. **Parallelization**: build_knn_idx, batch_lengths_jit, init_population_jit use prange
3. **Dont Look Bit**: Skip recently non-improving nodes in local search
4. **Pre-allocated Buffers**: Avoid in-loop memory allocation
5. **O(1) Position Lookup**: Use position mapping pos[city] = index
6. **KNN Candidate Filtering**: Reduce O(n^2) neighborhood to O(n*K)
