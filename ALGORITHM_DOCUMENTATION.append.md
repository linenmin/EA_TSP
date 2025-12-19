
### 15. Pacification Success & The "Tour500 Anomaly" (Final Analysis 2025-12-19)

#### 15.1 Pacification Strategy Results (Run: 194148)
实施 "Pacification" (Exploiter Mutation 0.1, Resync Threshold 1.0) 后，系统表现出极佳的稳定性与性能：
- **tour500**: **完全康复**。Population Gap 压制在 7% 左右，不再离散。
- **tour1000**: **历史最佳 (53,060)**。尽管仍产生 `inf`，但 Resync 有效充当了 Garbage Collector，清除死解，确保 Elite 持续进化。
- **结论**: "Elite-Driven Search + Garbage Collection" 是处理大规模非对称 TSP 的实用且高效模式。

#### 15.2 Why is tour500 so hard? (The "Mid-Size Trap" Theory)
相比于 tour750/1000 的持续下降 ("Staircase"), tour500 呈现出诡异的 "Flatline" (L型死线)。
- **地形分析**: tour500 似乎存在一个 **"Deep & Wide Basin"** (深且宽的引力坑)。
    - **Wide**: Scout 的 20% Ruin 不足以跳出这个坑的斜坡范围，LS 总是滑回同一个谷底。
    - **Deep**: 其他 Basin 都不如当前这个深，导致接受准则拒绝迁移。
- **种群陷阱**: `lam=300` 导致极高的选择压力，瞬间将所有个体吸入这个坑。Resync 只能把人拉回坑底，无法助推爬坡。
- **未来方向**: 若要突破 tour500，可能需要 **Hyper-Ruin (30%-40%)** 或反而 **减小种群** 以降低收敛速度。
