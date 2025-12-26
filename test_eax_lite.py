"""
EAX-lite 快速验证脚本
测试：
1. 基本可行性（无 Numba 编译错误）
2. EAX-lite 能否处理简单的 ATSP 实例
3. 生成的 tour 是否可行
"""

import numpy as np
import sys
sys.path.insert(0, r'd:\BaiduNetdiskWorkspace\Leuven\7th\Genetic Algorithms\project')
from r0927482 import build_knn_idx, _eax_lite_atsp_inplace, _tour_feasible_jit

# 创建一个简单的 ATSP 实例（50 个节点）
np.random.seed(42)
n = 50
D = np.random.rand(n, n) * 100
np.fill_diagonal(D, np.inf)

# 设置一些 inf 边来模拟真实 ATSP
for _ in range(n * 2):
    i, j = np.random.randint(0, n, 2)
    if i != j:
        D[i, j] = np.inf

finite_mask = np.isfinite(D)
np.fill_diagonal(finite_mask, False)

# 构建 KNN
knn_idx = build_knn_idx(D, finite_mask, 64)

# 创建两个父代 tour（简单的随机排列）
pA = np.random.permutation(n).astype(np.int32)
pB = np.random.permutation(n).astype(np.int32)

# 确保父代可行
from r0927482 import _repair_jit
_repair_jit(pA, D, finite_mask, 100)
_repair_jit(pB, D, finite_mask, 100)

if not _tour_feasible_jit(pA, finite_mask):
    print("警告：父代 A 不可行，测试可能失败")
if not _tour_feasible_jit(pB, finite_mask):
    print("警告：父代 B 不可行，测试可能失败")

# 准备 buffers
succA_buf = np.empty(n, dtype=np.int32)
predA_buf = np.empty(n, dtype=np.int32)
succB_buf = np.empty(n, dtype=np.int32)
predB_buf = np.empty(n, dtype=np.int32)
out_buf = np.empty(n, dtype=np.int32)
mark_buf = np.empty(n, dtype=np.int32)
cycle_u_buf = np.empty(n, dtype=np.int32)
cycle_v_buf = np.empty(n, dtype=np.int32)
nodes_buf = np.empty(n, dtype=np.int32)
child = np.empty(n, dtype=np.int32)

# 测试 EAX-lite
print("=" * 60)
print("EAX-lite 快速验证测试")
print("=" * 60)
print(f"问题规模: n = {n}")
print(f"父代 A 可行: {_tour_feasible_jit(pA, finite_mask)}")
print(f"父代 B 可行: {_tour_feasible_jit(pB, finite_mask)}")
print()

success_count = 0
fail_count = 0
fail_reasons = {1: 0, 2: 0, 3: 0}

for trial in range(20):
    result = _eax_lite_atsp_inplace(pA, pB, D, finite_mask, knn_idx, child,
                                    succA_buf, predA_buf, succB_buf, predB_buf,
                                    out_buf, mark_buf, cycle_u_buf, cycle_v_buf, 
                                    nodes_buf)
    
    if result == 0:
        success_count += 1
        # 验证 child 可行性
        if not _tour_feasible_jit(child, finite_mask):
            print(f"错误：Trial {trial + 1} 返回成功但 tour 不可行！")
    else:
        fail_count += 1
        fail_reasons[result] += 1

print(f"\n测试结果（20 次尝试）:")
print(f"  成功: {success_count} ({success_count / 20 * 100:.1f}%)")
print(f"  失败: {fail_count} ({fail_count / 20 * 100:.1f}%)")
if fail_count > 0:
    print(f"    - AB-cycle 构造失败: {fail_reasons[1]}")
    print(f"    - Subtour 合并失败: {fail_reasons[2]}")
    print(f"    - 可行性检查失败: {fail_reasons[3]}")

print()
if success_count > 0:
    print("✓ EAX-lite 基本功能正常")
else:
    print("✗ EAX-lite 未能成功生成任何 child")

print("=" * 60)
