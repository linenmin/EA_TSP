"""
TSP 问题矩阵诊断脚本
检查所有 tour*.csv 文件的对称性和稀疏性
"""
import numpy as np
import os

# 要检查的文件列表
FILES = ['tour50.csv', 'tour250.csv', 'tour500.csv', 'tour750.csv', 'tour1000.csv']

print("=" * 60)
print("TSP 问题矩阵诊断")
print("=" * 60)

for filename in FILES:
    if not os.path.exists(filename):
        print(f"\n{filename}: 文件不存在，跳过")
        continue
    
    # 加载距离矩阵
    D = np.loadtxt(filename, delimiter=',')
    n = D.shape[0]
    
    # 1. 对称性检查 (忽略 inf 值)
    # 只比较有限值
    finite_mask = np.isfinite(D) & np.isfinite(D.T)
    if np.any(finite_mask):
        is_symmetric = np.allclose(D[finite_mask], D.T[finite_mask], rtol=1e-5, atol=1e-8)
    else:
        is_symmetric = True  # 全是 inf，算对称
    
    # 计算非对称程度
    if not is_symmetric:
        with np.errstate(invalid='ignore'):
            diff = np.abs(D - D.T)
            diff = diff[finite_mask]
        max_diff = np.max(diff) if len(diff) > 0 else 0
        mean_diff = np.mean(diff) if len(diff) > 0 else 0
    
    # 2. 稀疏性检查 (inf 边的比例)
    finite_count = np.sum(np.isfinite(D))
    total_count = D.size
    sparsity = 1 - finite_count / total_count
    
    # 3. 距离统计
    finite_D = D[np.isfinite(D)]
    if len(finite_D) > 0:
        min_dist = np.min(finite_D[finite_D > 0])  # 排除自环
        max_dist = np.max(finite_D)
        mean_dist = np.mean(finite_D)
    
    # 输出结果
    print(f"\n{'='*40}")
    print(f"📁 {filename} ({n} 城市)")
    print(f"{'='*40}")
    
    # 对称性
    if is_symmetric:
        print(f"✅ 对称性: 对称 (可用 2-Opt)")
    else:
        print(f"⚠️  对称性: 非对称 (必须用 Or-Opt)")
        print(f"   最大差异: {max_diff:.4f}, 平均差异: {mean_diff:.4f}")
    
    # 稀疏性
    if sparsity > 0:
        print(f"⚠️  稀疏性: {sparsity:.2%} 的边不可行 (inf)")
    else:
        print(f"✅ 稀疏性: 完全连通图 (无 inf)")
    
    # 距离统计
    print(f"📊 距离范围: [{min_dist:.2f}, {max_dist:.2f}], 平均: {mean_dist:.2f}")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
