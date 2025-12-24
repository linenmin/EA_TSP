"""
TSP 最优解参考值获取脚本

方法 1: 使用 python-tsp 库（快速启发式）
方法 2: 使用 elkai 库（调用 LKH）
方法 3: 使用 Google OR-Tools（工业级求解器）
"""

import numpy as np
import time

def load_distance_matrix(filename):
    """加载距离矩阵"""
    return np.loadtxt(filename, delimiter=',')

def method_ortools(D):
    """使用 Google OR-Tools 求解 TSP（推荐，工业级）"""
    try:
        from ortools.constraint_solver import routing_enums_pb2
        from ortools.constraint_solver import pywrapcp
        
        n = D.shape[0]
        
        # 创建数据模型
        manager = pywrapcp.RoutingIndexManager(n, 1, 0)
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            d = D[from_node, to_node]
            if not np.isfinite(d):
                return 1000000000  # 大数表示不可达
            return int(d * 1000)  # 放大1000倍保持精度
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # 设置搜索参数（更强的搜索）
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = 60  # 60秒时间限制
        
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            return solution.ObjectiveValue() / 1000.0  # 除以1000恢复原始值
        return None
    except ImportError:
        print("  OR-Tools 未安装，运行: pip install ortools")
        return None

def method_elkai(D, precision=1, runs=10):
    """
    使用 elkai 库（LKH 的 Python 绑定）
    
    采用 DistanceMatrix 接口 + Big M 法 + 精度缩放，更稳健地处理：
    - 浮点数距离矩阵
    - inf 值（不可达边）
    - 非对称矩阵（ATSP）
    
    Args:
        D: 距离矩阵 (numpy array)
        precision: 精度缩放因子，默认 100 表示保留两位小数
        runs: LKH 迭代次数，对于含 inf 的矩阵建议设高一些
    """
    try:
        import elkai
        
        n = D.shape[0]
        D_copy = D.copy().astype(float)
        
        # 1. 处理 inf 值 (Big M 法)
        # 找到矩阵中现有的最大非 inf 值，并乘以一个足够大的系数
        finite_mask = np.isfinite(D_copy)
        if finite_mask.any():
            max_val = np.max(D_copy[finite_mask])
        else:
            max_val = 1.0  # 全是 inf 的极端情况
        
        # Big M = max_val * 1000，确保走一条 inf 边的代价超过绕行所有城市的总和
        big_m = max_val * 100
        D_copy[~finite_mask] = big_m
        
        # 2. 缩放并转为整数 (解决浮点数精度问题)
        int_matrix = (D_copy * precision).astype(int).tolist()
        
        # 3. 使用 DistanceMatrix 接口调用 elkai
        dm = elkai.DistanceMatrix(int_matrix)
        
        # runs 设高一点，因为 Big M 会增加解空间的复杂度
        result = dm.solve_tsp(runs=runs)
        
        # 4. 计算 tour 长度（使用原始浮点矩阵，确保精确）
        total = 0.0
        for i in range(len(result)):
            total += D[result[i], result[(i + 1) % len(result)]]
        
        return total, result  # 返回长度和路径
    except ImportError:
        print("elkai 未安装，运行: pip install elkai")
        return None, None
    except Exception as e:
        print(f"错误: {e}")
        return None, None

def method_python_tsp(D):
    """使用 python-tsp 库（多种启发式可选）"""
    try:
        from python_tsp.heuristics import solve_tsp_simulated_annealing
        from python_tsp.heuristics import solve_tsp_local_search
        
        # 处理 inf
        D_safe = D.copy()
        D_safe[~np.isfinite(D_safe)] = 1e10
        
        # 使用 SA + 局部搜索
        permutation, distance = solve_tsp_simulated_annealing(D_safe)
        permutation2, distance2 = solve_tsp_local_search(D_safe, permutation)
        
        return distance2
    except ImportError:
        print("  python-tsp 未安装，运行: pip install python-tsp")
        return None

def check_path_validity(D, path):
    """
    验证路径是否包含不可达的边（inf 值）
    
    Args:
        D: 原始距离矩阵
        path: 路径节点列表
    
    Returns:
        (is_valid, error_edge): 是否有效，如果无效返回故障边
    """
    for i in range(len(path)):
        u, v = path[i], path[(i + 1) % len(path)]
        if not np.isfinite(D[u, v]):
            return False, (u, v)
    return True, None


def save_route_to_file(route, filename):
    """保存路径到文件"""
    with open(filename, "w") as f:
        for node in route:
            f.write(f"{node}\n")
    print(f"  📁 路径已保存至: {filename}")


def main():
    print("=" * 60)
    print("TSP 最优解参考值获取")
    print("=" * 60)
    
    csv_files = ["tour750.csv", "tour1000.csv"]
    
    for filename in csv_files:
        try:
            D = load_distance_matrix(filename)
            n = D.shape[0]
            print(f"\n📊 {filename} (n={n})")
            print("-" * 40)
            
            # # 方法 1: OR-Tools
            # print("  OR-Tools (60s)...", end=" ", flush=True)
            # t0 = time.time()
            # result_ortools = method_ortools(D)
            # if result_ortools:
            #     print(f"✓ {result_ortools:.2f} ({time.time()-t0:.1f}s)")
            # else:
            #     print("✗")
            
            # 方法 2: elkai (LKH) - 使用 DistanceMatrix + Big M 法
            print("  elkai (LKH, runs=10)...", end=" ", flush=True)
            t0 = time.time()
            result_elkai, route_elkai = method_elkai(D, precision=1, runs=10)
            if result_elkai is not None:
                print(f"✓ {result_elkai:.2f} ({time.time()-t0:.1f}s)")
                
                # 验证路径是否有效（不包含 inf 边）
                is_valid, error_edge = check_path_validity(D, route_elkai)
                if is_valid:
                    print("  ✅ 路径验证通过（无不可达边）")
                else:
                    print(f"  ⚠️ 警告：路径包含不可达边！从 {error_edge[0]} 到 {error_edge[1]}")
                
                # 显示路径预览（前10个和后10个节点）
                if len(route_elkai) > 20:
                    preview = route_elkai[:10] + ["..."] + route_elkai[-10:]
                else:
                    preview = route_elkai
                print(f"  🛤️  路径预览: {preview}")
                
                # 保存路径到文件
                base_name = filename.replace(".csv", "")
                route_filename = f"best_route_{base_name}.txt"
                save_route_to_file(route_elkai, route_filename)
            else:
                print("✗")
            
            # 找最好的
            results = [r for r in [result_elkai] if r]
            if results:
                best = min(results)
                print(f"  ➡️  最佳参考值: {best:.2f}")
                
        except FileNotFoundError:
            print(f"\n⚠️ 文件不存在: {filename}")
    
    print("\n" + "=" * 60)
    print("💡 提示：安装更多库以获得更好的参考值：")
    print("   pip install ortools elkai python-tsp")
    print("=" * 60)

if __name__ == "__main__":
    main()
