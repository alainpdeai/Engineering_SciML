"""
M2.1 故障注入诊断代码 (供学生排查定位)
故障现象（实测，2026-10-16 复跑）：进程退出码 0，stdout 打印「求解成功: [nan nan ...]」，
只有 stderr 出现 MatrixRankWarning: Matrix is exactly singular——看退出码或前几行会误判成功。
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def buggy_poisson_solver(N=31, L=1.0):
    dx = L / (N - 1)
    total_nodes = N * N
    
    # 注入故障: 构造了刚度矩阵，但忘记在边界行对角元置 1，导致矩阵第一行全零而产生奇异！
    A = sp.lil_matrix((total_nodes, total_nodes))
    b = np.ones(total_nodes)
    
    for idx in range(total_nodes):
        A[idx, idx] = 4.0
        
    # 错误: 划行后未置主对角线为 1
    A[0, :] = 0.0  # 致命 Bug: 主对角元素也变成了 0!
    b[0] = 0.0
    
    print("尝试求解有缺陷的奇异矩阵...")
    try:
        u = spla.spsolve(A.tocsr(), b)
        print("求解成功:", u[:5])
    except Exception as e:
        print(f"[捕获预期崩溃] 错误类型: {type(e).__name__} | 错误详情: {e}")

if __name__ == "__main__":
    buggy_poisson_solver()
