"""
M2.1: 二维泊松/稳态热传导方程标准五点差分稀疏求解器 (NumPy/SciPy 实现)
"""
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def assemble_poisson_system(N=51, L=1.0, source_fn=None):
    """组装齐次 Dirichlet Poisson 系统，返回坐标、对称 CSR 矩阵和右端项。

    边界自由度保留为单位行/列；同步清理行与列可保持矩阵对称正定，
    使 CG 等要求对称正定的求解器满足其数学前提。

    与源稿④的 kron 内点版对照：同一物理问题、同一缩放下，两者的内点块逐元素一致
    （n=9 实测 allclose=True, max|差|=0.0, nnz 369=369）；差异只在边界行的存在形式
    （内点版隐去 / 本函数显式保留单位行）与本函数把 h^2 乘到右端（源稿④除进矩阵）。
    非零元口径：内点版 5N^2；本函数全节点版 N=41 实测 7609 ≈ 4.53N^2。
    """
    if N < 3:
        raise ValueError("N must be at least 3 to contain interior nodes")
    if L <= 0:
        raise ValueError("L must be positive")

    dx = L / (N - 1)
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)

    if source_fn is None:
        # 默认制造解在任意正方形 [0, L]^2 上满足齐次 Dirichlet：
        # u*=sin(pi*x/L)sin(pi*y/L)，-Δu*=2(pi/L)^2 u*。
        source_fn = lambda x, y: (2 * np.pi**2 / L**2) * np.sin(np.pi * x / L) * np.sin(np.pi * y / L)

    f_val = source_fn(X, Y)
    b = (dx**2) * f_val.flatten()

    # 构建块三对角稀疏矩阵 A（每行最多 5 个非零：主对角 + 偏移 ±1、±N 四条邻边）
    total_nodes = N * N
    main_diag = 4.0 * np.ones(total_nodes)
    off_diag_x = -1.0 * np.ones(total_nodes - 1)
    off_diag_y = -1.0 * np.ones(total_nodes - N)

    A = sp.diags(
        [main_diag, off_diag_x, off_diag_x, off_diag_y, off_diag_y],
        [0, 1, -1, N, -N], shape=(total_nodes, total_nodes), format='lil'
    )

    boundary = [
        i * N + j
        for i in range(N)
        for j in range(N)
        if i == 0 or i == N - 1 or j == 0 or j == N - 1
    ]
    # 齐次 Dirichlet 边界：清理对应行和列，再置单位对角。
    # 仅清理行会使 A 非对称，破坏 CG 的 SPD 前提。
    for idx in boundary:
        A[idx, :] = 0.0
        A[:, idx] = 0.0
        A[idx, idx] = 1.0
        b[idx] = 0.0

    return X, Y, A.tocsr(), b


def solve_poisson_2d(N=51, L=1.0, source_fn=None):
    """
    在 [0, L] x [0, L] 矩形域求解:
        -d^2u/dx^2 - d^2u/dy^2 = f(x, y)
    四周为齐次 Dirichlet 边界: u = 0
    """
    X, Y, A, b = assemble_poisson_system(N=N, L=L, source_fn=source_fn)
    u_vec = spla.spsolve(A, b)
    return X, Y, u_vec.reshape(X.shape)

def _fig_poisson_solution(out_dir, N=61, L=1.0):
    """数值解云图：main() 与 --figs 共用同一代码路径，保证 assets/ 与 code/ 产物同源可复现。"""
    X, Y, u = solve_poisson_2d(N=N, L=L)
    u_exact = np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
    h = L / (N - 1)
    l2_err = float(np.sqrt(h * h * np.sum((u - u_exact) ** 2)))
    print(f"[M2.1 求解成功] 网格节点: {N}x{N}, 空间步长 h={h:.4f}, L2 误差: {l2_err:.6e}")
    plt.figure(figsize=(7, 6))
    cp = plt.contourf(X, Y, u, 35, cmap='coolwarm')
    plt.colorbar(cp, label='场量数值解 u(x, y)')
    plt.title(f"二维泊松方程五点差分数值解 (网格 {N}x{N})")
    plt.xlabel("x")
    plt.ylabel("y")
    out_path = os.path.join(out_dir, "poisson_solution.png")
    plt.savefig(out_path, dpi=150)
    print(f"云图结果已保存至: {out_path}")
    plt.close()


def main():
    _fig_poisson_solution(os.path.dirname(os.path.abspath(__file__)))
def run_benchmarks(out_dir=None):
    """--all: 生成 benchmarks/poisson_mms/ 的 MMS 收敛证据（源稿⑥最小实验 + ⑦验证三件套）。

    误差口径：**真 L2 范数** sqrt(h^2 * sum(e^2))，与源稿⑤现场运行代码同一公式，
    覆盖含边界的全部 N^2 个节点（齐次 Dirichlet 下边界节点误差恒为 0，不影响阶数）。
    同时附列 L^inf 与代数残差 ||Au-b||_2——后者正是「残差小 ≠ 阶对」诊断页的取证列。
    节点 RMS 口径 sqrt(mean(e^2)) 与真 L2 的比例为 RMS/L2=(N-1)/N，
    即 L2/RMS=N/(N-1)=1+1/(N-1)；阶数落在
    1.967/1.983/1.991；真 L2 口径为 2.001/2.000/2.000。两种口径结论同为二阶，差异已入档。
    """
    import csv
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "benchmarks", "poisson_mms")
    os.makedirs(out_dir, exist_ok=True)
    rows, prev, prev_h = [], None, None
    for N in (21, 41, 81, 161):
        X, Y, u = solve_poisson_2d(N=N)
        u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
        err = (u - u_exact).ravel()
        h = 1.0 / (N - 1)
        l2 = float(np.sqrt(h * h * np.sum(err ** 2)))
        linf = float(np.max(np.abs(err)))
        _, _, A, b = assemble_poisson_system(N=N)
        residual = float(np.linalg.norm(A @ u.ravel() - b))
        order = "" if prev is None else f"{np.log(prev / l2) / np.log(prev_h / h):.3f}"
        rows.append({"N": N, "h": f"{h:.6e}", "l2_error": f"{l2:.6e}",
                     "linf_error": f"{linf:.6e}", "residual": f"{residual:.3e}",
                     "order": order})
        prev, prev_h = l2, h
        print(f"[MMS ] N={N:4d}  h={h:.4e}  L2={l2:.6e}  Linf={linf:.6e}  residual={residual:.3e}  order={order or '-'}")
    with open(os.path.join(out_dir, "mms_convergence.csv"), "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["N", "h", "l2_error", "linf_error", "residual", "order"])
        w.writeheader(); w.writerows(rows)
    print(f"[M2.1] benchmarks written: {out_dir}")


def _fig_five_point_stencil(out_dir, n=7):
    """五点差分模板示意图：中心 4/h^2，四邻居 -1/h^2（对应源稿 ③）。"""
    fig, ax = plt.subplots(figsize=(7.6, 6.9))
    for k in range(n):
        ax.plot([0, n - 1], [k, k], color="#cbd5e1", lw=0.8, zorder=1)
        ax.plot([k, k], [0, n - 1], color="#cbd5e1", lw=0.8, zorder=1)
    for i in range(n):
        for j in range(n):
            ax.plot(j, i, "o", ms=4.5, color="#94a3b8", zorder=3)
    ci = cj = n // 2
    for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        ax.plot([cj, cj + dj], [ci, ci + di], color="#1e3a8a", lw=3.2,
                zorder=2, solid_capstyle="round")
    ax.plot(cj, ci, "s", ms=25, color="#b91c1c", zorder=4)
    ax.text(cj, ci, r"$4/h^2$", color="white", ha="center", va="center",
            fontsize=10, zorder=5)
    for di, dj, ha, va in [(1, 0, "center", "bottom"), (-1, 0, "center", "top"),
                           (0, 1, "left", "center"), (0, -1, "right", "center")]:
        ax.plot(cj + dj, ci + di, "s", ms=14, color="#1e3a8a", zorder=4)
        ax.annotate(r"$-1/h^2$", xy=(cj + dj, ci + di),
                    xytext=(cj + dj * 1.75, ci + di * 1.75),
                    ha=ha, va=va, fontsize=12.5, color="#1e3a8a",
                    arrowprops=dict(arrowstyle="-", color="#1e3a8a", lw=0.9))
    ax.annotate("", xy=(0.0, -0.62), xytext=(1.0, -0.62),
                arrowprops=dict(arrowstyle="<->", color="#0f766e", lw=1.5))
    ax.text(0.5, -0.95, r"$h$", color="#0f766e", ha="center", va="top", fontsize=13)
    ax.text(1.7, -0.62, "网格步长", color="#0f766e", ha="left", va="center",
            fontsize=11)
    ax.text((n - 1) / 2, -1.75,
            "节点 $(i,j)$ 只与上、下、左、右四个邻居耦合\n"
            r"离散方程：$4u_{ij}-u_{i+1,j}-u_{i-1,j}-u_{i,j+1}-u_{i,j-1}=h^2 f_{ij}$",
            ha="center", va="top", fontsize=11.5, color="#334155")
    ax.set_title(r"五点差分模板：中心 $4/h^2$，四邻居 $-1/h^2$", fontsize=15)
    ax.set_xlim(-2.6, n - 1 + 2.6)
    ax.set_ylim(-3.6, n - 1 + 0.7)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    out = os.path.join(out_dir, "five_point_stencil.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def _fig_sparsity_pattern(out_dir, n=9):
    """稀疏矩阵结构与行主序映射：内部五点格式的块三对角结构（5 条对角带，对应源稿 ④）。"""
    N = n * n
    T = np.diag(2.0 * np.ones(n)) + np.diag(-1.0 * np.ones(n - 1), 1) \
        + np.diag(-1.0 * np.ones(n - 1), -1)
    A = np.kron(np.eye(n), T) + np.kron(T, np.eye(n))
    nnz = int(np.count_nonzero(A))
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.4),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax = axes[0]
    m = 5
    for k in range(m):
        ax.plot([0, m - 1], [k, k], color="#e2e8f0", lw=0.9, zorder=1)
        ax.plot([k, k], [0, m - 1], color="#e2e8f0", lw=0.9, zorder=1)
    for i in range(m):
        for j in range(m):
            p = i * m + j
            if (i, j) == (2, 2):
                ax.plot(j, i, "o", ms=26, color="#b91c1c", zorder=3)
                ax.text(j, i, f"{p}", color="white", ha="center", va="center",
                        fontsize=11, zorder=4)
            elif (i, j) in [(1, 2), (3, 2), (2, 1), (2, 3)]:
                ax.plot(j, i, "o", ms=19, color="#1e3a8a", zorder=3)
                ax.text(j, i, f"{p}", color="white", ha="center", va="center",
                        fontsize=9.5, zorder=4)
            else:
                ax.text(j, i, f"{p}", ha="center", va="center", fontsize=9.5,
                        color="#475569")
    for (di, dj), lab, tx, ty in [((1, 0), r"$\Delta p=+n$", 2, 5.6),
                                  ((-1, 0), r"$\Delta p=-n$", 2, -1.6),
                                  ((0, 1), r"$\Delta p=+1$", 5.9, 2),
                                  ((0, -1), r"$\Delta p=-1$", -1.9, 2)]:
        ax.annotate(lab, xy=(2 + dj, 2 + di), xytext=(tx, ty),
                    ha="center", va="center", fontsize=11, color="#1e3a8a",
                    arrowprops=dict(arrowstyle="-", color="#1e3a8a", lw=0.8))
    ax.set_title(r"行主序展开：$p=i\,n+j$（以 $n=5$ 示意）", fontsize=14)
    ax.text(2, -2.9,
            r"同一行内左右邻居 $\Delta p=\pm 1$，上下邻居 $\Delta p=\pm n$；" "\n"
            "内点网格的边界行只保留存在的邻居；整体为块三对角——"
            "5 条对角带（偏移 0、±1、±n），每行最多 5 个非零",
            ha="center", va="top", fontsize=11.5, color="#334155")
    ax.set_xlim(-3.8, m - 1 + 3.8)
    ax.set_ylim(-4.2, m - 1 + 2.8)
    ax.set_aspect("equal")
    ax.axis("off")

    ax = axes[1]
    ax.spy(A, markersize=4.0, color="#1e3a8a", marker="s")
    ax.set_title("稀疏结构：块三对角（5 条对角带，$n=%d$，非零 %d 个）" % (n, nnz),
                 fontsize=14)
    ax.set_xlabel("列索引 $p$")
    ax.set_ylabel("行索引 $p$")
    ax.text(0.97, 0.97,
            "对角带偏移量：\n"
            r"$\Delta p=0$  主对角（自身）" "\n"
            r"$\Delta p=\pm 1$  左/右邻居" "\n"
            r"$\Delta p=\pm n$  上/下邻居" "\n"
            "非零元占比 %.1f%%" % (100.0 * nnz / (N * N)),
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#eff6ff",
                      edgecolor="#93c5fd", lw=1.0))
    fig.tight_layout()
    out = os.path.join(out_dir, "sparse_pattern.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def make_teaching_figures(out_dir=None):
    """--figs: 生成教学配图（五点模板、稀疏矩阵结构与行主序映射）。"""
    if out_dir is None:
        out_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    os.makedirs(out_dir, exist_ok=True)
    _fig_five_point_stencil(out_dir)
    _fig_sparsity_pattern(out_dir)
    _fig_poisson_solution(out_dir)
    print(f"[M2.1] 教学配图已写入: {out_dir}")


if __name__ == "__main__":
    import sys
    if "--all" in sys.argv:
        run_benchmarks()
    elif "--figs" in sys.argv:
        make_teaching_figures()
    else:
        main()
