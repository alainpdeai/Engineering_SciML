# -*- coding: utf-8 -*-
"""
Unit M1.6: 柱/球坐标、特殊函数与声辐射
求解受日照圆柱二维横截面稳态温度场与均质球体瞬态径向降温曲线
"""
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def solve_cylinder_sunlight(R=1.0, q0=100.0, k=20.0, h=10.0, T_inf=20.0):
    # 解析级数求解受单侧日照圆柱截面温度场
    # 边界条件: -k ∂u/∂r|_{r=R} = h(u - T_inf) - q0*cos(theta)
    # 通解: u(r, theta) = A0 + A1 * (r/R) * cos(theta)
    A0 = T_inf
    A1 = (q0 * R) / (k + h * R)
    return A0, A1

def solve_sphere_transient(R=1.0, alpha=1e-4, T0=100.0, T_env=0.0, t_list=[5, 20, 60, 150]):
    """球对称瞬态导热的正弦级数解（v = r·u 化简后 v_t = alpha·v_rr）。

    系数推导：u(r,0) = T0 ⇒ v(r,0) = r·(T0 - T_env)。在 sin(n*pi*r/R) 上展开，
        b_n = (2/R)·∫_0^R r·(T0-T_env)·sin(n*pi*r/R) dr
            = 2·R·(T0 - T_env)·(-1)^(n+1) / (n*pi)
    于是 u(r,t) = T_env + Σ_n b_n·sin(n*pi*r/R)/r·exp(-alpha·(n*pi/R)^2·t)。

    注意 b_n 里已经带了一个 1/r 的量纲，**不能再多除一个 n·pi**。
    （tests/test_cylindrical_spherical.py::v_coefficients 用的是同一套系数，
      可作交叉校验：t→0 时本函数应收敛回 T0。）
    """
    r = np.linspace(1e-4, R, 200)
    results = {}
    for t in t_list:
        u = np.zeros_like(r)
        for n in range(1, 40):
            lam_n = (n * np.pi / R)**2
            # b_n / R，把 R 吸收进 coeff 后再乘回 R/r，即得 b_n·sin/r
            coeff = 2.0 * (T0 - T_env) * (-1)**(n + 1) / (n * np.pi)
            term = coeff * (R / r) * np.sin(n * np.pi * r / R) * np.exp(-alpha * lam_n * t)
            u += term
        results[t] = u + T_env
    return r, results

def main():
    print("==================================================")
    print(" Unit M1.6: 柱/球坐标、特殊函数与声辐射解析求解")
    print("==================================================")

    # 1. 柱坐标受日照截面温度场
    R = 1.0
    A0, A1 = solve_cylinder_sunlight(R=R, q0=200.0, k=15.0, h=10.0, T_inf=25.0)
    print(f"圆柱稳态解析参数: 基准环境温度 A0 = {A0:.2f}°C, 日照偏心一阶幅值 A1 = {A1:.2f}°C")
    # 可检查数字①：稳态边界应回到对流-热流条件
    #   边界：-k·∂u/∂r|_{r=R,θ=0} = h·(u(R,0) - T_inf) - q0
    #   u = A0 + A1·(r/R)·cosθ  ⇒  ∂u/∂r|_{R,0} = A1/R
    T_inf = 25.0
    u_sun = A0 + A1                       # 受日照侧（θ=0）表面温度
    flux_analytic = -15.0 * A1 / R        # -k·∂u/∂r
    flux_expected = 10.0 * (u_sun - T_inf) - 200.0
    print(f"  对流-热流边界校核: -k·u_r|R = {flux_analytic:.2f},  "
          f"h(u-T_inf) - q0 = {flux_expected:.2f}, 残差 = {abs(flux_analytic - flux_expected):.2e}")

    # 极坐标网格
    r = np.linspace(0, R, 80)
    theta = np.linspace(0, 2*np.pi, 120)
    R_grid, Theta_grid = np.meshgrid(r, theta)
    T_cylinder = A0 + A1 * (R_grid / R) * np.cos(Theta_grid)
    X = R_grid * np.cos(Theta_grid)
    Y = R_grid * np.sin(Theta_grid)

    # 2. 球坐标瞬态径向扩散
    r_sphere, sphere_res = solve_sphere_transient(R=0.05, alpha=1.2e-5, T0=200.0, T_env=20.0)
    # 可检查数字②：t=0 必须回到初值 T0（这是球级数系数的回归门禁）
    r_chk, res_chk = solve_sphere_transient(R=0.05, alpha=1.2e-5, T0=200.0, T_env=20.0, t_list=[0.0])
    u_chk = res_chk[0.0]
    mid = len(u_chk) // 2
    print(f"  球体瞬态 t→0 回归: u(r_mid={r_chk[mid]:.3f}m) = {u_chk[mid]:.3f}°C, "
          f"初值 T0 = 200.0, 相对误差 = {abs(u_chk[mid] - 200.0) / 200.0 * 100:.3f}%  (39 阶截断，n=1..39)")
    # 可检查数字③：扩散时标 R^2/alpha，决定时间轴取到哪
    print(f"  扩散时标 R^2/alpha = {0.05 ** 2 / 1.2e-5:.1f} s；"
          f"中心温度 {['%.1f' % sphere_res[t][0] for t in (5, 20, 60, 150)]} °C")

    # 3. 组合绘图
    fig = plt.figure(figsize=(13, 5))
    
    # 柱截面温度场云图
    ax1 = fig.add_subplot(1, 2, 1)
    cf = ax1.contourf(X, Y, T_cylinder, 30, cmap='plasma')
    ax1.set_title("圆柱横截面受单侧日照稳态温度场 (°C)")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_aspect('equal')
    fig.colorbar(cf, ax=ax1)

    # 球体径向降温曲线
    ax2 = fig.add_subplot(1, 2, 2)
    for t, u_prof in sphere_res.items():
        ax2.plot(r_sphere * 1000, u_prof, label=f"t = {t} s", lw=2)
    ax2.set_title("均质金属球体瞬态冷却径向温度分布")
    ax2.set_xlabel("球半径 r (mm)")
    ax2.set_ylabel("温度 T (°C)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    os.makedirs(output_dir, exist_ok=True)
    out_fig = os.path.join(output_dir, "coord_systems.png")
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"可视化图像已保存: {out_fig}")
    print("==================================================\n")

if __name__ == '__main__':
    main()
