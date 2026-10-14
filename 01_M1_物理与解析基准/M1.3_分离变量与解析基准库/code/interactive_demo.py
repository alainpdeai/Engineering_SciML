import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analytical_heat_1d(x, t, a=0.5, L=1.0, n_terms=50):
    """1D 热传导方程方波初温分离变量解: u_t = a^2 u_xx, u(0)=u(L)=0"""
    u = np.zeros_like(x)
    for n in range(1, n_terms + 1, 2):  # 仅奇数项非零
        k_n = n * np.pi / L
        # 方波展开系数: b_n = 4 / (n * pi)
        b_n = 4.0 / (n * np.pi)
        decay = np.exp(-(a * k_n)**2 * t)
        u += b_n * np.sin(k_n * x) * decay
    return u

def analytical_wave_1d(x, t, c=1.0, L=1.0, n_terms=30):
    """1D 弦振动初位移拨弦解 (三角波释放): u_tt = c^2 u_xx"""
    u = np.zeros_like(x)
    for n in range(1, n_terms + 1, 2):
        k_n = n * np.pi / L
        # 三角波初位移傅里叶系数: a_n = 8 / (pi^2 * n^2) * (-1)^((n-1)/2)
        sign = (-1)**((n - 1) // 2)
        a_n = (8.0 / (np.pi**2 * n**2)) * sign
        u += a_n * np.sin(k_n * x) * np.cos(c * k_n * t)
    return u

def run_demo():
    L = 1.0
    x = np.linspace(0, L, 200)
    
    plt.figure(figsize=(12, 5))
    
    # 1. 热传导耗散平滑
    plt.subplot(1, 2, 1)
    for t_val in [0.0, 0.005, 0.02, 0.1]:
        u_heat = analytical_heat_1d(x, t=t_val, a=0.5, L=L)
        plt.plot(x, u_heat, label=f"t = {t_val} s", lw=2)
    plt.title("1D 热传导分离变量解: 方波初温衰减平滑")
    plt.xlabel("x")
    plt.ylabel("温度 u(x,t)")
    plt.grid(True, ls='--', alpha=0.6)
    plt.legend()
    
    # 2. 波动方程驻波往复
    plt.subplot(1, 2, 2)
    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        u_wave = analytical_wave_1d(x, t=t_val, c=1.0, L=L)
        plt.plot(x, u_wave, label=f"t = {t_val} s", lw=2)
    plt.title("1D 弦振动三角波解析解: 驻波周期往复")
    plt.xlabel("x")
    plt.ylabel("位移 u(x,t)")
    plt.grid(True, ls='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    out_img = os.path.join(os.path.dirname(__file__), "analytical_demo.png")
    plt.savefig(out_img, dpi=150)
    print(f"Demo figure saved to {out_img}")
    plt.close()

if __name__ == "__main__":
    run_demo()
