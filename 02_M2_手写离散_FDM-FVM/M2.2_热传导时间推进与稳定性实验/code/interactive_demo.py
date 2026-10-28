# -*- coding: utf-8 -*-
"""
M2.2: 显式格式扩散数越界的数值爆炸现场（与隐式无条件稳定对照）

术语口径：抛物型热方程用「扩散数 r = α·Δt/Δx²」，不用 CFL（源稿① 术语口径 box）。
1D 显式中心差分的稳定边界是 r ≤ 1/2。

为什么步数必须给足：初值 u₀=sin(πx) 是纯最低模态，|G(π)| 的增长只能从
1e-16 量级的舍入噪声起步 —— r=0.6 也要约 130 步才首次越界、250 步才到 1e+19。
前 15 步曲线光滑得看不出任何问题（F4 卡故障一的「陷阱段」），所以本演示
故意把「陷阱段」和「爆炸段」画进同一张图：前 15 步仍平滑，300 步后发散。
"""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def simulate_ftcs_heat(r=0.4, n_steps=300):
    """显式 FTCS 模拟一维导热（零 Dirichlet 边界）。

    r 为扩散数 α·Δt/Δx²；n_steps 默认 300 —— 少于 ~130 步时即便 r=0.6
    也看不到爆炸（增长从 1e-16 舍入噪声起步），会重现「只截未爆炸窗口」的误判。
    """
    nx = 51
    x = np.linspace(0, 1.0, nx)
    u = np.sin(np.pi * x)  # 初始平滑温度场
    history = [u.copy()]

    for _ in range(n_steps):
        u_next = u.copy()
        # 内部节点显式递推
        u_next[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        # 边界绝热或恒温
        u_next[0] = 0.0
        u_next[-1] = 0.0
        u = u_next
        history.append(u.copy())

    return x, history


def main():
    # 对比 r = 0.4 (安全) 与 r = 0.6 (越界)；300 步让爆炸真正发生
    x, hist_safe = simulate_ftcs_heat(r=0.4, n_steps=300)
    _, hist_blow = simulate_ftcs_heat(r=0.6, n_steps=300)

    # 首次越界步数（max|u| > 10）：前 ~120 步曲线仍「看起来正常」
    over = next((i for i, u in enumerate(hist_blow) if np.max(np.abs(u)) > 10.0), None)
    peak = float(np.max(np.abs(hist_blow[-1])))

    plt.figure(figsize=(12, 5))

    # 1. 稳定模拟
    plt.subplot(1, 2, 1)
    for step in [0, 5, 15, 300]:
        plt.plot(x, hist_safe[step], label=f"Step {step}")
    plt.title("安全扩散数: r = 0.4 ≤ 1/2（平滑耗散）")
    plt.xlabel("空间坐标 x")
    plt.ylabel("温度 u")
    plt.grid(True, ls='--', alpha=0.6)
    plt.legend()

    # 2. 数值爆炸（陷阱段与爆炸段同图：前 15 步与左栏几乎重合）
    plt.subplot(1, 2, 2)
    for step in [0, 5, 15, 300]:
        plt.plot(x, hist_blow[step], label=f"Step {step}")
    plt.yscale('symlog', linthresh=1.0)
    # 手动制定无负值刻度：避开 symlog 自动生成的负刻度（SimHei 缺 U+2212 字形）
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 10, 1e2, 1e4, 1e8, 1e12, 1e16, 1e20, 1e24])
    plt.title(f"越界爆炸: r = 0.6 > 1/2（前 15 步仍光滑！）\n"
              f"第 {over} 步越界，300 步峰值 {peak:.1e}）")
    plt.xlabel("空间坐标 x")
    plt.ylabel("温度 u（symlog）")
    plt.grid(True, ls='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    os.makedirs(out_dir, exist_ok=True)
    out_img = os.path.join(out_dir, "cfl_explosion_demo.png")
    plt.savefig(out_img, dpi=150)
    print(f"扩散数越界对比结果已保存至: {out_img}")
    print(f"爆炸侧 r=0.6：第 {over} 步首次越界(max|u|>10)，300 步峰值 {peak:.3e}")
    print("陷阱段提示：前 15 步两条曲线几乎重合 —— 「看起来正常」不代表稳定")
    plt.close()


if __name__ == "__main__":
    main()
