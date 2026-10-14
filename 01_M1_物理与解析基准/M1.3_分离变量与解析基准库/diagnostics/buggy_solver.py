# -*- coding: utf-8 -*-
"""
M1.3 diagnostics: 分离变量级数的两类病态（截断阶数 / 系数对称性）

配套讲义 ⑪ 页「现场诊断挑战：把 F4 截断与离散误差彻底分开」，
与 benchmarks/heat_1d_square_wave.py 同口径。

问题设定（与讲义 ⑧ 页、benchmarks/ 完全一致）:
    PDE:  u_t = alpha * u_xx,  x in (0, L), t > 0
    BC:   u(0, t) = 0,  u(L, t) = 0
    IC:   u(x, 0) = +1 (0 < x < L/2),  -1 (L/2 < x < L)

正确展开: 目标方波关于中点反对称，由半周期正交性，系数仅在 n ≡ 2 (mod 4) 非零
          b_n = 8 / (n * pi)
          —— 这正是讲义 ⑪ 页诊断题 1「标准答案 C」所考察的知识点。

两类病态:
  病态 1 · 截断阶数 N 过小 -> 跳变处 Gibbs 振铃，振铃宽度随 N 缩小但峰值不消失
  病态 2 · 系数对称性取错 -> 误用全区间常数展开 b_n = 4/(n*pi)（仅奇数项），
          右半域整体反号，x > L/2 处误差恒为 2.0（⑪ 页诊断题 1 的反面教材）

自检要点: 所有误差一律对照 **解析真值 truth** 计算，而不是对照另一个级数解。
          若拿级数解当参考（无论它对不对），自检都会全绿 —— 那样的自检没有意义。
"""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

L_DEFAULT = 1.0
ALPHA_DEFAULT = 1.0


def exact_square_wave(x, L=L_DEFAULT):
    """解析真值: t=0 时刻的 ±1 方波初温（0 < x < L/2 取 +1，L/2 < x < L 取 -1）"""
    x = np.asarray(x, dtype=float)
    return np.where(x < L / 2, 1.0, -1.0)


def correct_series(x, t=0.0, L=L_DEFAULT, alpha=ALPHA_DEFAULT, n_terms=51):
    """正确解: 仅 n ≡ 2 (mod 4) 非零，b_n = 8/(n*pi)"""
    x = np.asarray(x, dtype=float)
    u = np.zeros_like(x)
    for n in range(1, n_terms + 1):
        if n % 4 != 2:
            continue
        k_n = n * np.pi / L
        b_n = 8.0 / (n * np.pi)
        u += b_n * np.sin(k_n * x) * np.exp(-alpha * k_n ** 2 * t)
    return u


def buggy_wrong_parity(x, t=0.0, L=L_DEFAULT, alpha=ALPHA_DEFAULT, n_terms=51):
    """
    BUGGY：误用全区间 (0, L) 常数展开 b_n = 4/(n*pi)，仅奇数项非零。

    这是讲义 ⑪ 页诊断题 1 的标准答案 C 所指认的错误：
    该展开对应的是常初值 f ≡ 1，而不是关于中点反对称的 ±1 方波，
    因此右半域 (x > L/2) 整体反号，误差恒为 2.0，且 N 再大也不会收敛。
    """
    x = np.asarray(x, dtype=float)
    u = np.zeros_like(x)
    for n in range(1, n_terms + 1, 2):
        k_n = n * np.pi / L
        b_n = 4.0 / (n * np.pi)
        u += b_n * np.sin(k_n * x) * np.exp(-alpha * k_n ** 2 * t)
    return u


def gibbs_overshoot(x, t=0.0, L=L_DEFAULT, alpha=ALPHA_DEFAULT, n_terms=51):
    """
    跳变点左侧的最大超调（绝对值，单位与初值幅值相同）。

    注意取样范围：超调峰位于跳变点左侧约半个最高模波长处，即距跳变点 ~L/n_terms。
    若用固定窄带（如 ±0.01）取样，N 较小时峰会落在带外，得到偏小甚至为负的值。
    因此这里直接取整个左半区间的最大值。
    """
    x = np.asarray(x, dtype=float)
    u = correct_series(x, t, L, alpha, n_terms)
    left = x < L / 2
    return float(np.max(u[left]) - 1.0)


def _l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def selfcheck():
    """
    自检: 全部对照解析真值 truth，并断言「正确 < 病态」这个次序成立。
    返回 (通过?, 说明行列表)
    """
    x = np.linspace(0.0, L_DEFAULT, 2001)
    truth = exact_square_wave(x)
    # 评价区间：排除端点与跳变邻域（那里 Dirichlet 边界与初值天然不相容）
    away = (np.abs(x - 0.5) > 0.03) & (x > 0.03) & (x < 0.97)

    rows = []
    correct = correct_series(x, 0.0, n_terms=51)
    wrong = buggy_wrong_parity(x, 0.0, n_terms=51)
    few = correct_series(x, 0.0, n_terms=3)

    e_correct = np.max(np.abs(correct[away] - truth[away]))
    e_wrong = np.max(np.abs(wrong[away] - truth[away]))
    e_few = np.max(np.abs(few[away] - truth[away]))
    l2_correct = _l2(correct[away], truth[away])
    l2_wrong = _l2(wrong[away], truth[away])
    l2_few = _l2(few[away], truth[away])

    rows.append('[正确 ] n ≡ 2 mod 4, b=8/(n*pi), N=51 -> Linf %.4f | L2 %.4f'
                % (e_correct, l2_correct))
    rows.append('[病态2] 误用 b=4/(n*pi) 仅奇数项, N=51 -> Linf %.4f | L2 %.4f'
                % (e_wrong, l2_wrong))
    rows.append('[病态1] 截断太少 N=3                      -> Linf %.4f | L2 %.4f'
                % (e_few, l2_few))

    checks = [
        ('正确解误差 < 1e-1', e_correct < 1e-1),
        ('病态2 误差 > 正确解一个数量级', e_wrong > 10 * e_correct),
        ('病态2 在 x>0.5 处反号（误差≈2）',
         bool(np.all(wrong[x > 0.6] > 0) and np.all(truth[x > 0.6] < 0))),
        ('截断太少误差 > 正确解', e_few > e_correct),
    ]
    ok = all(c[1] for c in checks)
    return ok, rows, checks, (x, truth, correct, wrong, few,
                              e_correct, e_wrong, e_few)


def diagnose():
    print("=" * 72)
    print(" M1.3 diagnostics: 分离变量级数病态诊断（截断阶数 / 系数对称性）")
    print("=" * 72)

    ok, rows, checks, pack = selfcheck()
    x, truth, correct, wrong, few, e_correct, e_wrong, e_few = pack
    for r in rows:
        print("  " + r)
    print()
    print("  Gibbs 绝对超调（跳跃幅 = 2）: N=51 -> %.4f | N=201 -> %.4f"
          % (gibbs_overshoot(np.linspace(0, 1, 20001), n_terms=51),
             gibbs_overshoot(np.linspace(0, 1, 20001), n_terms=201)))
    print("  理论极限 0.0895 x 2 = 0.1790 —— 振铃宽度随 N 缩小，峰值不随 N 消失。")
    print()
    print("  自检（全部对照解析真值）:")
    for name, passed in checks:
        print("    [%s] %s" % ("通过" if passed else "失败", name))
    print("  自检结论: %s" % ("通过" if ok else "★ 失败"))
    print("=" * 72)

    # 绘图：三种解并排对比
    plt.figure(figsize=(12, 4))
    for ax, (u, title) in zip(
            (plt.subplot(1, 3, 1), plt.subplot(1, 3, 2), plt.subplot(1, 3, 3)),
            ((few, "病态1: 截断阶数 N=3 过小"),
             (wrong, "病态2: 误用 b=4/(nπ)（第 11 页答案 C）"),
             (correct, "正确: n≡2 mod 4, b=8/(nπ)"))):
        ax.plot(x, truth, 'k-', lw=1.5, label='解析真值 ±1 方波')
        ax.plot(x, u, 'r--', lw=1.5, label='级数解')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    os.makedirs(out_dir, exist_ok=True)
    out_fig = os.path.join(out_dir, "truncation_pathology.png")
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"诊断对比图已保存: {out_fig}\n")
    return ok


if __name__ == '__main__':
    diagnose()
