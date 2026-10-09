# M1.1 基准库 · `control_volume/`

控制体守恒推导的**验证三件套**（量纲 → 解析残差 → 守恒账本）中的前两件。
运行任一脚本都会把结果合并写入本目录 `results.json`（v3 方案 §7 的数字单一来源）。

## 脚本

| 脚本 | 内容 | 通过判据 |
| :-- | :-- | :-- |
| `dimension_check.py` | ① 量纲检查：α=k/(ρc_p) 应为 [L²/T]，c=√(T/μ) 应为 [L/T] | 量纲指数逐一相等 |
| `analytical_residual.py` | ② 解析残差：热核、达朗贝尔行波代入算子残差为零；三大方程判别式分类 | 残差严格为 0 |

## 用法

```bash
python benchmarks/control_volume/dimension_check.py
python benchmarks/control_volume/analytical_residual.py
```

## 与其他交付物的关系

- 第③件（守恒账本离散检查）由 `../../tests/test_conservation.py` 以 pytest 断言承担；
- 实验记录与 POE 表见 `../../notebooks/M1.1_lab.ipynb`；
- 失效卡（F4）归档在 `../../diagnostics/M1.1_F4_cards.md`。
