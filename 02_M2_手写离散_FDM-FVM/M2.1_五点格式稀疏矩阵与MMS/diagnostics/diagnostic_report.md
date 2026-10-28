# M2.1 故障诊断与归因分析报告

## 1. 现象描述
运行 `python diagnostics/buggy_solver.py`（2026-10-16 实测）：**进程退出码为 0**，
stdout 打印「尝试求解有缺陷的奇异矩阵... / 求解成功: [nan nan nan nan nan]」，
只有 stderr 出现 `MatrixRankWarning: Matrix is exactly singular`。
不抛异常、不非零退出——只看退出码或前几行输出会误判为求解成功。

## 2. 根因剖析（Root Cause Analysis）
在施加 Dirichlet 边界条件时，代码将边界行所有的非零元清零：
```python
A[boundary_idx, :] = 0.0
```
但**遗漏了恢复主对角元**：
```python
A[boundary_idx, boundary_idx] = 1.0
```
这导致矩阵 $A$ 的该行全为 0，行列式 $\det(A) = 0$，矩阵退化为不可逆的奇异矩阵。

## 3. 修复方案
在清零边界行之后，务必紧跟对角元置 1；**并且必须同步清理边界列**（只划行不清列会让 $A\neq A^T$，
破坏 CG 的对称正定前提，见 F4 卡二）：
```python
A[boundary_idx, :] = 0.0
A[:, boundary_idx] = 0.0
A[boundary_idx, boundary_idx] = 1.0
b[boundary_idx] = g_boundary_value
```

## 4. 复现命令
```bash
python diagnostics/buggy_solver.py   # 退出码 0 + stdout 假成功 + stderr 告警
```
