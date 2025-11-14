# 氦原子极化率数值复现项目

本仓库旨在按照《氢原子体系能级与极化率的高精度计算》论文的理论与数值流程，严格复现实验。

## 环境依赖

- Python 3.11（推荐）
- NumPy
- SciPy
- Matplotlib

后续阶段将根据具体实现引入其它必要依赖。

## 项目结构概览

```
├── README.md
├── pyproject.toml
├── src
│   └── he_polarization
│       ├── __init__.py
│       ├── basis
│       │   ├── __init__.py
│       │   ├── nodes.py
│       │   ├── bspline.py
│       │   └── angular.py
│       ├── hamiltonian
│       │   ├── __init__.py
│       │   ├── operators.py
│       │   └── elements.py
│       ├── solver
│       │   ├── __init__.py
│       │   └── generalized_eigen.py
│       ├── observables
│       │   ├── __init__.py
│       │   ├── energies.py
│       │   ├── polarizability_static.py
│       │   └── polarizability_dynamic.py
│       ├── validation
│       │   ├── __init__.py
│       │   ├── hellmann.py
│       │   └── convergence.py
│       └── reporting
│           ├── __init__.py
│           └── plotting.py
└── scripts
    ├── run_energy_pipeline.py
    ├── run_polarizability_pipeline.py
    └── plot_dynamic_polarizability.py
```

> **说明**：当前阶段仅提供骨架，所有核心函数会在后续步骤中逐一按照论文各章节实现。

1. **基函数模块**：实现指数型节点、B 样条递推、导数、角动量耦合张量，与论文第 2 章公式对齐。
2. **哈密顿量与矩阵元**：根据论文推导完成广义本征方程涉及的全部矩阵元素计算。
3. **广义本征求解与外推**：重现变分能量计算、收敛分析、等差外推策略。
4. **极化率计算**：实现长度、速度规范下的静态极化率与误差评估；完成频率扫描与图形输出。
5. **验证**：完成 Hellmann 判据、数值对比、以及与文中数据一致的表格与图像导出。

## 使用说明

待所有模块填充完成后，可通过 `scripts/` 中的脚本串联能级与极化率的计算流程。
