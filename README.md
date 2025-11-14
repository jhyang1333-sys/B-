# 氦原子极化率数值复现项目

本仓库按照《氦原子体系能级与极化率的高精度计算》中的公式与数值流程，构建了从基函数生成、哈密顿矩阵装配到极化率评估的完整 Python 实现。核心矩阵均以稀疏格式存储，可支撑大规模 B 样条基组的数值实验。

## 1. 环境准备

- Python ≥ 3.11（已在 3.14 下测试）
- NumPy ≥ 1.26
- SciPy ≥ 1.11（稀疏矩阵与迭代广义本征求解必需）
- Matplotlib ≥ 3.8（绘制动态极化率曲线）

### 1.1 创建虚拟环境

在仓库根目录执行以下命令（Windows PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS 可将激活命令改为 `source .venv/bin/activate`。

### 1.2 安装依赖

```powershell
pip install -U pip
pip install -e .
```

`pip install -e .` 将基于 `pyproject.toml` 安装运行时依赖；如需后续维护工具，可在文件中补充 `extras` 后使用 `pip install -e .[dev]`。

> **提示**：若之前已激活过其他 Conda/venv 环境，请先退出或关闭旧终端，避免解释器混用。

## 2. 项目结构概览

```
├── README.md
├── pyproject.toml
├── scripts
│   ├── run_energy_pipeline.py
│   ├── run_polarizability_pipeline.py
│   └── plot_dynamic_polarizability.py
└── src
    └── he_polarization
        ├── basis                # B 样条基、角动量耦合与关联展开
        ├── hamiltonian          # 哈密顿算符与矩阵元装配（稀疏输出）
        ├── numerics             # 求积节点与积分工具
        ├── observables          # 能级、极化率、偶极矩等物理量
        ├── solver               # 稠密/稀疏广义本征求解器
        ├── validation           # Hellmann 判据、收敛分析
        └── reporting            # 图形化输出
```

## 3. 快速验证（小规模参数）

按照下列步骤可确认安装正确并熟悉计算流程。示例脚本均内置较小参数：`n=8`, `k=3`, `l_max=1`, `num_eigenvalues=5`。

1. **能量谱与 Hellmann 判据**

    ```powershell
    python scripts\run_energy_pipeline.py
    ```

    输出内容：

    - 低能级本征值列表（含前 5 个能级）。
    - Hellmann 判据 `η` 评估，数值越接近 0 表明动能/势能平衡越好。

2. **静态极化率**

    ```powershell
    python scripts\run_polarizability_pipeline.py
    ```

    输出内容：

    - 前三个本征态的长度规范 / 速度规范静态极化率。
    - 相对误差 `η = 2(α_L - α_V)/(α_L + α_V)`，用于监控波函数精度。

3. **动力学极化率与绘图**

    ```powershell
    python scripts\plot_dynamic_polarizability.py
    ```

    输出内容：

    - 终端提示图像保存路径。
    - `outputs/dynamic_polarizability_ground.png`：基态在给定频率网格上的长度/速度/加速度规范曲线。

如遇脚本运行失败，请先确认当前终端已激活 `.venv`，并阅读第 6 节故障排查。

> **缓存说明**：以上三个脚本默认会将求得的本征谱缓存在 `cache/energy_small/` 下，再次运行时会直接复用。如需强制重算，可加 `--refresh-cache`；若希望暂时关闭缓存，可加 `--no-cache`。

## 4. 脚本说明

- `run_energy_pipeline.py`
    - 生成指数型 B 样条节点与角动量通道。
    - 通过 `MatrixElementBuilder.assemble_matrices` 生成稀疏 `H`、`O` 及各分量矩阵。
    - 调用 `EnergyCalculator.diagonalize`（默认稀疏迭代求解）输出目标本征值与向量，并提供 Hellmann 判据。
- `run_polarizability_pipeline.py`
    - 复用能量管线结果，进一步构建稀疏偶极矩阵与速度规范动量矩阵。
    - 计算指定态的静态极化率，并在终端打印长度/速度规范匹配情况。
- `plot_dynamic_polarizability.py`
    - 在频率数组上评估长度/速度/加速度三种规范的动力学极化率。
    - 调用 `reporting.plot_dynamic_polarizability` 生成图像文件。

所有脚本默认位于仓库根目录执行，并使用稀疏矩阵流水线；无需手动配置即可完成小规模验证。

## 5. 参数调整与扩展

### 5.1 修改基组规模

脚本顶部给出了 `tau`, `r_max`, `k`, `n`, `l_max` 等参数，可根据实验需要修改。例如：

```python
tau = 0.038
r_max = 200.0
k = 5
n = 50
l_max = 4
```

增大参数前请确保机器拥有足够内存与时间预算；稀疏矩阵仍会消耗数 GB 工作空间。

### 5.2 控制本征态数量

`EnergyCalculator.diagonalize` 提供 `num_eigenvalues` 与 `solver_config`；示例：

```python
from he_polarization.solver import IterativeSolverConfig

config = IterativeSolverConfig(num_eigenvalues=12, which="SA", tol=1e-9)
energies, vectors, components = calculator.diagonalize(
    basis_states,
    weights=weights,
    points=points,
    solver_config=config,
)
```

当 `num_eigenvalues` 为空或大于矩阵阶数时，代码自动调用稠密求解器（需充足内存）。

### 5.3 自定义频率网格与输出位置

`plot_dynamic_polarizability.py` 中的 `freqs = np.linspace(0.0, 0.8, 40)` 与 `output_dir = Path("outputs")` 可按需调整；若脚本用于批量扫描，可在循环中改写输出文件名或将数据导出为 CSV。

### 5.4 光谱缓存开关

全部脚本均支持如下缓存参数：

- `--use-cache` / `--no-cache`：控制是否读写缓存（默认开启）。
- `--refresh-cache`：强制重新组装矩阵并覆盖缓存。
- `--cache-dir`：指定缓存根目录（默认 `cache`）。
- `--cache-key`：指定子目录名称，便于针对不同参数集保存多份结果。
- `--num-eigenvalues`：控制缓存中存储的低能本征态数量，影响后续极化率与绘图脚本的读取。

## 6. 常见问题与排查

- **ImportError: SciPy missing**
    - 确认已在激活的虚拟环境中执行 `pip install -e .`。
    - 使用 `python -c "import scipy; print(scipy.__version__)"` 验证安装。
- **运行脚本提示 DLL 加载失败**（Windows）
    - 关闭所有终端重新激活 `.venv`，或将 VS Code 配置为使用该解释器。
- **稠密矩阵导致内存不足**
    - 核心组件均为稀疏格式；如果在自定义脚本中出现 `matrix.toarray()`、`np.asarray(matrix)`，请改用稀疏运算。
- **迭代求解器不收敛**
    - 减少 `num_eigenvalues`。
    - 调整 `IterativeSolverConfig(sigma=<shift>)` 让谱转换更易收敛。
    - 检查基组参数是否过大。
- **绘图脚本没有生成 PNG**
    - 确认 `outputs/` 目录存在写权限；脚本会自动创建，但在某些网络文件系统上需手动授权。

欢迎根据研究需求扩展输出内容（例如保存偶极矩阵稀疏结构、导出极化率表格等），并结合 `validation/` 模块中的 Hellmann 判据与收敛分析工具进行精度评估。
