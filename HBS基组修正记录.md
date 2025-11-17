# HBS 基组修正记录

本文记录了近期为稳定 Hylleraas-B-spline (HBS) 基组而实施的全部操作，便于后续复现、排查与扩展。内容覆盖问题背景、代码改动、脚本入口调整以及推荐的验证步骤。

## 背景问题

- **通道内依赖性**：同一角动量/关联/交换通道内的基函数在数值上高度线性相关，导致重叠矩阵出现极小特征值，解广义本征问题时极易崩溃。
- **全局条件数失控**：即使全局对 `O` 进行正交化，若某些通道已经退化，求得的变换矩阵会放大误差，进而污染能谱与极化率。

为解决以上问题，本次修复将“逐通道正交 + 全局重叠调节”串联执行，并将关键控制参数透出到脚本入口。

## 理论基础与合理性

### 1. 广义本征问题的数值条件

HBS 基组的变分问题等价于求解广义本征方程

$$H c = E O c,$$

其中 $H$、$O$ 分别是哈密顿矩阵与重叠矩阵。若 $O$ 的特征值谱包含过小的 $\lambda_i$，则正交化矩阵 $T=V\Lambda^{-1/2}$（$O=V\Lambda V^\dagger$）会放大舍入误差，导致等效矩阵 $T^\dagger H T$ 条件数恶化。通过对通道子块执行同样的特征分解并丢弃 $\lambda_i<\text{tol}$ 的模式，相当于在每个物理子空间内进行最小范数投影，保留的列空间与原通道张成的稳定子空间完全一致，因此不会改变变分下界，也保证了 $O$ 在新基组内严格正定。

### 2. 通道分组的物理动机

Hylleraas-B-spline 函数以 $(l_1,l_2,L)$ 描述两个电子的角动量耦合，关联幂次与交换宇称进一步限定波函数的对称性。不同通道之间在角动量积分上互不耦合，因此同一通道产生的重叠矩阵退化往往来自径向部分的冗余。逐通道处理可以：

- 只在彼此真正独立的物理子空间内做裁剪，不会混入其它 $(l_1,l_2,L)$ 贡献；
- 保证每个通道都以正交归一的基函数输入哈密顿装配，避免在后续求和时不同通道出现“反向抵消”的伪迹；
- 与角动量选择定则一致（偶极、速度规范都以内禀通道为单位耦合），因此在物理上等价于对每个不可约表示实现最优正交基。

### 3. 与全局重叠调节的配合

逐通道正交化后的矩阵 $O'$ 已接近对角，但在将所有通道拼接后仍可能因为跨通道的线性相关或数值噪声产生新的小特征值。继续对 $O'$ 进行一次全局的 canonical orthonormalization 或对角正则化，即

$${H}'' = R^\dagger {H}' R, \quad {O}'' = R^\dagger {O}' R = I,$$

在数学上等价于对整套基函数做单位正交变换；在物理上，虽然丢弃了某些数值退化方向，但保留下来的波函数仍完整覆盖每个通道里线性独立的物理态。这样可以同时获得：

1. **局部稳定性**：每个通道都具备良好的主子式，矩阵装配阶段即可保证稀疏块的可逆性；
2. **全局稳定性**：最终传给广义本征求解器的是接近单位阵的重叠矩阵，迭代求解时的残差控制和 Hellmann 判据计算都更可靠。

这些操作遵循 Hylleraas 变分理论的基本要求：任何线性可逆的基变换都不会改变原本的谱，而在容差范围内删除的方向本就对能量泛函贡献可以忽略，故在数学与物理上都是合理的。

## 主要改动

### 1. 逐通道正交化模块

- 新增 `src/he_polarization/solver/channel_orthogonalizer.py`：
  - 按 `(l1, l2, L, correlation_power, exchange_parity)` 将基函数分组。
  - 针对每个分组提取 `O` 的子块并做稠密 `eigh`，丢弃特征值 < `tolerance` 的方向。
  - 对保留向量执行 `1/√λ` 归一，生成稀疏投影矩阵与反变换闭包。
  - 对过大子块提供 `max_block_dim` 保护，退化为恒等映射避免内存炸裂。

### 2. 能量求解流程集成

- `src/he_polarization/observables/energies.py`：
  - 在 `EnergyCalculator.diagonalize` 中优先应用 `ChannelOrthogonalizer`，保留反变换用于恢复全基函数系数。
  - 随后再调用既有 `OverlapConditioner`，并在日志中分别打印通道/全局裁剪信息。

### 3. CLI 层面的可配置开关

- `scripts/run_energy_pipeline.py`：
  - 新增 `--disable-channel-orthogonalization` 及 `--channel-ortho-*` 参数（容差、最大块尺寸）。
  - 原有的重叠调节参数亦重命名为 `--overlap-conditioning-*`，支持 `auto/dense/regularize/off` 策略切换。
  - 相关参数写入缓存 `metadata`，用于区分不同实验结果。
- `scripts/run_polarizability_pipeline.py` 与 `scripts/plot_dynamic_polarizability.py` 更新为默认启用逐通道正交与重叠调节，确保后续极化率流程读取到的是“清洁”本征谱。

### 4. 极化率相对误差的鲁棒性

- `src/he_polarization/observables/polarizability_static.py`：`relative_difference` 在长度/速度规范几乎抵消时返回 0 或带符号无穷，而非抛出异常，避免阻塞后续脚本。

## 使用与验证

1. **能量流程（带进度条）**

   ```powershell
   $env:PYTHONPATH='src'
   python scripts\run_energy_pipeline.py --no-cache --progress
   ```

   预期输出示例：
   - `Channel orthonormalization removed XXX states ...`
   - `Overlap conditioning removed YYY states ...`
   - 前 5 个能级与 Hellmann 判据。

2. **静态极化率**

   ```powershell
   $env:PYTHONPATH='src'
   python scripts\run_polarizability_pipeline.py --no-cache
   ```

   预期输出：前三个态的 `α_L/α_V/η`，若两规范抵消会打印 `η = 0.000e+00`。

3. **缓存/参数切换建议**

   - 调整 `--channel-ortho-tol` 可平衡“数值稳定”与“基函数数量”；推荐起始值 `1e-10`。
   - 若通道尺寸远超 `max_block_dim`，可先调大该参数或缩小 `l_max`/径向阶数后再尝试。
   - 当基组进一步做大，可将 `--overlap-conditioning-mode` 设为 `regularize`，并调节 `--overlap-conditioning-regularization`。

## 后续工作

- 针对特定通道的裁剪统计已在 `ChannelGroupStats` 中提供，可在调参时额外记录或可视化。
- 若未来需要混合更多关联幂次，可扩展键中的 `correlation_power` 标志或添加新的分组因子。这样即可在保持模块化的前提下继续拓展 HBS 基组。
