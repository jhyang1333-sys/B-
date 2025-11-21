"""串联论文第 3.3 节的静态极化率计算与误差评估。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.sparse.linalg import eigsh
import scipy.linalg as la

from he_polarization.basis import (
    AngularCoupling,
    AtomicChannel,
    BSplineBasis,
    ExponentialNodeConfig,
    generate_exponential_nodes,
    generate_hylleraas_bspline_functions,
)
from he_polarization.hamiltonian import HamiltonianOperators, MatrixElementBuilder
from he_polarization.numerics import generate_tensor_product_quadrature
from he_polarization.observables import StaticPolarizabilityCalculator, EnergyCalculator
from he_polarization.observables.dipole import (
    build_dipole_matrix,
    build_velocity_gauge_matrix,
)
from he_polarization.solver import ChannelOrthogonalizer, OverlapConditioner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute static polarizabilities with explicitly consistent basis transformation.")
    # ... (参数部分保持不变) ...
    parser.add_argument("--use-cache", dest="use_cache",
                        action="store_true", default=False)  # 建议默认关闭缓存以确保演示逻辑清晰
    parser.add_argument("--num-eigenvalues", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- 1. 物理参数定义 ---
    tau = 0.038
    r_max = 150.0
    k = 7
    n = 20
    l_max = 3

    # 物理常数
    m_nucleus = 7294.2995365
    mu = m_nucleus / (1.0 + m_nucleus)

    # --- 2. 构建基组 ---
    print("构建基组与算符...")
    config = ExponentialNodeConfig(
        r_min=0.0, r_max=r_max, k=k, n=n, gamma=r_max * tau)
    knots = generate_exponential_nodes(config)
    bspline = BSplineBasis(knots=knots, order=k)
    angular = AngularCoupling()
    operators = HamiltonianOperators(mu=mu, M=m_nucleus)

    channels = [
        AtomicChannel(l1=l1, l2=l2, L=L)
        for l1 in range(l_max + 1)
        for l2 in range(l_max + 1)
        for L in range(abs(l1 - l2), min(l1 + l2, l_max) + 1)
    ]
    basis_states = generate_hylleraas_bspline_functions(
        bspline, channels, n_radial=bspline.n_basis,
        correlation_powers=(0, 1), exchange_parity=1, symmetrize=True, unique_pairs=True,
    )
    points, weights = generate_tensor_product_quadrature(
        config.r_min, config.r_max, n_points=8)

    builder = MatrixElementBuilder(
        bspline=bspline, angular=angular, operators=operators)

    # --- 3. 显式装配与正交化 (关键修正步骤) ---
    print("装配原始矩阵 (H_orig, S_orig)...")
    # 获取原始基底下的矩阵 (Original Basis)
    h_total_orig, s_total_orig, components_orig = builder.assemble_matrices(
        basis_states, weights=weights, points=points, progress=True
    )

    print("执行逐通道正交化...")
    channel_ortho = ChannelOrthogonalizer(
        tolerance=1e-10,
        max_block_dim=2048,
        safe_mode=True,          # 启用你写的物理安全救援
        safe_mode_threshold=-2.0
    )

    # apply 返回: 新矩阵(Reduced), 变换函数(Back Transform), 统计信息
    h_red, s_red, back_transform_func, stats = channel_ortho.apply(
        h_total_orig, s_total_orig, basis_states
    )

    print(f"基组维度压缩: {stats.total_original} -> {stats.total_retained}")

    # --- 4. 求解本征值问题 (在 Reduced 空间) ---
    print("求解本征值 (Reduced Basis)...")
    # 使用 Shift-Invert 模式寻找基态附近
    sigma_val = -2.90372

    # 注意: h_red 和 s_red 是 scipy.sparse 矩阵
    vals, vecs_red = eigsh(
        h_red, k=args.num_eigenvalues, M=s_red, sigma=sigma_val, which='LM'
    )

    # eigsh 返回并未完全排序，需手动排序
    idx = vals.argsort()
    energies = vals[idx]
    vecs_red = vecs_red[:, idx]

    # --- 5. 回代变换 (Back Transformation) ---
    print("执行回代变换: Reduced -> Original ...")
    # 关键修正：利用 back_transform_func 将向量映射回原始 B-spline 基底
    # 这样才能与 dipole_matrix (原始基底) 匹配
    vecs_orig = back_transform_func(vecs_red)

    # --- 6. 构建偶极与动量算符 (原始基底) ---
    print("构建偶极与动量矩阵 (Original Basis)...")
    dipole_matrix_orig = build_dipole_matrix(
        bspline, basis_states, angular,
        correlation=builder.correlation, weights=weights, points=points
    )

    # 速度规范矩阵依赖于 [H, r]，这里传入 components_orig (原始动能/势能)
    # 确保这一步使用的是原始基底的组件
    momentum_matrix_orig = build_velocity_gauge_matrix(
        dipole_matrix_orig, components_orig, reduced_mass=operators.mu
    )

    # --- 7. 计算极化率 ---
    print("计算极化率...")

    # 现在维度匹配了: (K x N_orig) @ (N_orig x N_orig) @ (N_orig x K)
    dipole_eigen = vecs_orig.conj().T @ dipole_matrix_orig @ vecs_orig
    momentum_eigen = vecs_orig.conj().T @ momentum_matrix_orig @ vecs_orig

    # 为了兼容你的 StaticPolarizabilityCalculator，我们需要一个 Dummy Calculator
    # 或者直接实例化它 (它只依赖 energy_calculator 来获取一些元数据，这里我们可以简化使用)
    # 实际上你的 StaticPolarizabilityCalculator 只是个纯计算类，不需要复杂依赖
    # 这里稍微 Hack 一下传入 None 或者简单的占位符，因为 compute 方法只用到了 array

    class DummyEnergyCalc:
        pass
    polar_calc = StaticPolarizabilityCalculator(
        energy_calculator=DummyEnergyCalc())  # type: ignore

    print("\n=== 结果报告 ===")
    for state_idx in range(min(3, len(energies))):
        try:
            alpha_len = polar_calc.compute_length_gauge(
                energies, dipole_eigen, state_idx)
            alpha_vel = polar_calc.compute_velocity_gauge(
                energies, momentum_eigen, state_idx)
            eta = polar_calc.relative_difference(alpha_len, alpha_vel)

            print(f"态 {state_idx} (E={energies[state_idx]:.6f}):")
            print(f"  Length Gauge   = {alpha_len:.8e}")
            print(f"  Velocity Gauge = {alpha_vel:.8e}")
            print(f"  Gauge Diff (η) = {eta:.3e}")
        except ZeroDivisionError:
            print(f"态 {state_idx}: 跳过 (存在简并)")


if __name__ == "__main__":
    main()
