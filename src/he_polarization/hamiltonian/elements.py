"""矩阵元装配，对应论文第 2.4 节。"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from scipy.sparse import coo_matrix, load_npz, save_npz

from he_polarization.basis.bspline import (
    BSplineBasis,
    _bspline_derivative_njit,
    _bspline_recursive_njit,
)
from he_polarization.basis.angular import AngularCoupling
from he_polarization.basis.channels import AtomicChannel
from he_polarization.basis.functions import HylleraasBSplineFunction
from he_polarization.basis.correlation import CorrelationExpansion, CorrelationTerm
from he_polarization.numerics.quadrature import _gauss_legendre_nodes_weights
from he_polarization.hamiltonian.operators import HamiltonianOperators

try:  # Optional accelerator for inner-loop helpers.
    from numba import njit  # type: ignore
except ImportError:  # pragma: no cover - fallback when Numba absent.
    def njit(*args, **kwargs):  # type: ignore
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator


@njit()
def _safe_power_numba(base: float, exponent: int) -> float:
    if exponent == 0:
        return 1.0
    if base <= 0.0:
        return 0.0
    return base ** exponent


@njit()
def _radial_factor_components_numba(
    r1: float,
    r2: float,
    c_total: int,
    q: int,
    k_index: int,
) -> Tuple[float, float, float, float, float, float]:
    zero_tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    if c_total == -1:
        if k_index != 0:
            return zero_tuple

        if r1 <= r2:
            base_less = r1
            base_greater = r2
            swap = False
        else:
            base_less = r2
            base_greater = r1
            swap = True

        if base_greater <= 0.0:
            return zero_tuple

        pow_less_q = _safe_power_numba(base_less, q)
        pow_greater_q1 = _safe_power_numba(base_greater, q + 1)
        if pow_greater_q1 == 0.0:
            return zero_tuple

        factor = pow_less_q / pow_greater_q1

        pow_less_qm1 = 0.0
        pow_less_qm2 = 0.0
        if q == 0:
            d_less_1 = 0.0
            d2_less_1 = 0.0
        else:
            pow_less_qm1 = _safe_power_numba(base_less, q - 1)
            if pow_less_qm1 == 0.0:
                d_less_1 = 0.0
            else:
                d_less_1 = q * pow_less_qm1 / pow_greater_q1
            if q >= 2:
                pow_less_qm2 = _safe_power_numba(base_less, q - 2)
                if pow_less_qm2 == 0.0:
                    d2_less_1 = 0.0
                else:
                    d2_less_1 = q * (q - 1) * pow_less_qm2 / pow_greater_q1
            else:
                d2_less_1 = 0.0

        pow_greater_q2 = _safe_power_numba(base_greater, q + 2)
        if pow_greater_q2 == 0.0:
            return zero_tuple
        d_greater_1 = -((q + 1) * pow_less_q / pow_greater_q2)

        pow_greater_q3 = _safe_power_numba(base_greater, q + 3)
        if pow_greater_q3 == 0.0:
            return zero_tuple
        d2_greater_1 = (q + 1) * (q + 2) * pow_less_q / pow_greater_q3

        if q >= 1:
            if pow_less_qm1 == 0.0:
                cross = 0.0
            else:
                cross = -q * (q + 1) * pow_less_qm1 / pow_greater_q2
        else:
            cross = 0.0

        if not swap:
            return factor, d_less_1, d_greater_1, d2_less_1, d2_greater_1, cross

        return factor, d_greater_1, d_less_1, d2_greater_1, d2_less_1, cross

    exp_less = q + 2 * k_index
    exp_greater = c_total - q - 2 * k_index

    if r1 <= r2:
        r_less = r1
        r_greater = r2
        swap = False
    else:
        r_less = r2
        r_greater = r1
        swap = True

    pow_less = _safe_power_numba(r_less, exp_less)
    pow_greater = _safe_power_numba(r_greater, exp_greater)
    factor = pow_less * pow_greater

    if exp_less == 0 or r_less <= 0.0:
        d_less = 0.0
        d2_less = 0.0
    else:
        d_less = exp_less * _safe_power_numba(r_less, exp_less - 1)
        if exp_less <= 1:
            d2_less = 0.0
        else:
            d2_less = exp_less * (exp_less - 1) * \
                _safe_power_numba(r_less, exp_less - 2)

    if exp_greater == 0 or r_greater <= 0.0:
        d_greater = 0.0
        d2_greater = 0.0
    else:
        d_greater = exp_greater * _safe_power_numba(r_greater, exp_greater - 1)
        if exp_greater <= 1:
            d2_greater = 0.0
        else:
            d2_greater = exp_greater * (exp_greater - 1) * _safe_power_numba(
                r_greater, exp_greater - 2
            )

    if not swap:
        dr1 = d_less * pow_greater
        dr2 = pow_less * d_greater
        d2r1 = d2_less * pow_greater
        d2r2 = pow_less * d2_greater
    else:
        dr1 = pow_less * d_greater
        dr2 = d_less * pow_greater
        d2r1 = pow_less * d2_greater
        d2r2 = d2_less * pow_greater

    dr1r2 = d_less * d_greater

    return factor, dr1, dr2, d2r1, d2r2, dr1r2


_EPS_RADIAL = 1e-14


@njit()
def _compute_element_numba(
    i1_row: int,
    i2_row: int,
    i1_col: int,
    i2_col: int,
    knots: np.ndarray,
    bspline_order: int,
    gauss_nodes: np.ndarray,
    gauss_weights: np.ndarray,
    mu: float,
    M: float,
    pref_second: float,
    pref_first: float,
    pref_centrifugal: float,
    pref_cross: float,
    pref_mu_first_extra: float,
    pref_mass_first_extra: float,
    pref_mu_mix: float,
    pref_mass_mix: float,
    pref_mu_j: float,
    pref_mass_j: float,
    scalar_pref_nonzero: bool,
    l1: int,
    l2: int,
    c_col: int,
    neg_inv_M: float,
    coeff_c: np.ndarray,
    g1_c: np.ndarray,
    rhat_grad12_c: np.ndarray,
    rhat_grad21_c: np.ndarray,
    grad_grad_c: np.ndarray,
    c_power_c: np.ndarray,
    corr_q_c: np.ndarray,
    corr_k_c: np.ndarray,
    coeff_c_minus2: np.ndarray,
    g1_c_minus2: np.ndarray,
    rhat_grad12_c_minus2: np.ndarray,
    rhat_grad21_c_minus2: np.ndarray,
    grad_grad_c_minus2: np.ndarray,
    c_power_c_minus2: np.ndarray,
    corr_q_c_minus2: np.ndarray,
    corr_k_c_minus2: np.ndarray,
    coeff_c_minus1: np.ndarray,
    g1_c_minus1: np.ndarray,
    rhat_grad12_c_minus1: np.ndarray,
    rhat_grad21_c_minus1: np.ndarray,
    grad_grad_c_minus1: np.ndarray,
    c_power_c_minus1: np.ndarray,
    corr_q_c_minus1: np.ndarray,
    corr_k_c_minus1: np.ndarray,
    coeff_c_plus2: np.ndarray,
    g1_c_plus2: np.ndarray,
    rhat_grad12_c_plus2: np.ndarray,
    rhat_grad21_c_plus2: np.ndarray,
    grad_grad_c_plus2: np.ndarray,
    c_power_c_plus2: np.ndarray,
    corr_q_c_plus2: np.ndarray,
    corr_k_c_plus2: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    left_r1 = knots[i1_row]
    right_r1 = knots[i1_row + bspline_order]
    left_r1_col = knots[i1_col]
    right_r1_col = knots[i1_col + bspline_order]
    left_r1 = left_r1 if left_r1 > left_r1_col else left_r1_col
    right_r1 = right_r1 if right_r1 < right_r1_col else right_r1_col

    left_r2 = knots[i2_row]
    right_r2 = knots[i2_row + bspline_order]
    left_r2_col = knots[i2_col]
    right_r2_col = knots[i2_col + bspline_order]
    left_r2 = left_r2 if left_r2 > left_r2_col else left_r2_col
    right_r2 = right_r2 if right_r2 < right_r2_col else right_r2_col

    if (right_r1 - left_r1) <= _EPS_RADIAL or (right_r2 - left_r2) <= _EPS_RADIAL:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    total_ol = 0.0
    total_po = 0.0
    total_ki = 0.0
    total_ma = 0.0
    total_ve = 0.0

    gauss_size = gauss_nodes.shape[0]
    n_knots = knots.shape[0] - 1

    # 动能弱形式系数: 1/(2mu)。 pref_second 是 -0.5/mu，所以取反。
    pref_kinetic_weak = -pref_second

    for seg_r1_idx in range(n_knots):
        seg_left_r1 = knots[seg_r1_idx]
        seg_right_r1 = knots[seg_r1_idx + 1]
        if seg_right_r1 <= left_r1 or seg_left_r1 >= right_r1:
            continue
        actual_left_r1 = seg_left_r1 if seg_left_r1 > left_r1 else left_r1
        actual_right_r1 = seg_right_r1 if seg_right_r1 < right_r1 else right_r1
        if (actual_right_r1 - actual_left_r1) <= _EPS_RADIAL:
            continue

        half_r1 = 0.5 * (actual_right_r1 - actual_left_r1)
        mid_r1 = 0.5 * (actual_right_r1 + actual_left_r1)

        for seg_r2_idx in range(n_knots):
            seg_left_r2 = knots[seg_r2_idx]
            seg_right_r2 = knots[seg_r2_idx + 1]
            if seg_right_r2 <= left_r2 or seg_left_r2 >= right_r2:
                continue
            actual_left_r2 = seg_left_r2 if seg_left_r2 > left_r2 else left_r2
            actual_right_r2 = seg_right_r2 if seg_right_r2 < right_r2 else right_r2
            if (actual_right_r2 - actual_left_r2) <= _EPS_RADIAL:
                continue

            half_r2 = 0.5 * (actual_right_r2 - actual_left_r2)
            mid_r2 = 0.5 * (actual_right_r2 + actual_left_r2)

            for idx_r1 in range(gauss_size):
                node_r1 = gauss_nodes[idx_r1]
                weight_r1 = gauss_weights[idx_r1]
                r1 = mid_r1 + half_r1 * node_r1
                weight_node_r1 = half_r1 * weight_r1

                row_r1 = _bspline_recursive_njit(
                    knots, r1, i1_row, bspline_order)
                col_r1 = _bspline_recursive_njit(
                    knots, r1, i1_col, bspline_order)

                # === 关键修改：计算 row 的导数 (用于弱形式动能) ===
                d1_r1_row = _bspline_derivative_njit(
                    knots, r1, i1_row, bspline_order, 1)
                # ============================================

                d1_r1_col = _bspline_derivative_njit(
                    knots, r1, i1_col, bspline_order, 1)
                # 注意：弱形式不需要 row 或 col 的二阶导数 (d2)

                # 仍然计算 d2_r1_col 用于其他可能的混合项（如质量极化），如果需要的话。
                # 但标准动能项已不需要。这里保留以防万一其他项用到。
                # d2_r1_col = _bspline_derivative_njit(knots, r1, i1_col, bspline_order, 2)

                for idx_r2 in range(gauss_size):
                    node_r2 = gauss_nodes[idx_r2]
                    weight_r2 = gauss_weights[idx_r2]
                    r2 = mid_r2 + half_r2 * node_r2
                    weight = weight_node_r1 * half_r2 * weight_r2

                    row_r2 = _bspline_recursive_njit(
                        knots, r2, i2_row, bspline_order)
                    col_r2 = _bspline_recursive_njit(
                        knots, r2, i2_col, bspline_order)

                    # === 关键修改：计算 row 的导数 (用于弱形式动能) ===
                    d1_r2_row = _bspline_derivative_njit(
                        knots, r2, i2_row, bspline_order, 1)
                    # ============================================

                    d1_r2_col = _bspline_derivative_njit(
                        knots, r2, i2_col, bspline_order, 1)
                    # d2_r2_col = _bspline_derivative_njit(knots, r2, i2_col, bspline_order, 2)

                    if r1 <= 0.0 or r2 <= 0.0:
                        continue

                    phi_row = row_r1 * row_r2
                    if phi_row == 0.0 and d1_r1_row == 0.0 and d1_r2_row == 0.0:
                        # 优化：如果所有 row 相关项都为 0，则跳过
                        continue

                    phi_col = col_r1 * col_r2
                    col_dr1 = d1_r1_col * col_r2
                    col_dr2 = col_r1 * d1_r2_col
                    # col_d2r1 = d2_r1_col * col_r2
                    # col_d2r2 = col_r1 * d2_r2_col

                    # === 关键修改：恢复体积元 ===
                    measure = (r1 * r1) * (r2 * r2)
                    # ==========================

                    potential_pref = -2.0 / r1 - 2.0 / r2

                    # === 关键修改：数值稳定性增强 ===
                    _EPS_DENOM = 1e-20

                    if r1 > _EPS_DENOM:
                        # (r1-r2)*(r1+r2) 避免大数相减误差
                        ratio_mix_r1 = (r1 - r2) * (r1 + r2) / r1
                        ratio_j2 = r2 / r1
                    else:
                        ratio_mix_r1 = 0.0
                        ratio_j2 = 0.0

                    if r2 > _EPS_DENOM:
                        ratio_mix_r2 = (r2 - r1) * (r2 + r1) / r2
                        ratio_j1 = r1 / r2
                    else:
                        ratio_mix_r2 = 0.0
                        ratio_j1 = 0.0

                    if r1 > _EPS_DENOM and r2 > _EPS_DENOM:
                        ratio_cross = (r1 * r1 + r2 * r2) / (r1 * r2)
                    else:
                        ratio_cross = 0.0
                    # =============================

                    derivative_product = col_dr1 * col_dr2

                    point_ol = 0.0
                    point_po = 0.0
                    point_ki = 0.0
                    point_ma = 0.0
                    point_ve = 0.0

                    for idx in range(len(coeff_c)):
                        radial_values = _radial_factor_components_numba(
                            r1,
                            r2,
                            int(c_power_c[idx]),
                            int(corr_q_c[idx]),
                            int(corr_k_c[idx]),
                        )
                        radial_factor = radial_values[0]
                        if radial_factor == 0.0:
                            continue

                        coeff = coeff_c[idx]
                        g1 = g1_c[idx]
                        measure_factor = measure * radial_factor

                        if g1 != 0.0:
                            coeff_g1 = coeff * g1

                            # 交叠与势能
                            if phi_col != 0.0:
                                base_overlap = measure_factor * phi_row * phi_col
                                point_ol += coeff_g1 * base_overlap
                                point_po += coeff_g1 * base_overlap * potential_pref

                                # 离心势 (Angluar Kinetic Energy) - 保持原样
                                if r1 != 0.0:
                                    point_ki += (
                                        coeff_g1
                                        * pref_centrifugal
                                        * l1
                                        * (l1 + 1)
                                        * measure_factor
                                        * phi_row
                                        * phi_col
                                        / (r1 * r1)
                                    )
                                if r2 != 0.0:
                                    point_ki += (
                                        coeff_g1
                                        * pref_centrifugal
                                        * l2
                                        * (l2 + 1)
                                        * measure_factor
                                        * phi_row
                                        * phi_col
                                        / (r2 * r2)
                                    )

                            # === 关键修改：径向动能使用弱形式 (Weak Form) ===
                            # T = <phi' | psi'> * (1/2mu)
                            # 1. r1 方向: (d_row/dr1) * (d_col/dr1)
                            if d1_r1_row != 0.0 and d1_r1_col != 0.0:
                                # phi_row 的 r1 导数 = d1_r1_row * row_r2
                                # phi_col 的 r1 导数 = d1_r1_col * col_r2
                                # 乘积 = (d1_r1_row * d1_r1_col) * (row_r2 * col_r2)
                                term_r1 = (d1_r1_row * d1_r1_col) * \
                                    (row_r2 * col_r2)
                                point_ki += coeff_g1 * pref_kinetic_weak * measure_factor * term_r1

                            # 2. r2 方向: (d_row/dr2) * (d_col/dr2)
                            if d1_r2_row != 0.0 and d1_r2_col != 0.0:
                                # phi_row 的 r2 导数 = row_r1 * d1_r2_row
                                # phi_col 的 r2 导数 = col_r1 * d1_r2_col
                                term_r2 = (row_r1 * col_r1) * \
                                    (d1_r2_row * d1_r2_col)
                                point_ki += coeff_g1 * pref_kinetic_weak * measure_factor * term_r2
                            # ==============================================

                            # 强形式的二阶导数项已被移除 (pref_second)
                            # 强形式的一阶导数项已被移除 (pref_first)

                            if c_col != 0 and (col_dr1 != 0.0 or col_dr2 != 0.0):
                                factor_c = coeff_g1 * c_col
                                if col_dr1 != 0.0 and r1 != 0.0:
                                    # 关联项动能修正（这里保留原样，因为涉及到 dr12 的导数，
                                    # 在 Hylleraas 坐标下混合导数通常以这种形式处理是标准做法）
                                    point_ki += (
                                        factor_c
                                        * pref_mu_first_extra
                                        * measure_factor
                                        * phi_row
                                        * col_dr1
                                        / r1
                                    )
                                    point_ma += (
                                        factor_c
                                        * pref_mass_first_extra
                                        * measure_factor
                                        * phi_row
                                        * col_dr1
                                        / r1
                                    )
                                if col_dr2 != 0.0 and r2 != 0.0:
                                    point_ki += (
                                        factor_c
                                        * pref_mu_first_extra
                                        * measure_factor
                                        * phi_row
                                        * col_dr2
                                        / r2
                                    )
                                    point_ma += (
                                        factor_c
                                        * pref_mass_first_extra
                                        * measure_factor
                                        * phi_row
                                        * col_dr2
                                        / r2
                                    )

                            if (
                                pref_cross != 0.0
                                and derivative_product != 0.0
                                and ratio_cross != 0.0
                            ):
                                point_ma += (
                                    coeff_g1
                                    * pref_cross
                                    * measure_factor
                                    * phi_row
                                    * derivative_product
                                    * ratio_cross
                                )

                        rhat_grad_12 = rhat_grad12_c[idx]
                        if rhat_grad_12 != 0.0 and col_dr1 != 0.0 and r2 != 0.0:
                            point_ma += (
                                coeff
                                * rhat_grad_12
                                * neg_inv_M
                                * measure_factor
                                * phi_row
                                * col_dr1
                                / r2
                            )

                        rhat_grad_21 = rhat_grad21_c[idx]
                        if rhat_grad_21 != 0.0 and col_dr2 != 0.0 and r1 != 0.0:
                            point_ma += (
                                coeff
                                * rhat_grad_21
                                * neg_inv_M
                                * measure_factor
                                * phi_row
                                * col_dr2
                                / r1
                            )

                        grad_grad = grad_grad_c[idx]
                        if (
                            grad_grad != 0.0
                            and phi_col != 0.0
                            and r1 != 0.0
                            and r2 != 0.0
                        ):
                            point_ma += (
                                coeff
                                * grad_grad
                                * neg_inv_M
                                * measure_factor
                                * phi_row
                                * phi_col
                                / (r1 * r2)
                            )

                    if c_col != 0:
                        for idx in range(len(coeff_c_minus2)):
                            radial_values = _radial_factor_components_numba(
                                r1,
                                r2,
                                int(c_power_c_minus2[idx]),
                                int(corr_q_c_minus2[idx]),
                                int(corr_k_c_minus2[idx]),
                            )
                            radial_factor = radial_values[0]
                            if radial_factor == 0.0:
                                continue

                            coeff = coeff_c_minus2[idx] * c_col
                            g1 = g1_c_minus2[idx]
                            measure_factor = measure * radial_factor

                            if g1 != 0.0 and scalar_pref_nonzero:
                                if col_dr1 != 0.0 and ratio_mix_r1 != 0.0:
                                    base_mix_r1 = (
                                        measure_factor
                                        * phi_row
                                        * col_dr1
                                        * ratio_mix_r1
                                    )
                                    point_ki += coeff * g1 * pref_mu_mix * base_mix_r1
                                    point_ma += coeff * g1 * pref_mass_mix * base_mix_r1

                                if col_dr2 != 0.0 and ratio_mix_r2 != 0.0:
                                    base_mix_r2 = (
                                        measure_factor
                                        * phi_row
                                        * col_dr2
                                        * ratio_mix_r2
                                    )
                                    point_ki += coeff * g1 * pref_mu_mix * base_mix_r2
                                    point_ma += coeff * g1 * pref_mass_mix * base_mix_r2

                            rhat_grad_12 = rhat_grad12_c_minus2[idx]
                            rhat_grad_21 = rhat_grad21_c_minus2[idx]

                            if (
                                (rhat_grad_12 != 0.0 or rhat_grad_21 != 0.0)
                                and phi_col != 0.0
                            ):
                                if rhat_grad_12 != 0.0 and ratio_j1 != 0.0:
                                    base_j1 = measure_factor * phi_row * phi_col * ratio_j1
                                    point_ki += coeff * pref_mu_j * rhat_grad_12 * base_j1
                                    point_ma += coeff * pref_mass_j * rhat_grad_12 * base_j1

                                if rhat_grad_21 != 0.0 and ratio_j2 != 0.0:
                                    base_j2 = measure_factor * phi_row * phi_col * ratio_j2
                                    point_ki += coeff * pref_mu_j * rhat_grad_21 * base_j2
                                    point_ma += coeff * pref_mass_j * rhat_grad_21 * base_j2

                            if g1 != 0.0 and c_col > 0 and phi_col != 0.0:
                                base_r12 = measure_factor * phi_row * phi_col
                                coeff_mu = 0.0
                                coeff_mass = 0.0
                                if mu != 0.0:
                                    coeff_mu = (-2.0 / mu) * c_col
                                if M != 0.0:
                                    coeff_mass = (2.0 / M) * c_col
                                if c_col >= 2:
                                    if mu != 0.0:
                                        coeff_mu += (-1.0 / mu) * \
                                            c_col * (c_col - 1)
                                    if M != 0.0:
                                        coeff_mass += (1.0 / M) * \
                                            c_col * (c_col - 1)
                                if coeff_mu != 0.0:
                                    point_ki += coeff * g1 * coeff_mu * base_r12
                                if coeff_mass != 0.0:
                                    point_ma += coeff * g1 * coeff_mass * base_r12

                    for idx in range(len(coeff_c_plus2)):
                        radial_values = _radial_factor_components_numba(
                            r1,
                            r2,
                            int(c_power_c_plus2[idx]),
                            int(corr_q_c_plus2[idx]),
                            int(corr_k_c_plus2[idx]),
                        )
                        radial_factor = radial_values[0]
                        if (
                            radial_factor == 0.0
                            or g1_c_plus2[idx] == 0.0
                            or derivative_product == 0.0
                            or r1 == 0.0
                            or r2 == 0.0
                        ):
                            continue

                        measure_factor = measure * radial_factor
                        point_ma += (
                            coeff_c_plus2[idx]
                            * g1_c_plus2[idx]
                            * pref_cross
                            * measure_factor
                            * phi_row
                            * derivative_product
                            * (-1.0 / (r1 * r2))
                        )

                    for idx in range(len(coeff_c_minus1)):
                        radial_values = _radial_factor_components_numba(
                            r1,
                            r2,
                            int(c_power_c_minus1[idx]),
                            int(corr_q_c_minus1[idx]),
                            int(corr_k_c_minus1[idx]),
                        )
                        radial_factor = radial_values[0]
                        if (
                            radial_factor == 0.0
                            or g1_c_minus1[idx] == 0.0
                            or phi_col == 0.0
                        ):
                            continue

                        point_ve += (
                            coeff_c_minus1[idx]
                            * g1_c_minus1[idx]
                            * measure
                            * radial_factor
                            * phi_row
                            * phi_col
                        )

                    total_ol += weight * point_ol
                    total_po += weight * point_po
                    total_ki += weight * point_ki
                    total_ma += weight * point_ma
                    total_ve += weight * point_ve

    return total_ol, total_po, total_ki, total_ma, total_ve


@dataclass(frozen=True)
class _BasisTerm:
    coeff: float
    radial_indices: Tuple[int, int]
    channel: AtomicChannel
    correlation_power: int


@dataclass(frozen=True)
class _AngularComponents:
    g1: float
    rhat_dot: float
    rhat_grad_12: float
    rhat_grad_21: float
    grad_grad: float


@dataclass(frozen=True)
class _PrecomputedTerm:
    corr: CorrelationTerm
    angular: _AngularComponents
    coeff_base: float
    c_power: int


@dataclass(frozen=True)
class _PackedTerms:
    coeff: np.ndarray
    g1: np.ndarray
    rhat_grad_12: np.ndarray
    rhat_grad_21: np.ndarray
    grad_grad: np.ndarray
    c_power: np.ndarray
    corr_q: np.ndarray
    corr_k: np.ndarray


@dataclass(frozen=True)
class _AssemblyScalars:
    mu: float
    M: float
    pref_second: float
    pref_first: float
    pref_centrifugal: float
    pref_cross: float
    pref_mu_first_extra: float
    pref_mass_first_extra: float
    pref_mu_mix: float
    pref_mass_mix: float
    pref_mu_j: float
    pref_mass_j: float
    scalar_pref_nonzero: bool
    neg_inv_M: float


@dataclass
class _ChunkResult:
    chunk_id: int
    chunk_paths: Dict[str, str]
    time_precompute: float
    time_kernel: float
    time_overhead: float
    pairs_processed: int


@dataclass
class _WorkerPayload:
    builder: "MatrixElementBuilder"
    expanded_states: List[Tuple[_BasisTerm, ...]]
    scalars: _AssemblyScalars
    size: int


_WORKER_PAYLOAD: Optional[_WorkerPayload] = None


class _SimpleProgress:
    """Minimal progress reporter to avoid external dependencies."""

    __slots__ = (
        "total",
        "desc",
        "stream",
        "count",
        "_last_percent",
        "_last_print",
    )

    def __init__(self, total: int, desc: str) -> None:
        self.total = max(int(total), 0)
        self.desc = desc
        self.stream = sys.stderr
        self.count = 0
        self._last_percent = -1
        self._last_print = time.perf_counter()
        if self.total == 0:
            self._write_line(0)

    def update(self, step: int = 1) -> None:
        if self.total == 0:
            return
        self.count += step
        if self.count > self.total:
            self.count = self.total
        percent = int((self.count * 100) / self.total)
        now = time.perf_counter()
        if (
            percent != self._last_percent
            or now - self._last_print >= 0.25
            or self.count == self.total
        ):
            self._write_line(percent)
            self._last_percent = percent
            self._last_print = now

    def close(self) -> None:
        if self.total > 0:
            self._write_line(100)
        self.stream.write("\n")
        self.stream.flush()

    def _write_line(self, percent: int) -> None:
        message = f"\r{self.desc}: {percent:3d}% ({self.count}/{self.total})"
        self.stream.write(message)
        self.stream.flush()


@dataclass
class MatrixElementBuilder:
    """负责交叠矩阵和哈密顿矩阵的数值积分。"""

    bspline: BSplineBasis
    angular: AngularCoupling
    operators: HamiltonianOperators
    correlation: CorrelationExpansion = field(
        default_factory=CorrelationExpansion)
    quadrature_order: int = 8
    max_workers: Optional[int] = None
    rows_per_chunk: Optional[int] = None
    _angular_cache: Dict[Tuple[int, int, int, int, int, int, int], _AngularComponents] = field(
        init=False, default_factory=dict
    )
    _knots: np.ndarray = field(init=False, repr=False)
    _gauss_nodes: np.ndarray = field(init=False, repr=False)
    _gauss_weights: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._knots = np.ascontiguousarray(self.bspline.knots, dtype=float)
        nodes, weights = _gauss_legendre_nodes_weights(self.quadrature_order)
        self._gauss_nodes = np.ascontiguousarray(nodes, dtype=float)
        self._gauss_weights = np.ascontiguousarray(weights, dtype=float)

    def assemble_matrices(
        self,
        basis_states: Iterable[
            Union[
                HylleraasBSplineFunction,
                Tuple[Tuple[int, int], AtomicChannel]
            ]
        ],
        *,
        weights: Iterable[float],
        points: Iterable[Tuple[float, float]],
        progress: bool | str | None = None,
    ) -> tuple[coo_matrix, coo_matrix, Dict[str, coo_matrix]]:
        """批量构建 ``H`` 与 ``O`` 以及单独的算符分量矩阵。"""

        _ = points, weights

        states = [
            state
            if isinstance(state, HylleraasBSplineFunction)
            else HylleraasBSplineFunction(radial_indices=state[0], channel=state[1])
            for state in basis_states
        ]
        size = len(states)

        expanded_states = [self._expand_state(state) for state in states]

        if isinstance(progress, str):
            progress_desc = progress
            show_progress = True
        elif isinstance(progress, bool) or progress is None:
            progress_desc = "Assembling matrix elements"
            show_progress = bool(progress)
        else:
            raise TypeError("progress must be a bool, str, or None")

        total_pairs = size * (size + 1) // 2
        reporter = _SimpleProgress(
            total_pairs, progress_desc) if show_progress else None

        mu = self.operators.mu
        M = self.operators.M
        pref_second = -0.5 / mu if mu != 0.0 else 0.0
        pref_first = -1.0 / mu if mu != 0.0 else 0.0
        pref_centrifugal = 0.5 / mu if mu != 0.0 else 0.0

        # --- 质量极化系数修正 (论文更正) ---
        # 论文公式 (2.37) 的 -1/4M 疑似笔误，正确应为 -1/2M (H_MP = -1/M grad1.grad2)
        pref_cross = -0.5 / M if M != 0.0 else 0.0
        # --------------------------------

        pref_mu_first_extra = -2.0 * pref_second
        pref_mass_first_extra = -2.0 * (0.5 / M if M != 0.0 else 0.0)
        pref_mu_mix = -2.0 * pref_second
        pref_mass_mix = pref_mass_first_extra
        pref_mu_j = 1.0 / mu if mu != 0.0 else 0.0
        pref_mass_j = -1.0 / M if M != 0.0 else 0.0
        scalar_pref_nonzero = pref_mu_mix != 0.0 or pref_mass_mix != 0.0
        neg_inv_M = -1.0 / M if M != 0.0 else 0.0

        scalars = _AssemblyScalars(
            mu=float(mu),
            M=float(M),
            pref_second=float(pref_second),
            pref_first=float(pref_first),
            pref_centrifugal=float(pref_centrifugal),
            pref_cross=float(pref_cross),
            pref_mu_first_extra=float(pref_mu_first_extra),
            pref_mass_first_extra=float(pref_mass_first_extra),
            pref_mu_mix=float(pref_mu_mix),
            pref_mass_mix=float(pref_mass_mix),
            pref_mu_j=float(pref_mu_j),
            pref_mass_j=float(pref_mass_j),
            scalar_pref_nonzero=bool(scalar_pref_nonzero),
            neg_inv_M=float(neg_inv_M),
        )

        worker_count = self._resolve_worker_count()
        used_parallel = worker_count > 1 and size > 1

        chunk_root = Path("cache") / "assembly_chunks"
        chunk_root.mkdir(parents=True, exist_ok=True)
        chunk_dir = Path(
            tempfile.mkdtemp(prefix="chunk_run_", dir=str(chunk_root))
        )
        print(f"Using out-of-core temporary directory: {chunk_dir}")

        chunk_results: List[_ChunkResult] = []
        final_matrices: Dict[str, coo_matrix] = {}
        time_start = time.perf_counter()

        try:
            if worker_count <= 1 or size <= 1:
                chunk_results.append(
                    self._assemble_rows(
                        expanded_states,
                        size,
                        scalars,
                        0,
                        size,
                        reporter,
                        chunk_id=0,
                        chunk_dir=chunk_dir,
                    )
                )
            else:
                chunk_results.extend(
                    self._assemble_parallel(
                        expanded_states,
                        size,
                        scalars,
                        reporter,
                        worker_count,
                        chunk_dir,
                    )
                )

            if not chunk_results:
                raise RuntimeError("Matrix assembly failed before completion.")

            total_elapsed = time.perf_counter() - time_start
            if total_elapsed > 0.0:
                time_precompute = sum(
                    chunk.time_precompute for chunk in chunk_results)
                time_kernel = sum(chunk.time_kernel for chunk in chunk_results)
                time_overhead = sum(
                    chunk.time_overhead for chunk in chunk_results)
                print("\n--- Assembly timing (seconds) ---")
                print(
                    f"  Python precompute : {time_precompute:10.2f}"
                    f" ({time_precompute / total_elapsed:6.2%})"
                )
                print(
                    f"  Numba kernel      : {time_kernel:10.2f}"
                    f" ({time_kernel / total_elapsed:6.2%})"
                )
                print(
                    f"  Python overhead   : {time_overhead:10.2f}"
                    f" ({time_overhead / total_elapsed:6.2%})"
                )
                print(f"  Total elapsed     : {total_elapsed:10.2f} (100.00%)")
                if used_parallel:
                    print("  (timings reflect summed worker CPU seconds)")

            print("Assembly complete. Merging sparse matrices from disk...")
            chunk_paths: Dict[str, List[str]] = defaultdict(list)
            for chunk in sorted(chunk_results, key=lambda c: c.chunk_id):
                for key, path in chunk.chunk_paths.items():
                    chunk_paths[key].append(path)

            final_matrices = {}
            for key, paths in chunk_paths.items():
                matrix_sum: Optional[coo_matrix] = None
                for path in paths:
                    chunk_matrix = load_npz(path).tocoo()
                    if matrix_sum is None:
                        matrix_sum = chunk_matrix
                    else:
                        matrix_sum = matrix_sum + chunk_matrix
                if matrix_sum is not None:
                    final_matrices[key] = matrix_sum.tocoo()

            print("Matrix merge complete.")
        finally:
            if reporter is not None:
                reporter.close()
            if chunk_dir.exists():
                print(f"Cleaning up temporary directory: {chunk_dir}")
                shutil.rmtree(chunk_dir, ignore_errors=True)

        overlap = final_matrices.get("overlap", coo_matrix((size, size)))
        potential = final_matrices.get("potential", coo_matrix((size, size)))
        kinetic = final_matrices.get("kinetic", coo_matrix((size, size)))
        mass = final_matrices.get("mass", coo_matrix((size, size)))

        h_total = kinetic + mass + potential
        components = {
            "overlap": overlap,
            "potential": potential,
            "kinetic": kinetic,
            "mass": mass,
        }
        return h_total, overlap, components

    def _assemble_parallel(
        self,
        expanded_states: List[Tuple[_BasisTerm, ...]],
        size: int,
        scalars: _AssemblyScalars,
        reporter: Optional[_SimpleProgress],
        worker_count: int,
        chunk_dir: Path,
    ) -> List[_ChunkResult]:
        chunks = self._make_row_chunks(size, worker_count)
        if not chunks:
            return []

        # 自动选择启动模式 (Linux -> fork, 其他 -> spawn)
        try:
            ctx = mp.get_context(None)
        except ValueError:
            ctx = mp.get_context("spawn")

        payload = _WorkerPayload(
            builder=self,
            expanded_states=expanded_states,
            scalars=scalars,
            size=size,
        )

        chunk_tasks = [
            (chunk_id, row_range, str(chunk_dir))
            for chunk_id, row_range in enumerate(chunks)
        ]
        chunk_results: List[_ChunkResult] = []
        with ctx.Pool(
            processes=worker_count,
            initializer=_parallel_worker_init,
            initargs=(payload,),
        ) as pool:
            for chunk in pool.imap_unordered(
                _parallel_process_chunk, chunk_tasks, chunksize=1
            ):
                chunk_results.append(chunk)
                if reporter is not None and chunk.pairs_processed:
                    reporter.update(chunk.pairs_processed)

        return chunk_results

    def _assemble_rows(
        self,
        expanded_states: List[Tuple[_BasisTerm, ...]],
        size: int,
        scalars: _AssemblyScalars,
        row_start: int,
        row_end: int,
        reporter: Optional[_SimpleProgress],
        chunk_id: int,
        chunk_dir: Path,
    ) -> _ChunkResult:
        overlap_entries: Dict[Tuple[int, int], float] = {}
        potential_entries: Dict[Tuple[int, int], float] = {}
        kinetic_entries: Dict[Tuple[int, int], float] = {}
        mass_entries: Dict[Tuple[int, int], float] = {}
        time_precompute = 0.0
        time_kernel = 0.0
        time_overhead = 0.0
        pairs_processed = 0

        if row_start >= row_end:
            return _ChunkResult(
                chunk_id=chunk_id,
                chunk_paths={},
                time_precompute=time_precompute,
                time_kernel=time_kernel,
                time_overhead=time_overhead,
                pairs_processed=pairs_processed,
            )

        for row in range(row_start, row_end):
            row_terms = expanded_states[row]
            for col in range(row, size):
                col_terms = expanded_states[col]

                total_overlap = 0.0
                total_potential = 0.0
                total_kinetic_mu = 0.0
                total_mass = 0.0
                total_vee = 0.0

                for term_row in row_terms:
                    for term_col in col_terms:
                        c_row = term_row.correlation_power
                        c_col = term_col.correlation_power
                        c_total = c_row + c_col

                        t_pre_start = time.perf_counter()
                        terms_c = self._precompute_terms(
                            term_row, term_col, c_total)
                        terms_c_minus2 = self._precompute_terms(
                            term_row, term_col, c_total - 2)
                        terms_c_minus1 = self._precompute_terms(
                            term_row, term_col, c_total - 1)
                        terms_c_plus2 = self._precompute_terms(
                            term_row, term_col, c_total + 2)
                        time_precompute += time.perf_counter() - t_pre_start

                        if (
                            not terms_c
                            and not terms_c_minus2
                            and not terms_c_minus1
                            and not terms_c_plus2
                        ):
                            continue

                        l1 = term_col.channel.l1
                        l2 = term_col.channel.l2

                        packed_c = self._pack_terms(terms_c)
                        packed_c_minus2 = self._pack_terms(terms_c_minus2)
                        packed_c_minus1 = self._pack_terms(terms_c_minus1)
                        packed_c_plus2 = self._pack_terms(terms_c_plus2)

                        t_kernel_start = time.perf_counter()
                        overlap_val, potential_val, kinetic_val, mass_val, vee_val = _compute_element_numba(
                            int(term_row.radial_indices[0]),
                            int(term_row.radial_indices[1]),
                            int(term_col.radial_indices[0]),
                            int(term_col.radial_indices[1]),
                            self._knots,
                            int(self.bspline.order),
                            self._gauss_nodes,
                            self._gauss_weights,
                            float(scalars.mu),
                            float(scalars.M),
                            float(scalars.pref_second),
                            float(scalars.pref_first),
                            float(scalars.pref_centrifugal),
                            float(scalars.pref_cross),
                            float(scalars.pref_mu_first_extra),
                            float(scalars.pref_mass_first_extra),
                            float(scalars.pref_mu_mix),
                            float(scalars.pref_mass_mix),
                            float(scalars.pref_mu_j),
                            float(scalars.pref_mass_j),
                            bool(scalars.scalar_pref_nonzero),
                            int(l1),
                            int(l2),
                            int(c_col),
                            float(scalars.neg_inv_M),
                            packed_c.coeff,
                            packed_c.g1,
                            packed_c.rhat_grad_12,
                            packed_c.rhat_grad_21,
                            packed_c.grad_grad,
                            packed_c.c_power,
                            packed_c.corr_q,
                            packed_c.corr_k,
                            packed_c_minus2.coeff,
                            packed_c_minus2.g1,
                            packed_c_minus2.rhat_grad_12,
                            packed_c_minus2.rhat_grad_21,
                            packed_c_minus2.grad_grad,
                            packed_c_minus2.c_power,
                            packed_c_minus2.corr_q,
                            packed_c_minus2.corr_k,
                            packed_c_minus1.coeff,
                            packed_c_minus1.g1,
                            packed_c_minus1.rhat_grad_12,
                            packed_c_minus1.rhat_grad_21,
                            packed_c_minus1.grad_grad,
                            packed_c_minus1.c_power,
                            packed_c_minus1.corr_q,
                            packed_c_minus1.corr_k,
                            packed_c_plus2.coeff,
                            packed_c_plus2.g1,
                            packed_c_plus2.rhat_grad_12,
                            packed_c_plus2.rhat_grad_21,
                            packed_c_plus2.grad_grad,
                            packed_c_plus2.c_power,
                            packed_c_plus2.corr_q,
                            packed_c_plus2.corr_k,
                        )
                        time_kernel += time.perf_counter() - t_kernel_start

                        total_overlap += overlap_val
                        total_potential += potential_val
                        total_kinetic_mu += kinetic_val
                        total_mass += mass_val
                        total_vee += vee_val

                potential_total = total_potential + total_vee
                t_overhead_start = time.perf_counter()
                overlap_entries[(row, col)] = total_overlap
                potential_entries[(row, col)] = potential_total
                kinetic_entries[(row, col)] = total_kinetic_mu
                mass_entries[(row, col)] = total_mass

                if row != col:
                    overlap_entries[(col, row)] = total_overlap
                    potential_entries[(col, row)] = potential_total
                    kinetic_entries[(col, row)] = total_kinetic_mu
                    mass_entries[(col, row)] = total_mass
                time_overhead += time.perf_counter() - t_overhead_start

                pairs_processed += 1
                if reporter is not None:
                    reporter.update()
        write_start = time.perf_counter()
        chunk_paths: Dict[str, str] = {}
        matrices = {
            "overlap": overlap_entries,
            "potential": potential_entries,
            "kinetic": kinetic_entries,
            "mass": mass_entries,
        }

        for key, entries in matrices.items():
            if not entries:
                continue
            rows, cols, data = zip(*((i, j, v)
                                   for (i, j), v in entries.items()))
            sparse_chunk = coo_matrix(
                (data, (rows, cols)), shape=(size, size)).tocsr()
            file_path = chunk_dir / f"chunk_{chunk_id:04d}_{key}.npz"
            save_npz(file_path, sparse_chunk)
            chunk_paths[key] = str(file_path)
            entries.clear()

        time_overhead += time.perf_counter() - write_start

        return _ChunkResult(
            chunk_id=chunk_id,
            chunk_paths=chunk_paths,
            time_precompute=time_precompute,
            time_kernel=time_kernel,
            time_overhead=time_overhead,
            pairs_processed=pairs_processed,
        )

    def _resolve_worker_count(self) -> int:
        if self.max_workers is not None:
            return max(1, int(self.max_workers))
        env_value = os.getenv("HEP_ASSEMBLY_WORKERS")
        if env_value:
            try:
                value = int(env_value)
                if value >= 1:
                    return value
            except ValueError:
                pass
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count)

    def _make_row_chunks(self, size: int, worker_count: int) -> List[Tuple[int, int]]:
        if size <= 0:
            return []
        if worker_count <= 0:
            worker_count = 1
        if self.rows_per_chunk is not None and self.rows_per_chunk > 0:
            row_span = int(self.rows_per_chunk)
        else:
            row_span = max(1, math.ceil(size / (worker_count * 4)))

        chunks: List[Tuple[int, int]] = []
        start = 0
        while start < size:
            end = min(size, start + row_span)
            chunks.append((start, end))
            start = end
        return chunks

    def _angular_components(
        self,
        channel_i: AtomicChannel,
        channel_j: AtomicChannel,
        q: int,
    ) -> _AngularComponents:
        key = (
            channel_i.l1,
            channel_i.l2,
            channel_i.L,
            channel_j.l1,
            channel_j.l2,
            channel_j.L,
            q,
        )
        cache = self._angular_cache
        if key in cache:
            return cache[key]

        params = {
            "l1": channel_i.l1,
            "l2": channel_i.l2,
            "L": channel_i.L,
            "l1p": channel_j.l1,
            "l2p": channel_j.l2,
            "Lp": channel_j.L,
            "q": q,
        }
        if channel_i == channel_j and q == 0:
            g1 = 1.0
        else:
            g1 = self.angular.angular_integral_g1(params)

        l1, l2, L = channel_i.l1, channel_i.l2, channel_i.L
        l1p, l2p, Lp = channel_j.l1, channel_j.l2, channel_j.L
        rhat_dot = self.angular.angular_tensor_rhat_dot(
            l1, l2, L, l1p, l2p, Lp, q
        )
        rhat_grad_12 = self.angular.angular_tensor_rhat_grad_12(
            l1, l2, L, l1p, l2p, Lp, q
        )
        rhat_grad_21 = self.angular.angular_tensor_rhat_grad_21(
            l1, l2, L, l1p, l2p, Lp, q
        )
        grad_grad = self.angular.angular_tensor_grad_grad(
            l1, l2, L, l1p, l2p, Lp, q
        )

        components = _AngularComponents(
            g1=g1,
            rhat_dot=rhat_dot,
            rhat_grad_12=rhat_grad_12,
            rhat_grad_21=rhat_grad_21,
            grad_grad=grad_grad,
        )
        cache[key] = components
        return components

    def _iter_correlation_terms(self, c_total: int) -> Tuple[CorrelationTerm, ...]:
        key = (c_total,)
        if not hasattr(self, "_correlation_cache"):
            self._correlation_cache: Dict[Tuple[int],
                                          Tuple[CorrelationTerm, ...]] = {}
        cache = self._correlation_cache
        if key not in cache:
            cache[key] = tuple(self.correlation.iter_terms(c_total))
        return cache[key]

    def _precompute_terms(
        self,
        term_row: _BasisTerm,
        term_col: _BasisTerm,
        c_power: int,
    ) -> Tuple[_PrecomputedTerm, ...]:
        if c_power < -1:
            return ()

        terms: List[_PrecomputedTerm] = []
        for corr_term in self._iter_correlation_terms(c_power):
            angular = self._angular_components(
                term_row.channel,
                term_col.channel,
                corr_term.q,
            )
            coeff_base = (
                term_row.coeff
                * term_col.coeff
                * corr_term.coefficient
            )
            if coeff_base == 0.0 and (
                angular.g1 == 0.0
                and angular.rhat_grad_12 == 0.0
                and angular.rhat_grad_21 == 0.0
                and angular.grad_grad == 0.0
            ):
                continue
            terms.append(
                _PrecomputedTerm(
                    corr=corr_term,
                    angular=angular,
                    coeff_base=coeff_base,
                    c_power=c_power,
                )
            )

        return tuple(terms)

    @staticmethod
    def _pack_terms(terms: Tuple[_PrecomputedTerm, ...]) -> _PackedTerms:
        if not terms:
            empty_float = np.empty(0, dtype=np.float64)
            empty_int = np.empty(0, dtype=np.int64)
            return _PackedTerms(
                coeff=empty_float,
                g1=empty_float,
                rhat_grad_12=empty_float,
                rhat_grad_21=empty_float,
                grad_grad=empty_float,
                c_power=empty_int,
                corr_q=empty_int,
                corr_k=empty_int,
            )

        size = len(terms)
        coeff = np.empty(size, dtype=np.float64)
        g1 = np.empty(size, dtype=np.float64)
        rhat_grad_12 = np.empty(size, dtype=np.float64)
        rhat_grad_21 = np.empty(size, dtype=np.float64)
        grad_grad = np.empty(size, dtype=np.float64)
        c_power = np.empty(size, dtype=np.int64)
        corr_q = np.empty(size, dtype=np.int64)
        corr_k = np.empty(size, dtype=np.int64)

        for idx, term in enumerate(terms):
            coeff[idx] = term.coeff_base
            g1[idx] = term.angular.g1
            rhat_grad_12[idx] = term.angular.rhat_grad_12
            rhat_grad_21[idx] = term.angular.rhat_grad_21
            grad_grad[idx] = term.angular.grad_grad
            c_power[idx] = term.c_power
            corr_q[idx] = term.corr.q
            corr_k[idx] = term.corr.k

        return _PackedTerms(
            coeff=coeff,
            g1=g1,
            rhat_grad_12=rhat_grad_12,
            rhat_grad_21=rhat_grad_21,
            grad_grad=grad_grad,
            c_power=c_power,
            corr_q=corr_q,
            corr_k=corr_k,
        )

    @staticmethod
    def _safe_power(base: float, exponent: int) -> float:
        return float(_safe_power_numba(float(base), int(exponent)))

    @classmethod
    def _radial_factor_components(
        cls,
        r1: float,
        r2: float,
        c_total: int,
        q: int,
        k_index: int,
    ) -> Tuple[float, float, float, float, float, float]:
        result = _radial_factor_components_numba(
            float(r1),
            float(r2),
            int(c_total),
            int(q),
            int(k_index),
        )
        return (
            float(result[0]),
            float(result[1]),
            float(result[2]),
            float(result[3]),
            float(result[4]),
            float(result[5]),
        )

    def _expand_state(self, state: HylleraasBSplineFunction) -> Tuple[_BasisTerm, ...]:
        coeff_primary = 1.0
        primary = _BasisTerm(
            coeff=coeff_primary,
            radial_indices=state.radial_indices,
            channel=state.channel,
            correlation_power=state.correlation_power,
        )

        if not state.symmetrized:
            return (primary,)

        swapped = state.swapped()
        parity = float(state.exchange_parity)

        if swapped.radial_indices == state.radial_indices and swapped.channel == state.channel:
            combined_coeff = coeff_primary + parity
            if abs(combined_coeff) < 1e-12:
                return tuple()
            return (
                _BasisTerm(
                    coeff=combined_coeff,
                    radial_indices=state.radial_indices,
                    channel=state.channel,
                    correlation_power=state.correlation_power,
                ),
            )

        return (
            primary,
            _BasisTerm(
                coeff=parity,
                radial_indices=swapped.radial_indices,
                channel=swapped.channel,
                correlation_power=swapped.correlation_power,
            ),
        )


def _parallel_worker_init(payload: _WorkerPayload) -> None:
    global _WORKER_PAYLOAD
    _WORKER_PAYLOAD = payload


def _parallel_process_chunk(task: Tuple[int, Tuple[int, int], str]) -> _ChunkResult:
    if _WORKER_PAYLOAD is None:
        raise RuntimeError("Worker payload is not initialized.")
    builder = _WORKER_PAYLOAD.builder
    expanded_states = _WORKER_PAYLOAD.expanded_states
    scalars = _WORKER_PAYLOAD.scalars
    size = _WORKER_PAYLOAD.size
    chunk_id, row_range, chunk_dir_str = task
    start, end = row_range
    chunk_dir = Path(chunk_dir_str)
    return builder._assemble_rows(
        expanded_states,
        size,
        scalars,
        start,
        end,
        reporter=None,
        chunk_id=chunk_id,
        chunk_dir=chunk_dir,
    )
