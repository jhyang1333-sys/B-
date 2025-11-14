"""矩阵元装配，对应论文第 2.4 节。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple, Union

import numpy as np

from he_polarization.basis.bspline import BSplineBasis
from he_polarization.basis.angular import AngularCoupling
from he_polarization.basis.channels import AtomicChannel
from he_polarization.basis.functions import HylleraasBSplineFunction
from he_polarization.basis.correlation import CorrelationExpansion, CorrelationTerm
from he_polarization.numerics import RadialQuadrature2D, RadialPointData
from he_polarization.hamiltonian.operators import HamiltonianOperators


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


@dataclass
class MatrixElementBuilder:
    """负责交叠矩阵和哈密顿矩阵的数值积分。"""

    bspline: BSplineBasis
    angular: AngularCoupling
    operators: HamiltonianOperators
    correlation: CorrelationExpansion = field(
        default_factory=CorrelationExpansion)
    _angular_cache: Dict[Tuple[int, int, int, int, int, int, int], _AngularComponents] = field(
        init=False, default_factory=dict
    )

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
    ) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """批量构建 ``H`` 与 ``O`` 以及单独的算符分量矩阵。"""

        # 外部提供的张量积求积节点暂未使用，保留参数以兼容先前接口。
        _ = points, weights

        states = [
            state
            if isinstance(state, HylleraasBSplineFunction)
            else HylleraasBSplineFunction(radial_indices=state[0], channel=state[1])
            for state in basis_states
        ]
        size = len(states)

        overlap = np.zeros((size, size), dtype=float)
        potential = np.zeros((size, size), dtype=float)
        kinetic = np.zeros((size, size), dtype=float)
        mass = np.zeros((size, size), dtype=float)

        quadrature = RadialQuadrature2D(self.bspline, order=8)

        expanded_states = [self._expand_state(state) for state in states]

        for row, row_terms in enumerate(expanded_states):
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

                        # I-type integrals that share the same correlation power.
                        for corr_term in self._iter_correlation_terms(c_total):
                            q = corr_term.q
                            ang = self._angular_components(
                                term_row.channel, term_col.channel, q
                            )

                            def integrand(point: RadialPointData) -> Tuple[
                                float,
                                float,
                                float,
                                float,
                                float,
                                float,
                                float,
                                float,
                                float,
                                float,
                                float,
                            ]:
                                r1 = point.r1
                                r2 = point.r2
                                if r1 <= 0.0 or r2 <= 0.0:
                                    return (
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                    )

                                (radial_factor, *_) = self._radial_factor_components(
                                    r1, r2, c_total, q, corr_term.k
                                )

                                measure = (r1 * r1) * (r2 * r2)

                                row_r1 = point.value_r1[0]
                                row_r2 = point.value_r2[0]
                                col_r1 = point.value_r1[1]
                                col_r2 = point.value_r2[1]

                                phi_row = row_r1 * row_r2
                                phi_col = col_r1 * col_r2

                                if (
                                    phi_row == 0.0
                                    and phi_col == 0.0
                                    and radial_factor == 0.0
                                ):
                                    return (
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                    )

                                d_col_r1 = point.d1_r1[1]
                                d_col_r2 = point.d1_r2[1]
                                d2_col_r1 = point.d2_r1[1]
                                d2_col_r2 = point.d2_r2[1]

                                col_dr1 = d_col_r1 * col_r2
                                col_dr2 = col_r1 * d_col_r2
                                col_d2r1 = d2_col_r1 * col_r2
                                col_d2r2 = col_r1 * d2_col_r2

                                overlap_val = measure * radial_factor * phi_row * phi_col
                                potential_val = overlap_val * self.operators.potential_terms(
                                    r1, r2
                                )

                                kin_second_r1 = measure * radial_factor * phi_row * col_d2r1
                                kin_second_r2 = measure * radial_factor * phi_row * col_d2r2

                                kin_first_r1 = 0.0
                                if r1 > 0.0:
                                    kin_first_r1 = (
                                        measure
                                        * radial_factor
                                        * phi_row
                                        * col_dr1
                                        / r1
                                    )

                                kin_first_r2 = 0.0
                                if r2 > 0.0:
                                    kin_first_r2 = (
                                        measure
                                        * radial_factor
                                        * phi_row
                                        * col_dr2
                                        / r2
                                    )

                                centrifugal_r1 = 0.0
                                if r1 > 0.0:
                                    centrifugal_r1 = (
                                        measure
                                        * radial_factor
                                        * phi_row
                                        * phi_col
                                        / (r1 * r1)
                                    )

                                centrifugal_r2 = 0.0
                                if r2 > 0.0:
                                    centrifugal_r2 = (
                                        measure
                                        * radial_factor
                                        * phi_row
                                        * phi_col
                                        / (r2 * r2)
                                    )

                                j1_term1 = 0.0
                                if r2 > 0.0:
                                    j1_term1 = (
                                        measure
                                        * radial_factor
                                        * phi_row
                                        * col_dr1
                                        / r2
                                    )

                                j2_term1 = 0.0
                                if r1 > 0.0:
                                    j2_term1 = (
                                        measure
                                        * radial_factor
                                        * phi_row
                                        * col_dr2
                                        / r1
                                    )

                                j3_term = 0.0
                                if r1 > 0.0 and r2 > 0.0:
                                    j3_term = (
                                        measure
                                        * radial_factor
                                        * phi_row
                                        * phi_col
                                        / (r1 * r2)
                                    )

                                return (
                                    overlap_val,
                                    potential_val,
                                    kin_second_r1,
                                    kin_first_r1,
                                    kin_second_r2,
                                    kin_first_r2,
                                    centrifugal_r1,
                                    centrifugal_r2,
                                    j1_term1,
                                    j2_term1,
                                    j3_term,
                                )

                            (
                                overlap_val,
                                potential_val,
                                kin_second_r1,
                                kin_first_r1,
                                kin_second_r2,
                                kin_first_r2,
                                centrifugal_r1,
                                centrifugal_r2,
                                j1_term1,
                                j2_term1,
                                j3_term,
                            ) = quadrature.integrate(
                                term_row.radial_indices,
                                term_col.radial_indices,
                                integrand,
                            )

                            coeff_base = (
                                term_row.coeff
                                * term_col.coeff
                                * corr_term.coefficient
                            )

                            g1 = ang.g1
                            if g1 != 0.0:
                                total_overlap += coeff_base * g1 * overlap_val
                                total_potential += coeff_base * g1 * potential_val

                                pure_second = -0.5 / self.operators.mu * (
                                    kin_second_r1 + kin_second_r2
                                )
                                pure_first = -1.0 / self.operators.mu * (
                                    kin_first_r1 + kin_first_r2
                                )
                                centrifugal = (
                                    0.5
                                    / self.operators.mu
                                    * (
                                        term_col.channel.l1
                                        * (term_col.channel.l1 + 1)
                                        * centrifugal_r1
                                        + term_col.channel.l2
                                        * (term_col.channel.l2 + 1)
                                        * centrifugal_r2
                                    )
                                )

                                kinetic_contrib = pure_second + pure_first + centrifugal
                                if kinetic_contrib != 0.0:
                                    total_kinetic_mu += coeff_base * g1 * kinetic_contrib

                                if c_col != 0:
                                    pref_mu = -0.5 / self.operators.mu
                                    pref_mass = 0.5 / self.operators.M
                                    factor = coeff_base * g1 * c_col
                                    if kin_first_r1 != 0.0:
                                        if pref_mu != 0.0:
                                            total_kinetic_mu += factor * pref_mu * kin_first_r1
                                        if pref_mass != 0.0:
                                            total_mass += factor * pref_mass * kin_first_r1
                                    if kin_first_r2 != 0.0:
                                        if pref_mu != 0.0:
                                            total_kinetic_mu += factor * pref_mu * kin_first_r2
                                        if pref_mass != 0.0:
                                            total_mass += factor * pref_mass * kin_first_r2

                                pref_cross = -0.25 / self.operators.M

                                # Mixed ∂_{r1}∂_{r2} contribution using c_total.
                                if pref_cross != 0.0:

                                    def integrand_cross(point: RadialPointData) -> Tuple[float]:
                                        r1 = point.r1
                                        r2 = point.r2
                                        if r1 <= 0.0 or r2 <= 0.0:
                                            return (0.0,)

                                        (radial_factor, *_) = self._radial_factor_components(
                                            r1, r2, c_total, q, corr_term.k
                                        )
                                        if radial_factor == 0.0:
                                            return (0.0,)

                                        row_r1 = point.value_r1[0]
                                        row_r2 = point.value_r2[0]
                                        phi_row = row_r1 * row_r2
                                        if phi_row == 0.0:
                                            return (0.0,)

                                        d_col_r1 = point.d1_r1[1]
                                        d_col_r2 = point.d1_r2[1]
                                        derivative_product = d_col_r1 * d_col_r2
                                        if derivative_product == 0.0:
                                            return (0.0,)

                                        denom = r1 * r2
                                        if denom == 0.0:
                                            return (0.0,)

                                        ratio = (r1 * r1 + r2 * r2) / denom
                                        measure = (r1 * r1) * (r2 * r2)
                                        return (
                                            measure
                                            * radial_factor
                                            * phi_row
                                            * derivative_product
                                            * ratio,
                                        )

                                    (cross_base,) = quadrature.integrate(
                                        term_row.radial_indices,
                                        term_col.radial_indices,
                                        integrand_cross,
                                    )

                                    if cross_base != 0.0:
                                        total_mass += coeff_base * g1 * pref_cross * cross_base

                            if ang.rhat_grad_12 != 0.0 and j1_term1 != 0.0:
                                total_mass += (
                                    coeff_base
                                    * ang.rhat_grad_12
                                    * (-1.0 / self.operators.M)
                                    * j1_term1
                                )

                            if ang.rhat_grad_21 != 0.0 and j2_term1 != 0.0:
                                total_mass += (
                                    coeff_base
                                    * ang.rhat_grad_21
                                    * (-1.0 / self.operators.M)
                                    * j2_term1
                                )

                            if ang.grad_grad != 0.0 and j3_term != 0.0:
                                total_mass += (
                                    coeff_base
                                    * ang.grad_grad
                                    * (-1.0 / self.operators.M)
                                    * j3_term
                                )

                        pref_mu_mix = -0.5 / self.operators.mu
                        pref_mass_mix = 0.5 / self.operators.M

                        # Terms stemming from ∂_{ri}∂_{r12} acting on the correlation factor.

                        if c_col != 0:
                            c_shift_mix = c_total - 2
                            scalar_pref_nonzero = (
                                pref_mu_mix != 0.0 or pref_mass_mix != 0.0
                            )
                            if c_shift_mix >= -1:
                                pref_mu_j = 1.0 / self.operators.mu
                                pref_mass_j = -1.0 / self.operators.M
                                for corr_term_mix in self._iter_correlation_terms(c_shift_mix):
                                    q_mix = corr_term_mix.q
                                    ang_mix = self._angular_components(
                                        term_row.channel, term_col.channel, q_mix
                                    )

                                    coeff_base_mix = (
                                        term_row.coeff
                                        * term_col.coeff
                                        * corr_term_mix.coefficient
                                    )

                                    g1_mix = ang_mix.g1
                                    has_scalar_pref = (
                                        g1_mix != 0.0 and scalar_pref_nonzero
                                    )

                                    if has_scalar_pref:

                                        def integrand_mix_r1(point: RadialPointData) -> Tuple[float]:
                                            r1 = point.r1
                                            r2 = point.r2
                                            if r1 <= 0.0 or r2 <= 0.0:
                                                return (0.0,)

                                            (radial_factor, *_) = self._radial_factor_components(
                                                r1, r2, c_shift_mix, q_mix, corr_term_mix.k
                                            )
                                            if radial_factor == 0.0:
                                                return (0.0,)

                                            row_r1 = point.value_r1[0]
                                            row_r2 = point.value_r2[0]
                                            phi_row = row_r1 * row_r2
                                            if phi_row == 0.0:
                                                return (0.0,)

                                            d_col_r1 = point.d1_r1[1]
                                            col_r2 = point.value_r2[1]
                                            col_dr1 = d_col_r1 * col_r2
                                            if col_dr1 == 0.0:
                                                return (0.0,)

                                            ratio = (r1 * r1 - r2 * r2) / \
                                                r1 if r1 != 0.0 else 0.0
                                            measure = (r1 * r1) * (r2 * r2)
                                            return (
                                                measure
                                                * radial_factor
                                                * phi_row
                                                * col_dr1
                                                * ratio,
                                            )

                                        (mix_r1_val,) = quadrature.integrate(
                                            term_row.radial_indices,
                                            term_col.radial_indices,
                                            integrand_mix_r1,
                                        )

                                        if mix_r1_val != 0.0:
                                            factor = coeff_base_mix * c_col
                                            if pref_mu_mix != 0.0:
                                                total_kinetic_mu += (
                                                    factor
                                                    * pref_mu_mix
                                                    * g1_mix
                                                    * mix_r1_val
                                                )
                                            if pref_mass_mix != 0.0:
                                                total_mass += (
                                                    factor
                                                    * pref_mass_mix
                                                    * g1_mix
                                                    * mix_r1_val
                                                )

                                    if ang_mix.rhat_grad_12 != 0.0 and (
                                        pref_mu_j != 0.0 or pref_mass_j != 0.0
                                    ):

                                        def integrand_j1_term2(point: RadialPointData) -> Tuple[float]:
                                            r1 = point.r1
                                            r2 = point.r2
                                            if r1 <= 0.0 or r2 <= 0.0:
                                                return (0.0,)

                                            (radial_factor, *_) = self._radial_factor_components(
                                                r1, r2, c_shift_mix, q_mix, corr_term_mix.k
                                            )
                                            if radial_factor == 0.0 or r2 == 0.0:
                                                return (0.0,)

                                            row_r1 = point.value_r1[0]
                                            row_r2 = point.value_r2[0]
                                            phi_row = row_r1 * row_r2
                                            if phi_row == 0.0:
                                                return (0.0,)

                                            col_r1 = point.value_r1[1]
                                            col_r2 = point.value_r2[1]
                                            phi_col = col_r1 * col_r2
                                            if phi_col == 0.0:
                                                return (0.0,)

                                            measure = (r1 * r1) * (r2 * r2)
                                            return (
                                                measure
                                                * radial_factor
                                                * phi_row
                                                * phi_col
                                                * (r1 / r2),
                                            )

                                        (j1_term2_val,) = quadrature.integrate(
                                            term_row.radial_indices,
                                            term_col.radial_indices,
                                            integrand_j1_term2,
                                        )

                                        if j1_term2_val != 0.0:
                                            factor = coeff_base_mix * c_col
                                            if pref_mu_j != 0.0:
                                                total_kinetic_mu += (
                                                    factor
                                                    * pref_mu_j
                                                    * ang_mix.rhat_grad_12
                                                    * j1_term2_val
                                                )
                                            if pref_mass_j != 0.0:
                                                total_mass += (
                                                    factor
                                                    * pref_mass_j
                                                    * ang_mix.rhat_grad_12
                                                    * j1_term2_val
                                                )

                                    if has_scalar_pref:

                                        def integrand_mix_r2(point: RadialPointData) -> Tuple[float]:
                                            r1 = point.r1
                                            r2 = point.r2
                                            if r1 <= 0.0 or r2 <= 0.0:
                                                return (0.0,)

                                            (radial_factor, *_) = self._radial_factor_components(
                                                r1, r2, c_shift_mix, q_mix, corr_term_mix.k
                                            )
                                            if radial_factor == 0.0:
                                                return (0.0,)

                                            row_r1 = point.value_r1[0]
                                            row_r2 = point.value_r2[0]
                                            phi_row = row_r1 * row_r2
                                            if phi_row == 0.0:
                                                return (0.0,)

                                            col_r1 = point.value_r1[1]
                                            d_col_r2 = point.d1_r2[1]
                                            col_dr2 = col_r1 * d_col_r2
                                            if col_dr2 == 0.0:
                                                return (0.0,)

                                            ratio = (r2 * r2 - r1 * r1) / \
                                                r2 if r2 != 0.0 else 0.0
                                            measure = (r1 * r1) * (r2 * r2)
                                            return (
                                                measure
                                                * radial_factor
                                                * phi_row
                                                * col_dr2
                                                * ratio,
                                            )

                                        (mix_r2_val,) = quadrature.integrate(
                                            term_row.radial_indices,
                                            term_col.radial_indices,
                                            integrand_mix_r2,
                                        )

                                        if mix_r2_val != 0.0:
                                            factor = coeff_base_mix * c_col
                                            if pref_mu_mix != 0.0:
                                                total_kinetic_mu += (
                                                    factor
                                                    * pref_mu_mix
                                                    * g1_mix
                                                    * mix_r2_val
                                                )
                                            if pref_mass_mix != 0.0:
                                                total_mass += (
                                                    factor
                                                    * pref_mass_mix
                                                    * g1_mix
                                                    * mix_r2_val
                                                )

                                    if ang_mix.rhat_grad_21 != 0.0 and (
                                        pref_mu_j != 0.0 or pref_mass_j != 0.0
                                    ):

                                        def integrand_j2_term2(point: RadialPointData) -> Tuple[float]:
                                            r1 = point.r1
                                            r2 = point.r2
                                            if r1 <= 0.0 or r2 <= 0.0:
                                                return (0.0,)

                                            (radial_factor, *_) = self._radial_factor_components(
                                                r1, r2, c_shift_mix, q_mix, corr_term_mix.k
                                            )
                                            if radial_factor == 0.0 or r1 == 0.0:
                                                return (0.0,)

                                            row_r1 = point.value_r1[0]
                                            row_r2 = point.value_r2[0]
                                            phi_row = row_r1 * row_r2
                                            if phi_row == 0.0:
                                                return (0.0,)

                                            col_r1 = point.value_r1[1]
                                            col_r2 = point.value_r2[1]
                                            phi_col = col_r1 * col_r2
                                            if phi_col == 0.0:
                                                return (0.0,)

                                            measure = (r1 * r1) * (r2 * r2)
                                            return (
                                                measure
                                                * radial_factor
                                                * phi_row
                                                * phi_col
                                                * (r2 / r1),
                                            )

                                        (j2_term2_val,) = quadrature.integrate(
                                            term_row.radial_indices,
                                            term_col.radial_indices,
                                            integrand_j2_term2,
                                        )

                                        if j2_term2_val != 0.0:
                                            factor = coeff_base_mix * c_col
                                            if pref_mu_j != 0.0:
                                                total_kinetic_mu += (
                                                    factor
                                                    * pref_mu_j
                                                    * ang_mix.rhat_grad_21
                                                    * j2_term2_val
                                                )
                                            if pref_mass_j != 0.0:
                                                total_mass += (
                                                    factor
                                                    * pref_mass_j
                                                    * ang_mix.rhat_grad_21
                                                    * j2_term2_val
                                                )

                            if scalar_pref_nonzero and c_total >= -1:
                                for corr_term_mix_zero in self._iter_correlation_terms(c_total):
                                    q_mix_zero = corr_term_mix_zero.q
                                    ang_mix_zero = self._angular_components(
                                        term_row.channel, term_col.channel, q_mix_zero
                                    )

                                    g1_mix_zero = ang_mix_zero.g1
                                    if g1_mix_zero == 0.0:
                                        continue

                                    coeff_base_zero = (
                                        term_row.coeff
                                        * term_col.coeff
                                        * corr_term_mix_zero.coefficient
                                    )
                                    factor_zero = coeff_base_zero * c_col

                                    def integrand_mix_r1_zero(point: RadialPointData) -> Tuple[float]:
                                        r1 = point.r1
                                        r2 = point.r2
                                        if r1 <= 0.0 or r2 <= 0.0 or r1 == 0.0:
                                            return (0.0,)

                                        (radial_factor, *_) = self._radial_factor_components(
                                            r1, r2, c_total, q_mix_zero, corr_term_mix_zero.k
                                        )
                                        if radial_factor == 0.0:
                                            return (0.0,)

                                        row_r1 = point.value_r1[0]
                                        row_r2 = point.value_r2[0]
                                        phi_row = row_r1 * row_r2
                                        if phi_row == 0.0:
                                            return (0.0,)

                                        d_col_r1 = point.d1_r1[1]
                                        col_r2 = point.value_r2[1]
                                        col_dr1 = d_col_r1 * col_r2
                                        if col_dr1 == 0.0:
                                            return (0.0,)

                                        measure = (r1 * r1) * (r2 * r2)
                                        return (
                                            measure
                                            * radial_factor
                                            * phi_row
                                            * col_dr1
                                            / r1,
                                        )

                                    (mix_r1_zero,) = quadrature.integrate(
                                        term_row.radial_indices,
                                        term_col.radial_indices,
                                        integrand_mix_r1_zero,
                                    )

                                    if mix_r1_zero != 0.0:
                                        if pref_mu_mix != 0.0:
                                            total_kinetic_mu += (
                                                factor_zero
                                                * pref_mu_mix
                                                * g1_mix_zero
                                                * mix_r1_zero
                                            )
                                        if pref_mass_mix != 0.0:
                                            total_mass += (
                                                factor_zero
                                                * pref_mass_mix
                                                * g1_mix_zero
                                                * mix_r1_zero
                                            )

                                    def integrand_mix_r2_zero(point: RadialPointData) -> Tuple[float]:
                                        r1 = point.r1
                                        r2 = point.r2
                                        if r1 <= 0.0 or r2 <= 0.0 or r2 == 0.0:
                                            return (0.0,)

                                        (radial_factor, *_) = self._radial_factor_components(
                                            r1, r2, c_total, q_mix_zero, corr_term_mix_zero.k
                                        )
                                        if radial_factor == 0.0:
                                            return (0.0,)

                                        row_r1 = point.value_r1[0]
                                        row_r2 = point.value_r2[0]
                                        phi_row = row_r1 * row_r2
                                        if phi_row == 0.0:
                                            return (0.0,)

                                        col_r1 = point.value_r1[1]
                                        d_col_r2 = point.d1_r2[1]
                                        col_dr2 = col_r1 * d_col_r2
                                        if col_dr2 == 0.0:
                                            return (0.0,)

                                        measure = (r1 * r1) * (r2 * r2)
                                        return (
                                            measure
                                            * radial_factor
                                            * phi_row
                                            * col_dr2
                                            / r2,
                                        )

                                    (mix_r2_zero,) = quadrature.integrate(
                                        term_row.radial_indices,
                                        term_col.radial_indices,
                                        integrand_mix_r2_zero,
                                    )

                                    if mix_r2_zero != 0.0:
                                        if pref_mu_mix != 0.0:
                                            total_kinetic_mu += (
                                                factor_zero
                                                * pref_mu_mix
                                                * g1_mix_zero
                                                * mix_r2_zero
                                            )
                                        if pref_mass_mix != 0.0:
                                            total_mass += (
                                                factor_zero
                                                * pref_mass_mix
                                                * g1_mix_zero
                                                * mix_r2_zero
                                            )

                        c_shift_r12 = c_total - 2
                        # Pure r12-derivative terms after lowering c_total.
                        if c_shift_r12 >= -1 and c_col > 0:
                            for corr_term_r12 in self._iter_correlation_terms(c_shift_r12):
                                q_r12 = corr_term_r12.q
                                ang_r12 = self._angular_components(
                                    term_row.channel, term_col.channel, q_r12
                                )

                                def integrand_r12(point: RadialPointData) -> Tuple[float]:
                                    r1 = point.r1
                                    r2 = point.r2
                                    if r1 <= 0.0 or r2 <= 0.0:
                                        return (0.0,)

                                    phi_row = point.value_r1[0] * \
                                        point.value_r2[0]
                                    phi_col = point.value_r1[1] * \
                                        point.value_r2[1]
                                    if phi_row == 0.0 or phi_col == 0.0:
                                        return (0.0,)

                                    (radial_factor, *_) = self._radial_factor_components(
                                        r1, r2, c_shift_r12, q_r12, corr_term_r12.k
                                    )
                                    if radial_factor == 0.0:
                                        return (0.0,)

                                    measure = (r1 * r1) * (r2 * r2)
                                    return (
                                        measure * radial_factor * phi_row * phi_col,
                                    )

                                (integral_r12,) = quadrature.integrate(
                                    term_row.radial_indices,
                                    term_col.radial_indices,
                                    integrand_r12,
                                )

                                if integral_r12 == 0.0:
                                    continue

                                coeff_base_r12 = (
                                    term_row.coeff
                                    * term_col.coeff
                                    * corr_term_r12.coefficient
                                )

                                g1_r12 = ang_r12.g1
                                if g1_r12 == 0.0:
                                    continue

                                coeff_mu = 0.0
                                coeff_mass = 0.0
                                if c_col >= 2:
                                    coeff_mu += (
                                        -1.0
                                        / self.operators.mu
                                        * c_col
                                        * (c_col - 1)
                                    )
                                    coeff_mass += (
                                        1.0
                                        / self.operators.M
                                        * c_col
                                        * (c_col - 1)
                                    )
                                coeff_mu += (
                                    -2.0 / self.operators.mu * c_col
                                )
                                coeff_mass += (
                                    2.0 / self.operators.M * c_col
                                )

                                if coeff_mu != 0.0:
                                    total_kinetic_mu += (
                                        coeff_base_r12
                                        * g1_r12
                                        * coeff_mu
                                        * integral_r12
                                    )
                                if coeff_mass != 0.0:
                                    total_mass += (
                                        coeff_base_r12
                                        * g1_r12
                                        * coeff_mass
                                        * integral_r12
                                    )

                        c_plus = c_total + 2
                        # Remaining part of ∂_{r1}∂_{r2} that raises the correlation power.
                        if c_plus >= -1:
                            for corr_term_plus in self._iter_correlation_terms(c_plus):
                                q_plus = corr_term_plus.q
                                ang_plus = self._angular_components(
                                    term_row.channel, term_col.channel, q_plus
                                )

                                def integrand_cross_plus(point: RadialPointData) -> Tuple[float]:
                                    r1 = point.r1
                                    r2 = point.r2
                                    if r1 <= 0.0 or r2 <= 0.0:
                                        return (0.0,)

                                    (radial_factor, *_) = self._radial_factor_components(
                                        r1, r2, c_plus, q_plus, corr_term_plus.k
                                    )
                                    if radial_factor == 0.0:
                                        return (0.0,)

                                    row_r1 = point.value_r1[0]
                                    row_r2 = point.value_r2[0]
                                    phi_row = row_r1 * row_r2
                                    if phi_row == 0.0:
                                        return (0.0,)

                                    d_col_r1 = point.d1_r1[1]
                                    d_col_r2 = point.d1_r2[1]
                                    derivative_product = d_col_r1 * d_col_r2
                                    if derivative_product == 0.0 or r1 == 0.0 or r2 == 0.0:
                                        return (0.0,)

                                    measure = (r1 * r1) * (r2 * r2)
                                    return (
                                        measure
                                        * radial_factor
                                        * phi_row
                                        * derivative_product
                                        * (-1.0 / (r1 * r2)),
                                    )

                                (cross_plus_val,) = quadrature.integrate(
                                    term_row.radial_indices,
                                    term_col.radial_indices,
                                    integrand_cross_plus,
                                )

                                if cross_plus_val == 0.0:
                                    continue

                                coeff_base_plus = (
                                    term_row.coeff
                                    * term_col.coeff
                                    * corr_term_plus.coefficient
                                )

                                g1_plus = ang_plus.g1
                                if g1_plus == 0.0:
                                    continue

                                total_mass += (
                                    coeff_base_plus
                                    * g1_plus
                                    * (-0.25 / self.operators.M)
                                    * cross_plus_val
                                )

                        c_vee = c_total - 1
                        if c_vee >= -1:
                            for corr_term_vee in self._iter_correlation_terms(c_vee):
                                q_vee = corr_term_vee.q
                                ang_vee = self._angular_components(
                                    term_row.channel, term_col.channel, q_vee
                                )
                                if ang_vee.g1 == 0.0:
                                    continue

                                def integrand_vee(point: RadialPointData) -> Tuple[float]:
                                    r1 = point.r1
                                    r2 = point.r2
                                    if r1 <= 0.0 or r2 <= 0.0:
                                        return (0.0,)

                                    phi_row = point.value_r1[0] * \
                                        point.value_r2[0]
                                    phi_col = point.value_r1[1] * \
                                        point.value_r2[1]
                                    if phi_row == 0.0 or phi_col == 0.0:
                                        return (0.0,)

                                    (radial_factor, *_) = self._radial_factor_components(
                                        r1, r2, c_vee, q_vee, corr_term_vee.k
                                    )
                                    if radial_factor == 0.0:
                                        return (0.0,)

                                    measure = (r1 * r1) * (r2 * r2)
                                    return (
                                        measure * radial_factor * phi_row * phi_col,
                                    )

                                (integral_vee,) = quadrature.integrate(
                                    term_row.radial_indices,
                                    term_col.radial_indices,
                                    integrand_vee,
                                )

                                if integral_vee == 0.0:
                                    continue

                                coeff_base_vee = (
                                    term_row.coeff
                                    * term_col.coeff
                                    * corr_term_vee.coefficient
                                )
                                total_vee += coeff_base_vee * ang_vee.g1 * integral_vee

                overlap[row, col] = total_overlap
                potential[row, col] = total_potential + total_vee
                kinetic[row, col] = total_kinetic_mu
                mass[row, col] = total_mass

                if row != col:
                    overlap[col, row] = total_overlap
                    potential[col, row] = total_potential + total_vee
                    kinetic[col, row] = total_kinetic_mu
                    mass[col, row] = total_mass

        def _symmetrize(matrix: np.ndarray) -> np.ndarray:
            return 0.5 * (matrix + matrix.T)

        overlap = _symmetrize(overlap)
        potential = _symmetrize(potential)
        kinetic = _symmetrize(kinetic)
        mass = _symmetrize(mass)

        h_total = kinetic + mass + potential
        components = {
            "overlap": overlap,
            "potential": potential,
            "kinetic": kinetic,
            "mass": mass,
        }
        return h_total, overlap, components

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

    @staticmethod
    def _safe_power(base: float, exponent: int) -> float:
        if exponent == 0:
            return 1.0
        if base <= 0.0:
            if exponent > 0:
                return 0.0
            return 0.0
        try:
            return base ** exponent
        except (OverflowError, ZeroDivisionError):
            return 0.0

    @classmethod
    def _radial_factor_components(
        cls,
        r1: float,
        r2: float,
        c_total: int,
        q: int,
        k_index: int,
    ) -> Tuple[float, float, float, float, float, float]:
        if c_total == -1:
            if k_index != 0:
                return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

            r_less = min(r1, r2)
            r_greater = max(r1, r2)
            if r_greater <= 0.0:
                return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

            if r1 <= r2:
                base_less, base_greater = r1, r2
                factor = (base_less ** q) / (base_greater ** (q + 1))
                if q == 0:
                    dr1 = 0.0
                    d2r1 = 0.0
                else:
                    dr1 = q * (base_less ** (q - 1)) / \
                        (base_greater ** (q + 1))
                    d2r1 = (
                        q * (q - 1)
                        * (base_less ** (q - 2))
                        / (base_greater ** (q + 1))
                        if q >= 2
                        else 0.0
                    )
                dr2 = (
                    -(q + 1)
                    * (base_less ** q)
                    / (base_greater ** (q + 2))
                )
                d2r2 = (
                    (q + 1)
                    * (q + 2)
                    * (base_less ** q)
                    / (base_greater ** (q + 3))
                )
                dr1r2 = (
                    -q
                    * (q + 1)
                    * (base_less ** (q - 1))
                    / (base_greater ** (q + 2))
                    if q >= 1
                    else 0.0
                )
            else:
                base_less, base_greater = r2, r1
                factor = (base_less ** q) / (base_greater ** (q + 1))
                dr1 = (
                    -(q + 1)
                    * (base_less ** q)
                    / (base_greater ** (q + 2))
                )
                d2r1 = (
                    (q + 1)
                    * (q + 2)
                    * (base_less ** q)
                    / (base_greater ** (q + 3))
                )
                if q == 0:
                    dr2 = 0.0
                    d2r2 = 0.0
                else:
                    dr2 = q * (base_less ** (q - 1)) / \
                        (base_greater ** (q + 1))
                    d2r2 = (
                        q * (q - 1)
                        * (base_less ** (q - 2))
                        / (base_greater ** (q + 1))
                        if q >= 2
                        else 0.0
                    )
                dr1r2 = (
                    -q
                    * (q + 1)
                    * (base_less ** (q - 1))
                    / (base_greater ** (q + 2))
                    if q >= 1
                    else 0.0
                )

            return factor, dr1, dr2, d2r1, d2r2, dr1r2

        exp_less = q + 2 * k_index
        exp_greater = c_total - q - 2 * k_index

        r_less = min(r1, r2)
        r_greater = max(r1, r2)

        def power(base: float, exponent: int) -> float:
            return cls._safe_power(base, exponent)

        def first_derivative(base: float, exponent: int) -> float:
            if exponent == 0 or base <= 0.0:
                return 0.0
            return exponent * power(base, exponent - 1)

        def second_derivative(base: float, exponent: int) -> float:
            if exponent <= 1 or base <= 0.0:
                return 0.0
            return exponent * (exponent - 1) * power(base, exponent - 2)

        pow_less = power(r_less, exp_less)
        pow_greater = power(r_greater, exp_greater)
        factor = pow_less * pow_greater

        d_less = first_derivative(r_less, exp_less)
        d_greater = first_derivative(r_greater, exp_greater)
        d2_less = second_derivative(r_less, exp_less)
        d2_greater = second_derivative(r_greater, exp_greater)

        if r1 <= r2:
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
