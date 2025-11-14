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


@dataclass
class MatrixElementBuilder:
    """负责交叠矩阵和哈密顿矩阵的数值积分。"""

    bspline: BSplineBasis
    angular: AngularCoupling
    operators: HamiltonianOperators
    correlation: CorrelationExpansion = field(
        default_factory=CorrelationExpansion)

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
        radial_pairs = [state.radial_indices for state in states]
        channels = [state.channel for state in states]
        size = len(states)

        overlap = np.zeros((size, size), dtype=float)
        potential = np.zeros((size, size), dtype=float)
        kinetic = np.zeros((size, size), dtype=float)
        mass = np.zeros((size, size), dtype=float)

        quadrature = RadialQuadrature2D(self.bspline, order=8)

        expanded_states = [self._expand_state(state) for state in states]

        for row, row_state in enumerate(states):
            row_terms = expanded_states[row]
            for col in range(row, size):
                col_state = states[col]
                col_terms = expanded_states[col]

                total_overlap = 0.0
                total_potential = 0.0
                total_kinetic = 0.0
                total_mass = 0.0

                for term_row in row_terms:
                    for term_col in col_terms:
                        c_total = term_row.correlation_power + term_col.correlation_power
                        for corr_term in self._iter_correlation_terms(c_total):
                            q = corr_term.q
                            angular_factor = self._angular_factor(
                                term_row.channel, term_col.channel, q
                            )
                            if angular_factor == 0.0:
                                continue

                            def integrand(point: RadialPointData) -> Tuple[float, float, float, float]:
                                r1 = point.r1
                                r2 = point.r2
                                if r1 <= 0.0 or r2 <= 0.0:
                                    return 0.0, 0.0, 0.0, 0.0

                                phi_row = point.value_r1[0] * point.value_r2[0]
                                phi_col = point.value_r1[1] * point.value_r2[1]

                                if phi_row == 0.0 and phi_col == 0.0:
                                    return 0.0, 0.0, 0.0, 0.0

                                radial_factor = self._radial_power_factor(
                                    r1, r2, c_total, q, corr_term.k
                                )
                                if radial_factor == 0.0:
                                    return 0.0, 0.0, 0.0, 0.0

                                measure = (r1 * r1) * (r2 * r2)
                                base = measure * radial_factor

                                pot_val = self.operators.potential_terms(
                                    r1, r2, max(r1, r2)
                                )

                                lap_r1 = self._radial_laplacian_term(
                                    r1,
                                    point.d1_r1[1],
                                    point.d2_r1[1],
                                    term_col.channel.l1,
                                    point.value_r1[1],
                                )
                                lap_r2 = self._radial_laplacian_term(
                                    r2,
                                    point.d1_r2[1],
                                    point.d2_r2[1],
                                    term_col.channel.l2,
                                    point.value_r2[1],
                                )
                                laplacian_phi = (
                                    lap_r1 * point.value_r2[1]
                                    + lap_r2 * point.value_r1[1]
                                )

                                grad_i_r1 = point.d1_r1[0] * point.value_r2[0]
                                grad_i_r2 = point.value_r1[0] * point.d1_r2[0]
                                grad_j_r1 = point.d1_r1[1] * point.value_r2[1]
                                grad_j_r2 = point.value_r1[1] * point.d1_r2[1]

                                overlap_val = base * phi_row * phi_col
                                potential_val = base * phi_row * pot_val * phi_col
                                kinetic_val = base * phi_row * (
                                    -(0.5 / self.operators.mu) * laplacian_phi
                                )
                                mass_val = base * (-(1.0 / self.operators.M)) * (
                                    grad_i_r1 * grad_j_r1 + grad_i_r2 * grad_j_r2
                                )
                                return overlap_val, potential_val, kinetic_val, mass_val

                            o, v, k_val, m = quadrature.integrate(
                                term_row.radial_indices,
                                term_col.radial_indices,
                                integrand,
                            )

                            coeff = (
                                term_row.coeff
                                * term_col.coeff
                                * corr_term.coefficient
                                * angular_factor
                            )
                            total_overlap += coeff * o
                            total_potential += coeff * v
                            total_kinetic += coeff * k_val
                            total_mass += coeff * m

                overlap[row, col] = total_overlap
                potential[row, col] = total_potential
                kinetic[row, col] = total_kinetic
                mass[row, col] = total_mass

                if row != col:
                    overlap[col, row] = total_overlap
                    potential[col, row] = total_potential
                    kinetic[col, row] = total_kinetic
                    mass[col, row] = total_mass

        h_total = kinetic + mass + potential
        components = {
            "overlap": overlap,
            "potential": potential,
            "kinetic": kinetic,
            "mass": mass,
        }
        return h_total, overlap, components

    @staticmethod
    def _radial_correction(r: float, first_derivative: float) -> float:
        if np.isclose(r, 0.0):
            return 0.0
        return 2.0 * first_derivative / r

    @staticmethod
    def _radial_laplacian_term(
        r: float,
        first_derivative: float,
        second_derivative: float,
        angular_momentum: int,
        basis_value: float,
    ) -> float:
        """返回 ``R'' + 2/r R' - l(l+1)/r^2 R`` 的数值，实现论文式 (2.38)-(2.39) 的径向部分。"""

        laplacian = second_derivative + MatrixElementBuilder._radial_correction(
            r, first_derivative
        )
        if angular_momentum > 0 and r > 1e-12:
            laplacian -= angular_momentum * \
                (angular_momentum + 1) * basis_value / (r * r)
        return laplacian

    def _angular_factor(self, channel_i: AtomicChannel, channel_j: AtomicChannel, q: int) -> float:
        if channel_i == channel_j and q == 0:
            return 1.0
        params = {
            "l1": channel_i.l1,
            "l2": channel_i.l2,
            "L": channel_i.L,
            "l1p": channel_j.l1,
            "l2p": channel_j.l2,
            "Lp": channel_j.L,
            "q": q,
        }
        return self.angular.angular_integral_g1(params)

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
    def _radial_power_factor(
        cls,
        r1: float,
        r2: float,
        c_total: int,
        q: int,
        k_index: int,
    ) -> float:
        r_less = min(r1, r2)
        r_greater = max(r1, r2)
        exp_less = q + 2 * k_index
        exp_greater = c_total - q - 2 * k_index
        factor_less = cls._safe_power(r_less, exp_less)
        factor_greater = cls._safe_power(r_greater, exp_greater)
        return factor_less * factor_greater

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
