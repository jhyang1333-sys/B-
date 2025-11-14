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
        radial_pairs = [state.radial_indices for state in states]
        channels = [state.channel for state in states]
        size = len(states)

        overlap = np.zeros((size, size), dtype=float)
        potential = np.zeros((size, size), dtype=float)
        kinetic = np.zeros((size, size), dtype=float)
        diff_rhat_dot_matrix = np.zeros((size, size), dtype=float)
        diff_rhat_grad_12_matrix = np.zeros((size, size), dtype=float)
        diff_rhat_grad_21_matrix = np.zeros((size, size), dtype=float)
        diff_grad_grad_matrix = np.zeros((size, size), dtype=float)

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
                total_diff_rhat_dot = 0.0
                total_diff_rhat_grad_12 = 0.0
                total_diff_rhat_grad_21 = 0.0
                total_diff_grad_grad = 0.0

                for term_row in row_terms:
                    for term_col in col_terms:
                        c_total = term_row.correlation_power + term_col.correlation_power
                        for corr_term in self._iter_correlation_terms(c_total):
                            q = corr_term.q
                            ang = self._angular_components(
                                term_row.channel, term_col.channel, q
                            )
                            if (
                                ang.g1 == 0.0
                                and ang.rhat_dot == 0.0
                                and ang.rhat_grad_12 == 0.0
                                and ang.rhat_grad_21 == 0.0
                                and ang.grad_grad == 0.0
                            ):
                                continue

                            def integrand(point: RadialPointData) -> Tuple[float, float, float, float, float, float, float]:
                                r1 = point.r1
                                r2 = point.r2
                                if r1 <= 0.0 or r2 <= 0.0:
                                    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                                (
                                    radial_factor,
                                    radial_dr1,
                                    radial_dr2,
                                    radial_d2r1,
                                    radial_d2r2,
                                    _,
                                ) = self._radial_factor_components(
                                    r1, r2, c_total, q, corr_term.k
                                )

                                measure = (r1 * r1) * (r2 * r2)

                                row_r1 = point.value_r1[0]
                                row_r2 = point.value_r2[0]
                                col_r1 = point.value_r1[1]
                                col_r2 = point.value_r2[1]

                                phi_row = row_r1 * row_r2
                                phi_col = col_r1 * col_r2

                                if phi_row == 0.0 and phi_col == 0.0 and radial_factor == 0.0:
                                    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                                d_col_r1 = point.d1_r1[1]
                                d_col_r2 = point.d1_r2[1]
                                d2_col_r1 = point.d2_r1[1]
                                d2_col_r2 = point.d2_r2[1]

                                col_dr1 = d_col_r1 * col_r2
                                col_dr2 = col_r1 * d_col_r2
                                col_d2r1 = d2_col_r1 * col_r2
                                col_d2r2 = col_r1 * d2_col_r2

                                col_full = phi_col * radial_factor

                                d_r1_total = col_dr1 * radial_factor + phi_col * radial_dr1
                                d_r2_total = col_dr2 * radial_factor + phi_col * radial_dr2

                                d2_r1_total = (
                                    col_d2r1 * radial_factor
                                    + 2.0 * col_dr1 * radial_dr1
                                    + phi_col * radial_d2r1
                                )
                                d2_r2_total = (
                                    col_d2r2 * radial_factor
                                    + 2.0 * col_dr2 * radial_dr2
                                    + phi_col * radial_d2r2
                                )
                                lap_r1_total = d2_r1_total
                                if r1 > 0.0:
                                    lap_r1_total += (2.0 / r1) * d_r1_total
                                    if term_col.channel.l1 > 0:
                                        lap_r1_total -= (
                                            term_col.channel.l1
                                            * (term_col.channel.l1 + 1)
                                            * col_full
                                            / (r1 * r1)
                                        )
                                    if q > 0:
                                        lap_r1_total -= (
                                            q
                                            * (q + 1)
                                            * col_full
                                            / (r1 * r1)
                                        )

                                lap_r2_total = d2_r2_total
                                if r2 > 0.0:
                                    lap_r2_total += (2.0 / r2) * d_r2_total
                                    if term_col.channel.l2 > 0:
                                        lap_r2_total -= (
                                            term_col.channel.l2
                                            * (term_col.channel.l2 + 1)
                                            * col_full
                                            / (r2 * r2)
                                        )
                                    if q > 0:
                                        lap_r2_total -= (
                                            q
                                            * (q + 1)
                                            * col_full
                                            / (r2 * r2)
                                        )

                                pot_val = self.operators.potential_terms(
                                    r1, r2, max(r1, r2)
                                )

                                overlap_val = measure * radial_factor * phi_row * phi_col
                                potential_val = overlap_val * pot_val
                                kinetic_val = measure * phi_row * (
                                    -(0.5 / self.operators.mu)
                                    * (lap_r1_total + lap_r2_total)
                                )

                                diff_rhat_dot = measure * phi_row * (
                                    d_r1_total * d_r2_total
                                )

                                diff_rhat_grad_12 = 0.0
                                if r2 > 0.0:
                                    diff_rhat_grad_12 = (
                                        measure
                                        * phi_row
                                        * d_r1_total
                                        / r2
                                    )

                                diff_rhat_grad_21 = 0.0
                                if r1 > 0.0:
                                    diff_rhat_grad_21 = (
                                        measure
                                        * phi_row
                                        * d_r2_total
                                        / r1
                                    )

                                diff_grad_grad = 0.0
                                if r1 > 0.0 and r2 > 0.0:
                                    diff_grad_grad = (
                                        measure
                                        * phi_row
                                        * col_full
                                        / (r1 * r2)
                                    )

                                return (
                                    overlap_val,
                                    potential_val,
                                    kinetic_val,
                                    diff_rhat_dot,
                                    diff_rhat_grad_12,
                                    diff_rhat_grad_21,
                                    diff_grad_grad,
                                )

                            (
                                o,
                                v,
                                k_val,
                                diff_rhat_dot,
                                diff_rhat_grad_12,
                                diff_rhat_grad_21,
                                diff_grad_grad,
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
                            total_overlap += coeff_base * ang.g1 * o
                            total_potential += coeff_base * ang.g1 * v
                            total_kinetic += coeff_base * ang.g1 * k_val

                            total_diff_rhat_dot += coeff_base * ang.rhat_dot * diff_rhat_dot
                            total_diff_rhat_grad_12 += coeff_base * ang.rhat_grad_12 * diff_rhat_grad_12
                            total_diff_rhat_grad_21 += coeff_base * ang.rhat_grad_21 * diff_rhat_grad_21
                            total_diff_grad_grad += coeff_base * ang.grad_grad * diff_grad_grad

                overlap[row, col] = total_overlap
                potential[row, col] = total_potential
                kinetic[row, col] = total_kinetic

                diff_rhat_dot_matrix[row, col] = total_diff_rhat_dot
                diff_rhat_grad_12_matrix[row, col] = total_diff_rhat_grad_12
                diff_rhat_grad_21_matrix[row, col] = total_diff_rhat_grad_21
                diff_grad_grad_matrix[row, col] = total_diff_grad_grad

                if row != col:
                    overlap[col, row] = total_overlap
                    potential[col, row] = total_potential
                    kinetic[col, row] = total_kinetic

                    diff_rhat_dot_matrix[col, row] = total_diff_rhat_dot
                    diff_rhat_grad_12_matrix[col,
                                             row] = total_diff_rhat_grad_12
                    diff_rhat_grad_21_matrix[col,
                                             row] = total_diff_rhat_grad_21
                    diff_grad_grad_matrix[col, row] = total_diff_grad_grad

        diff_components = {
            "diff_rhat_dot": diff_rhat_dot_matrix,
            "diff_rhat_grad_12": diff_rhat_grad_12_matrix,
            "diff_rhat_grad_21": diff_rhat_grad_21_matrix,
            "diff_grad_grad": diff_grad_grad_matrix,
        }

        mass_prefactor = -1.0 / self.operators.M
        mass = self.combine_differential_components(
            diff_components,
            rhat_dot=mass_prefactor,
            rhat_grad_12=mass_prefactor,
            rhat_grad_21=mass_prefactor,
            grad_grad=mass_prefactor,
        )

        mass_rhat_dot_matrix = self.combine_differential_components(
            diff_components,
            rhat_dot=mass_prefactor,
        )
        mass_rhat_grad_12_matrix = self.combine_differential_components(
            diff_components,
            rhat_grad_12=mass_prefactor,
        )
        mass_rhat_grad_21_matrix = self.combine_differential_components(
            diff_components,
            rhat_grad_21=mass_prefactor,
        )
        mass_grad_grad_matrix = self.combine_differential_components(
            diff_components,
            grad_grad=mass_prefactor,
        )

        def _symmetrize(matrix: np.ndarray) -> np.ndarray:
            return 0.5 * (matrix + matrix.T)

        mass = _symmetrize(mass)
        mass_rhat_dot_matrix = _symmetrize(mass_rhat_dot_matrix)
        mass_rhat_grad_12_matrix = _symmetrize(mass_rhat_grad_12_matrix)
        mass_rhat_grad_21_matrix = _symmetrize(mass_rhat_grad_21_matrix)
        mass_grad_grad_matrix = _symmetrize(mass_grad_grad_matrix)

        h_total = kinetic + mass + potential
        components = {
            "overlap": overlap,
            "potential": potential,
            "kinetic": kinetic,
            "mass": mass,
            "mass_rhat_dot": mass_rhat_dot_matrix,
            "mass_rhat_grad_12": mass_rhat_grad_12_matrix,
            "mass_rhat_grad_21": mass_rhat_grad_21_matrix,
            "mass_grad_grad": mass_grad_grad_matrix,
        }
        # Raw differential integrals without the nuclear-mass prefactor.
        components.update(diff_components)
        return h_total, overlap, components

    @staticmethod
    def combine_differential_components(
        diff_components: Dict[str, np.ndarray],
        *,
        rhat_dot: float = 0.0,
        rhat_grad_12: float = 0.0,
        rhat_grad_21: float = 0.0,
        grad_grad: float = 0.0,
    ) -> np.ndarray:
        """Linearly combine the differential matrices produced by Eq. (2.70)-(2.74).

        Parameters
        ----------
        diff_components
            Mapping that must contain the raw differential matrices returned by
            :meth:`assemble_matrices` under the keys ``diff_rhat_dot``,
            ``diff_rhat_grad_12``, ``diff_rhat_grad_21`` 与 ``diff_grad_grad``。
        rhat_dot, rhat_grad_12, rhat_grad_21, grad_grad
            权重系数，将分别乘以对应的差分矩阵再累加。

        Returns
        -------
        numpy.ndarray
            与输入矩阵同型的线性组合结果。
        """

        required = (
            "diff_rhat_dot",
            "diff_rhat_grad_12",
            "diff_rhat_grad_21",
            "diff_grad_grad",
        )
        arrays: Dict[str, np.ndarray] = {}
        for name in required:
            if name not in diff_components:
                raise KeyError(f"Missing differential component '{name}'.")
            arrays[name] = diff_components[name]

        result = np.zeros_like(arrays["diff_rhat_dot"], dtype=float)
        if rhat_dot:
            result = result + rhat_dot * arrays["diff_rhat_dot"]
        if rhat_grad_12:
            result = result + rhat_grad_12 * arrays["diff_rhat_grad_12"]
        if rhat_grad_21:
            result = result + rhat_grad_21 * arrays["diff_rhat_grad_21"]
        if grad_grad:
            result = result + grad_grad * arrays["diff_grad_grad"]

        return result

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
