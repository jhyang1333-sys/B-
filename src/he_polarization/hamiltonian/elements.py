"""矩阵元装配，对应论文第 2.4 节。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
from scipy.sparse import coo_matrix

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


@dataclass(frozen=True)
class _PrecomputedTerm:
    corr: CorrelationTerm
    angular: _AngularComponents
    coeff_base: float
    c_power: int


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
    ) -> tuple[coo_matrix, coo_matrix, Dict[str, coo_matrix]]:
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

        overlap_entries: Dict[Tuple[int, int], float] = {}
        potential_entries: Dict[Tuple[int, int], float] = {}
        kinetic_entries: Dict[Tuple[int, int], float] = {}
        mass_entries: Dict[Tuple[int, int], float] = {}

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

                        terms_c = self._precompute_terms(
                            term_row, term_col, c_total)
                        terms_c_minus2 = self._precompute_terms(
                            term_row, term_col, c_total - 2)
                        terms_c_minus1 = self._precompute_terms(
                            term_row, term_col, c_total - 1)
                        terms_c_plus2 = self._precompute_terms(
                            term_row, term_col, c_total + 2)

                        if (
                            not terms_c
                            and not terms_c_minus2
                            and not terms_c_minus1
                            and not terms_c_plus2
                        ):
                            continue

                        mu = self.operators.mu
                        M = self.operators.M
                        pref_second = -0.5 / mu if mu != 0.0 else 0.0
                        pref_first = -1.0 / mu if mu != 0.0 else 0.0
                        pref_centrifugal = 0.5 / mu if mu != 0.0 else 0.0
                        pref_cross = -0.25 / M if M != 0.0 else 0.0
                        pref_mu_first_extra = pref_second
                        pref_mass_first_extra = 0.5 / M if M != 0.0 else 0.0
                        pref_mu_mix = pref_second
                        pref_mass_mix = pref_mass_first_extra
                        pref_mu_j = 1.0 / mu if mu != 0.0 else 0.0
                        pref_mass_j = -1.0 / M if M != 0.0 else 0.0
                        scalar_pref_nonzero = (
                            pref_mu_mix != 0.0 or pref_mass_mix != 0.0
                        )

                        l1 = term_col.channel.l1
                        l2 = term_col.channel.l2

                        def integrand(point: RadialPointData) -> Tuple[float, float, float, float, float]:
                            r1 = point.r1
                            r2 = point.r2
                            if r1 <= 0.0 or r2 <= 0.0:
                                return (0.0, 0.0, 0.0, 0.0, 0.0)

                            row_r1 = point.value_r1[0]
                            row_r2 = point.value_r2[0]
                            col_r1 = point.value_r1[1]
                            col_r2 = point.value_r2[1]

                            phi_row = row_r1 * row_r2
                            if phi_row == 0.0:
                                return (0.0, 0.0, 0.0, 0.0, 0.0)

                            d_col_r1 = point.d1_r1[1]
                            d_col_r2 = point.d1_r2[1]
                            d2_col_r1 = point.d2_r1[1]
                            d2_col_r2 = point.d2_r2[1]

                            col_dr1 = d_col_r1 * col_r2
                            col_dr2 = col_r1 * d_col_r2
                            col_d2r1 = d2_col_r1 * col_r2
                            col_d2r2 = col_r1 * d2_col_r2

                            phi_col = col_r1 * col_r2

                            measure = (r1 * r1) * (r2 * r2)
                            potential_pref = self.operators.potential_terms(
                                r1, r2)

                            ratio_mix_r1 = (
                                (r1 * r1 - r2 * r2) / r1 if r1 != 0.0 else 0.0
                            )
                            ratio_mix_r2 = (
                                (r2 * r2 - r1 * r1) / r2 if r2 != 0.0 else 0.0
                            )
                            ratio_cross = (
                                (r1 * r1 + r2 * r2) / (r1 * r2)
                                if r1 != 0.0 and r2 != 0.0
                                else 0.0
                            )
                            ratio_j1 = r1 / r2 if r2 != 0.0 else 0.0
                            ratio_j2 = r2 / r1 if r1 != 0.0 else 0.0

                            derivative_product = col_dr1 * col_dr2

                            total_ol = 0.0
                            total_po = 0.0
                            total_ki = 0.0
                            total_ma = 0.0
                            total_ve = 0.0

                            for term in terms_c:
                                radial_factor, _, _, _, _, _ = self._radial_factor_components(
                                    r1, r2, term.c_power, term.corr.q, term.corr.k
                                )
                                if radial_factor == 0.0:
                                    continue

                                coeff = term.coeff_base
                                ang = term.angular
                                measure_factor = measure * radial_factor

                                g1 = ang.g1
                                if g1 != 0.0:
                                    coeff_g1 = coeff * g1

                                    if phi_col != 0.0:
                                        base_overlap = measure_factor * phi_row * phi_col
                                        total_ol += coeff_g1 * base_overlap
                                        total_po += coeff_g1 * base_overlap * potential_pref

                                        if r1 != 0.0:
                                            total_ki += (
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
                                            total_ki += (
                                                coeff_g1
                                                * pref_centrifugal
                                                * l2
                                                * (l2 + 1)
                                                * measure_factor
                                                * phi_row
                                                * phi_col
                                                / (r2 * r2)
                                            )

                                    if col_d2r1 != 0.0:
                                        total_ki += (
                                            coeff_g1
                                            * pref_second
                                            * measure_factor
                                            * phi_row
                                            * col_d2r1
                                        )
                                    if col_d2r2 != 0.0:
                                        total_ki += (
                                            coeff_g1
                                            * pref_second
                                            * measure_factor
                                            * phi_row
                                            * col_d2r2
                                        )
                                    if col_dr1 != 0.0 and r1 != 0.0:
                                        total_ki += (
                                            coeff_g1
                                            * pref_first
                                            * measure_factor
                                            * phi_row
                                            * col_dr1
                                            / r1
                                        )
                                    if col_dr2 != 0.0 and r2 != 0.0:
                                        total_ki += (
                                            coeff_g1
                                            * pref_first
                                            * measure_factor
                                            * phi_row
                                            * col_dr2
                                            / r2
                                        )

                                    if c_col != 0 and (col_dr1 != 0.0 or col_dr2 != 0.0):
                                        factor_c = coeff_g1 * c_col
                                        if col_dr1 != 0.0 and r1 != 0.0:
                                            total_ki += (
                                                factor_c
                                                * pref_mu_first_extra
                                                * measure_factor
                                                * phi_row
                                                * col_dr1
                                                / r1
                                            )
                                            total_ma += (
                                                factor_c
                                                * pref_mass_first_extra
                                                * measure_factor
                                                * phi_row
                                                * col_dr1
                                                / r1
                                            )
                                        if col_dr2 != 0.0 and r2 != 0.0:
                                            total_ki += (
                                                factor_c
                                                * pref_mu_first_extra
                                                * measure_factor
                                                * phi_row
                                                * col_dr2
                                                / r2
                                            )
                                            total_ma += (
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
                                        total_ma += (
                                            coeff_g1
                                            * pref_cross
                                            * measure_factor
                                            * phi_row
                                            * derivative_product
                                            * ratio_cross
                                        )

                                if ang.rhat_grad_12 != 0.0 and col_dr1 != 0.0 and r2 != 0.0:
                                    total_ma += (
                                        coeff
                                        * ang.rhat_grad_12
                                        * (-1.0 / M if M != 0.0 else 0.0)
                                        * measure_factor
                                        * phi_row
                                        * col_dr1
                                        / r2
                                    )

                                if ang.rhat_grad_21 != 0.0 and col_dr2 != 0.0 and r1 != 0.0:
                                    total_ma += (
                                        coeff
                                        * ang.rhat_grad_21
                                        * (-1.0 / M if M != 0.0 else 0.0)
                                        * measure_factor
                                        * phi_row
                                        * col_dr2
                                        / r1
                                    )

                                if ang.grad_grad != 0.0 and phi_col != 0.0 and r1 != 0.0 and r2 != 0.0:
                                    total_ma += (
                                        coeff
                                        * ang.grad_grad
                                        * (-1.0 / M if M != 0.0 else 0.0)
                                        * measure_factor
                                        * phi_row
                                        * phi_col
                                        / (r1 * r2)
                                    )

                            if c_col != 0:
                                for term in terms_c_minus2:
                                    radial_factor, _, _, _, _, _ = self._radial_factor_components(
                                        r1, r2, term.c_power, term.corr.q, term.corr.k
                                    )
                                    if radial_factor == 0.0:
                                        continue

                                    coeff = term.coeff_base * c_col
                                    ang = term.angular
                                    measure_factor = measure * radial_factor

                                    if ang.g1 != 0.0 and scalar_pref_nonzero:
                                        if col_dr1 != 0.0 and ratio_mix_r1 != 0.0:
                                            base_mix_r1 = (
                                                measure_factor
                                                * phi_row
                                                * col_dr1
                                                * ratio_mix_r1
                                            )
                                            total_ki += coeff * ang.g1 * pref_mu_mix * base_mix_r1
                                            total_ma += coeff * ang.g1 * pref_mass_mix * base_mix_r1

                                        if col_dr2 != 0.0 and ratio_mix_r2 != 0.0:
                                            base_mix_r2 = (
                                                measure_factor
                                                * phi_row
                                                * col_dr2
                                                * ratio_mix_r2
                                            )
                                            total_ki += coeff * ang.g1 * pref_mu_mix * base_mix_r2
                                            total_ma += coeff * ang.g1 * pref_mass_mix * base_mix_r2

                                    if (
                                        (ang.rhat_grad_12 !=
                                         0.0 or ang.rhat_grad_21 != 0.0)
                                        and phi_col != 0.0
                                    ):
                                        if ang.rhat_grad_12 != 0.0 and ratio_j1 != 0.0:
                                            base_j1 = (
                                                measure_factor
                                                * phi_row
                                                * phi_col
                                                * ratio_j1
                                            )
                                            total_ki += coeff * pref_mu_j * ang.rhat_grad_12 * base_j1
                                            total_ma += coeff * pref_mass_j * ang.rhat_grad_12 * base_j1

                                        if ang.rhat_grad_21 != 0.0 and ratio_j2 != 0.0:
                                            base_j2 = (
                                                measure_factor
                                                * phi_row
                                                * phi_col
                                                * ratio_j2
                                            )
                                            total_ki += coeff * pref_mu_j * ang.rhat_grad_21 * base_j2
                                            total_ma += coeff * pref_mass_j * ang.rhat_grad_21 * base_j2

                                    if ang.g1 != 0.0 and c_col > 0 and phi_col != 0.0:
                                        base_r12 = measure_factor * phi_row * phi_col
                                        coeff_mu = (-2.0 / mu) * \
                                            c_col if mu != 0.0 else 0.0
                                        coeff_mass = (2.0 / M) * \
                                            c_col if M != 0.0 else 0.0
                                        if c_col >= 2:
                                            if mu != 0.0:
                                                coeff_mu += (
                                                    -1.0
                                                    / mu
                                                    * c_col
                                                    * (c_col - 1)
                                                )
                                            if M != 0.0:
                                                coeff_mass += (
                                                    1.0
                                                    / M
                                                    * c_col
                                                    * (c_col - 1)
                                                )
                                        if coeff_mu != 0.0:
                                            total_ki += coeff * ang.g1 * coeff_mu * base_r12
                                        if coeff_mass != 0.0:
                                            total_ma += coeff * ang.g1 * coeff_mass * base_r12

                                if scalar_pref_nonzero and terms_c:
                                    for term in terms_c:
                                        radial_factor, _, _, _, _, _ = self._radial_factor_components(
                                            r1, r2, term.c_power, term.corr.q, term.corr.k
                                        )
                                        if radial_factor == 0.0:
                                            continue

                                        ang = term.angular
                                        if ang.g1 == 0.0:
                                            continue

                                        coeff = term.coeff_base * c_col
                                        measure_factor = measure * radial_factor

                                        if col_dr1 != 0.0 and r1 != 0.0:
                                            base_mix_r1_zero = (
                                                measure_factor
                                                * phi_row
                                                * col_dr1
                                                / r1
                                            )
                                            total_ki += coeff * ang.g1 * pref_mu_mix * base_mix_r1_zero
                                            total_ma += coeff * ang.g1 * pref_mass_mix * base_mix_r1_zero

                                        if col_dr2 != 0.0 and r2 != 0.0:
                                            base_mix_r2_zero = (
                                                measure_factor
                                                * phi_row
                                                * col_dr2
                                                / r2
                                            )
                                            total_ki += coeff * ang.g1 * pref_mu_mix * base_mix_r2_zero
                                            total_ma += coeff * ang.g1 * pref_mass_mix * base_mix_r2_zero

                            for term in terms_c_plus2:
                                radial_factor, _, _, _, _, _ = self._radial_factor_components(
                                    r1, r2, term.c_power, term.corr.q, term.corr.k
                                )
                                if (
                                    radial_factor == 0.0
                                    or term.angular.g1 == 0.0
                                    or derivative_product == 0.0
                                    or r1 == 0.0
                                    or r2 == 0.0
                                ):
                                    continue

                                measure_factor = measure * radial_factor
                                total_ma += (
                                    term.coeff_base
                                    * term.angular.g1
                                    * pref_cross
                                    * measure_factor
                                    * phi_row
                                    * derivative_product
                                    * (-1.0 / (r1 * r2))
                                )

                            for term in terms_c_minus1:
                                radial_factor, _, _, _, _, _ = self._radial_factor_components(
                                    r1, r2, term.c_power, term.corr.q, term.corr.k
                                )
                                if (
                                    radial_factor == 0.0
                                    or term.angular.g1 == 0.0
                                    or phi_col == 0.0
                                ):
                                    continue

                                total_ve += (
                                    term.coeff_base
                                    * term.angular.g1
                                    * measure
                                    * radial_factor
                                    * phi_row
                                    * phi_col
                                )

                            return total_ol, total_po, total_ki, total_ma, total_ve

                        (
                            overlap_val,
                            potential_val,
                            kinetic_val,
                            mass_val,
                            vee_val,
                        ) = quadrature.integrate(
                            term_row.radial_indices,
                            term_col.radial_indices,
                            integrand,
                        )

                        total_overlap += overlap_val
                        total_potential += potential_val
                        total_kinetic_mu += kinetic_val
                        total_mass += mass_val
                        total_vee += vee_val

                potential_total = total_potential + total_vee
                overlap_entries[(row, col)] = total_overlap
                potential_entries[(row, col)] = potential_total
                kinetic_entries[(row, col)] = total_kinetic_mu
                mass_entries[(row, col)] = total_mass

                if row != col:
                    overlap_entries[(col, row)] = total_overlap
                    potential_entries[(col, row)] = potential_total
                    kinetic_entries[(col, row)] = total_kinetic_mu
                    mass_entries[(col, row)] = total_mass

        def _to_sparse(entries: Dict[Tuple[int, int], float]) -> coo_matrix:
            if not entries:
                return coo_matrix((size, size))
            rows, cols, data = zip(*((i, j, v)
                                   for (i, j), v in entries.items()))
            return coo_matrix((data, (rows, cols)), shape=(size, size))

        overlap = _to_sparse(overlap_entries)
        potential = _to_sparse(potential_entries)
        kinetic = _to_sparse(kinetic_entries)
        mass = _to_sparse(mass_entries)

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
