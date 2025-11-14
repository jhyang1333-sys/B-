"""计算偶极矩阵元。"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, NamedTuple, Sequence, Tuple, Union

import numpy as np

from he_polarization.basis.bspline import BSplineBasis
from he_polarization.basis.channels import AtomicChannel
from he_polarization.basis.angular import AngularCoupling
from he_polarization.basis.functions import HylleraasBSplineFunction
from he_polarization.numerics import RadialQuadrature2D, RadialPointData
from he_polarization.basis.correlation import CorrelationExpansion, CorrelationTerm


class _DipoleBasisTerm(NamedTuple):
    coeff: float
    radial_indices: Tuple[int, int]
    channel: AtomicChannel
    correlation_power: int


def _safe_power(base: float, exponent: int) -> float:
    if exponent == 0:
        return 1.0
    if base <= 0.0:
        if exponent > 0:
            return 0.0
        return 0.0
    try:
        return base ** exponent
    except (ZeroDivisionError, OverflowError):
        return 0.0


def _radial_power_factor(r1: float, r2: float, c_total: int, q: int, k_index: int) -> float:
    r_less = min(r1, r2)
    r_greater = max(r1, r2)
    exp_less = q + 2 * k_index
    exp_greater = c_total - q - 2 * k_index
    return _safe_power(r_less, exp_less) * _safe_power(r_greater, exp_greater)


def _expand_state(state: HylleraasBSplineFunction) -> Tuple[_DipoleBasisTerm, ...]:
    """展开对称化的基函数，匹配哈密顿矩阵装配逻辑。"""

    primary = _DipoleBasisTerm(
        1.0,
        state.radial_indices,
        state.channel,
        state.correlation_power,
    )

    if not state.symmetrized:
        return (primary,)

    swapped = state.swapped()
    parity = float(state.exchange_parity)

    if (
        swapped.radial_indices == state.radial_indices
        and swapped.channel == state.channel
    ):
        combined = primary.coeff + parity
        if abs(combined) < 1e-12:
            return tuple()
        return (
            _DipoleBasisTerm(
                combined,
                state.radial_indices,
                state.channel,
                state.correlation_power,
            ),
        )

    return (
        primary,
        _DipoleBasisTerm(
            parity,
            swapped.radial_indices,
            swapped.channel,
            swapped.correlation_power,
        ),
    )


def build_dipole_matrix(
    bspline: BSplineBasis,
    basis_states: Iterable[
        Union[
            HylleraasBSplineFunction,
            Tuple[Tuple[int, int], AtomicChannel]
        ]
    ],
    angular: AngularCoupling,
    *,
    correlation: CorrelationExpansion | None = None,
    weights: Iterable[float],
    points: Iterable[Tuple[float, float]],
) -> np.ndarray:
    """构建偶极矩阵 ``<φ_i| r_1 + r_2 |φ_j>``，包含角向耦合。"""

    _ = weights, points  # 外部节点保留以兼容旧接口

    corr = correlation or CorrelationExpansion()

    states = [
        state
        if isinstance(state, HylleraasBSplineFunction)
        else HylleraasBSplineFunction(radial_indices=state[0], channel=state[1])
        for state in basis_states
    ]
    expanded_states: Sequence[Tuple[_DipoleBasisTerm, ...]] = tuple(
        _expand_state(state) for state in states
    )
    size = len(states)
    dipole = np.zeros((size, size), dtype=float)

    quadrature = RadialQuadrature2D(bspline=bspline, order=8)

    correlation_cache: Dict[int, Tuple[CorrelationTerm, ...]] = {}

    for row, row_terms in enumerate(expanded_states):
        for col in range(row, size):
            col_terms = expanded_states[col]
            total = 0.0

            for term_row in row_terms:
                for term_col in col_terms:
                    c_total = term_row.correlation_power + term_col.correlation_power
                    corr_terms = correlation_cache.get(c_total)
                    if corr_terms is None:
                        corr_terms = tuple(corr.iter_terms(c_total))
                        correlation_cache[c_total] = corr_terms

                    for corr_term in corr_terms:
                        q = corr_term.q
                        ang_r1 = angular.angular_tensor_ry(
                            acting_on=1,
                            l=1,
                            m=0,
                            l1=term_row.channel.l1,
                            l2=term_row.channel.l2,
                            L=term_row.channel.L,
                            M=term_row.channel.M,
                            l1p=term_col.channel.l1,
                            l2p=term_col.channel.l2,
                            Lp=term_col.channel.L,
                            Mp=term_col.channel.M,
                            q=q,
                        )
                        ang_r2 = angular.angular_tensor_ry(
                            acting_on=2,
                            l=1,
                            m=0,
                            l1=term_row.channel.l1,
                            l2=term_row.channel.l2,
                            L=term_row.channel.L,
                            M=term_row.channel.M,
                            l1p=term_col.channel.l1,
                            l2p=term_col.channel.l2,
                            Lp=term_col.channel.L,
                            Mp=term_col.channel.M,
                            q=q,
                        )

                        if abs(ang_r1) < 1e-14 and abs(ang_r2) < 1e-14:
                            continue

                        def integrand(point: RadialPointData) -> Tuple[float, float]:
                            r1 = point.r1
                            r2 = point.r2
                            if r1 <= 0.0 or r2 <= 0.0:
                                return 0.0, 0.0

                            phi_row = point.value_r1[0] * point.value_r2[0]
                            phi_col = point.value_r1[1] * point.value_r2[1]
                            if phi_row == 0.0 or phi_col == 0.0:
                                return 0.0, 0.0

                            radial_factor = _radial_power_factor(
                                r1, r2, c_total, q, corr_term.k
                            )
                            if radial_factor == 0.0:
                                return 0.0, 0.0

                            measure = (r1 * r1) * (r2 * r2)
                            base = measure * radial_factor * phi_row * phi_col
                            return base * r1, base * r2

                        integral_r1, integral_r2 = quadrature.integrate(
                            term_row.radial_indices,
                            term_col.radial_indices,
                            integrand,
                        )

                        coeff = (
                            term_row.coeff
                            * term_col.coeff
                            * corr_term.coefficient
                        )
                        total += coeff * (
                            ang_r1 * integral_r1 + ang_r2 * integral_r2
                        )

            dipole[row, col] = total
            if row != col:
                dipole[col, row] = total

    return dipole


def build_velocity_gauge_matrix(
    dipole_matrix: np.ndarray,
    components: Mapping[str, np.ndarray],
    *,
    reduced_mass: float = 1.0,
) -> np.ndarray:
    """Construct the velocity-gauge momentum operator via ``p = i μ [H, R]``.

    Parameters
    ----------
    dipole_matrix
        Length-gauge dipole operator in the configuration basis.
    components
        Matrix components returned by :class:`MatrixElementBuilder`; these
        must include the ``"kinetic"``, ``"mass"`` and ``"potential"``
        blocks produced during Hamiltonian assembly.

    reduced_mass
        Reduced mass ``μ`` appearing in ``T = p^2 / (2μ)``. Defaults to
        atomic-unit electron mass.

    Returns
    -------
    numpy.ndarray
        Velocity-gauge momentum operator matrix corresponding to论文式
        ``i μ [H, R]``.
    """

    kinetic = components.get("kinetic")
    if kinetic is None:
        raise KeyError(
            "Kinetic matrix is required to build the velocity gauge operator.")

    potential = components.get("potential")
    if potential is None:
        raise KeyError(
            "Potential matrix is required to build the velocity gauge operator.")

    mass = components.get("mass")

    hamiltonian = kinetic.copy()
    if mass is not None:
        hamiltonian = hamiltonian + mass
    hamiltonian = hamiltonian + potential

    commutator = hamiltonian @ dipole_matrix - dipole_matrix @ hamiltonian
    return 1j * reduced_mass * commutator
