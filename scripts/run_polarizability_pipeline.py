"""串联论文第 3.3 节的静态极化率计算与误差评估。"""
from __future__ import annotations

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
from he_polarization.observables import EnergyCalculator, StaticPolarizabilityCalculator
from he_polarization.observables.dipole import build_dipole_matrix


def main() -> None:
    tau = 0.038
    r_max = 200.0
    k = 5
    n = 15

    config = ExponentialNodeConfig(
        r_min=0.0, r_max=r_max, k=k, n=n, gamma=r_max * tau)
    knots = generate_exponential_nodes(config)
    bspline = BSplineBasis(knots=knots, order=k)

    angular = AngularCoupling()
    operators = HamiltonianOperators(mu=1.0, M=1836.15267389)
    builder = MatrixElementBuilder(
        bspline=bspline, angular=angular, operators=operators)

    channels = [
        AtomicChannel(l1=0, l2=0, L=0),
        AtomicChannel(l1=1, l2=0, L=1),
        AtomicChannel(l1=0, l2=1, L=1),
    ]
    basis_states = generate_hylleraas_bspline_functions(
        bspline,
        channels,
        n_radial=bspline.n_basis,
        correlation_powers=(0,),
        exchange_parity=1,
        symmetrize=False,
        unique_pairs=False,
    )
    points, weights = generate_tensor_product_quadrature(
        config.r_min, config.r_max, n_points=8)

    calculator = EnergyCalculator(builder=builder)
    energies, eigenvectors, components = calculator.diagonalize(
        basis_states, weights=weights, points=points)

    dipole_basis = build_dipole_matrix(
        bspline,
        basis_states,
        angular,
        correlation=builder.correlation,
        weights=weights,
        points=points,
    )
    dipole_eigen = eigenvectors.conj().T @ dipole_basis @ eigenvectors

    polar_calc = StaticPolarizabilityCalculator(energy_calculator=calculator)
    for state_idx in range(3):
        alpha_len = polar_calc.compute_length_gauge(
            energies, dipole_eigen, state_idx)
        alpha_vel = polar_calc.compute_velocity_gauge(
            energies, dipole_eigen, state_idx)
        eta = polar_calc.relative_difference(alpha_len, alpha_vel)
        print(
            f"态 {state_idx}: α_L = {alpha_len:.6e}, α_V = {alpha_vel:.6e}, η = {eta:.3e}")


if __name__ == "__main__":
    main()
