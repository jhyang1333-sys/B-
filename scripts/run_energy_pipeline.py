"""串联论文第 3.2 节的能量计算流程。"""
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
from he_polarization.observables import EnergyCalculator
from he_polarization.observables.expectation import expectation_from_matrix


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

    channel = AtomicChannel(l1=0, l2=0, L=0)
    basis_states = generate_hylleraas_bspline_functions(
        bspline,
        [channel],
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

    ground_vec = eigenvectors[:, 0]
    kinetic_expect = expectation_from_matrix(ground_vec, components["kinetic"])
    potential_expect = expectation_from_matrix(
        ground_vec, components["potential"])
    eta = calculator.hellmann_eta(
        expect_T=kinetic_expect, expect_V=potential_expect)

    print("最低几个能级 (a.u.):")
    for idx, energy in enumerate(energies[:5]):
        print(f"  E_{idx} = {energy:.8f}")

    print(f"Hellmann 判据 η = {eta:.3e}")


if __name__ == "__main__":
    main()
