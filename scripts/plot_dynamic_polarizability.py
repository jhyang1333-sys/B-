"""绘制论文第 3.4 节的动力学极化率频率扫描曲线。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

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
from he_polarization.observables import DynamicPolarizabilityCalculator, EnergyCalculator
from he_polarization.observables.dipole import (
    build_dipole_matrix,
    build_velocity_gauge_matrix,
)
from he_polarization.reporting import plot_dynamic_polarizability


def main() -> None:
    tau = 0.038
    r_max = 20.0
    k = 3
    n = 8
    l_max = 1

    config = ExponentialNodeConfig(
        r_min=0.0, r_max=r_max, k=k, n=n, gamma=r_max * tau)
    knots = generate_exponential_nodes(config)
    bspline = BSplineBasis(knots=knots, order=k)

    angular = AngularCoupling()
    m_nucleus = 1836.15267389
    mu = m_nucleus / (1.0 + m_nucleus)
    operators = HamiltonianOperators(mu=mu, M=m_nucleus)
    builder = MatrixElementBuilder(
        bspline=bspline, angular=angular, operators=operators)

    channels = [
        AtomicChannel(l1=l1, l2=l2, L=L)
        for l1 in range(l_max + 1)
        for l2 in range(l_max + 1)
        for L in range(abs(l1 - l2), min(l1 + l2, l_max) + 1)
    ]
    basis_states = generate_hylleraas_bspline_functions(
        bspline,
        channels,
        n_radial=bspline.n_basis,
        correlation_powers=(0, 1),
        exchange_parity=1,
        symmetrize=True,
        unique_pairs=True,
    )
    points, weights = generate_tensor_product_quadrature(
        config.r_min, config.r_max, n_points=8)

    calculator = EnergyCalculator(builder=builder)
    energies, eigenvectors, components = calculator.diagonalize(
        basis_states,
        weights=weights,
        points=points,
        num_eigenvalues=5,
    )

    dipole_basis = build_dipole_matrix(
        bspline,
        basis_states,
        angular,
        correlation=builder.correlation,
        weights=weights,
        points=points,
    )
    momentum_basis = build_velocity_gauge_matrix(
        dipole_basis,
        components,
        reduced_mass=builder.operators.mu,
    )

    dipole_eigen = eigenvectors.conj().T @ dipole_basis @ eigenvectors
    momentum_eigen = eigenvectors.conj().T @ momentum_basis @ eigenvectors

    dynamic_calc = DynamicPolarizabilityCalculator(
        energy_calculator=calculator)
    freqs = np.linspace(0.0, 0.8, 40)
    length, velocity, acceleration = dynamic_calc.evaluate(
        energies,
        dipole_eigen,
        momentum_eigen,
        state_index=0,
        freqs=freqs,
    )

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "dynamic_polarizability_ground.png"
    plot_dynamic_polarizability(
        freqs, length, velocity, acceleration, title="基态动力学极化率", output_path=str(plot_path))

    print(f"动力学极化率曲线已保存至 {plot_path}")


if __name__ == "__main__":
    main()
