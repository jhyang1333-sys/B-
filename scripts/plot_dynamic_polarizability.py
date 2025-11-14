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
from he_polarization.observables.dipole import build_dipole_matrix
from he_polarization.reporting import plot_dynamic_polarizability


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
    energies, eigenvectors, _ = calculator.diagonalize(
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

    dynamic_calc = DynamicPolarizabilityCalculator(
        energy_calculator=calculator)
    freqs = np.linspace(0.0, 0.8, 40)
    length, velocity, acceleration = dynamic_calc.evaluate(
        energies, dipole_eigen, state_index=0, freqs=freqs)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "dynamic_polarizability_ground.png"
    plot_dynamic_polarizability(
        freqs, length, velocity, acceleration, title="基态动力学极化率", output_path=str(plot_path))

    print(f"动力学极化率曲线已保存至 {plot_path}")


if __name__ == "__main__":
    main()
