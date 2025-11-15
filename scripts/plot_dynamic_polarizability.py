"""绘制论文第 3.4 节的动力学极化率频率扫描曲线。"""
from __future__ import annotations

import argparse
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
from he_polarization.io import SolverResultCache
from he_polarization.numerics import generate_tensor_product_quadrature
from he_polarization.observables import DynamicPolarizabilityCalculator, EnergyCalculator
from he_polarization.observables.dipole import (
    build_dipole_matrix,
    build_velocity_gauge_matrix,
)
from he_polarization.reporting import plot_dynamic_polarizability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot dynamic polarizability curves with cached spectra support.")
    parser.add_argument("--use-cache", dest="use_cache", action="store_true",
                        default=True, help="Load/save cached spectra when available (default: on).")
    parser.add_argument("--no-cache", dest="use_cache",
                        action="store_false", help="Disable cache usage for this run.")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Force recomputation even if cached data exist.")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"),
                        help="Directory used to store cached solver outputs.")
    parser.add_argument("--cache-key", default="energy_small",
                        help="Cache subdirectory name to use.")
    parser.add_argument("--num-eigenvalues", type=int, default=5,
                        help="Number of lowest eigenpairs to compute when cache is refreshed.")
    parser.add_argument("--freq-max", type=float, default=0.8,
                        help="Upper bound of frequency scan (a.u.).")
    parser.add_argument("--freq-count", type=int, default=40,
                        help="Number of frequency samples in the scan.")
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/dynamic_polarizability_ground.png"), help="Output PNG path for the plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    m_nucleus = 7294.2995365
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

    metadata = {
        "tau": tau,
        "r_max": r_max,
        "k": k,
        "n": n,
        "l_max": l_max,
        "num_eigenvalues": args.num_eigenvalues,
        "correlation_powers": [0, 1],
        "exchange_parity": 1,
        "symmetrize": True,
        "unique_pairs": True,
        "basis_size": len(basis_states),
        "mu": mu,
    }

    cache = SolverResultCache(args.cache_dir)
    if args.use_cache and not args.refresh_cache and cache.available(args.cache_key, metadata=metadata):
        cached = cache.load(args.cache_key)
        energies = cached.energies
        eigenvectors = cached.eigenvectors
        components = cached.components
    else:
        energies, eigenvectors, components = calculator.diagonalize(
            basis_states,
            weights=weights,
            points=points,
            num_eigenvalues=args.num_eigenvalues,
        )

        cache_components = dict(components)
        kinetic = cache_components.get("kinetic")
        potential = cache_components.get("potential")
        mass = cache_components.get("mass")
        if kinetic is not None and potential is not None:
            total_h = kinetic + potential
            if mass is not None:
                total_h = total_h + mass
            cache_components["hamiltonian"] = total_h
        if args.use_cache:
            cache.save(
                args.cache_key,
                energies=energies,
                eigenvectors=eigenvectors,
                components=cache_components,
                metadata=metadata,
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
    freqs = np.linspace(0.0, args.freq_max, args.freq_count)
    length, velocity, acceleration = dynamic_calc.evaluate(
        energies,
        dipole_eigen,
        momentum_eigen,
        state_index=0,
        freqs=freqs,
    )

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_dynamic_polarizability(
        freqs,
        length,
        velocity,
        acceleration,
        title="基态动力学极化率",
        output_path=str(output_path),
    )

    print(f"动力学极化率曲线已保存至 {output_path}")


if __name__ == "__main__":
    main()
