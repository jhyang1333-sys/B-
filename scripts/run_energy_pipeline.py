"""串联论文第 3.2 节的能量计算流程。"""
from __future__ import annotations

import argparse
from pathlib import Path

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
from he_polarization.observables import EnergyCalculator
from he_polarization.observables.expectation import expectation_from_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute helium energy spectrum with optional caching.")
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
                        help="Number of lowest eigenpairs to compute (and cache).")
    parser.add_argument("--progress", action="store_true",
                        help="Display a progress bar during matrix assembly.")
    parser.add_argument("--assembly-workers", type=int, default=None,
                        help="Number of worker processes for matrix assembly (default: auto).")
    parser.add_argument("--assembly-chunk-rows", type=int, default=None,
                        help="Number of consecutive matrix rows per worker chunk.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tau = 0.038
    r_max = 20.0
    k = 7
    n = 20
    l_max = 2

    config = ExponentialNodeConfig(
        r_min=0.0, r_max=r_max, k=k, n=n, gamma=r_max * tau)
    knots = generate_exponential_nodes(config)
    bspline = BSplineBasis(knots=knots, order=k)

    angular = AngularCoupling()
    # Use the 4He nucleus-to-electron mass ratio (CODATA 2022) for the reduced mass.
    m_nucleus = 7294.2995365
    mu = m_nucleus / (1.0 + m_nucleus)
    operators = HamiltonianOperators(mu=mu, M=m_nucleus)
    builder = MatrixElementBuilder(
        bspline=bspline,
        angular=angular,
        operators=operators,
        max_workers=args.assembly_workers,
        rows_per_chunk=args.assembly_chunk_rows,
    )

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
            progress="矩阵装配进度" if args.progress else None,
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
