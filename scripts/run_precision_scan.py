"""批量扫描参数集以逼近论文精度。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

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
from he_polarization.observables import EnergyCalculator
from he_polarization.observables.expectation import expectation_from_matrix
from he_polarization.solver import ChannelOrthogonalizer, OverlapConditioner


@dataclass
class StageConfig:
    name: str
    tau: float
    r_max: float
    k: int
    n: int
    l_max: int
    n_radial: int | None = None
    correlation_powers: Sequence[int] = (0, 1)
    exchange_parity: int = 1
    symmetrize: bool = True
    unique_pairs: bool = True
    quadrature_points: int = 8
    num_eigenvalues: int = 5
    reference_state_index: int = 0
    reference_energy: float | None = None
    target_abs_error: float | None = None
    channel_ortho_tol: float = 1e-10
    channel_ortho_max_dim: int = 512
    overlap_tol: float = 1e-10
    overlap_max_dim: int = 4096
    overlap_mode: str = "auto"
    overlap_regularization: float = 1e-8

    metadata: dict = field(default_factory=dict)


STAGES: dict[str, StageConfig] = {
    "baseline": StageConfig(
        name="baseline",
        tau=0.038,
        r_max=20.0,
        k=5,
        n=5,
        l_max=2,
        reference_energy=-2.903724377034119,
        target_abs_error=5e-4,
    ),
    "extended": StageConfig(
        name="extended",
        tau=0.030,
        r_max=40.0,
        k=6,
        n=10,
        l_max=3,
        channel_ortho_tol=5e-11,
        channel_ortho_max_dim=1024,
        overlap_tol=5e-11,
        overlap_max_dim=12000,
        reference_energy=-2.903724377034119,
        target_abs_error=5e-5,
    ),
    "final": StageConfig(
        name="final",
        tau=0.025,
        r_max=80.0,
        k=7,
        n=18,
        l_max=4,
        channel_ortho_tol=1e-10,
        channel_ortho_max_dim=2048,
        overlap_tol=1e-10,
        overlap_max_dim=20000,
        overlap_mode="regularize",
        overlap_regularization=1e-7,
        reference_energy=-2.903724377034119,
        target_abs_error=5e-6,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan predefined parameter sets and compare against reference precision.")
    parser.add_argument("--stage", action="append", choices=sorted(STAGES.keys()),
                        help="Stage names to run (default: all). Can be supplied multiple times.")
    parser.add_argument("--use-cache", dest="use_cache", action="store_true", default=True,
                        help="Load/save cached spectra when available (default: on).")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false",
                        help="Disable cache usage for this run.")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Force recomputation even if cached data exist.")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"),
                        help="Directory used to store cached solver outputs.")
    parser.add_argument("--progress", action="store_true",
                        help="Display a progress bar during matrix assembly.")
    return parser.parse_args()


def _build_channels(l_max: int) -> list[AtomicChannel]:
    channels: list[AtomicChannel] = []
    for l1 in range(l_max + 1):
        for l2 in range(l_max + 1):
            for L in range(abs(l1 - l2), min(l1 + l2, l_max) + 1):
                channels.append(AtomicChannel(l1=l1, l2=l2, L=L))
    return channels


def _run_stage(stage: StageConfig, *, use_cache: bool, refresh_cache: bool, cache: SolverResultCache,
               show_progress: bool) -> dict:
    config = ExponentialNodeConfig(
        r_min=0.0,
        r_max=stage.r_max,
        k=stage.k,
        n=stage.n,
        gamma=stage.r_max * stage.tau,
    )
    knots = generate_exponential_nodes(config)
    bspline = BSplineBasis(knots=knots, order=stage.k)

    angular = AngularCoupling()
    m_nucleus = 7294.2995365
    mu = m_nucleus / (1.0 + m_nucleus)
    operators = HamiltonianOperators(mu=mu, M=m_nucleus)
    builder = MatrixElementBuilder(
        bspline=bspline, angular=angular, operators=operators)

    channels = _build_channels(stage.l_max)
    n_radial = stage.n_radial or bspline.n_basis
    basis_states = generate_hylleraas_bspline_functions(
        bspline,
        channels,
        n_radial=n_radial,
        correlation_powers=tuple(stage.correlation_powers),
        exchange_parity=stage.exchange_parity,
        symmetrize=stage.symmetrize,
        unique_pairs=stage.unique_pairs,
    )

    points, weights = generate_tensor_product_quadrature(
        config.r_min, config.r_max, n_points=stage.quadrature_points)

    channel_ortho: ChannelOrthogonalizer | None = ChannelOrthogonalizer(
        tolerance=stage.channel_ortho_tol,
        max_block_dim=stage.channel_ortho_max_dim,
    )

    overlap_conditioner: OverlapConditioner | None = OverlapConditioner(
        tolerance=stage.overlap_tol,
        max_dense_dim=stage.overlap_max_dim,
        mode=stage.overlap_mode,
        regularization=stage.overlap_regularization,
    )

    calculator = EnergyCalculator(
        builder=builder,
        channel_orthogonalizer=channel_ortho,
        overlap_conditioner=overlap_conditioner,
    )

    metadata = {
        "stage": stage.name,
        "tau": stage.tau,
        "r_max": stage.r_max,
        "k": stage.k,
        "n": stage.n,
        "l_max": stage.l_max,
        "n_radial": n_radial,
        "correlation_powers": list(stage.correlation_powers),
        "exchange_parity": stage.exchange_parity,
        "symmetrize": stage.symmetrize,
        "unique_pairs": stage.unique_pairs,
        "num_eigenvalues": stage.num_eigenvalues,
        "channel_ortho_tol": stage.channel_ortho_tol,
        "channel_ortho_max_dim": stage.channel_ortho_max_dim,
        "overlap_tol": stage.overlap_tol,
        "overlap_max_dim": stage.overlap_max_dim,
        "overlap_mode": stage.overlap_mode,
        "overlap_regularization": stage.overlap_regularization,
    }

    cache_key = f"precision_{stage.name}"
    if use_cache and not refresh_cache and cache.available(cache_key, metadata=metadata):
        cached = cache.load(cache_key)
        energies = cached.energies
        eigenvectors = cached.eigenvectors
        components = cached.components
    else:
        energies, eigenvectors, components = calculator.diagonalize(
            basis_states,
            weights=weights,
            points=points,
            num_eigenvalues=stage.num_eigenvalues,
            progress=f"{stage.name} 装配" if show_progress else None,
        )
        if use_cache:
            cache.save(
                cache_key,
                energies=energies,
                eigenvectors=eigenvectors,
                components=components,
                metadata=metadata,
            )

    idx = stage.reference_state_index
    ref_energy = energies[idx]
    abs_error = None
    if stage.reference_energy is not None:
        abs_error = float(abs(ref_energy - stage.reference_energy))

    hellmann = None
    kinetic = components.get("kinetic")
    potential = components.get("potential")
    mass = components.get("mass")
    if kinetic is not None and potential is not None:
        vector = eigenvectors[:, idx]
        kinetic_expect = expectation_from_matrix(vector, kinetic)
        total_kinetic = kinetic_expect
        if mass is not None:
            total_kinetic += expectation_from_matrix(vector, mass)
        potential_expect = expectation_from_matrix(vector, potential)
        hellmann = float(abs((total_kinetic + potential_expect) /
                             (total_kinetic - potential_expect)))

    return {
        "stage": stage.name,
        "basis_size": len(basis_states),
        "energy": float(ref_energy),
        "abs_error": abs_error,
        "target_abs_error": stage.target_abs_error,
        "hellmann": hellmann,
    }


def main() -> None:
    args = parse_args()
    selected: Iterable[StageConfig]
    if args.stage:
        selected = [STAGES[name] for name in args.stage]
    else:
        selected = STAGES.values()

    cache = SolverResultCache(args.cache_dir)
    rows: list[dict] = []
    for stage in selected:
        result = _run_stage(stage,
                            use_cache=args.use_cache,
                            refresh_cache=args.refresh_cache,
                            cache=cache,
                            show_progress=args.progress)
        rows.append(result)
        abs_err = result["abs_error"]
        target = result["target_abs_error"]
        status = "--"
        if abs_err is not None and target is not None:
            status = "OK" if abs_err <= target else "NOT MET"
        err_str = "nan"
        if abs_err is not None:
            err_str = f"{abs_err:.3e}"
        print(
            f"[{stage.name}] basis={result['basis_size']} energy={result['energy']:.12f} "
            f"abs_err={err_str} target={target} status={status}"
        )
        if result["hellmann"] is not None:
            print(f"    Hellmann η = {result['hellmann']:.3e}")

    print("\n汇总：")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
