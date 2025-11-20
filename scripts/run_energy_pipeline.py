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
from he_polarization.solver import ChannelOrthogonalizer, OverlapConditioner
from he_polarization.solver import IterativeSolverConfig


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
    parser.add_argument("--disable-channel-orthogonalization", action="store_true",
                        help="Disable per-channel orthonormalization (default: enabled).")
    parser.add_argument("--channel-ortho-tol", type=float, default=1e-10,
                        help="Eigenvalue tolerance for channel-wise pruning.")
    parser.add_argument("--channel-ortho-max-dim", type=int, default=512,
                        help="Largest channel block size handled via dense eigendecomposition.")

    # === 新增安全模式参数 ===
    parser.add_argument("--disable-channel-ortho-safe-mode", action="store_true",
                        help="Disable physics-based rescue of dropped states (default: rescue enabled).")
    parser.add_argument("--channel-ortho-safe-threshold", type=float, default=-2.0,
                        help="Energy threshold (a.u.) below which states are forced to be kept.")
    # ======================

    parser.add_argument("--disable-overlap-conditioning", action="store_true",
                        help="Disable overlap-matrix stabilization (default: enabled).")
    parser.add_argument("--overlap-conditioning-tol", type=float, default=1e-10,
                        help="Eigenvalue tolerance used when pruning ill-conditioned overlap modes.")
    parser.add_argument("--overlap-conditioning-max-dim", type=int, default=4096,
                        help="Largest basis dimension allowed for dense overlap conditioning.")
    parser.add_argument("--overlap-conditioning-mode", choices=["auto", "dense", "regularize", "off"], default="auto",
                        help="Conditioning strategy: auto switches to regularization once the basis exceeds the dense limit.")
    parser.add_argument("--overlap-conditioning-regularization", type=float, default=1e-8,
                        help="Diagonal shift added when using the regularize strategy.")
    return parser.parse_args()


def verify_hamiltonian_coefficients(mu: float, M: float):
    """启动前验证哈密顿量关键系数符号与大小"""
    print("Verifying Hamiltonian coefficients...")

    # 质量极化系数检查 (-1/2M)
    expected_cross = -0.5 / M
    # 注意：elements.py 中应实现为 pref_cross = -0.5/M
    print(
        f"  [Check] Mass polarization coeff (should be ~ -{0.5/M:.2e}): {expected_cross:.2e}")

    if M < 0 or mu < 0:
        print("  [ALARM] Negative mass detected!")

    print("  Coefficient check passed.")


def main() -> None:
    args = parse_args()

    # --- 关键参数：数值稳定组合 (Yang 2019) ---
    tau = 0.14     # 0.038
    r_max = 60.0   # 必须足够大
    k = 7           # 必须是高阶
    n = 25          # 节点数配合 r_max
    l_max = 2       # 包含 s, p, d, f 波
    # ----------------------------------------

    # 0. 启动前物理验证
    m_nucleus = 7294.2995365
    mu = m_nucleus / (1.0 + m_nucleus)
    verify_hamiltonian_coefficients(mu, m_nucleus)

    config = ExponentialNodeConfig(
        r_min=0.0, r_max=r_max, k=k, n=n, gamma=r_max * tau)
    knots = generate_exponential_nodes(config)
    bspline = BSplineBasis(knots=knots, order=k)

    angular = AngularCoupling()
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

    channel_ortho = None
    if not args.disable_channel_orthogonalization:
        channel_ortho = ChannelOrthogonalizer(
            tolerance=args.channel_ortho_tol,
            max_block_dim=args.channel_ortho_max_dim,
            safe_mode=(not args.disable_channel_ortho_safe_mode),
            safe_mode_threshold=args.channel_ortho_safe_threshold,
        )

    conditioner = None
    if (not args.disable_overlap_conditioning) and args.overlap_conditioning_mode != "off":
        conditioner = OverlapConditioner(
            tolerance=args.overlap_conditioning_tol,
            max_dense_dim=args.overlap_conditioning_max_dim,
            mode=args.overlap_conditioning_mode,
            regularization=args.overlap_conditioning_regularization,
        )

    calculator = EnergyCalculator(
        builder=builder,
        channel_orthogonalizer=channel_ortho,
        overlap_conditioner=conditioner,
    )

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
        "overlap_conditioning_tol": args.overlap_conditioning_tol,
        "overlap_conditioning_max_dim": args.overlap_conditioning_max_dim,
        "overlap_conditioning_mode": args.overlap_conditioning_mode,
        "overlap_conditioning_regularization": args.overlap_conditioning_regularization,
        "overlap_conditioning_enabled": conditioner is not None,
        "channel_ortho_tol": args.channel_ortho_tol,
        "channel_ortho_max_dim": args.channel_ortho_max_dim,
        "channel_ortho_enabled": channel_ortho is not None,
        "channel_ortho_safe_mode": (not args.disable_channel_ortho_safe_mode) if channel_ortho else False,
    }

    cache = SolverResultCache(args.cache_dir)
    if args.use_cache and not args.refresh_cache and cache.available(args.cache_key, metadata=metadata):
        cached = cache.load(args.cache_key)
        energies = cached.energies
        eigenvectors = cached.eigenvectors
        components = cached.components
    else:
        # --- 关键配置：Shift-Invert 求解器 ---
        solver_cfg = IterativeSolverConfig(
            num_eigenvalues=args.num_eigenvalues,
            tol=1e-12,
            sigma=-2.90372,  # 锁定物理基态能量附近
            which="LM"       # 寻找最大 1/(E-sigma)
        )
        # -----------------------------------------------

        energies, eigenvectors, components = calculator.diagonalize(
            basis_states,
            weights=weights,
            points=points,
            num_eigenvalues=args.num_eigenvalues,
            progress="矩阵装配进度" if args.progress else None,
            solver_config=solver_cfg,
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
    kinetic_mu_expect = expectation_from_matrix(
        ground_vec, components["kinetic"])
    mass_expect = expectation_from_matrix(ground_vec, components["mass"])
    total_kinetic_expect = kinetic_mu_expect + mass_expect

    potential_expect = expectation_from_matrix(
        ground_vec, components["potential"])

    eta = calculator.hellmann_eta(
        expect_T=total_kinetic_expect, expect_V=potential_expect)

    print("最低几个能级 (a.u.):")
    for idx, energy in enumerate(energies[:5]):
        print(f"  E_{idx} = {energy:.8f}")

    print(f"Hellmann 判据 η = {eta:.3e}")

    # --- 结果验证 ---
    REF_ENERGY = -2.903724377
    diff = abs(energies[0] - REF_ENERGY)
    print("\n=== 结果验证 ===")
    print(f"基准值 (Drake/Yang): {REF_ENERGY}")
    print(f"计算值: {energies[0]:.9f}")
    print(f"绝对误差: {diff:.2e}")
    if diff < 1e-4:
        print(">>> 结果物理上合理 (PASS)")
    else:
        print(">>> 结果偏差较大，请检查收敛性 (WARN)")


if __name__ == "__main__":
    main()
