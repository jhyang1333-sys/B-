"""能级计算，对应论文第 3.2 节与表 3.3-3.6。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Optional, Tuple, Union, cast, Sequence

import numpy as np

from he_polarization.basis.channels import AtomicChannel
from he_polarization.basis.functions import HylleraasBSplineFunction
from he_polarization.hamiltonian.elements import MatrixElementBuilder
from he_polarization.solver import (
    ChannelOrthogonalizer,
    IterativeSolverConfig,
    OverlapConditioner,
    solve_generalized_eigen,
    solve_sparse_generalized_eigen,
)
from he_polarization.validation.convergence import ConvergenceTracker
from he_polarization.validation.hellmann import hellmann_indicator


@dataclass
class EnergyCalculator:
    """包装能级求解与外推策略。"""

    builder: MatrixElementBuilder
    channel_orthogonalizer: Optional[ChannelOrthogonalizer] = None
    overlap_conditioner: Optional[OverlapConditioner] = None

    def diagonalize(
        self,
        basis_states: Iterable[
            Union[
                HylleraasBSplineFunction,
                Tuple[Tuple[int, int], AtomicChannel]
            ]
        ],
        *,
        weights,
        points,
        num_eigenvalues: int | None = None,
        solver_config: IterativeSolverConfig | None = None,
        progress: bool | str | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """对广义本征问题进行求解，返回能量与系数矩阵。"""
        state_list = list(basis_states)
        H, O, components = self.builder.assemble_matrices(
            state_list, weights=weights, points=points, progress=progress)

        def _summarize(matrix, label: str) -> None:
            if matrix is None:
                print(f"{label}: <missing>")
                return
            try:
                data = matrix.data if hasattr(
                    matrix, "data") else np.asarray(matrix)
                finite = np.isfinite(data)
                total = data[finite] if np.any(finite) else data
                if total.size == 0:
                    print(f"{label}: empty")
                else:
                    print(
                        f"{label}: min={total.min():.3e} max={total.max():.3e} "
                        f"abs_mean={np.mean(np.abs(total)):.3e}"
                    )
            except Exception as err:  # pragma: no cover
                print(f"{label}: stats unavailable ({err})")

        # _summarize(H, "H matrix")
        # _summarize(O, "O matrix")
        # _summarize(components.get("kinetic"), "Kinetic block")
        # _summarize(components.get("potential"), "Potential block")
        # if "mass" in components:
        #     _summarize(components.get("mass"), "Mass block")

        H_eff, O_eff = H, O
        back_transform: Optional[BackTransform] = None

        if self.channel_orthogonalizer is not None:
            H_eff, O_eff, channel_back, channel_stats = self.channel_orthogonalizer.apply(
                H_eff, O_eff, cast(Sequence[HylleraasBSplineFunction], state_list))
            if channel_stats and channel_stats.total_retained != channel_stats.total_original:
                removed = channel_stats.total_original - channel_stats.total_retained
                print(
                    "Channel orthonormalization removed "
                    f"{removed} states (tol={self.channel_orthogonalizer.tolerance:.1e})."
                )
            back_transform = channel_back

        conditioned = self.overlap_conditioner.condition(
            H_eff, O_eff) if self.overlap_conditioner else (H_eff, O_eff, None, None)
        H_eff, O_eff, overlap_back, metadata = conditioned
        if metadata:
            if metadata.strategy == "dense" and metadata.discarded_dimension:
                print(
                    "Overlap conditioning removed "
                    f"{metadata.discarded_dimension} states (min λ={metadata.min_kept_eigenvalue:.2e})."
                )
            elif metadata.strategy == "regularize" and self.overlap_conditioner is not None:
                print(
                    "Overlap conditioning added diagonal regularization "
                    f"(ε={self.overlap_conditioner.regularization:.2e})."
                )

        back_transform = _compose_back_transforms(back_transform, overlap_back)

        if _should_use_iterative_solver(H_eff, O_eff, num_eigenvalues):
            config = solver_config or IterativeSolverConfig()
            if num_eigenvalues is not None:
                config = replace(config, num_eigenvalues=num_eigenvalues)
            eigvals, eigvecs = solve_sparse_generalized_eigen(
                H_eff, O_eff, config=config)
        else:
            eigvals, eigvecs = solve_generalized_eigen(
                cast(Any, H_eff),
                cast(Any, O_eff),
            )

        if back_transform is not None:
            eigvecs = back_transform(eigvecs)
        return eigvals, eigvecs, components

    def extrapolate(self, energies: np.ndarray, n_values: np.ndarray) -> Tuple[float, float]:
        """实现论文式 (3.2)-(3.3) 的等差外推，返回外推值与不确定度。"""
        tracker = ConvergenceTracker(n_values=np.asarray(
            n_values, dtype=float), observable_values=np.asarray(energies, dtype=float))
        return tracker.extrapolate()

    def hellmann_eta(self, expect_T: float, expect_V: float) -> float:
        """根据公式 (3.6) 评估 Hellmann 判据误差。"""
        return hellmann_indicator(expect_T, expect_V)


def _should_use_iterative_solver(H, O, num_eigenvalues: int | None) -> bool:
    try:
        from scipy.sparse import issparse as _issparse  # type: ignore
    except ModuleNotFoundError:
        def _issparse(obj: Any) -> bool:
            return False

    if num_eigenvalues is not None:
        return True
    return _issparse(H) or _issparse(O)


BackTransform = Callable[[np.ndarray], np.ndarray]


def _compose_back_transforms(
    first: Optional[BackTransform],
    second: Optional[BackTransform],
) -> Optional[BackTransform]:
    if first is None:
        return second
    if second is None:
        return first

    def combined(vecs: np.ndarray) -> np.ndarray:
        return first(second(vecs))

    return combined
