"""能级计算，对应论文第 3.2 节与表 3.3-3.6。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Tuple, Union, cast

import numpy as np

from he_polarization.basis.channels import AtomicChannel
from he_polarization.basis.functions import HylleraasBSplineFunction
from he_polarization.hamiltonian.elements import MatrixElementBuilder
from he_polarization.solver import (
    IterativeSolverConfig,
    solve_generalized_eigen,
    solve_sparse_generalized_eigen,
)
from he_polarization.validation.convergence import ConvergenceTracker
from he_polarization.validation.hellmann import hellmann_indicator


@dataclass
class EnergyCalculator:
    """包装能级求解与外推策略。"""

    builder: MatrixElementBuilder

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
        H, O, components = self.builder.assemble_matrices(
            basis_states, weights=weights, points=points, progress=progress)
        if _should_use_iterative_solver(H, O, num_eigenvalues):
            config = solver_config or IterativeSolverConfig()
            if num_eigenvalues is not None:
                config = replace(config, num_eigenvalues=num_eigenvalues)
            eigvals, eigvecs = solve_sparse_generalized_eigen(
                H, O, config=config)
        else:
            eigvals, eigvecs = solve_generalized_eigen(
                cast(Any, H),
                cast(Any, O),
            )
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
    if num_eigenvalues is not None:
        return True
    try:
        from scipy.sparse import issparse
    except ModuleNotFoundError:
        return False
    return issparse(H) or issparse(O)
