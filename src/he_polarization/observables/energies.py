"""能级计算，对应论文第 3.2 节与表 3.3-3.6。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

from typing import Iterable

from he_polarization.basis.channels import AtomicChannel
from he_polarization.basis.functions import HylleraasBSplineFunction
from he_polarization.hamiltonian.elements import MatrixElementBuilder
from he_polarization.solver import solve_generalized_eigen
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
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """对广义本征问题进行求解，返回能量与系数矩阵。"""
        H, O, components = self.builder.assemble_matrices(
            basis_states, weights=weights, points=points)
        eigvals, eigvecs = solve_generalized_eigen(H, O)
        return eigvals, eigvecs, components

    def extrapolate(self, energies: np.ndarray, n_values: np.ndarray) -> Tuple[float, float]:
        """实现论文式 (3.2)-(3.3) 的等差外推，返回外推值与不确定度。"""
        tracker = ConvergenceTracker(n_values=np.asarray(
            n_values, dtype=float), observable_values=np.asarray(energies, dtype=float))
        return tracker.extrapolate()

    def hellmann_eta(self, expect_T: float, expect_V: float) -> float:
        """根据公式 (3.6) 评估 Hellmann 判据误差。"""
        return hellmann_indicator(expect_T, expect_V)
