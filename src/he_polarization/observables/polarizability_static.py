"""静态偶极极化率计算，复现论文第 3.3 节内容。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from he_polarization.observables.energies import EnergyCalculator


@dataclass
class StaticPolarizabilityCalculator:
    """长度/速度规范下的静态极化率与误差评估。"""

    energy_calculator: EnergyCalculator

    def compute_length_gauge(self, energies: np.ndarray, dipole_matrix: np.ndarray, state_index: int) -> float:
        """实现论文式 (1.5) 的离散求和版本。"""
        energies = np.asarray(energies, dtype=float)
        dipole_matrix = np.asarray(dipole_matrix, dtype=float)

        E0 = energies[state_index]
        diffs = energies - E0
        mask = np.ones_like(diffs, dtype=bool)
        mask[state_index] = False

        diffs = diffs[mask]
        matrix_elements = dipole_matrix[state_index, mask]

        if np.any(np.isclose(diffs, 0.0)):
            raise ZeroDivisionError("存在与参考态简并的能级，无法直接使用长度规范公式。")

        contributions = 2.0 * np.abs(matrix_elements) ** 2 / diffs
        return float(np.sum(contributions))

    def compute_velocity_gauge(self, energies: np.ndarray, momentum_matrix: np.ndarray, state_index: int) -> float:
        """Velocity-gauge polarizability via momentum matrix elements."""
        energies = np.asarray(energies, dtype=float)
        momentum_matrix = np.asarray(momentum_matrix, dtype=complex)

        E0 = energies[state_index]
        diffs = energies - E0
        mask = np.ones_like(diffs, dtype=bool)
        mask[state_index] = False

        diffs = diffs[mask]
        matrix_elements = momentum_matrix[state_index, mask]

        if np.any(np.isclose(diffs, 0.0)):
            raise ZeroDivisionError("存在与参考态简并的能级，无法直接使用速度规范公式。")

        contributions = 2.0 * np.abs(matrix_elements) ** 2 / (diffs ** 3)
        return float(np.sum(contributions))

    def relative_difference(self, length_value: float, velocity_value: float) -> float:
        """实现公式 (3.9) 定义的相对差异 η。"""
        numerator = 2.0 * (length_value - velocity_value)
        denominator = length_value + velocity_value
        if np.isclose(denominator, 0.0):
            raise ZeroDivisionError("极化率求和结果出现抵消导致分母为零。")
        return float(numerator / denominator)
