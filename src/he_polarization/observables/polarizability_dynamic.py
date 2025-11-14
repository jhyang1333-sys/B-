"""动力学偶极极化率，复现论文第 3.4 节。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from he_polarization.observables.energies import EnergyCalculator


@dataclass
class DynamicPolarizabilityCalculator:
    """在给定频率网格上计算多种规范下的动力学极化率。"""

    energy_calculator: EnergyCalculator

    def evaluate(self, energies: np.ndarray, dipole_matrix: np.ndarray, state_index: int, freqs: Iterable[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回长度、速度、加速度三种规范的极化率序列。"""
        energies = np.asarray(energies, dtype=float)
        dipole_matrix = np.asarray(dipole_matrix, dtype=float)
        freqs = np.asarray(tuple(freqs), dtype=float)

        E0 = energies[state_index]
        diffs = energies - E0
        mask = np.ones_like(diffs, dtype=bool)
        mask[state_index] = False

        delta = diffs[mask]
        matrix_elements = dipole_matrix[state_index, mask]
        oscillator_strengths = 2.0 * np.abs(matrix_elements) ** 2 * delta
        acceleration_elements = delta ** 2 * matrix_elements  # 近似：来自双对易关系

        length_values = np.zeros_like(freqs)
        velocity_values = np.zeros_like(freqs)
        acceleration_values = np.zeros_like(freqs)

        for idx, omega in enumerate(freqs):
            denom = delta ** 2 - omega ** 2
            length_values[idx] = np.sum(
                2.0 * np.abs(matrix_elements) ** 2 * delta / denom)
            velocity_values[idx] = np.sum(oscillator_strengths / denom)
            acceleration_values[idx] = np.sum(
                np.abs(acceleration_elements) ** 2 / denom)

        return length_values, velocity_values, acceleration_values

    def magic_wavelengths(self, alpha_ground: np.ndarray, alpha_excited: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """寻找魔幻波长，即两态极化率相等的频率。"""
        diff = alpha_ground - alpha_excited
        zeros = self._find_zero_crossings(diff, freqs)
        return zeros

    def tune_out_wavelengths(self, alpha: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """寻找幻零波长，极化率为零的频率。"""
        return self._find_zero_crossings(alpha, freqs)

    @staticmethod
    def _find_zero_crossings(values: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        sign_changes = np.where(
            np.sign(values[:-1]) * np.sign(values[1:]) < 0)[0]
        zeros = []
        for idx in sign_changes:
            x0, x1 = freqs[idx], freqs[idx + 1]
            y0, y1 = values[idx], values[idx + 1]
            if np.isclose(y1 - y0, 0.0):
                zeros.append(0.5 * (x0 + x1))
            else:
                zeros.append(x0 - y0 * (x1 - x0) / (y1 - y0))
        return np.asarray(zeros, dtype=float)
