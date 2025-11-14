"""收敛与外推工具，对应论文第 3.2 节。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def convergence_ratio(values: np.ndarray) -> np.ndarray:
    """实现论文式 (3.2) 中的 R(N) 定义。"""
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        raise ValueError("至少需要 3 个数据点才能计算收敛率。")
    ratios = []
    for idx in range(2, values.size):
        numerator = values[idx - 1] - values[idx - 2]
        denominator = values[idx] - values[idx - 1]
        if np.isclose(denominator, 0):
            ratios.append(np.inf)
        else:
            ratios.append(numerator / denominator)
    return np.asarray(ratios)


@dataclass
class ConvergenceTracker:
    """记录能量/极化率随样条数、分波展开的收敛行为。"""

    n_values: np.ndarray
    observable_values: np.ndarray

    def extrapolate(self) -> tuple[float, float]:
        """返回外推值与误差，使用公式 (3.3)。"""
        if self.n_values.size != self.observable_values.size:
            raise ValueError("n_values 与 observable_values 长度必须一致。")
        if self.n_values.size < 3:
            raise ValueError("外推至少需要 3 个样本点。")

        ratios = convergence_ratio(self.observable_values)
        last_ratio = ratios[-1]
        if not np.isfinite(last_ratio) or np.isclose(last_ratio, 1.0):
            raise ValueError("收敛率不稳定，无法执行等差外推。")

        E_N = self.observable_values[-1]
        E_prev = self.observable_values[-2]
        extrapolated = E_N + (E_N - E_prev) / (last_ratio - 1.0)
        uncertainty = abs(extrapolated - E_N)
        return extrapolated, uncertainty
