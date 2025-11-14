"""利用矩阵结果计算期望值，并与论文中判据对接。"""
from __future__ import annotations

import numpy as np


def expectation_from_matrix(vector: np.ndarray, operator_matrix: np.ndarray) -> float:
    """计算给定算符矩阵的期望值 ``<ψ|A|ψ>``。"""
    return float(np.real_if_close(vector.conj().T @ operator_matrix @ vector))
