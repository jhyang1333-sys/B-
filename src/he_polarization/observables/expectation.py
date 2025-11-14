"""利用矩阵结果计算期望值，并与论文中判据对接。"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - type checking helper
    from scipy.sparse import spmatrix as _SparseMatrix
else:  # noqa: SIM108 - runtime fallback without SciPy
    _SparseMatrix = Any

MatrixLike = Union[np.ndarray, _SparseMatrix]


def expectation_from_matrix(vector: np.ndarray, operator_matrix: MatrixLike) -> float:
    """计算给定算符矩阵的期望值 ``<ψ|A|ψ>``。"""

    op_vec = operator_matrix @ vector  # 支持稀疏/稠密乘法
    value = vector.conj().T @ op_vec
    return float(np.real_if_close(value))
