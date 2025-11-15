"""B 样条构造与导数，实现论文公式 (2.19)-(2.28)。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

try:  # Optional acceleration via numba JIT.
    from numba import njit  # type: ignore

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - numba not installed.
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator


_EPS = 1e-12


@njit(cache=True)
def _bspline_recursive_njit(
    knots: np.ndarray,
    r: float,
    i: int,
    order: int,
) -> float:
    if order == 1:
        left = knots[i]
        right = knots[i + 1]
        if (left <= r < right) or (
            abs(r - knots[-1]) < _EPS and abs(r - right) < _EPS
        ):
            return 1.0
        return 0.0

    left_denom = knots[i + order - 1] - knots[i]
    right_denom = knots[i + order] - knots[i + 1]

    term_left = 0.0
    if abs(left_denom) > _EPS:
        term_left = (
            (r - knots[i])
            / left_denom
            * _bspline_recursive_njit(knots, r, i, order - 1)
        )

    term_right = 0.0
    if abs(right_denom) > _EPS:
        term_right = (
            (knots[i + order] - r)
            / right_denom
            * _bspline_recursive_njit(knots, r, i + 1, order - 1)
        )

    return term_left + term_right


@njit(cache=True)
def _bspline_derivative_njit(
    knots: np.ndarray,
    r: float,
    i: int,
    order: int,
    deriv_order: int,
) -> float:
    if deriv_order <= 0:
        return _bspline_recursive_njit(knots, r, i, order)
    if order <= 1:
        return 0.0

    left_denom = knots[i + order - 1] - knots[i]
    right_denom = knots[i + order] - knots[i + 1]

    term_left = 0.0
    if abs(left_denom) > _EPS:
        factor_left = (order - 1) / left_denom
        term_left = factor_left * _bspline_derivative_njit(
            knots, r, i, order - 1, deriv_order - 1
        )

    term_right = 0.0
    if abs(right_denom) > _EPS:
        factor_right = -(order - 1) / right_denom
        term_right = factor_right * _bspline_derivative_njit(
            knots, r, i + 1, order - 1, deriv_order - 1
        )

    return term_left + term_right


@dataclass
class BSplineBasis:
    """Hylleraas-B-spline 径向部分的样条基。

    Attributes
    ----------
    knots : np.ndarray
        节点序列，需满足单调非减。
    order : int
        样条阶数 ``k``。
    """

    knots: np.ndarray
    order: int
    _eps: float = _EPS

    def __post_init__(self) -> None:
        self.knots = np.asarray(self.knots, dtype=float)
        if self.order < 1:
            raise ValueError("B 样条阶数必须大于等于 1。")
        if len(self.knots) < self.order + 1:
            raise ValueError("节点数量不足以构成给定阶数的样条基。")
        if np.any(np.diff(self.knots) < -self._eps):
            raise ValueError("节点必须单调非减。")

    @property
    def n_basis(self) -> int:
        """返回样条基函数数量。"""
        return len(self.knots) - self.order

    def evaluate(self, r: float, i: int) -> float:
        """计算第 ``i`` 个样条在 ``r`` 处的值。"""
        r = float(r)
        i = int(i)
        if i < 0 or i >= self.n_basis:
            raise IndexError("样条索引越界。")
        if _HAS_NUMBA:
            return float(_bspline_recursive_njit(self.knots, r, i, self.order))
        return self._bspline_recursive_python(r, i, self.order)

    def derivative(self, r: float, i: int, order: int = 1) -> float:
        """计算第 ``i`` 个样条的高阶导数，使用论文式 (2.28) 的递推。"""
        r = float(r)
        i = int(i)
        order = int(order)

        if order < 0:
            raise ValueError("导数阶数必须为非负整数。")
        if order == 0:
            return self.evaluate(r, i)
        if _HAS_NUMBA:
            return float(
                _bspline_derivative_njit(
                    self.knots,
                    r,
                    i,
                    self.order,
                    order,
                )
            )

        coeffs: List[Tuple[float, int, int]] = [(1.0, i, self.order)]
        for _ in range(order):
            new_coeffs: List[Tuple[float, int, int]] = []
            for coeff, idx, current_order in coeffs:
                if current_order <= 1:
                    continue
                left_denom = self.knots[idx +
                                        current_order - 1] - self.knots[idx]
                if abs(left_denom) > self._eps:
                    factor_left = coeff * (current_order - 1) / left_denom
                    new_coeffs.append((factor_left, idx, current_order - 1))

                right_denom = self.knots[idx +
                                         current_order] - self.knots[idx + 1]
                if abs(right_denom) > self._eps:
                    factor_right = -coeff * (current_order - 1) / right_denom
                    new_coeffs.append(
                        (factor_right, idx + 1, current_order - 1))
            coeffs = new_coeffs
            if not coeffs:
                return 0.0

        result = 0.0
        for coeff, idx, current_order in coeffs:
            if idx < 0 or idx >= self.n_basis:
                continue
            result += coeff * \
                self._bspline_recursive_python(r, idx, current_order)
        return result

    def support(self, i: int) -> tuple[float, float]:
        """返回第 ``i`` 个样条的支撑区间。"""
        i = int(i)
        left = self.knots[i]
        right = self.knots[i + self.order]
        return float(left), float(right)

    def as_matrix(self, points: Iterable[float]) -> np.ndarray:
        """批量评估所有样条，方便组装矩阵元素。"""
        pts = np.asarray(tuple(float(p) for p in points))
        n_basis = self.n_basis
        matrix = np.zeros((pts.size, n_basis), dtype=float)
        for idx, r in enumerate(pts):
            for i in range(n_basis):
                matrix[idx, i] = self.evaluate(r, i)
        return matrix

    def _bspline_recursive_python(self, r: float, i: int, order: int) -> float:
        # k=1 时为分段常数函数
        if order == 1:
            left = self.knots[i]
            right = self.knots[i + 1]
            if (left <= r < right) or (
                np.isclose(r, self.knots[-1]) and np.isclose(r, right)
            ):
                return 1.0
            return 0.0

        left_denom = self.knots[i + order - 1] - self.knots[i]
        right_denom = self.knots[i + order] - self.knots[i + 1]

        term_left = 0.0
        if abs(left_denom) > self._eps:
            term_left = (r - self.knots[i]) / left_denom * \
                self._bspline_recursive_python(r, i, order - 1)

        term_right = 0.0
        if abs(right_denom) > self._eps:
            term_right = (self.knots[i + order] - r) / right_denom * \
                self._bspline_recursive_python(r, i + 1, order - 1)

        return term_left + term_right
