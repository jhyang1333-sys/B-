"""径向双积分求积工具，用于 B 样条基函数矩阵元计算。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Tuple

import numpy as np

from .quadrature import _gauss_legendre_nodes_weights
from he_polarization.basis.bspline import BSplineBasis

_EPS = 1e-14


@dataclass(slots=True)
class RadialPointData:
    """存储给定积分节点处的径向基函数数值与导数。"""

    r1: float
    r2: float
    value_r1: Tuple[float, float]
    d1_r1: Tuple[float, float]
    d2_r1: Tuple[float, float]
    value_r2: Tuple[float, float]
    d1_r2: Tuple[float, float]
    d2_r2: Tuple[float, float]


@dataclass(slots=True)
class _SegmentCache:
    """缓存单个区段上所需的样条数值，避免在内层循环中重复递归。"""

    nodes: np.ndarray
    weights: np.ndarray
    value_row: np.ndarray
    value_col: np.ndarray
    d1_row: np.ndarray
    d1_col: np.ndarray
    d2_row: np.ndarray
    d2_col: np.ndarray


class RadialQuadrature2D:
    """在给定 B 样条基下对 ``r_1``、``r_2`` 的双积分执行高斯求积。"""

    def __init__(self, bspline: BSplineBasis, order: int = 8) -> None:
        if order <= 0:
            raise ValueError("求积分点数必须为正整数。")
        self._bspline = bspline
        self._order = int(order)
        self._nodes, self._weights = _gauss_legendre_nodes_weights(self._order)
        self._knots = np.asarray(bspline.knots, dtype=float)

    def _support_intersection(self, idx_a: int, idx_b: int) -> Tuple[float, float]:
        left_a, right_a = self._bspline.support(idx_a)
        left_b, right_b = self._bspline.support(idx_b)
        left = max(left_a, left_b)
        right = min(right_a, right_b)
        if right - left <= _EPS:
            return 0.0, 0.0
        return left, right

    def _segments(self, left: float, right: float) -> Iterable[Tuple[float, float]]:
        if right - left <= _EPS:
            return []
        segments = []
        for idx in range(len(self._knots) - 1):
            seg_left = max(self._knots[idx], left)
            seg_right = min(self._knots[idx + 1], right)
            if seg_right - seg_left > _EPS:
                segments.append((seg_left, seg_right))
        return segments

    def _map_nodes(self, segment: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        a, b = segment
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        nodes = mid + half * self._nodes
        weights = half * self._weights
        return nodes, weights

    def integrate(
        self,
        row_indices: Tuple[int, int],
        col_indices: Tuple[int, int],
        callback: Callable[[RadialPointData], Sequence[float]],
    ) -> Tuple[float, ...]:
        """对给定行列基函数配对执行双重高斯求积。

        Parameters
        ----------
        row_indices, col_indices : Tuple[int, int]
            行/列基函数的径向样条索引对 ``(i_1, i_2)``。
        callback : Callable[[RadialPointData], Sequence[float]]
            在每个积分节点上评估 integrand，返回一个或多个累加值。
        """

        i1_row, i2_row = row_indices
        i1_col, i2_col = col_indices

        left_r1, right_r1 = self._support_intersection(i1_row, i1_col)
        left_r2, right_r2 = self._support_intersection(i2_row, i2_col)
        if (right_r1 - left_r1) <= _EPS or (right_r2 - left_r2) <= _EPS:
            zeros = tuple(0.0 for _ in callback(
                RadialPointData(
                    r1=0.0,
                    r2=0.0,
                    value_r1=(0.0, 0.0),
                    d1_r1=(0.0, 0.0),
                    d2_r1=(0.0, 0.0),
                    value_r2=(0.0, 0.0),
                    d1_r2=(0.0, 0.0),
                    d2_r2=(0.0, 0.0),
                )
            ))
            return zeros

        segments_r1 = self._segments(left_r1, right_r1)
        segments_r2 = self._segments(left_r2, right_r2)
        if not segments_r1 or not segments_r2:
            zeros = tuple(0.0 for _ in callback(
                RadialPointData(
                    r1=0.0,
                    r2=0.0,
                    value_r1=(0.0, 0.0),
                    d1_r1=(0.0, 0.0),
                    d2_r1=(0.0, 0.0),
                    value_r2=(0.0, 0.0),
                    d1_r2=(0.0, 0.0),
                    d2_r2=(0.0, 0.0),
                )
            ))
            return zeros

        accumulator: Tuple[float, ...] | None = None

        segment_cache_r1 = tuple(
            self._build_segment_cache(segment, i1_row, i1_col)
            for segment in segments_r1
        )
        segment_cache_r2 = tuple(
            self._build_segment_cache(segment, i2_row, i2_col)
            for segment in segments_r2
        )

        for cache_r1 in segment_cache_r1:
            nodes_r1 = cache_r1.nodes
            weights_r1 = cache_r1.weights
            for cache_r2 in segment_cache_r2:
                nodes_r2 = cache_r2.nodes
                weights_r2 = cache_r2.weights
                for idx_r1, r1 in enumerate(nodes_r1):
                    weight_r1 = weights_r1[idx_r1]
                    val_r1_row = cache_r1.value_row[idx_r1]
                    val_r1_col = cache_r1.value_col[idx_r1]
                    d1_r1_row = cache_r1.d1_row[idx_r1]
                    d1_r1_col = cache_r1.d1_col[idx_r1]
                    d2_r1_row = cache_r1.d2_row[idx_r1]
                    d2_r1_col = cache_r1.d2_col[idx_r1]

                    for idx_r2, r2 in enumerate(nodes_r2):
                        weight = weight_r1 * weights_r2[idx_r2]
                        val_r2_row = cache_r2.value_row[idx_r2]
                        val_r2_col = cache_r2.value_col[idx_r2]
                        d1_r2_row = cache_r2.d1_row[idx_r2]
                        d1_r2_col = cache_r2.d1_col[idx_r2]
                        d2_r2_row = cache_r2.d2_row[idx_r2]
                        d2_r2_col = cache_r2.d2_col[idx_r2]

                        point = RadialPointData(
                            r1=float(r1),
                            r2=float(r2),
                            value_r1=(float(val_r1_row), float(val_r1_col)),
                            d1_r1=(float(d1_r1_row), float(d1_r1_col)),
                            d2_r1=(float(d2_r1_row), float(d2_r1_col)),
                            value_r2=(float(val_r2_row), float(val_r2_col)),
                            d1_r2=(float(d1_r2_row), float(d1_r2_col)),
                            d2_r2=(float(d2_r2_row), float(d2_r2_col)),
                        )
                        values = callback(point)
                        if accumulator is None:
                            accumulator = tuple(0.0 for _ in values)
                        accumulator = tuple(
                            acc + weight * val for acc, val in zip(accumulator, values)
                        )
        if accumulator is None:
            zeros = tuple(0.0 for _ in callback(
                RadialPointData(
                    r1=0.0,
                    r2=0.0,
                    value_r1=(0.0, 0.0),
                    d1_r1=(0.0, 0.0),
                    d2_r1=(0.0, 0.0),
                    value_r2=(0.0, 0.0),
                    d1_r2=(0.0, 0.0),
                    d2_r2=(0.0, 0.0),
                )
            ))
            return zeros
        return accumulator

    def _build_segment_cache(
        self,
        segment: Tuple[float, float],
        idx_row: int,
        idx_col: int,
    ) -> _SegmentCache:
        nodes, weights = self._map_nodes(segment)
        count = nodes.size

        value_row = np.empty(count, dtype=float)
        value_col = np.empty(count, dtype=float)
        d1_row = np.empty(count, dtype=float)
        d1_col = np.empty(count, dtype=float)
        d2_row = np.empty(count, dtype=float)
        d2_col = np.empty(count, dtype=float)

        for idx, radius in enumerate(nodes):
            radius_float = float(radius)
            value_row[idx] = self._bspline.evaluate(radius_float, idx_row)
            value_col[idx] = self._bspline.evaluate(radius_float, idx_col)
            d1_row[idx] = self._bspline.derivative(radius_float, idx_row, 1)
            d1_col[idx] = self._bspline.derivative(radius_float, idx_col, 1)
            d2_row[idx] = self._bspline.derivative(radius_float, idx_row, 2)
            d2_col[idx] = self._bspline.derivative(radius_float, idx_col, 2)

        return _SegmentCache(
            nodes=nodes,
            weights=weights,
            value_row=value_row,
            value_col=value_col,
            d1_row=d1_row,
            d1_col=d1_col,
            d2_row=d2_row,
            d2_col=d2_col,
        )
