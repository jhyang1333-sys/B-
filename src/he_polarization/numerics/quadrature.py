"""高斯求积相关函数，对应论文中径向积分的数值实现。"""
from __future__ import annotations

from typing import Iterable, Tuple

import math
import numpy as np


def _gauss_legendre_nodes_weights(n: int, tol: float = 1e-14) -> tuple[np.ndarray, np.ndarray]:
    """返回 n 点 Gauss-Legendre 求积的节点和权重，避免依赖 ``leggauss``。"""

    if n <= 0:
        raise ValueError("n 必须为正整数。")

    nodes = np.zeros(n, dtype=float)
    weights = np.zeros(n, dtype=float)
    m = (n + 1) // 2

    for i in range(m):
        # 初值来自 F.J.Stieltjes 的余弦近似
        x = math.cos(math.pi * (i + 0.75) / (n + 0.5))
        dp = 0.0

        for _ in range(100):
            p0 = 1.0
            p1 = x
            for k in range(2, n + 1):
                pk = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
                p0, p1 = p1, pk
            pn = p1
            pn_minus1 = p0
            dp = n * (pn_minus1 - x * pn) / (1.0 - x * x)
            delta = pn / dp
            x -= delta
            if abs(delta) < tol:
                break

        if dp == 0.0:
            raise RuntimeError("Gauss-Legendre 节点迭代未能收敛。")

        nodes[i] = -x
        nodes[n - 1 - i] = x
        weights_value = 2.0 / ((1.0 - x * x) * (dp * dp))
        weights[i] = weights_value
        weights[n - 1 - i] = weights_value

    return nodes, weights


def generate_tensor_product_quadrature(r_min: float, r_max: float, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """生成二维径向张量积的高斯求积点与权重。

    Parameters
    ----------
    r_min, r_max : float
        径向积分区间。
    n_points : int
        每个维度的高斯点数。

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``points`` 形状为 ``(n_points**2, 2)``，``weights`` 为对应权重（包含 ``r^2`` 体积元）。
    """

    if n_points <= 0:
        raise ValueError("n_points 必须为正整数。")

    x, w = _gauss_legendre_nodes_weights(n_points)
    half_interval = 0.5 * (r_max - r_min)
    shift = 0.5 * (r_max + r_min)
    radii = half_interval * x + shift
    base_weights = half_interval * w

    points = np.zeros((n_points * n_points, 2), dtype=float)
    weights = np.zeros(n_points * n_points, dtype=float)

    index = 0
    for i, r1 in enumerate(radii):
        for j, r2 in enumerate(radii):
            points[index] = (r1, r2)
            weights[index] = base_weights[i] * \
                base_weights[j] * (r1**2) * (r2**2)
            index += 1

    return points, weights
