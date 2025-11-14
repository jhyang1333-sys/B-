"""节点生成工具，对应论文公式 (2.30) 与 (3.1)。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ExponentialNodeConfig:
    r"""指数型节点配置。

    Attributes
    ----------
    r_min : float
        定义域左端点，通常取 0。
    r_max : float
        定义域右端点，即盒子半径。
    k : int
        B 样条阶数。
    n : int
        去除两端重复节点后的节点数量（包含端点）。
    gamma : float
    指数型节点参数，论文中使用 \(\gamma = r_{max} \times \tau\)。
    """

    r_min: float
    r_max: float
    k: int
    n: int
    gamma: float


def generate_exponential_nodes(config: ExponentialNodeConfig) -> np.ndarray:
    r"""生成指数型节点序列。

    该函数严格对应论文式 (3.1)，并在端点加入阶数要求的重复节点：

    .. math::

        t_i = \frac{r_{\max} \left(e^{\gamma \frac{i-1}{N-1}} - 1\right)}{e^{\gamma} - 1}.

    Parameters
    ----------
    config : ExponentialNodeConfig
        节点参数配置。

    Returns
    -------
    np.ndarray
        单调非减的节点数组，长度等于 ``config.n + 2 * (config.k - 1)``。
    """

    r_min = float(config.r_min)
    r_max = float(config.r_max)

    if config.k < 1:
        raise ValueError("B 样条阶数 k 必须为正整数。")
    if config.n < 2:
        raise ValueError("节点数量 n 至少为 2，以覆盖整个区间。")
    if r_min > r_max:
        raise ValueError("要求 r_min <= r_max。")

    # 内部节点（包含两端点，各出现一次，不含额外重复节点）
    indices = np.linspace(0.0, 1.0, config.n)
    if np.isclose(config.gamma, 0.0):
        scaled = indices
    else:
        scaled = np.expm1(config.gamma * indices) / np.expm1(config.gamma)

    interior = r_min + (r_max - r_min) * scaled

    # 按照论文式 (3.1)，在两端补齐 k-1 个重复节点以满足 B 样条边界条件
    left = np.full(config.k - 1, r_min)
    right = np.full(config.k - 1, r_max)

    knots = np.concatenate((left, interior, right))
    return knots
