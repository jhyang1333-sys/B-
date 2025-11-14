"""定义论文式 (2.34)-(2.41) 中的哈密顿算符组件。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HamiltonianOperators:
    """封装薛定谔哈密顿在质心坐标下的各类算符。"""

    mu: float
    M: float

    def potential_terms(self, r1: float, r2: float) -> float:
        """返回电子-核势能 ``V_{en}``，对应论文式 (2.9) 的前两项。"""
        if r1 <= 0 or r2 <= 0:
            raise ValueError("径向距离必须为正。")
        return -2.0 / r1 - 2.0 / r2
