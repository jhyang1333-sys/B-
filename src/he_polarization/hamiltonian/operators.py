"""定义论文式 (2.34)-(2.41) 中的哈密顿算符组件。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HamiltonianOperators:
    """封装薛定谔哈密顿在质心坐标下的各类算符。"""

    mu: float
    M: float

    def potential_terms(self, r1: float, r2: float, r12: float) -> float:
        """返回库仑势能项，对应论文式 (2.9)。

        其中 ``r12`` 取角向积分后的有效 ``r_>``，避免近零差引发数值爆炸。
        """
        if r1 <= 0 or r2 <= 0 or r12 <= 0:
            raise ValueError("径向距离必须为正。")
        return -2.0 / r1 - 2.0 / r2 + 1.0 / r12
