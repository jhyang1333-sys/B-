"""关联项勒让德展开系数生成工具，对应论文公式 (2.43)-(2.44)。"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, Iterator, Tuple

import math


@dataclass(frozen=True)
class CorrelationTerm:
    """表示 ``r_{12}^c`` 展开中的单个项。

    Attributes
    ----------
    q : int
        勒让德多项式阶 ``P_q``。
    k : int
        径向幂次求和指标，对应式 (2.43) 中的 ``k``。
    coefficient : float
        系数 ``C_{cpk}``。
    """

    q: int
    k: int
    coefficient: float


class CorrelationExpansion:
    """生成整数 ``c`` 的 ``r_{12}^c`` 扩展系数。"""

    def __init__(self, *, q_cutoff: Dict[int, int] | None = None) -> None:
        self._q_cutoff = q_cutoff or {}

    def iter_terms(self, c: int) -> Iterator[CorrelationTerm]:
        """遍历 ``r_{12}^c`` 的展开项。

        对奇数阶 ``c``，展开为无穷级数；此处按照 ``q_cutoff`` 给出的上限截断。
        若未指定则按照 ``2 * (c + 5)`` 作为经验截断。
        """

        if c < 0:
            raise ValueError("关联幂次 c 必须为非负整数。")

        if c % 2 == 0:
            q_max = c // 2
        else:
            q_max = self._q_cutoff.get(c, 2 * (c + 5))

        for q in range(0, q_max + 1):
            for k in self._iter_k_indices(c, q):
                coeff = self._coefficient(c, q, k)
                if abs(coeff) < 1e-18:
                    continue
                yield CorrelationTerm(q=q, k=k, coefficient=coeff)

    @staticmethod
    def _iter_k_indices(c: int, q: int) -> Iterable[int]:
        if c % 2 == 0:
            return range(0, c // 2 - q + 1)
        upper = (c + 1) // 2
        return range(0, upper + 1)

    @staticmethod
    @lru_cache(maxsize=None)
    def _coefficient(c: int, q: int, k: int) -> float:
        numerator = (2 * q + 1) / (c + 2)
        numerator *= (c + 2) / (2 * k + 1)

        s_qc = min(q - 1, (c + 1) // 2)
        product = 1.0
        for t in range(0, s_qc + 1):
            product *= (2 * k + 2 * t - c) / (2 * k + 2 * q - 2 * t + 1)
        return numerator * product


__all__ = ["CorrelationExpansion", "CorrelationTerm"]
