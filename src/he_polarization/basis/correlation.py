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

        if c < -1:
            raise ValueError("关联幂次 c 必须大于等于 -1。")

        if c == -1:
            q_max = self._q_cutoff.get(-1, 2 * (abs(c) + 5))
        elif c % 2 == 0:
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
        if c == -1:
            return (0,)
        if c % 2 == 0:
            return range(0, c // 2 - q + 1)
        upper = (c + 1) // 2
        return range(0, upper + 1)

    @staticmethod
    @lru_cache(maxsize=None)
    def _coefficient(c: int, q: int, k: int) -> float:
        if c == -1:
            if k != 0 or q < 0:
                return 0.0
            return 1.0
        if c < 0 or q < 0 or k < 0:
            return 0.0

        index = 2 * k + 1
        if index > c + 2:
            return 0.0

        try:
            binomial = math.comb(c + 2, index)
        except ValueError:
            return 0.0

        numerator = (2 * q + 1) / (c + 2)
        numerator *= binomial

        s_qc = min(q - 1, (c + 1) // 2)
        product = 1.0
        for t in range(0, s_qc + 1):
            product *= (2 * k + 2 * t - c) / (2 * k + 2 * q - 2 * t + 1)
        return numerator * product

    @staticmethod
    def estimate_truncation_error(c: int, q_max: int, r_ratio: float = 0.9) -> float:
        """
        估算勒让德展开的截断误差。

        Parameters
        ----------
        r_ratio : float
            r_</r_> 的比值。取 0.9 表示较坏情况（收敛慢）。
        """
        if c % 2 == 0:
            return 0.0  # 偶数次幂是有限项，无截断误差

        # 对于奇数次幂，系数衰减大约为 O(q^{-2.5}) * r_ratio^q
        # 这里计算最后一项的相对贡献作为误差估计

        # 取展开式中最后一项的 k=0 的系数近似
        # C_{c, q_max, 0} ~ 1/q_max
        last_coeff = CorrelationExpansion._coefficient(c, q_max, 0)

        # 相对误差估计
        error = abs(last_coeff) * (r_ratio ** q_max)
        return error

    def check_convergence(self, c: int, tolerance: float = 1e-12) -> bool:
        """检查当前 q_cutoff 设置是否满足精度要求。"""
        if c % 2 == 0:
            return True

        # 获取当前使用的 q_max
        q_max = self._q_cutoff.get(c, 2 * (c + 5))

        # 在 r_</r_> = 0.9 处估算误差
        error = self.estimate_truncation_error(c, q_max, r_ratio=0.9)

        if error > tolerance:
            print(f"Warning: Legendre expansion for c={c} (q_max={q_max}) "
                  f"may not converge to {tolerance:.1e} (est. error={error:.1e})")
            return False
        return True


__all__ = ["CorrelationExpansion", "CorrelationTerm"]
