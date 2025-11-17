r"""角动量耦合与张量运算，对应论文公式 (2.16)-(2.58)。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import numpy as np
from scipy.special import sph_harm
from sympy.physics.wigner import wigner_3j, wigner_6j


@dataclass
class AngularCoupling:
    """封装角动量耦合系数与相关积分计算。


    - 矢量耦合积 Lambda_{l1 l2}^{LM}
    - 3j、6j 符号与 Clebsch-Gordan 系数
    - 角向积分 G1 / G_{r1,r2}
    """

    cache_3j: Dict[Tuple[int, int, int, int, int, int],
                   float] = field(default_factory=dict)
    cache_6j: Dict[Tuple[int, int, int, int, int, int],
                   float] = field(default_factory=dict)
    cache_rhat_dot: Dict[Tuple[int, int, int, int,
                               int, int], float] = field(default_factory=dict)
    cache_rhat_grad_12: Dict[Tuple[int, int, int, int,
                                   int, int], float] = field(default_factory=dict)
    cache_rhat_grad_21: Dict[Tuple[int, int, int, int,
                                   int, int], float] = field(default_factory=dict)
    cache_grad_grad: Dict[Tuple[int, int, int, int,
                                int, int], float] = field(default_factory=dict)

    def clebsch_gordan(self, l1: int, m1: int, l2: int, m2: int, L: int, M: int) -> float:
        """返回 Clebsch-Gordan 系数。"""
        phase = (-1) ** (l1 - l2 + M)
        coeff = phase * np.sqrt(2 * L + 1.0) * self.wigner_3j(
            l1, l2, L, m1, m2, -M
        )
        return float(coeff)

    def wigner_3j(self, j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
        """返回 Wigner 3j 符号。"""
        key = (j1, j2, j3, m1, m2, m3)
        if key not in self.cache_3j:
            self.cache_3j[key] = float(
                wigner_3j(j1, j2, j3, m1, m2, m3).evalf())
        return self.cache_3j[key]

    def wigner_6j(self, j1: int, j2: int, j3: int, l1: int, l2: int, l3: int) -> float:
        """返回 Wigner 6j 符号。"""
        key = (j1, j2, j3, l1, l2, l3)
        if key not in self.cache_6j:
            self.cache_6j[key] = float(
                wigner_6j(j1, j2, j3, l1, l2, l3).evalf())
        return self.cache_6j[key]

    @staticmethod
    def _triangle_condition(j1: int, j2: int, j3: int) -> bool:
        return abs(j1 - j2) <= j3 <= j1 + j2 and ((j1 + j2 + j3) % 2 == 0)

    def angular_tensor_g1_explicit(
        self,
        l1: int,
        l2: int,
        L: int,
        l1p: int,
        l2p: int,
        Lp: int,
        q: int,
    ) -> float:
        r"""实现论文式 (2.58) 的角向张量积分 ``G_1``。"""

        if not self._triangle_condition(l1p, q, l1):
            return 0.0
        if not self._triangle_condition(l2p, q, l2):
            return 0.0
        dims = math.sqrt((2 * l1 + 1) * (2 * l2 + 1) *
                         (2 * l1p + 1) * (2 * l2p + 1))

        prefactor = (-1) ** (l1 + l2 + Lp) * (2 * q + 1) * dims

        three1 = self.wigner_3j(l1p, q, l1, 0, 0, 0)
        if abs(three1) < 1e-14:
            return 0.0
        three2 = self.wigner_3j(l2p, l2, q, 0, 0, 0)
        if abs(three2) < 1e-14:
            return 0.0
        six = self.wigner_6j(l1p, l2p, Lp, l2, l1, q)
        return prefactor * three1 * three2 * six

    def coupled_spherical_harmonics(self, l1: int, l2: int, L: int, M: int, theta1: float, phi1: float, theta2: float, phi2: float) -> complex:
        """构建论文式 (2.18)。"""
        result = 0.0 + 0.0j
        for m1 in range(-l1, l1 + 1):
            for m2 in range(-l2, l2 + 1):
                if m1 + m2 != M:
                    continue
                cg = self.clebsch_gordan(l1, m1, l2, m2, L, M)
                y1 = sph_harm(m1, l1, phi1, theta1)
                y2 = sph_harm(m2, l2, phi2, theta2)
                result += cg * y1 * y2
        return result

    def angular_integral_g1(self, params: dict) -> float:
        """实现论文式 (2.58) 的角向部分。"""
        l1 = params["l1"]
        l2 = params["l2"]
        L = params["L"]
        l1p = params["l1p"]
        l2p = params["l2p"]
        Lp = params["Lp"]
        q = params["q"]

        return self.angular_tensor_g1_explicit(l1, l2, L, l1p, l2p, Lp, q)

    @staticmethod
    def _b_coefficient(l: int, T: int) -> float:
        if T == l - 1:
            return float(l + 1)
        if T == l + 1:
            return float(-l)
        return 0.0

    def _angular_tensor_rhat_weighted(
        self,
        l1: int,
        l2: int,
        L: int,
        l1p: int,
        l2p: int,
        Lp: int,
        q: int,
        *,
        weight_fn: Callable[[int, int], float],
    ) -> float:
        if L != Lp:
            return 0.0
        if not (self._triangle_condition(l1p, q, l1) and self._triangle_condition(l2p, q, l2)):
            return 0.0

        prefactor = math.sqrt(
            (2 * l1 + 1)
            * (2 * l2 + 1)
            * (2 * l1p + 1)
            * (2 * l2p + 1)
        )

        total = 0.0
        for T1 in range(abs(l1 - 1), l1 + 2):
            if not self._triangle_condition(l1, 1, T1):
                continue
            three_1 = self.wigner_3j(1, l1, T1, 0, 0, 0)
            if abs(three_1) < 1e-14:
                continue
            if not self._triangle_condition(l1p, T1, q):
                continue
            three_3 = self.wigner_3j(l1p, T1, q, 0, 0, 0)
            if abs(three_3) < 1e-14:
                continue
            for T2 in range(abs(l2 - 1), l2 + 2):
                if not self._triangle_condition(l2, 1, T2):
                    continue
                three_2 = self.wigner_3j(1, l2, T2, 0, 0, 0)
                if abs(three_2) < 1e-14:
                    continue
                if not self._triangle_condition(l2p, T2, q):
                    continue
                three_4 = self.wigner_3j(l2p, T2, q, 0, 0, 0)
                if abs(three_4) < 1e-14:
                    continue

                weight = weight_fn(T1, T2)
                if abs(weight) < 1e-14:
                    continue

                six_1 = self.wigner_6j(T2, l2, 1, l1, T1, L)
                six_2 = self.wigner_6j(l2p, T2, q, T1, l1p, Lp)

                term = (
                    weight
                    * (2 * T1 + 1)
                    * (2 * T2 + 1)
                    * three_1
                    * three_2
                    * three_3
                    * three_4
                    * six_1
                    * six_2
                )
                total += term

        return prefactor * total

    def angular_tensor_rhat_dot(
        self,
        l1: int,
        l2: int,
        L: int,
        l1p: int,
        l2p: int,
        Lp: int,
        q: int,
    ) -> float:
        r"""实现论文式 (2.70) 中 ``\hat{r}_1 \cdot \hat{r}_2`` 的角向部分。"""

        key = (l1, l2, L, l1p, l2p, q)
        cache = self.cache_rhat_dot
        if key in cache:
            return cache[key]

        value = self._angular_tensor_rhat_weighted(
            l1,
            l2,
            L,
            l1p,
            l2p,
            Lp,
            q,
            weight_fn=lambda _T1, _T2: 1.0,
        )
        cache[key] = value
        return value

    def angular_tensor_rhat_grad_12(
        self,
        l1: int,
        l2: int,
        L: int,
        l1p: int,
        l2p: int,
        Lp: int,
        q: int,
    ) -> float:
        r"""对应 ``\hat{r}_1 \cdot \hat{\nabla}_2^Y`` 的角向部分。"""

        key = (l1, l2, L, l1p, l2p, q)
        cache = self.cache_rhat_grad_12
        if key in cache:
            return cache[key]

        value = self._angular_tensor_rhat_weighted(
            l1,
            l2,
            L,
            l1p,
            l2p,
            Lp,
            q,
            weight_fn=lambda _T1, T2: self._b_coefficient(l2, T2),
        )
        cache[key] = value
        return value

    def angular_tensor_rhat_grad_21(
        self,
        l1: int,
        l2: int,
        L: int,
        l1p: int,
        l2p: int,
        Lp: int,
        q: int,
    ) -> float:
        r"""对应 ``\hat{r}_2 \cdot \hat{\nabla}_1^Y`` 的角向部分。"""

        key = (l1, l2, L, l1p, l2p, q)
        cache = self.cache_rhat_grad_21
        if key in cache:
            return cache[key]

        value = self._angular_tensor_rhat_weighted(
            l1,
            l2,
            L,
            l1p,
            l2p,
            Lp,
            q,
            weight_fn=lambda T1, _T2: self._b_coefficient(l1, T1),
        )
        cache[key] = value
        return value

    def angular_tensor_grad_grad(
        self,
        l1: int,
        l2: int,
        L: int,
        l1p: int,
        l2p: int,
        Lp: int,
        q: int,
    ) -> float:
        r"""对应 ``\hat{\nabla}_1^Y \cdot \hat{\nabla}_2^Y`` 的角向部分。"""

        key = (l1, l2, L, l1p, l2p, q)
        cache = self.cache_grad_grad
        if key in cache:
            return cache[key]

        value = self._angular_tensor_rhat_weighted(
            l1,
            l2,
            L,
            l1p,
            l2p,
            Lp,
            q,
            weight_fn=lambda T1, T2: self._b_coefficient(
                l1, T1) * self._b_coefficient(l2, T2),
        )
        cache[key] = value
        return value

    def angular_tensor_ry(
        self,
        acting_on: int,
        l: int,
        m: int,
        l1: int,
        l2: int,
        L: int,
        M: int,
        l1p: int,
        l2p: int,
        Lp: int,
        Mp: int,
        q: int,
    ) -> float:
        r"""实现论文式 (2.74) 中 ``r_i^Y Y_{lm}`` 积分的角向部分。"""

        if acting_on not in (1, 2):
            raise ValueError("acting_on 只能为 1 或 2。")

        if abs(M) > L or abs(Mp) > Lp:
            return 0.0

        if acting_on == 1:
            phase = (-1) ** (L - Mp + l + l1p)
            three_L = self.wigner_3j(Lp, l, L, -Mp, m, M)
            if abs(three_L) < 1e-14:
                return 0.0

            prefactor = phase * math.sqrt((2 * l + 1) / (4 * math.pi))
            prefactor *= math.sqrt(
                (2 * l1 + 1)
                * (2 * l2 + 1)
                * (2 * l1p + 1)
                * (2 * l2p + 1)
            )

            total = 0.0
            for T in range(abs(l - l1), l + l1 + 1):
                if not self._triangle_condition(l, l1, T):
                    continue
                three1 = self.wigner_3j(l, l1, T, 0, 0, 0)
                if abs(three1) < 1e-14:
                    continue
                if not self._triangle_condition(l1p, T, q):
                    continue
                three2 = self.wigner_3j(l1p, T, q, 0, 0, 0)
                if abs(three2) < 1e-14:
                    continue
                if not self._triangle_condition(l2p, l2, q):
                    continue
                three3 = self.wigner_3j(l2p, l2, q, 0, 0, 0)
                if abs(three3) < 1e-14:
                    continue
                six1 = self.wigner_6j(L, l1, l2, T, Lp, l)
                six2 = self.wigner_6j(l2p, l2, q, T, l1p, Lp)
                term = (2 * T + 1) * three1 * three2 * three3 * six1 * six2
                total += term

            return prefactor * three_L * total

        # acting_on == 2
        phase = (-1) ** (L - Mp + l + l2p)
        three_L = self.wigner_3j(Lp, l, L, -Mp, m, M)
        if abs(three_L) < 1e-14:
            return 0.0

        prefactor = phase * math.sqrt((2 * l + 1) / (4 * math.pi))
        prefactor *= math.sqrt(
            (2 * l1 + 1)
            * (2 * l2 + 1)
            * (2 * l1p + 1)
            * (2 * l2p + 1)
        )

        total = 0.0
        for T in range(abs(l - l2), l + l2 + 1):
            if not self._triangle_condition(l, l2, T):
                continue
            three1 = self.wigner_3j(l, l2, T, 0, 0, 0)
            if abs(three1) < 1e-14:
                continue
            if not self._triangle_condition(l2p, T, q):
                continue
            three2 = self.wigner_3j(l2p, T, q, 0, 0, 0)
            if abs(three2) < 1e-14:
                continue
            if not self._triangle_condition(l1p, l1, q):
                continue
            three3 = self.wigner_3j(l1p, l1, q, 0, 0, 0)
            if abs(three3) < 1e-14:
                continue
            six1 = self.wigner_6j(L, l2, l1, T, Lp, l)
            six2 = self.wigner_6j(l1p, l1, q, T, l2p, Lp)
            term = (2 * T + 1) * three1 * three2 * three3 * six1 * six2
            total += term

        return prefactor * three_L * total
