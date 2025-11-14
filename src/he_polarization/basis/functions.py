"""Hylleraas-B-spline 基函数定义与生成工具。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .bspline import BSplineBasis
from .channels import AtomicChannel


@dataclass(frozen=True)
class HylleraasBSplineFunction:
    r"""Hylleraas-B-spline 基函数描述符。

    Parameters
    ----------
    radial_indices : Tuple[int, int]
        径向样条索引对 ``(i, j)``，遵循论文式 (2.29) 中的 ``B_{i,k}(r_1) B_{j,k}(r_2)`` 约定。
    channel : AtomicChannel
        对应的角动量通道 ``(l_1, l_2, L, M)``。
    correlation_power : int
        电子关联项 ``r_{12}^c`` 的指数 ``c``，参见论文式 (2.29) 与 (2.33)。
    exchange_parity : int
        交换对称性，``+1`` 表示对称组合 ``(1 + P_{12})``，``-1`` 表示反对称组合 ``(1 - P_{12})``。
    symmetrized : bool
        若为 ``True``，则在实际构造波函数时需考虑 ``(r_1 \leftrightarrow r_2)`` 项。
    """

    radial_indices: Tuple[int, int]
    channel: AtomicChannel
    correlation_power: int = 0
    exchange_parity: int = 1
    symmetrized: bool = True

    def swapped(self) -> "HylleraasBSplineFunction":
        """返回电子交换后的基函数描述符。"""
        i, j = self.radial_indices
        swapped_channel = AtomicChannel(
            l1=self.channel.l2,
            l2=self.channel.l1,
            L=self.channel.L,
            M=self.channel.M,
        )
        return HylleraasBSplineFunction(
            radial_indices=(j, i),
            channel=swapped_channel,
            correlation_power=self.correlation_power,
            exchange_parity=self.exchange_parity,
            symmetrized=self.symmetrized,
        )


def generate_hylleraas_bspline_functions(
    bspline: BSplineBasis,
    channels: Sequence[AtomicChannel],
    *,
    n_radial: int | None = None,
    correlation_powers: Sequence[int] = (0,),
    exchange_parity: int = 1,
    symmetrize: bool = True,
    unique_pairs: bool = True,
) -> List[HylleraasBSplineFunction]:
    """批量生成 Hylleraas-B-spline 基函数描述符。

    该函数实现论文式 (2.33) 的参数选择策略：对给定的径向样条指数
    满足 ``0 \\le i \\le j < n`` 的条件（在代码中采用 0 基索引），并为每个角
    动量通道分配所需的电子关联幂次 ``c``。

    Parameters
    ----------
    bspline : BSplineBasis
        径向 B 样条基组。
    channels : Sequence[AtomicChannel]
        角动量通道集合。
    n_radial : int, optional
        限制参与组态的径向样条个数；若为 ``None``，则使用 ``bspline.n_basis``。
    correlation_powers : Sequence[int]
        关联指数集合，例如 ``(0, 1)`` 对应论文中的 ``c = 0, 1``。
    exchange_parity : int
        指定交换对称性，``+1`` 对应空间对称组合，``-1`` 对应反对称组合。
    symmetrize : bool
    控制是否在后续构造波函数时加入 ``(r_1 \\leftrightarrow r_2)`` 项。

    Returns
    -------
    list[HylleraasBSplineFunction]
        基函数描述符列表。
    """

    total_radial = bspline.n_basis if n_radial is None else min(
        n_radial, bspline.n_basis)
    if total_radial <= 0:
        raise ValueError("n_radial 必须大于 0。")

    states: List[HylleraasBSplineFunction] = []
    for j in range(total_radial):
        i_range = range(j + 1) if unique_pairs else range(total_radial)
        for i in i_range:
            if unique_pairs and i > j:
                continue
            radial_pair = (i, j)
            for channel in channels:
                for c in correlation_powers:
                    states.append(
                        HylleraasBSplineFunction(
                            radial_indices=radial_pair,
                            channel=channel,
                            correlation_power=int(c),
                            exchange_parity=exchange_parity,
                            symmetrized=symmetrize,
                        )
                    )
    return states


def iter_with_exchange(
    basis_functions: Iterable[HylleraasBSplineFunction],
) -> Iterable[Tuple[HylleraasBSplineFunction, HylleraasBSplineFunction]]:
    """生成原始与交换后的基函数对，便于组装矩阵元时统一处理。"""

    for func in basis_functions:
        yield func, func.swapped()
