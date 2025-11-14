r"""角动量通道定义，与论文中 \(l_1, l_2, L, M\) 量子数一致。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AtomicChannel:
    l1: int
    l2: int
    L: int
    M: int = 0
