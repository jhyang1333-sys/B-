"""Hellmann 判据实现，参考论文公式 (3.5)-(3.6)。"""
from __future__ import annotations


def hellmann_indicator(expect_T: float, expect_V: float) -> float:
    """计算 η = |1 + <V> / (2 <T>)|。"""
    if expect_T == 0:
        raise ZeroDivisionError("<T> 不能为 0。")
    return abs(1.0 + expect_V / (2.0 * expect_T))
