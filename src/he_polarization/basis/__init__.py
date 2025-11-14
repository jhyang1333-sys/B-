"""基函数构建模块，覆盖论文第 2.3 节的 Hylleraas-B-spline 基组细节。"""

from .nodes import ExponentialNodeConfig, generate_exponential_nodes
from .bspline import BSplineBasis
from .angular import AngularCoupling
from .channels import AtomicChannel
from .functions import (
    HylleraasBSplineFunction,
    generate_hylleraas_bspline_functions,
    iter_with_exchange,
)
from .correlation import CorrelationExpansion, CorrelationTerm

__all__ = [
    "ExponentialNodeConfig",
    "generate_exponential_nodes",
    "BSplineBasis",
    "AngularCoupling",
    "AtomicChannel",
    "HylleraasBSplineFunction",
    "generate_hylleraas_bspline_functions",
    "CorrelationExpansion",
    "CorrelationTerm",
    "iter_with_exchange",
]
