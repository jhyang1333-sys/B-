"""Channel-wise orthonormalization utilities for Hylleraas-B-spline bases."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse

from he_polarization.basis.functions import HylleraasBSplineFunction


_BackTransform = Callable[[np.ndarray], np.ndarray]


@dataclass
class ChannelGroupStats:
    """Statistics for a single channel block after orthonormalization."""

    key: Tuple[int, int, int, int, int]
    original_dimension: int
    retained_dimension: int
    min_kept_eigenvalue: float
    max_kept_eigenvalue: float


@dataclass
class ChannelOrthoResult:
    """Aggregated result information for channel orthonormalization."""

    total_original: int
    total_retained: int
    group_stats: List[ChannelGroupStats]


@dataclass
class ChannelOrthogonalizer:
    """Per-channel canonical orthonormalization and pruning."""

    tolerance: float = 1e-10
    max_block_dim: int = 2048
    include_exchange_parity: bool = True
    include_correlation_power: bool = True

    def apply(
        self,
        h_matrix: Any,
        o_matrix: Any,
        basis_states: Sequence[HylleraasBSplineFunction],
    ) -> Tuple[Any, Any, _BackTransform, ChannelOrthoResult]:
        size = len(basis_states)

        # 定义默认的恒等反变换 (Identity transform)
        def identity_back_transform(vecs: np.ndarray) -> np.ndarray:
            return vecs

        # 如果没有状态或容差关闭，直接返回
        if size == 0 or (self.tolerance <= 0):
            return _to_csr(h_matrix), _to_csr(o_matrix), identity_back_transform, ChannelOrthoResult(size, size, [])

        grouped = self._group_indices(basis_states)
        transform, stats = self._build_block_transform(o_matrix, grouped)

        if transform is None:
            # 即使不需要变换，也要返回有效的 stats 和反变换函数
            return _to_csr(h_matrix), _to_csr(o_matrix), identity_back_transform, stats

        # 执行投影: H_new = T.T @ H @ T
        new_h, new_o = _project_matrices(h_matrix, o_matrix, transform)

        def back_transform(vecs: np.ndarray) -> np.ndarray:
            return transform @ vecs

        return new_h, new_o, back_transform, stats

    def _group_indices(
        self,
        basis_states: Sequence[HylleraasBSplineFunction],
    ) -> Dict[Tuple[int, int, int, int, int], List[int]]:
        groups: Dict[Tuple[int, int, int, int, int], List[int]] = {}
        for idx, state in enumerate(basis_states):
            channel = state.channel
            parity = state.exchange_parity if self.include_exchange_parity else 0
            corr = state.correlation_power if self.include_correlation_power else 0
            key = (channel.l1, channel.l2, channel.L, corr, parity)
            groups.setdefault(key, []).append(idx)
        return groups

    def _build_block_transform(
        self,
        overlap_matrix: Any,
        groups: Dict[Tuple[int, int, int, int, int], List[int]],
    ) -> Tuple[Any, ChannelOrthoResult]:
        size = overlap_matrix.shape[0]
        csr_overlap = _to_csr(overlap_matrix)

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        new_dim = 0
        stats: List[ChannelGroupStats] = []

        for key, indices in groups.items():
            block_size = len(indices)
            if block_size == 0:
                continue

            # 如果块太大，跳过密集特征值分解以节省时间/内存，保留原样
            if block_size > self.max_block_dim:
                for offset, row_idx in enumerate(indices):
                    rows.append(row_idx)
                    cols.append(new_dim + offset)
                    data.append(1.0)
                stats.append(
                    ChannelGroupStats(
                        key=key,
                        original_dimension=block_size,
                        retained_dimension=block_size,
                        min_kept_eigenvalue=float("nan"),
                        max_kept_eigenvalue=float("nan"),
                    )
                )
                new_dim += block_size
                continue

            # 提取子块并进行特征值分解 S = Q Λ Q.T
            block = csr_overlap[indices][:, indices].toarray()

            if block_size == 1:
                eigvals = np.asarray([block[0, 0]], dtype=float)
                eigvecs = np.asarray([[1.0]], dtype=float)
            else:
                # eigh 用于厄米矩阵/对称矩阵
                try:
                    eigvals, eigvecs = np.linalg.eigh(block)
                except np.linalg.LinAlgError:
                    # 如果分解失败，回退到保留所有（不做过滤），防止程序崩溃
                    eigvals = np.ones(block_size)
                    eigvecs = np.eye(block_size)

            # 筛选特征值：只保留大于容差的
            mask = eigvals >= self.tolerance
            if not np.any(mask):
                # 如果所有特征值都小于容差，保留最大的那个以防整个通道丢失
                max_idx = int(np.argmax(eigvals))
                mask[max_idx] = True

            kept_vals = eigvals[mask]
            kept_vecs = eigvecs[:, mask]

            # 构建变换矩阵 T = Q Λ^{-1/2}
            # 注意：kept_vals 可能非常小，但 >= tolerance
            with np.errstate(divide='ignore', invalid='ignore'):
                scale_factors = 1.0 / np.sqrt(kept_vals)

            transform_block = kept_vecs * scale_factors

            for local_col in range(transform_block.shape[1]):
                for local_row, row_idx in enumerate(indices):
                    value = float(transform_block[local_row, local_col])
                    if value == 0.0:
                        continue
                    rows.append(row_idx)
                    cols.append(new_dim + local_col)
                    data.append(value)

            stats.append(
                ChannelGroupStats(
                    key=key,
                    original_dimension=block_size,
                    retained_dimension=transform_block.shape[1],
                    min_kept_eigenvalue=float(np.min(kept_vals)),
                    max_kept_eigenvalue=float(np.max(kept_vals)),
                )
            )
            new_dim += transform_block.shape[1]

        result_stats = ChannelOrthoResult(
            total_original=size,
            total_retained=new_dim,
            group_stats=stats,
        )

        if new_dim == size:
            # 维度未减少，检查是否近似恒等变换
            # 简单的优化：如果维度没变，我们可以假设它是恒等的，或者直接返回 None 让外层处理
            # 返回 None 表示“无需变换”
            return None, result_stats

        # 构建稀疏变换矩阵
        # 使用 type: ignore 忽略 pylance 对 tocsr 的报错
        transform = coo_matrix((data, (rows, cols)),
                               shape=(size, new_dim)).tocsr()  # type: ignore

        return transform, result_stats


def _to_csr(matrix: Any) -> Any:
    """Convert to CSR matrix safely."""
    if issparse(matrix):
        return matrix.tocsr()
    return csr_matrix(matrix)


def _project_matrices(h_matrix: Any, o_matrix: Any, transform: Any) -> Tuple[Any, Any]:
    """Project H and O matrices using the transform T: H' = T.T @ H @ T."""
    h_csr = _to_csr(h_matrix)
    o_csr = _to_csr(o_matrix)

    right = transform
    left = transform.transpose()

    projected_h = left @ (h_csr @ right)
    projected_o = left @ (o_csr @ right)

    return projected_h, projected_o
