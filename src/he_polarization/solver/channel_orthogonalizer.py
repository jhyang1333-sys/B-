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

        def identity_back_transform(vecs: np.ndarray) -> np.ndarray:
            return vecs

        if size == 0 or (self.tolerance <= 0):
            return _to_csr(h_matrix), _to_csr(o_matrix), identity_back_transform, ChannelOrthoResult(size, size, [])

        grouped = self._group_indices(basis_states)
        # 传入 h_matrix 用于物理验证
        transform, stats = self._build_block_transform(
            h_matrix, o_matrix, grouped)

        if transform is None:
            return _to_csr(h_matrix), _to_csr(o_matrix), identity_back_transform, stats

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
        h_matrix: Any,
        overlap_matrix: Any,
        groups: Dict[Tuple[int, int, int, int, int], List[int]],
    ) -> Tuple[Any, ChannelOrthoResult]:
        size = overlap_matrix.shape[0]
        csr_overlap = _to_csr(overlap_matrix)
        csr_hamiltonian = _to_csr(h_matrix)  # 转为 CSR 以便快速切片

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        new_dim = 0
        stats: List[ChannelGroupStats] = []

        for key, indices in groups.items():
            block_size = len(indices)
            if block_size == 0:
                continue

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

            # 提取重叠矩阵子块
            block = csr_overlap[indices][:, indices].toarray()

            if block_size == 1:
                eigvals = np.asarray([block[0, 0]], dtype=float)
                eigvecs = np.asarray([[1.0]], dtype=float)
            else:
                try:
                    eigvals, eigvecs = np.linalg.eigh(block)
                except np.linalg.LinAlgError:
                    eigvals = np.ones(block_size)
                    eigvecs = np.eye(block_size)

            # --- 验证逻辑 ---
            # 1. 检查条件数
            max_eval = np.max(eigvals)
            min_eval = np.min(eigvals)
            if min_eval < 1e-16:
                min_eval = 1e-16
            cond_num = max_eval / min_eval
            # if cond_num > 1e14:
            # print(f"  [WARN] Channel {key}: Condition number {cond_num:.1e} is very large!")

            # 2. 检查将被丢弃态的物理能量 (Rayleigh Quotient)
            discard_mask = eigvals < self.tolerance
            if np.any(discard_mask):
                h_block = csr_hamiltonian[indices][:, indices].toarray()

                # 获取被丢弃的向量 (N, k)
                discarded_vecs = eigvecs[:, discard_mask]
                discarded_vals = eigvals[discard_mask]

                # --- [修复] 使用矩阵乘法代替 einsum 以避免广播错误 ---
                # 计算 H @ V
                hv_product = h_block @ discarded_vecs
                # 计算对角元 <v|H|v> = sum(v.conj * (H @ v), axis=0)
                h_expects = np.sum(discarded_vecs.conj() * hv_product, axis=0)

                # 瑞利商 E = <H>/<S> = <H>/lambda
                pseudo_energies = np.real(h_expects) / (discarded_vals + 1e-30)

                min_pseudo_E = np.min(pseudo_energies)
                # 物理基态约为 -2.9 a.u.。如果丢弃了能量 < -2.0 的态，说明误删了重要物理成分。
                if min_pseudo_E < -2.0:
                    print(
                        f"  [ALARM] Channel {key}: Dropping state with physical energy {min_pseudo_E:.2f} a.u.!")
                    print(
                        f"          Tolerance {self.tolerance:.1e} might be too high.")
            # -------------------------

            mask = eigvals >= self.tolerance
            if not np.any(mask):
                max_idx = int(np.argmax(eigvals))
                mask[max_idx] = True

            kept_vals = eigvals[mask]
            kept_vecs = eigvecs[:, mask]

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
            return None, result_stats

        # type: ignore
        transform = coo_matrix((data, (rows, cols)),
                               shape=(size, new_dim)).tocsr()

        return transform, result_stats


def _to_csr(matrix: Any) -> Any:
    if issparse(matrix):
        return matrix.tocsr()
    return csr_matrix(matrix)


def _project_matrices(h_matrix: Any, o_matrix: Any, transform: Any) -> Tuple[Any, Any]:
    h_csr = _to_csr(h_matrix)
    o_csr = _to_csr(o_matrix)

    right = transform
    left = transform.transpose()

    projected_h = left @ (h_csr @ right)
    projected_o = left @ (o_csr @ right)

    return projected_h, projected_o
