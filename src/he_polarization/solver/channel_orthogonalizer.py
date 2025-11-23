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

    tolerance: float = 1e-12
    max_block_dim: int = 2048
    include_exchange_parity: bool = True
    include_correlation_power: bool = True

    # === 新增参数：安全模式 ===
    safe_mode: bool = False
    safe_mode_threshold: float = -1e15  # 如果能量低于此值（更负），则强制保留
    # ========================

    def apply(
        self,
        h_matrix: Any,
        o_matrix: Any,
        basis_states: Sequence[HylleraasBSplineFunction],
    ) -> Tuple[Any, Any, _BackTransform, ChannelOrthoResult]:
        size = len(basis_states)

        def identity_back_transform(vecs: np.ndarray) -> np.ndarray:
            return vecs

        if size == 0:
            return _to_csr(h_matrix), _to_csr(o_matrix), identity_back_transform, ChannelOrthoResult(size, size, [])

        grouped = self._group_indices(basis_states)
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
        """
        [终极理论版] 严格依据群表示论分组：
        仅依据对称群 G 的守恒量子数 (L, Parity) 进行分组。

        理论依据：
        1. (l1, l2) 是组态标记，不是守恒量 -> 合并
        2. correlation_power (c) 是标量算符 r12 的幂次，不改变对称性 -> 合并

        结果：
        每个 Group 对应一个严格的不可约表示子空间 H_Gamma。
        根据舒尔引理，不同 Group 之间的矩阵元在理论上严格为 0。
        """
        # Key 格式: (l1, l2, L, corr, parity)
        # 我们只保留 L 和 parity，其他位置填 -1 占位
        groups: Dict[Tuple[int, int, int, int, int], List[int]] = {}

        for idx, state in enumerate(basis_states):
            channel = state.channel
            # exchange_parity 对应自旋单/三重态 (S) 和空间宇称 (P) 的组合
            # 对于氦原子，它是严格的好量子数。
            parity = state.exchange_parity if self.include_exchange_parity else 0

            # 终极修改：l1, l2, corr 全部视为内部自由度，用 -1 屏蔽
            key = (-1, -1, channel.L, -1, parity)

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
        csr_hamiltonian = _to_csr(h_matrix)

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
                # Block too large, keep all
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

            # 1. 对角化重叠矩阵子块
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

            # 2. 初步决定保留哪些态 (基于 tolerance)
            mask = eigvals >= self.tolerance

            # === 关键逻辑：物理安全检查与自动救援 ===
            if self.safe_mode:
                # 找出初步决定丢弃的索引
                discard_indices = np.where(~mask)[0]

                if len(discard_indices) > 0:
                    # 提取对应的哈密顿量矩阵元
                    h_block = csr_hamiltonian[indices][:, indices].toarray()

                    # 获取被丢弃的向量
                    discarded_vecs = eigvecs[:, discard_indices]
                    discarded_vals = eigvals[discard_indices]

                    # 计算 Rayleigh Quotient: E = <v|H|v> / <v|S|v>
                    # <v|S|v> 就是对应的特征值 lambda (discarded_vals)

                    hv = h_block @ discarded_vecs
                    # 计算对角元 sum(v_i * (Hv)_i)
                    h_expects = np.sum(discarded_vecs.conj() * hv, axis=0)

                    # 加上 1e-30 防止除零
                    pseudo_energies = np.real(
                        h_expects) / (discarded_vals + 1e-30)

                    # 判断哪些态虽然 lambda 小，但能量很低（物理态）
                    to_rescue_local_indices = np.where(
                        pseudo_energies < self.safe_mode_threshold)[0]

                    if len(to_rescue_local_indices) > 0:
                        # 将这些态强制加回 mask
                        indices_to_rescue = discard_indices[to_rescue_local_indices]
                        mask[indices_to_rescue] = True

                        rescued_min_lambda = np.min(
                            discarded_vals[to_rescue_local_indices])
                        print(f"  [RESCUE] Channel {key}: Rescued {len(to_rescue_local_indices)} physical states "
                              f"(min λ={rescued_min_lambda:.1e})")

            # 防止 mask 全空（至少保留一个最大的）
            if not np.any(mask):
                max_idx = int(np.argmax(eigvals))
                mask[max_idx] = True

            kept_vals = eigvals[mask]
            kept_vecs = eigvecs[:, mask]

            # 3. 构建变换矩阵 T = U * Lambda^(-1/2)
            # 使用 abs 防止数值噪声导致的微小负值在 sqrt 时产生 NaN
            with np.errstate(divide='ignore', invalid='ignore'):
                scale_factors = 1.0 / np.sqrt(np.abs(kept_vals))

            transform_block = kept_vecs * scale_factors

            # 4. 组装稀疏矩阵
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
