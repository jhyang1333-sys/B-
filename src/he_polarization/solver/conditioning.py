"""Tools for stabilizing generalized eigenproblems via overlap conditioning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.sparse import identity as sparse_identity
from scipy.sparse import issparse


_BackTransform = Callable[[np.ndarray], np.ndarray]


def _as_dense(matrix) -> np.ndarray:
    if issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix, dtype=float)


def _project_matrix(matrix, transform: np.ndarray) -> np.ndarray:
    if issparse(matrix):
        product = matrix.tocsr().dot(transform)
    else:
        product = np.asarray(matrix, dtype=float) @ transform
    reduced = transform.T @ product
    # Enforce symmetry to damp numerical noise
    return 0.5 * (reduced + reduced.T)


@dataclass
class OverlapConditionResult:
    """Contains conditioning metadata for logging or inspection."""

    strategy: str
    retained_dimension: int
    discarded_dimension: int
    min_kept_eigenvalue: float
    max_kept_eigenvalue: float


@dataclass
class OverlapConditioner:
    """Canonically orthonormalize the overlap matrix to improve stability."""

    tolerance: float = 1e-10
    max_dense_dim: int = 4096
    enabled: bool = True
    mode: str = "auto"
    regularization: float = 1e-8

    def condition(self, h_matrix, o_matrix) -> Tuple[np.ndarray, np.ndarray, Optional[_BackTransform], OverlapConditionResult | None]:
        if not self.enabled:
            return h_matrix, o_matrix, None, None

        size = o_matrix.shape[0]
        if size == 0:
            return h_matrix, o_matrix, None, None

        strategy = self.mode.lower()
        if strategy not in {"auto", "dense", "regularize"}:
            strategy = "auto"

        use_dense = strategy == "dense" or (
            strategy == "auto" and size <= self.max_dense_dim)
        if not use_dense:
            conditioned_overlap = _apply_regularization(
                o_matrix, self.regularization)
            result = OverlapConditionResult(
                strategy="regularize",
                retained_dimension=size,
                discarded_dimension=0,
                min_kept_eigenvalue=float("nan"),
                max_kept_eigenvalue=float("nan"),
            )
            return h_matrix, conditioned_overlap, None, result

        dense_overlap = _as_dense(o_matrix)
        eigvals, eigvecs = np.linalg.eigh(dense_overlap)
        mask = eigvals >= self.tolerance
        retained = int(np.count_nonzero(mask))

        if retained == 0:
            raise RuntimeError(
                "Overlap conditioning failed: no eigenvalues above tolerance.")
        if retained == size:
            return h_matrix, o_matrix, None, OverlapConditionResult(
                strategy="dense",
                retained_dimension=retained,
                discarded_dimension=0,
                min_kept_eigenvalue=float(np.min(eigvals)),
                max_kept_eigenvalue=float(np.max(eigvals)),
            )

        kept_eigvals = eigvals[mask]
        kept_eigvecs = eigvecs[:, mask]
        transform = kept_eigvecs / np.sqrt(kept_eigvals)

        projected_h = _project_matrix(h_matrix, transform)
        projected_o = np.eye(retained, dtype=float)

        def back_transform(vecs: np.ndarray) -> np.ndarray:
            return transform @ vecs

        result = OverlapConditionResult(
            strategy="dense",
            retained_dimension=retained,
            discarded_dimension=size - retained,
            min_kept_eigenvalue=float(np.min(kept_eigvals)),
            max_kept_eigenvalue=float(np.max(kept_eigvals)),
        )
        return projected_h, projected_o, back_transform, result


def _apply_regularization(matrix, value: float):
    if value <= 0.0:
        return matrix
    size = matrix.shape[0]
    if issparse(matrix):
        dtype = getattr(matrix, "dtype", None)
        ident = sparse_identity(
            size, dtype=dtype, format="csr") if dtype is not None else sparse_identity(size, format="csr")
        return matrix + value * ident
    dense = np.asarray(matrix, dtype=float).copy()
    diag_idx = np.diag_indices(size)
    dense[diag_idx] += value
    return dense
