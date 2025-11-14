"""Iterative sparse generalized eigen solvers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


@dataclass
class IterativeSolverConfig:
    """Configuration for the sparse generalized eigen solver."""

    num_eigenvalues: int = 10
    which: str = "SA"
    sigma: Optional[float] = None
    tol: float = 1e-10
    maxiter: Optional[int] = None


def solve_sparse_generalized_eigen(
        H,
        O,
        *,
        config: Optional[IterativeSolverConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve ``H C = E O C`` using sparse iterative methods.

    Parameters
    ----------
    H, O
            Hamiltonian and overlap matrices. Either sparse matrices or ndarray.
    config
            Solver configuration. Defaults target the lowest eigenpairs.
    """

    try:
        from scipy.sparse import csc_matrix, csr_matrix, issparse
        from scipy.sparse.linalg import eigsh
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "需要安装 SciPy 才能使用稀疏广义本征求解器。"
        ) from exc

    if config is None:
        config = IterativeSolverConfig()

    if config.num_eigenvalues <= 0:
        raise ValueError("num_eigenvalues 必须为正整数。")

    H_sparse = H if issparse(H) else np.asarray(H)
    O_sparse = O if issparse(O) else np.asarray(O)

    n = H_sparse.shape[0]
    if n != H_sparse.shape[1] or O_sparse.shape != H_sparse.shape:
        raise ValueError("H 与 O 必须是方阵且维度一致。")

    if config.num_eigenvalues >= n:
        # Defer to dense solver when all eigenpairs are requested.
        from .generalized_eigen import solve_generalized_eigen

        dense_H = _to_dense(H_sparse)
        dense_O = _to_dense(O_sparse)
        return solve_generalized_eigen(dense_H, dense_O)

    # Enforce Hermitian structure before calling eigsh.
    H_sparse = 0.5 * (H_sparse + H_sparse.T)
    O_sparse = 0.5 * (O_sparse + O_sparse.T)

    # SciPy 的 eigsh 要求 CSR/CSC 格式才能高效执行。
    H_sparse = _to_csr(H_sparse, csr_matrix, issparse)
    O_sparse = _to_csr(O_sparse, csr_matrix, issparse)

    eigvals, eigvecs = eigsh(
        H_sparse,
        k=config.num_eigenvalues,
        M=O_sparse,
        sigma=config.sigma,
        which=config.which,
        tol=float(config.tol),  # type: ignore[arg-type]
        maxiter=config.maxiter,
        return_eigenvectors=True,
    )

    eigvals, eigvecs = _postprocess_eigensystem(
        eigvals, eigvecs, O_sparse
    )
    return eigvals, eigvecs


def _to_dense(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def _to_csr(matrix: Any, csr_matrix, issparse):
    if issparse(matrix):
        return matrix.tocsr()
    return csr_matrix(np.asarray(matrix))


def _postprocess_eigensystem(eigvals, eigvecs, overlap_matrix):
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    O_eig = overlap_matrix @ eigvecs
    gram = eigvecs.T @ O_eig
    diag = np.diag(gram)
    diag = np.where(diag > 0.0, diag, 1.0)
    norms = np.sqrt(diag)
    eigvecs = eigvecs / norms
    return eigvals, eigvecs
