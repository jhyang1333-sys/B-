"""广义本征问题求解，对应论文式 (2.15)。

实现一个不依赖 SciPy 的数值流程，适配 Python 3.14 环境中
LAPACK 接口不稳定的问题。
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def solve_generalized_eigen(H: np.ndarray, O: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """求解 ``HC = EOC``，返回本征值与 O-正交归一化的本征矢。"""

    if H.shape != O.shape:
        raise ValueError("H 与 O 的维度必须一致。")

    H = np.asarray(H, dtype=float)
    O = np.asarray(O, dtype=float)

    H = 0.5 * (H + H.T)
    O = 0.5 * (O + O.T)

    O_reg = _regularize_overlap(O)
    L = _cholesky_lower(O_reg)

    invL_H = _forward_substitution(L, H)
    A_std = _back_substitution(L.T, invL_H)
    A_std = 0.5 * (A_std + A_std.T)

    eigvals, eigvecs_std = _jacobi_eigh(A_std)

    eigenvectors = _back_substitution(L.T, eigvecs_std)
    gram = eigenvectors.T @ O @ eigenvectors
    norms = np.sqrt(np.clip(np.diag(gram), 1e-30, None))
    eigenvectors = eigenvectors / norms

    order = np.argsort(eigvals)
    return eigvals[order], eigenvectors[:, order]


def _regularize_overlap(O: np.ndarray) -> np.ndarray:
    diag = np.abs(np.diag(O))
    scale = float(np.max(diag)) if np.max(diag) > 0 else 1.0
    eps = 1e-10 * scale
    for _ in range(12):
        try:
            O_reg = O.copy()
            np.fill_diagonal(O_reg, np.diag(O_reg) + eps)
            _cholesky_lower(O_reg)
            return O_reg
        except ValueError:
            eps *= 10.0
    raise ValueError("无法对交叠矩阵进行正则化，仍非正定。")


def _cholesky_lower(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    L = np.zeros_like(matrix)
    for i in range(n):
        for j in range(i + 1):
            acc = 0.0
            row_i = L[i]
            row_j = L[j]
            for k in range(j):
                acc += row_i[k] * row_j[k]
            summation = matrix[i, j] - acc
            if i == j:
                if summation <= 0:
                    raise ValueError("交叠矩阵非正定，无法进行 Cholesky 分解。")
                L[i, j] = math.sqrt(summation)
            else:
                if L[j, j] == 0:
                    raise ValueError("交叠矩阵在 Cholesky 分解中出现零主元。")
                L[i, j] = summation / L[j, j]
    return L


def _forward_substitution(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    B = np.asarray(B, dtype=float)
    n = L.shape[0]
    m = B.shape[1]
    Y = np.zeros_like(B)
    for col in range(m):
        for i in range(n):
            acc = 0.0
            row = L[i]
            for k in range(i):
                acc += row[k] * Y[k, col]
            value = B[i, col] - acc
            Y[i, col] = value / L[i, i]
    return Y


def _back_substitution(U: np.ndarray, B: np.ndarray) -> np.ndarray:
    B = np.asarray(B, dtype=float)
    n = U.shape[0]
    m = B.shape[1]
    X = np.zeros_like(B)
    for col in range(m):
        for i in range(n - 1, -1, -1):
            acc = 0.0
            row = U[i]
            for k in range(i + 1, n):
                acc += row[k] * X[k, col]
            value = B[i, col] - acc
            X[i, col] = value / U[i, i]
    return X


def _jacobi_eigh(matrix: np.ndarray, tol: float = 1e-12, max_sweeps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    A = np.array(matrix, dtype=float, copy=True)
    n = A.shape[0]
    V = np.eye(n, dtype=float)

    for _ in range(max_sweeps):
        max_off = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p, q]
                if abs(apq) <= tol:
                    continue
                app = A[p, p]
                aqq = A[q, q]
                tau = (aqq - app) / (2.0 * apq)
                t = math.copysign(1.0, tau) / (abs(tau) +
                                               math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c

                for k in range(n):
                    if k != p and k != q:
                        akp = A[k, p]
                        akq = A[k, q]
                        A[k, p] = akp * c - akq * s
                        A[p, k] = A[k, p]
                        A[k, q] = akq * c + akp * s
                        A[q, k] = A[k, q]

                A[p, p] = app - t * apq
                A[q, q] = aqq + t * apq
                A[p, q] = 0.0
                A[q, p] = 0.0

                for k in range(n):
                    vip = V[k, p]
                    viq = V[k, q]
                    V[k, p] = vip * c - viq * s
                    V[k, q] = viq * c + vip * s

                max_off = max(max_off, abs(apq))
        if max_off < tol:
            break

    eigenvalues = np.diag(A).copy()
    return eigenvalues, V
