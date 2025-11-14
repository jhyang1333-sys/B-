"""Persistence helpers for solver outputs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

try:
    from scipy.sparse import issparse, save_npz, load_npz, coo_matrix
except ModuleNotFoundError:  # pragma: no cover
    issparse = None  # type: ignore[assignment]
    save_npz = load_npz = None  # type: ignore[assignment]
    coo_matrix = None  # type: ignore[assignment]


@dataclass
class SolverResult:
    energies: np.ndarray
    eigenvectors: np.ndarray
    components: Dict[str, Any]


class SolverResultCache:
    """Cache dense/sparse solver outputs on disk."""

    spectra_filename = "spectra.npz"
    metadata_filename = "metadata.json"

    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path

    def _resolve(self, name: str) -> Path:
        return self.base_path / name

    def available(self, name: str, *, metadata: Mapping[str, object]) -> bool:
        path = self._resolve(name)
        spectra = path / self.spectra_filename
        meta_file = path / self.metadata_filename
        if not spectra.exists() or not meta_file.exists():
            return False
        try:
            with meta_file.open("r", encoding="utf-8") as handle:
                stored = json.load(handle)
        except json.JSONDecodeError:
            return False
        return stored == dict(metadata)

    def load(self, name: str) -> SolverResult:
        path = self._resolve(name)
        spectra = np.load(path / self.spectra_filename)
        energies = spectra["energies"]
        eigenvectors = spectra["eigenvectors"]
        components: Dict[str, Any] = {}
        if issparse is None or load_npz is None:
            raise ModuleNotFoundError(
                "SciPy is required to load sparse matrices from cache.")
        assert issparse is not None  # type: ignore[redundant-cast]
        assert load_npz is not None
        for key in ("overlap", "kinetic", "mass", "potential", "hamiltonian"):
            file = path / f"{key}.npz"
            if file.exists():
                components[key] = load_npz(file)
        if "hamiltonian" not in components and {"kinetic", "potential"}.issubset(components):
            total_h = components["kinetic"] + components["potential"]
            mass = components.get("mass")
            if mass is not None:
                total_h = total_h + mass
            components["hamiltonian"] = total_h
        return SolverResult(energies=energies, eigenvectors=eigenvectors, components=components)

    def save(
        self,
        name: str,
        *,
        energies: np.ndarray,
        eigenvectors: np.ndarray,
        components: Mapping[str, Any],
        metadata: Mapping[str, object],
    ) -> None:
        path = self._resolve(name)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / self.spectra_filename,
                 energies=energies, eigenvectors=eigenvectors)
        if issparse is None or save_npz is None or coo_matrix is None:
            raise ModuleNotFoundError(
                "SciPy is required to persist sparse matrices.")
        assert issparse is not None  # type: ignore[redundant-cast]
        assert save_npz is not None
        assert coo_matrix is not None
        for key, matrix in components.items():
            if matrix is None:
                continue
            if not issparse(matrix):
                matrix = coo_matrix(matrix)
            save_npz(path / f"{key}.npz", matrix)
        with (path / self.metadata_filename).open("w", encoding="utf-8") as handle:
            json.dump(dict(metadata), handle, ensure_ascii=False,
                      indent=2, sort_keys=True)

    def drop(self, name: str) -> None:
        path = self._resolve(name)
        if not path.exists():
            return
        for item in path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                self._remove_tree(item)
        path.rmdir()

    def _remove_tree(self, root: Path) -> None:
        for item in root.iterdir():
            if item.is_dir():
                self._remove_tree(item)
            else:
                item.unlink()
        root.rmdir()
