import numpy as np

from he_polarization.basis import (
    AngularCoupling,
    AtomicChannel,
    BSplineBasis,
    ExponentialNodeConfig,
    generate_exponential_nodes,
    generate_hylleraas_bspline_functions,
)
from he_polarization.hamiltonian import HamiltonianOperators, MatrixElementBuilder
from he_polarization.numerics import generate_tensor_product_quadrature
from he_polarization.solver import (
    IterativeSolverConfig,
    solve_generalized_eigen,
    solve_sparse_generalized_eigen,
)

try:
    from scipy.sparse import issparse
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "SciPy is required for this comparison script.") from exc


def main() -> None:
    tau = 0.038
    r_max = 200.0
    k = 5
    n = 5
    l_max = 1

    config = ExponentialNodeConfig(
        r_min=0.0, r_max=r_max, k=k, n=n, gamma=r_max * tau)
    knots = generate_exponential_nodes(config)
    bspline = BSplineBasis(knots=knots, order=k)
    angular = AngularCoupling()
    m_nucleus = 7294.2995365
    mu = m_nucleus / (1.0 + m_nucleus)
    operators = HamiltonianOperators(mu=mu, M=m_nucleus)
    builder = MatrixElementBuilder(
        bspline=bspline, angular=angular, operators=operators)

    channels = [
        AtomicChannel(l1=l1, l2=l2, L=L)
        for l1 in range(l_max + 1)
        for l2 in range(l_max + 1)
        for L in range(abs(l1 - l2), min(l1 + l2, l_max) + 1)
    ]

    basis_states = generate_hylleraas_bspline_functions(
        bspline,
        channels,
        n_radial=bspline.n_basis,
        correlation_powers=(0, 1),
        exchange_parity=1,
        symmetrize=True,
        unique_pairs=True,
    )

    points, weights = generate_tensor_product_quadrature(
        config.r_min,
        config.r_max,
        n_points=8,
    )

    H, O, _ = builder.assemble_matrices(
        basis_states, weights=weights, points=points)

    if issparse(H) or issparse(O):
        eigvals, eigvecs = solve_sparse_generalized_eigen(
            H,
            O,
            config=IterativeSolverConfig(num_eigenvalues=2),
        )
    else:
        dense_H = np.asarray(H.toarray() if hasattr(H, "toarray") else H)
        dense_O = np.asarray(O.toarray() if hasattr(O, "toarray") else O)
        eigvals, eigvecs = solve_generalized_eigen(dense_H, dense_O)

    print("Eigenvalues:", eigvals[:2])


if __name__ == "__main__":
    main()
