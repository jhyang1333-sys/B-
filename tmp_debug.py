from he_polarization.numerics import generate_tensor_product_quadrature
from he_polarization.hamiltonian import HamiltonianOperators, MatrixElementBuilder
from he_polarization.basis import (
    AngularCoupling,
    AtomicChannel,
    BSplineBasis,
    ExponentialNodeConfig,
    generate_exponential_nodes,
    generate_hylleraas_bspline_functions,
)
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent / "src"))


config = ExponentialNodeConfig(r_min=0.0, r_max=5.0, k=3, n=3, gamma=0.1)
knots = generate_exponential_nodes(config)
bspline = BSplineBasis(knots=knots, order=3)
angular = AngularCoupling()
M = 7294.2995365
mu = M / (1.0 + M)
operators = HamiltonianOperators(mu=mu, M=M)

builder = MatrixElementBuilder(
    bspline=bspline, angular=angular, operators=operators)
channels = [AtomicChannel(l1=0, l2=0, L=0)]
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
    config.r_min, config.r_max, n_points=4)

print("basis states:", len(basis_states))

H, S, components = builder.assemble_matrices(
    basis_states,
    weights=weights,
    points=points,
)
print("matrix shapes:", H.shape, S.shape)
