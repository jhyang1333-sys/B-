import numpy as np
import traceback

try:
    from he_polarization.solver import solve_generalized_eigen
except Exception as exc:
    print("import error", exc)
    traceback.print_exc()
    raise

rng = np.random.default_rng(0)
A = rng.random((4, 4))
H = 0.5 * (A + A.T)
B = rng.random((4, 4))
O = 0.5 * (B + B.T) + 4.0 * np.eye(4)

try:
    vals, vecs = solve_generalized_eigen(H, O)
    print("vals", vals)
    print("orth", np.allclose(vecs.T @ O @ vecs, np.eye(4), atol=1e-6))
except Exception as exc:
    print("solve error", exc)
    traceback.print_exc()
    raise
