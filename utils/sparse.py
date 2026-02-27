import numpy as np
from scipy import sparse

import moreau


def sparse_to_csr(mat):
    """Convert a scipy sparse matrix to CSR components.

    Parameters
    ----------
    mat : scipy.sparse matrix
        Any scipy sparse format. Will be converted to CSR.

    Returns
    -------
    row_offsets : np.ndarray[int64]
    col_indices : np.ndarray[int64]
    values : np.ndarray[float64]
    """
    csr = sparse.csr_array(mat, dtype=np.float64)
    csr.sort_indices()
    return (
        csr.indptr.astype(np.int64),
        csr.indices.astype(np.int64),
        csr.data.astype(np.float64),
    )


def build_solver(P, q, A, b, cones, batch_size=1, device="auto", **kwargs):
    """Build a CompiledSolver from scipy sparse matrices.

    Parameters
    ----------
    P : scipy.sparse matrix, shape (n, n)
        Quadratic cost (full symmetric).
    q : np.ndarray, shape (n,) or (batch, n)
        Linear cost.
    A : scipy.sparse matrix, shape (m, n)
        Constraint matrix.
    b : np.ndarray, shape (m,) or (batch, m)
        Constraint RHS.
    cones : moreau.Cones
    batch_size : int
    device : str
    **kwargs : passed to moreau.Settings

    Returns
    -------
    solver : moreau.CompiledSolver (already set up)
    """
    P_ro, P_ci, P_vals = sparse_to_csr(P)
    A_ro, A_ci, A_vals = sparse_to_csr(A)

    n = P.shape[0]
    m = A.shape[0]

    settings = moreau.Settings(batch_size=batch_size, device=device, **kwargs)
    solver = moreau.CompiledSolver(
        n=n, m=m,
        P_row_offsets=P_ro,
        P_col_indices=P_ci,
        A_row_offsets=A_ro,
        A_col_indices=A_ci,
        cones=cones,
        settings=settings,
    )
    solver.setup(P_vals, A_vals)
    return solver
