r"""Symmetric Sinkhorn normalization solvers.

Given a symmetric positive kernel $K$ and masses $m$ (the diagonal
of $M$), these solvers compute the unique positive diagonal scaling
$\Lambda$ such that $Q = \Lambda K M \Lambda$ is a diffusion
operator: $Q 1 = 1$, $Q$ is self-adjoint for the mass-weighted
inner product, and (for PSD kernels) has spectrum in $[0, 1]$.

Two families of solvers are provided:

- **Log-domain** (``sinkhorn_log``, ``sinkhorn_log_sparse``, and the general
  ``sinkhorn_log_kernel``): for kernels given through a **log-kernel**
  $\log K_{ij}$ (from squared distances for the Gaussian and exponential
  kernels, or an arbitrary matrix, e.g. a Gaussian mixture).
  Numerically stable for any bandwidth; they return log-potentials ``f`` with
  $\Lambda = \mathrm{diag}(e^f)$. The update is;

  $$
     f_i \leftarrow \tfrac12 f_i - \tfrac12 \log \sum_j
        \exp\big(\log K_{ij} + f_j + \log m_j\big).
  $$

  This avoids underflow (Gaussian, exponential, GMM), especially in float32.

- **Multiplicative** (``sinkhorn``): Algorithm 1 of the paper, for arbitrary
  kernels given as a matrix or a black-box matvec (``scipy`` ``LinearOperator``),
  e.g. a grid convolution. Returns scalings ``scaling`` ($\lambda = e^f$, with 0 allowed outside the support):

  $$
     \lambda \leftarrow \sqrt{\lambda \oslash (K M \lambda)}.
  $$

  Use it for kernels that don't underflow, or given as black boxes.

In practice 5-10 iterations is enough for convergence.
"""

import numpy as np
import scipy.sparse as sparse
from scipy.special import logsumexp

_KERNELS = ("gaussian", "exponential")


def _marginal_error(rowsums, masses):
    """L1(mu)-averaged violation of the unit row-sum constraint."""
    if masses is None:
        return np.abs(rowsums - 1.0).mean()
    return np.average(np.abs(rowsums - 1.0), weights=masses)


def _log_kernel_from_sqdist(sqdist_values, sigma, kernel):
    """Log-kernel values from (squared) distances.

    Gaussian: $\\log K = -d^2 / 2\\sigma^2$ ($\\varepsilon = \\sigma^2$).
    Exponential: $\\log K = -\\|x-y\\| / \\sigma$ ($\\varepsilon = \\sigma$).
    Works on dense arrays or on the ``.data`` of a sparse matrix.
    """
    if kernel == "gaussian":
        return -sqdist_values / (2 * sigma**2)
    if kernel == "exponential":
        return -np.sqrt(sqdist_values) / sigma
    raise ValueError(f"Unknown kernel {kernel!r}; expected one of {_KERNELS}.")


def _sinkhorn_log_dense(
    log_kernel, masses=None, n_iter=10, tol=None, f_init=None, full_output=False
):
    """Shared dense log-domain solver operating on a full log-kernel matrix.

    ``log_kernel`` is $\\log K$ (N, N), diagonal included.
    """
    log_kernel = np.asarray(log_kernel)  # (N, N)
    N = log_kernel.shape[0]
    dtype = log_kernel.dtype

    # f: (N,) log-potentials;  log_m: (N,) log-masses
    f = np.zeros(N, dtype=dtype) if f_init is None else np.asarray(f_init).astype(dtype).copy()
    log_m = None if masses is None else np.log(masses).astype(dtype)

    def _lse(f):
        # arg: (N, N) exponents;  logsumexp over columns -> (N,) row reductions
        arg = log_kernel + f[None, :] if log_m is None else log_kernel + (f + log_m)[None, :]
        return logsumexp(arg, axis=1)

    n_done = 0
    for _ in range(n_iter):
        lse = _lse(f)
        if tol is not None and _marginal_error(np.exp(f + lse), masses) <= tol:
            break
        f = 0.5 * f - 0.5 * lse
        n_done += 1

    if not full_output:
        return f
    err = _marginal_error(np.exp(f + _lse(f)), masses)
    return f, {"n_iter": n_done, "marginal_error": err}


def sinkhorn_log(
    sqdist,
    sigma,
    masses=None,
    kernel="gaussian",
    n_iter=10,
    tol=None,
    f_init=None,
    full_output=False,
):
    """Dense log-domain symmetric Sinkhorn from (squared) distances.

    Normalizes the Gaussian kernel $K_{ij} = \\exp(-d_{ij}^2 / 2\\sigma^2)$
    (``kernel="gaussian"``) or the exponential/Laplace kernel
    $K_{ij} = \\exp(-\\|x_i-x_j\\| / \\sigma)$ (``kernel="exponential"``)
    so that $Q = \\Lambda K M \\Lambda$ has unit row sums, with
    $\\Lambda = \\mathrm{diag}(e^f)$.

    Parameters
    ----------
    sqdist : (N, N) ndarray
        Squared pairwise distances.
    sigma : float
        Kernel bandwidth.
    masses : (N,) ndarray, optional
        Positive point masses. Defaults to uniform ones.
    kernel : {"gaussian", "exponential"}
        Kernel family.
    n_iter : int
        Maximum number of updates.
    tol : float, optional
        Early-stopping tolerance on the L1(mu) marginal violation.
        ``None`` (default) runs exactly ``n_iter`` updates.
    f_init : (N,) ndarray, optional
        Warm-start log-potentials.
    full_output : bool
        If True, also return an info dict with ``n_iter`` (updates performed)
        and ``marginal_error`` (achieved violation).

    Returns
    -------
    f : (N, ) ndarray
        Log-potentials.
    info : dict, only if ``full_output=True``.
    """
    log_kernel = _log_kernel_from_sqdist(np.asarray(sqdist), sigma, kernel)
    return _sinkhorn_log_dense(
        log_kernel, masses=masses, n_iter=n_iter, tol=tol, f_init=f_init, full_output=full_output
    )


def sinkhorn_log_kernel(
    log_kernel,
    masses=None,
    n_iter=10,
    tol=None,
    include_diagonal=False,
    f_init=None,
    full_output=False,
):
    """Log-domain symmetric Sinkhorn from an arbitrary log-kernel matrix.

    The lowest-level log-domain entry point: normalizes any symmetric kernel
    given as its log $\\log K$ (dense array or sparse matrix).

    Parameters
    ----------
    log_kernel : (N, N) ndarray or sparse matrix
        The log-kernel $\\log K$ (symmetric). For a dense matrix the
        diagonal is taken as given; for a sparse matrix only the stored
        entries participate (see ``include_diagonal``).
    masses, n_iter, tol, f_init, full_output :
        See :func:`sinkhorn_log`.
    include_diagonal : bool
        Sparse input only: if True, add the analytic self term
        ($\\log K_{ii} = 0$) to each row (use when the stored pattern
        omits the diagonal, e.g. a k-NN graph). Ignored for dense input.

    Returns
    -------
    f : (N,) ndarray
        Log-potentials (and an info dict if ``full_output=True``).
    """
    if sparse.issparse(log_kernel):
        return _sinkhorn_log_sparse(
            sparse.csr_matrix(log_kernel),
            masses=masses,
            n_iter=n_iter,
            tol=tol,
            include_diagonal=include_diagonal,
            f_init=f_init,
            full_output=full_output,
        )
    return _sinkhorn_log_dense(
        log_kernel, masses=masses, n_iter=n_iter, tol=tol, f_init=f_init, full_output=full_output
    )


def _csr_logsumexp(data, indptr, out_len):
    """Row-wise stabilized logsumexp over the values of a CSR layout.
    A bit tricky.

    Empty rows get ``-inf``.
    """
    counts = np.diff(indptr)  # (out_len,) entries per row
    valid = counts > 0  # (out_len,) rows with at least one stored entry
    ptrs = indptr[:-1][valid]  # (n_valid,) segment starts for reduceat

    row_max = np.full(out_len, -np.inf, dtype=data.dtype)  # (out_len,)
    if data.size:
        row_max[valid] = np.maximum.reduceat(data, ptrs)

    row_idx = np.repeat(np.arange(out_len), counts)  # (nnz,) row of each stored entry
    sums = np.zeros(out_len, dtype=data.dtype)  # (out_len,)
    if data.size:
        sums[valid] = np.add.reduceat(np.exp(data - row_max[row_idx]), ptrs)

    with np.errstate(divide="ignore"):
        return row_max + np.log(sums)  # (out_len,)


def _sinkhorn_log_sparse(
    log_kernel,
    masses=None,
    n_iter=10,
    tol=None,
    include_diagonal=False,
    f_init=None,
    full_output=False,
):
    """Shared sparse log-domain solver operating on a CSR log-kernel.

    ``log_kernel`` is a CSR matrix of $\\log K$ on the stored pattern.
    With ``include_diagonal=True`` the analytic self term
    ($\\log K_{ii} = 0$) is added to each row.
    """
    D = sparse.csr_matrix(log_kernel)  # (N, N)
    N = D.shape[0]

    if include_diagonal:
        rows = np.repeat(np.arange(N), np.diff(D.indptr))  # (nnz,) row of each entry
        if np.any(rows == D.indices):
            raise ValueError(
                "include_diagonal=True but the sparse matrix stores diagonal entries; "
                "pass include_diagonal=False or remove them."
            )

    log_k = D.data  # (nnz,) stored log-kernel values
    dtype = log_k.dtype

    # f: (N,) log-potentials;  log_m: (N,) log-masses
    f = np.zeros(N, dtype=dtype) if f_init is None else np.asarray(f_init).astype(dtype).copy()
    log_m = None if masses is None else np.log(masses).astype(dtype)

    def _lse(f):
        # data: (nnz,) exponents on the stored pattern -> (N,) row logsumexp
        data = log_k + f[D.indices] if log_m is None else log_k + (f + log_m)[D.indices]
        lse = _csr_logsumexp(data, D.indptr, N)
        if include_diagonal:
            # add the analytic self term exp(log K_ii + f_i + log m_i), log K_ii = 0
            self_term = f if log_m is None else f + log_m
            lse = np.logaddexp(lse, self_term)
        return lse

    n_done = 0
    for _ in range(n_iter):
        lse = _lse(f)
        if tol is not None and _marginal_error(np.exp(f + lse), masses) <= tol:
            break
        f = 0.5 * f - 0.5 * lse
        n_done += 1

    if not full_output:
        return f
    err = _marginal_error(np.exp(f + _lse(f)), masses)
    return f, {"n_iter": n_done, "marginal_error": err}


def sinkhorn_log_sparse(
    sqdist,
    sigma,
    masses=None,
    kernel="gaussian",
    n_iter=10,
    tol=None,
    include_diagonal=True,
    f_init=None,
    full_output=False,
):
    """Sparse log-domain symmetric Sinkhorn for a truncated Gaussian/exponential kernel.

    Same fixed point as :func:`sinkhorn_log`, but the kernel support is
    restricted to the stored entries of a sparse matrix of squared distances

    Parameters
    ----------
    sqdist : (N, N) sparse matrix
        Squared distances on the sparsity pattern; converted to CSR.
        Must not store diagonal entries when ``include_diagonal=True``.
    sigma, masses, kernel, n_iter, tol, f_init, full_output :
        See :func:`sinkhorn_log`.
    include_diagonal : bool
        If True (default), the self-interaction term
        $K_{ii} = 1$ is added analytically to each row, so that a
        complete k-NN graph reproduces the dense solution exactly.
        Set to False if the stored pattern already covers the diagonal.

    Returns
    -------
    f : (N,) ndarray
        Log-potentials (and an info dict if ``full_output=True``).
    """
    D = sparse.csr_matrix(sqdist)  # (N, N)
    log_kernel = D.copy()  # (N, N)
    log_kernel.data = _log_kernel_from_sqdist(D.data, sigma, kernel)  # (nnz,)
    return _sinkhorn_log_sparse(
        log_kernel,
        masses=masses,
        n_iter=n_iter,
        tol=tol,
        include_diagonal=include_diagonal,
        f_init=f_init,
        full_output=full_output,
    )


def sinkhorn(K, masses=None, n_iter=10, tol=None, scaling_init=None, full_output=False):
    """Multiplicative symmetric Sinkhorn (Algorithm 1 of the paper).

    Works with any symmetric non-negative kernel given as a dense array, a
    sparse matrix, or a black-box matvec (``scipy.sparse.linalg.LinearOperator``,
    e.g. :class:`sinkhornkernels.grid.GridGaussian`). Each iteration costs one
    kernel matvec:

    $$
        \\lambda \\leftarrow \\sqrt{\\lambda \\oslash (K (m \\odot \\lambda))}.
    $$

    Prefer the log-domain solvers over this one. This solver is
    for kernels only available as matvecs (grid convolutions, graphs).

    Parameters
    ----------
    K : (N, N) ndarray, sparse matrix or LinearOperator
        Symmetric kernel with non-negative entries.
    masses : (N,) ndarray, optional
        Non-negative point masses. Defaults to uniform ones.
    n_iter, tol, full_output :
        See :func:`sinkhorn_log`.
    scaling_init : (N,) ndarray, optional
        Warm-start scalings.

    Returns
    -------
    scaling : (N,) ndarray
        Linear-domain scalings $\\lambda$ (the diagonal of
        $\\Lambda$), 0 wherever the kernel row sum vanishes.
    info : dict, only if ``full_output=True``.
    """
    N = K.shape[0]
    dtype = np.dtype(getattr(K, "dtype", np.float64) or np.float64)
    if not np.issubdtype(dtype, np.floating):
        dtype = np.dtype(np.float64)
    # scaling: (N,) linear-domain lambda;  m: (N,) masses
    scaling = (
        np.ones(N, dtype=dtype)
        if scaling_init is None
        else np.asarray(scaling_init).astype(dtype).copy()
    )
    m = None if masses is None else np.asarray(masses)

    def _denom(scaling):
        v = scaling if m is None else m * scaling  # (N,)
        return np.asarray(K @ v)  # (N,) = K M lambda

    n_done = 0
    denom = None
    for _ in range(n_iter):
        denom = _denom(scaling)
        if tol is not None and _marginal_error(scaling * denom, m) <= tol:
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            scaling = np.where(denom > 0, np.sqrt(scaling / denom), 0.0)
        n_done += 1

    if not full_output:
        return scaling
    err = _marginal_error(scaling * _denom(scaling), m)
    return scaling, {"n_iter": n_done, "marginal_error": err}
