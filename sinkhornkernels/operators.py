"""Normalized diffusion operators as matrix-free ``LinearOperator`` objects.

:class:`NormalizedKernel` wraps any symmetric non-negative kernel (dense,
sparse, or black-box matvec) together with point masses, and applies one of
the normalizations compared in the paper:

- ``"none"``:               $Q = K M$
- ``"row"``:                $Q = D^{-1} K M$, $D = \\mathrm{diag}(K m)$
- ``"symmetric_one_step"``: $Q = \\Lambda K M \\Lambda$, $\\lambda = 1/\\sqrt{K m}$
- ``"sinkhorn"``:           $Q = \\Lambda K M \\Lambda$ with $\\Lambda$
  the (unique) Sinkhorn scaling making $Q 1 = 1$.

The scalings are computed once at construction; each application then costs a
single kernel matvec.

For exponentially-decaying kernels (Gaussian, exponential, GMM) the constructor
applies the kernel in the **primal** domain, can suffer from underflows especialy when using float32.
Use
:meth:`NormalizedKernel.from_log_kernel` (or the ``gaussian_diffusion`` /
``exponential_diffusion`` / ``gmm_diffusion`` factories), which both solve and
apply in the log domain and are stable in either dtype. The primal path is
correct for kernels whose entries do not underflow (e.g. the sum-one grid
convolution).
"""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import LinearOperator

from .kernels import (
    exponential_kernel,
    gaussian_kernel,
    gmm_log_kernel,
    knn_graph,
    knn_to_sparse_sqdist,
    squared_distances,
)
from .sinkhorn import (
    _log_kernel_from_sqdist,
    _marginal_error,
    sinkhorn,
    sinkhorn_log_kernel,
)

_MODES = ("none", "row", "symmetric_one_step", "sinkhorn")


def _materialize_Q(log_kernel, f, masses):
    """Build the stable normalized operator $Q_{ij} = e^{\\log K_{ij} + f_i + f_j + \\log m_j}$.

    The full exponent is formed before the single ``exp``, so no kernel entry
    underflows on its own. ``log_kernel`` is a dense array (diagonal included)
    or a CSR matrix (the analytic diagonal $\\log K_{ii}=0$ is added).
    Returns a dense array or CSR matrix matching the input.
    """
    log_m = np.log(masses)  # (N,)
    if sparse.issparse(log_kernel):
        D = sparse.csr_matrix(log_kernel)  # (N, N)
        N = D.shape[0]
        rows = np.repeat(np.arange(N), np.diff(D.indptr))  # (nnz,) row of each entry
        Q = D.copy()  # (N, N)
        Q.data = np.exp(D.data + f[rows] + (f + log_m)[D.indices])  # (nnz,)
        diag = sparse.diags(np.exp(2.0 * f + log_m))  # (N, N), log K_ii = 0
        return (Q + diag).tocsr()
    return np.exp(np.asarray(log_kernel) + f[:, None] + (f + log_m)[None, :])  # (N, N)


class NormalizedKernel(LinearOperator):
    """Diffusion operator $Q$ obtained by normalizing $S = K M$.

    Parameters
    ----------
    kernel : (N, N) ndarray, sparse matrix, or LinearOperator
        Symmetric kernel with non-negative entries (strictly positive for
        guaranteed convergence of the Sinkhorn mode).
    masses : (N,) ndarray, optional
        Non-negative point masses (diagonal of $M$). Defaults to ones.
    mode : {"sinkhorn", "row", "symmetric_one_step", "none"}
    n_iter, tol :
        Passed to :func:`sinkhornkernels.sinkhorn.sinkhorn` in mode
        ``"sinkhorn"`` (ignored if ``scaling`` is given).
    scaling : (N,) ndarray, optional
        Precomputed symmetric scalings $\\lambda$ (e.g. ``np.exp(f)``
        from a log-domain solver). Only valid with ``mode="sinkhorn"``.

    Attributes
    ----------
    scaling : (N,) ndarray
        Symmetric scaling $\\lambda$ for the symmetric modes; the left
        row scaling $1/(Km)$ for ``mode="row"``; ones for ``"none"``.
    masses : (N,) ndarray
    self_adjoint_measure : (N,) ndarray
        Weights $w$ such that $\\mathrm{diag}(w)\\,Q$ is symmetric:
        the masses for the symmetric modes, the stationary measure
        $m \\odot K m$ for ``mode="row"``. Feed to
        :func:`sinkhornkernels.spectral.diffusion_eigsh`.
    """

    def __init__(self, kernel, masses=None, mode="sinkhorn", n_iter=10, tol=None, scaling=None):
        if mode not in _MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {_MODES}.")
        if scaling is not None and mode != "sinkhorn":
            raise ValueError("A precomputed `scaling` is only meaningful with mode='sinkhorn'.")

        N = kernel.shape[0]
        dtype = np.dtype(getattr(kernel, "dtype", np.float64) or np.float64)
        if not np.issubdtype(dtype, np.floating):
            dtype = np.dtype(np.float64)

        self.kernel = kernel
        self.mode = mode
        self.masses = (
            np.ones(N, dtype=dtype) if masses is None else np.asarray(masses).astype(dtype)
        )
        # materialized stable operator (set by from_log_kernel); primal path when None
        self._Q = None
        self._measure = None

        # left / right: (N,) row / column scalings applied around the kernel
        if mode == "none":
            left = right = np.ones(N, dtype=dtype)
        elif mode == "row":
            degrees = self._kernel_matvec(self.masses)  # (N,) = K m
            with np.errstate(divide="ignore"):
                left = np.where(degrees > 0, 1.0 / degrees, 0.0)
            right = np.ones(N, dtype=dtype)
        elif mode == "symmetric_one_step":
            degrees = self._kernel_matvec(self.masses)  # (N,) = K m
            with np.errstate(divide="ignore"):
                left = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
            right = left
        else:  # "sinkhorn"
            if scaling is None:
                scaling = sinkhorn(kernel, masses=self.masses, n_iter=n_iter, tol=tol)  # (N,)
            left = right = np.asarray(scaling).astype(dtype)

        self.scaling = left  # (N,)
        self._left, self._right = left, right
        super().__init__(dtype=dtype, shape=(N, N))

    @classmethod
    def from_log_kernel(
        cls, log_kernel, masses=None, n_iter=10, tol=None, include_diagonal=False, scaling=None
    ):
        r"""Stable Sinkhorn normalization from a log-kernel $\log K$.

        Solves for the potentials in the log domain and then **materializes the
        normalized operator** directly as
        $Q_{ij} = \exp(\log K_{ij} + f_i + f_j + \log m_j)$ — the full
        exponent is formed before the single ``exp``, so both the solve and the
        application avoid underflowing any kernel entry. Numerically stable in
        either float32 or float64, and the natural entry point for the Gaussian,
        exponential and Gaussian-mixture kernels.

        Parameters
        ----------
        log_kernel : (N, N) ndarray or sparse matrix
            The log-kernel $\log K$ (symmetric). Dense: diagonal included.
            Sparse: stored pattern only (see ``include_diagonal``).
        masses : (N,) ndarray, optional
            Point masses; defaults to uniform ones.
        n_iter, tol :
            Sinkhorn stopping parameters (ignored if ``scaling`` is given).
        include_diagonal : bool
            Sparse input only: add the analytic diagonal $\log K_{ii}=0$
            (use for a k-NN pattern that omits the diagonal).
        scaling : (N,) ndarray, optional
            Precomputed symmetric scalings $\lambda = e^f$ to inject
            instead of solving.

        Returns
        -------
        Q : :class:`NormalizedKernel`
            With a materialized, log-stable operator; ``self_adjoint_measure``
            is the masses (Q is M-self-adjoint).
        """
        N = log_kernel.shape[0]
        m = np.ones(N) if masses is None else np.asarray(masses, dtype=float)  # (N,)

        if scaling is None:
            f = sinkhorn_log_kernel(  # (N,) log-potentials
                log_kernel, masses=m, n_iter=n_iter, tol=tol, include_diagonal=include_diagonal
            )
        else:
            with np.errstate(divide="ignore"):
                f = np.log(np.asarray(scaling, dtype=float))  # (N,)

        Q = _materialize_Q(log_kernel, f, m)  # (N, N) dense or CSR

        obj = cls.__new__(cls)
        LinearOperator.__init__(obj, dtype=np.dtype(np.float64), shape=(N, N))
        obj.kernel = Q
        obj.mode = "sinkhorn"
        obj.masses = m
        obj._Q = Q
        obj._measure = m
        obj.scaling = np.exp(f)
        obj._left = obj._right = np.ones(N)
        return obj

    def _kernel_matvec(self, x):
        return np.asarray(self.kernel @ x)

    def _matvec(self, x):
        x = np.asarray(x).reshape(-1)
        if self._Q is not None:
            return np.asarray(self._Q @ x)
        return self._left * self._kernel_matvec(self.masses * (self._right * x))

    def _rmatvec(self, x):
        x = np.asarray(x).reshape(-1)
        if self._Q is not None:
            return np.asarray(self._Q.T @ x)
        # K and M are symmetric, so Q^T x = R (K (m . L x)) with L/R swapped.
        return self._right * (self.masses * self._kernel_matvec(self._left * x))

    @property
    def log_potentials(self):
        """Log-domain scalings $\\log \\lambda$ (``-inf`` on empty support)."""
        with np.errstate(divide="ignore"):
            return np.log(self.scaling)

    @property
    def self_adjoint_measure(self):
        if self._measure is not None:
            return self._measure
        if self.mode == "row":
            return self.masses * self._kernel_matvec(self.masses)
        return self.masses

    def marginal_error(self):
        """L1(mu)-averaged violation of $Q 1 = 1$."""
        rowsums = self @ np.ones(self.shape[0], dtype=self.dtype)  # (N,) = Q 1
        return _marginal_error(rowsums, self.masses)

    def toarray(self):
        """Materialize $Q$ as a dense array (small problems / tests)."""
        return self @ np.eye(self.shape[0], dtype=self.dtype)


def gaussian_diffusion(
    points, sigma, masses=None, kernel="gaussian", mode="sinkhorn", knn=None, n_iter=10, tol=None
):
    """Sinkhorn-normalized Gaussian/exponential diffusion operator on a point cloud.

    Convenience factory. For ``mode="sinkhorn"`` (default) it builds the
    log-kernel (dense, or truncated to a symmetrized k-NN graph) and calls
    :meth:`NormalizedKernel.from_log_kernel`, so both the normalization and the
    application are numerically stable in the log domain — the recommended path,
    especially at small ``sigma`` or in float32. Other modes use the primal
    degree-based normalizations.

    Parameters
    ----------
    points : (N, d) ndarray
    sigma : float
        Kernel bandwidth.
    masses : (N,) ndarray, optional
        E.g. vertex areas (:func:`sinkhornkernels.mesh.vertex_areas`) or
        uniform ``1/N`` weights.
    kernel : {"gaussian", "exponential"}
        Kernel family (see :func:`sinkhornkernels.sinkhorn.sinkhorn_log`).
    mode : see :class:`NormalizedKernel`.
    knn : int, optional
        If given, truncate the kernel to the symmetrized ``knn``-nearest
        neighbor graph (plus the analytic diagonal) — O(kN) storage.
    n_iter, tol :
        Sinkhorn stopping parameters.

    Returns
    -------
    Q : :class:`NormalizedKernel`
    """
    points = np.asarray(points)  # (N, d)
    N = points.shape[0]

    if mode == "sinkhorn":
        if knn is None:
            # log_kernel: (N, N) dense log-kernel
            log_kernel = _log_kernel_from_sqdist(squared_distances(points), sigma, kernel)
            return NormalizedKernel.from_log_kernel(
                log_kernel, masses=masses, n_iter=n_iter, tol=tol
            )
        knn_sqdist, knn_indices = knn_graph(points, knn, include_self=False)  # (N, k), (N, k)
        sq_sparse = knn_to_sparse_sqdist(knn_sqdist, knn_indices, symmetrize="max")  # (N, N)
        log_kernel = sq_sparse.copy()  # (N, N)
        log_kernel.data = _log_kernel_from_sqdist(sq_sparse.data, sigma, kernel)  # (nnz,)
        return NormalizedKernel.from_log_kernel(
            log_kernel, masses=masses, n_iter=n_iter, tol=tol, include_diagonal=True
        )

    # primal degree-based modes (row / symmetric_one_step / none)
    K_dense = (
        gaussian_kernel(points, sigma)
        if kernel == "gaussian"
        else exponential_kernel(points, sigma)
    )
    if knn is None:
        return NormalizedKernel(K_dense, masses=masses, mode=mode)
    knn_sqdist, knn_indices = knn_graph(points, knn, include_self=False)  # (N, k), (N, k)
    sq_sparse = knn_to_sparse_sqdist(knn_sqdist, knn_indices, symmetrize="max")  # (N, N)
    log_kernel = sq_sparse.copy()  # (N, N)
    log_kernel.data = np.exp(_log_kernel_from_sqdist(sq_sparse.data, sigma, kernel))  # (nnz,)
    K = log_kernel + sparse.eye(N, format="csr")  # (N, N)
    return NormalizedKernel(K, masses=masses, mode=mode)


def exponential_diffusion(
    points, sigma, masses=None, mode="sinkhorn", knn=None, n_iter=10, tol=None
):
    """Sinkhorn-normalized exponential (Laplace) diffusion operator on a point cloud.

    Thin wrapper over :func:`gaussian_diffusion` with ``kernel="exponential"``
    (kernel $\\exp(-\\|x-y\\|/\\sigma)$, $\\varepsilon = \\sigma$).
    Note that the Laplacian-eigenvalue heuristic
    :func:`sinkhornkernels.spectral.laplacian_eigenvalues` is derived for the
    Gaussian kernel and does not transfer to the exponential kernel.
    """
    return gaussian_diffusion(
        points,
        sigma,
        masses=masses,
        kernel="exponential",
        mode=mode,
        knn=knn,
        n_iter=n_iter,
        tol=tol,
    )


def gmm_diffusion(means, covariances, sigma, masses=None, divide_by_det=False, n_iter=10, tol=None):
    """Sinkhorn-normalized diffusion operator on a Gaussian mixture (stable).

    Builds the anisotropic log-kernel
    (:func:`sinkhornkernels.kernels.gmm_log_kernel`) and normalizes it via
    :meth:`NormalizedKernel.from_log_kernel`, so the small-``sigma`` regime
    (where the primal GMM kernel underflows) is handled in the log domain.

    Parameters
    ----------
    means : (N, d) ndarray
    covariances : (N, d, d) ndarray
    sigma : float
        Isotropic smoothing bandwidth.
    masses : (N,) ndarray, optional
        Mixture weights (e.g. ``GaussianMixture.weights_``). Defaults to uniform.
    divide_by_det : bool
        Include the $\\det(C_{ij})^{-1/2}$ prefactor (default False; see
        :func:`sinkhornkernels.kernels.gmm_kernel`).
    n_iter, tol :
        Sinkhorn stopping parameters.

    Returns
    -------
    Q : :class:`NormalizedKernel`
    """
    log_kernel = gmm_log_kernel(means, covariances, sigma, divide_by_det=divide_by_det)  # (N, N)
    return NormalizedKernel.from_log_kernel(log_kernel, masses=masses, n_iter=n_iter, tol=tol)
