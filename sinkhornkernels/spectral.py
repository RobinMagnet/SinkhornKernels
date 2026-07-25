"""Spectral analysis of normalized diffusion operators.

A normalized operator $Q$ is self-adjoint for the weighted inner
product $\\langle f, g \\rangle_w = f^\\top \\mathrm{diag}(w) g$
so we can use standard eigsh.

$$
    \\mathrm{diag}(w)\\, Q\\, \\phi = \\lambda\\, \\mathrm{diag}(w)\\, \\phi,
$$

which only requires matvecs with $Q$. The leading eigenvector estimate the smallest Laplacian eigenvectors.
"""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import LinearOperator, eigsh


def diffusion_eigsh(Q, k=64, measure=None, which="LM", **eigsh_kwargs):
    """Leading eigenpairs of a diffusion operator.

    Parameters
    ----------
    Q : LinearOperator or ndarray
        The diffusion operator, typically a
        :class:`sinkhornkernels.operators.NormalizedKernel`.
    k : int
        Number of eigenpairs.
    measure : (N,) ndarray, optional
        Strictly positive weights $w$ making $\\mathrm{diag}(w) Q$
        symmetric. Defaults to ``Q.self_adjoint_measure`` when available
        (masses for symmetric modes, the stationary measure for row mode).
    which : str
        Passed to ``scipy.sparse.linalg.eigsh`` (default ``"LM"``: largest,
        i.e. the low-frequency end of the diffusion).
    **eigsh_kwargs :
        Forwarded to ``eigsh`` (e.g. ``tol``, ``maxiter``, ``v0``).

    Returns
    -------
    evals : (k,) ndarray
        Eigenvalues of $Q$, sorted in decreasing order.
    evecs : (N, k) ndarray
        Corresponding eigenvectors, orthonormal for the ``measure``-weighted
        inner product.
    """
    if measure is None:
        measure = getattr(Q, "self_adjoint_measure", None)
        if measure is None:
            raise ValueError("`measure` is required when Q does not expose `self_adjoint_measure`.")
    measure = np.asarray(measure)  # (N,)
    if np.any(measure <= 0):
        raise ValueError("`measure` must be strictly positive for the generalized eigsh.")

    N = Q.shape[0]
    dtype = np.dtype(getattr(Q, "dtype", np.float64) or np.float64)
    if not np.issubdtype(dtype, np.floating):
        dtype = np.dtype(np.float64)

    # symmetric operator W Q, self-adjoint for the diag(measure) inner product
    WQ = LinearOperator(
        shape=(N, N),
        dtype=dtype,
        matvec=lambda x: measure * np.asarray(Q @ np.asarray(x).reshape(-1)),
    )
    evals, evecs = eigsh(WQ, k=k, M=sparse.diags(measure).tocsc(), which=which, **eigsh_kwargs)
    # evals: (k,)  evecs: (N, k)

    order = np.argsort(-evals)  # (k,) sort descending
    return evals[order], evecs[:, order]


def laplacian_eigenvalues(evals_Q, sigma):
    """Laplacian eigenvalue estimates from diffusion eigenvalues.

    Uses the heat-kernel correspondence $\\lambda^Q = e^{-t \\lambda^\\Delta}$
    at time $t = \\sigma^2 / 2$:

    $$
        \\lambda^\\Delta = -\\frac{2}{\\sigma^2} \\log \\lambda^Q.
    $$

    Eigenvalues are clipped to $(0, 1]$ before the logarithm.
    For Gaussian-mixture operators, pass
    ``sigma=np.sqrt(gmm_effective_sigma2(...))``.

    .. note::
        This heuristic is derived for the **Gaussian** kernel (and its GMM
        variant). The exponential/Laplace kernel has no equivalent heat-kernel
        correspondence, so this conversion should not be applied to
        exponential-kernel spectra. In all cases the relation to the true
        Laplacian spectrum is qualitative, not quantitative.
    """
    evals_Q = np.clip(np.asarray(evals_Q), np.finfo(float).tiny, 1.0)
    return -(2.0 / sigma**2) * np.log(evals_Q)


def gmm_effective_sigma2(sigma, covariances, dim, masses=None):
    """Effective squared bandwidth of a Gaussian-mixture kernel.

    The anisotropic kernel of :func:`sinkhornkernels.kernels.gmm_kernel`
    behaves, on average, like a Gaussian kernel of squared bandwidth

    $$
        \\sigma_\\text{eff}^2 = \\sigma^2 + \\frac{2}{d}\\,
            \\frac{\\sum_i m_i \\operatorname{tr} \\Sigma_i}{\\sum_i m_i},
    $$

    where $d$ is the *intrinsic* dimension of the shape
    (2 for surfaces, 3 for volumes).

    Parameters
    ----------
    sigma : float
        Isotropic smoothing bandwidth used in :func:`gmm_kernel`.
    covariances : (N, d_ambient, d_ambient) ndarray
        Component covariances.
    dim : int
        Intrinsic dimension (2 for surfaces, 3 for volumes).
    masses : (N,) ndarray, optional
        Component weights. Defaults to uniform.

    Returns
    -------
    sigma2_eff : float
    """
    traces = np.trace(np.asarray(covariances), axis1=-2, axis2=-1)  # (N,) tr Sigma_i
    mean_trace = np.average(traces, weights=masses)
    return sigma**2 + 2.0 * mean_trace / dim
