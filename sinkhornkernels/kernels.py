"""Kernel constructions for the modalities considered in the paper.

- Point clouds: dense or k-NN-truncated Gaussian kernels.
- Gaussian mixtures: anisotropic kernel with pairwise covariance
  $C_{ij} = \\sigma^2 I + \\Sigma_i + \\Sigma_j$.
- Graphs: $K = D + A + \\varepsilon \\, 1 1^\\top$.

Everything here is numpy/scipy only.
"""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import LinearOperator
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from sklearn.neighbors import NearestNeighbors


def squared_distances(X, Y=None):
    """Pairwise squared Euclidean distances.

    Parameters
    ----------
    X : (N, d) ndarray
    Y : (M, d) ndarray, optional
        Defaults to ``X``.

    Returns
    -------
    sqdist : (N, M) ndarray
    """
    X = np.asarray(X)
    Y = X if Y is None else np.asarray(Y)
    return cdist(X, Y, metric="sqeuclidean")


def gaussian_kernel_from_sqdist(sqdist, sigma):
    """Gaussian kernel $K = \\exp(-d^2 / 2\\sigma^2)$ from squared distances.

    Accepts a dense array or a sparse matrix
    """
    if sparse.issparse(sqdist):
        K = sqdist.copy().tocsr()
        K.data = np.exp(-K.data / (2 * sigma**2))
        return K
    return np.exp(-np.asarray(sqdist) / (2 * sigma**2))


def gaussian_kernel(points, sigma):
    """Dense Gaussian kernel $K_{ij} = \\exp(-\\|x_i-x_j\\|^2/2\\sigma^2)$."""
    return gaussian_kernel_from_sqdist(squared_distances(points), sigma)


def exponential_kernel_from_dist(dist, sigma):
    """Exponential (Laplace) kernel $K = \\exp(-\\|x-y\\| / \\sigma)$ from distances.

    Accepts a dense array or a sparse matrix. ``dist`` is Euclidean distances, not
    squared.
    """
    if sparse.issparse(dist):
        K = dist.copy().tocsr()
        K.data = np.exp(-K.data / sigma)
        return K
    return np.exp(-np.asarray(dist) / sigma)


def exponential_kernel(points, sigma):
    """Dense exponential kernel $K_{ij} = \\exp(-\\|x_i-x_j\\| / \\sigma)$.

    The OT cost is $\\|x-y\\|$ with $\\varepsilon = \\sigma$ (vs the
    Gaussian's $\\tfrac12\\|x-y\\|^2$ with $\\varepsilon = \\sigma^2$).
    """
    return np.exp(-np.sqrt(squared_distances(points)) / sigma)


def knn_graph(points, k, include_self=False):
    """k-nearest-neighbor graph of a point cloud.

    Parameters
    ----------
    points : (N, d) ndarray
    k : int
        Number of neighbors per point. ``k`` always counts *other* points
        unless ``include_self=True``, in which case the point itself occupies
        one of the ``k`` slots.
    include_self : bool
        Whether to keep the point itself among its neighbors.

    Returns
    -------
    knn_sqdist : (N, k) ndarray
        Squared distances to the neighbors.
    knn_indices : (N, k) ndarray
        Neighbor indices.
    """
    points = np.asarray(points)  # (N, d)
    N = points.shape[0]

    tree = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(points)

    dists, indices = tree.kneighbors(return_distance=True)  # (N, k), (N, k)

    if include_self:
        # prepend the point itself (distance 0) -> (N, k+1)
        dists = np.hstack([np.zeros((N, 1)), dists])
        indices = np.hstack([np.arange(N)[:, None], indices])

    return dists**2, indices


def knn_to_sparse_sqdist(knn_sqdist, knn_indices, symmetrize="max"):
    """CSR matrix of squared distances from k-NN arrays.

    Parameters
    ----------
    knn_sqdist, knn_indices : (N, k) ndarrays
        As returned by :func:`knn_graph` (without self entries).
    symmetrize : {"max", "min", None}
        ``"max"`` (default) keeps the union of the directed neighborhoods
        (elementwise maximum with the transpose; both directions store the
        same distance whenever present). ``"min"`` keeps only mutual
        neighbors. ``None`` returns the directed graph as-is.

    Returns
    -------
    sqdist : (N, N) CSR matrix
    """
    N, k = knn_indices.shape
    rows = np.repeat(np.arange(N), k)  # (N * k,)
    D = sparse.csr_matrix(
        (np.asarray(knn_sqdist).ravel(), (rows, np.asarray(knn_indices).ravel())),
        shape=(N, N),
    )  # (N, N)
    if symmetrize == "max":
        return D.maximum(D.T).tocsr()
    if symmetrize == "min":
        return D.minimum(D.T).tocsr()
    if symmetrize is None:
        return D
    raise ValueError(f"Unknown symmetrize mode {symmetrize!r}; use 'max', 'min' or None.")


def _gmm_quad_logdet(means, covariances, sigma):
    """Quadratic form and log-determinant of the GMM pairwise covariance.

    Returns ``quad`` (N, N) with $\\delta_{ij}^\\top C_{ij}^{-1} \\delta_{ij}$
    and ``logdet`` (N, N) with $\\log\\det C_{ij}$, where
    $C_{ij} = \\sigma^2 I + \\Sigma_i + \\Sigma_j$.
    """
    means = np.asarray(means)
    covariances = np.asarray(covariances)
    N, d = means.shape
    if covariances.shape != (N, d, d):
        raise ValueError(
            f"covariances must have shape {(N, d, d)}, got {covariances.shape}; "
            "use covariance_type='full' (or expand spherical/diagonal covariances)."
        )

    C = sigma**2 * np.eye(d) + covariances[:, None] + covariances[None, :]  # (N, N, d, d)
    delta = means[:, None, :] - means[None, :, :]  # (N, N, d)
    sol = np.linalg.solve(C, delta[..., None])[..., 0]  # C_ij^{-1} delta_ij
    quad = np.einsum("ijd,ijd->ij", delta, sol)
    _, logdet = np.linalg.slogdet(C)
    return quad, logdet


def gmm_log_kernel(means, covariances, sigma, divide_by_det=False):
    """Log of the anisotropic Gaussian-mixture kernel (see :func:`gmm_kernel`).

    Returns $\\log K_{ij} = -\\tfrac12 \\delta_{ij}^\\top C_{ij}^{-1}
    \\delta_{ij}$ (and, with ``divide_by_det=True``, the extra term
    $-\\tfrac12 \\log\\det C_{ij}$). Feed to
    :func:`sinkhornkernels.sinkhorn.sinkhorn_log_kernel` or
    :func:`sinkhornkernels.operators.NormalizedKernel.from_log_kernel` for a
    numerically stable normalization at small ``sigma``.
    """
    quad, logdet = _gmm_quad_logdet(means, covariances, sigma)
    log_k = -0.5 * quad
    if divide_by_det:
        log_k = log_k - 0.5 * logdet
    return log_k


def gmm_kernel(means, covariances, sigma, divide_by_det=False):
    """Anisotropic kernel between the components of a Gaussian mixture.

    $$
        K_{ij} = \\det(C_{ij})^{-1/2}\\;
                 \\exp\\big(-\\tfrac12 (\\mu_i-\\mu_j)^\\top
                 C_{ij}^{-1} (\\mu_i-\\mu_j)\\big),
        \\qquad C_{ij} = \\sigma^2 I + \\Sigma_i + \\Sigma_j,
    $$

    the (rescaled) $L^2$ inner product of the component densities
    convolved with an isotropic Gaussian of variance $\\sigma^2/2$

    Parameters
    ----------
    means : (N, d) ndarray
        Component means.
    covariances : (N, d, d) ndarray
        Full component covariances (e.g. ``GaussianMixture.covariances_``
        with ``covariance_type="full"``).
    sigma : float
        Isotropic smoothing bandwidth.
    divide_by_det : bool
        Whether to include the $\\det(C_{ij})^{-1/2}$ prefactor of the
        exact inner product (default ``False``). Unlike the $(2\\pi)^{-d/2}$
        constant, this factor this is not absorbed by the Sinkhorn normalization.

    Returns
    -------
    K : (N, N) ndarray

    Notes
    -----
    Use the mixture weights as ``masses`` in the Sinkhorn solvers, and
    :func:`sinkhornkernels.spectral.gmm_effective_sigma2` when converting the
    normalized spectrum to Laplacian eigenvalues. For small ``sigma`` prefer
    the stable :func:`sinkhornkernels.operators.gmm_diffusion` factory (which
    normalizes via :func:`gmm_log_kernel`) over normalizing this matrix directly.
    """
    return np.exp(gmm_log_kernel(means, covariances, sigma, divide_by_det=divide_by_det))


class _SparsePlusConstant(LinearOperator):
    """Matvec for ``B + eps * 1 1^T`` without densifying the rank-one term. For the graph kernel."""

    def __init__(self, base, eps):
        self.base = base.tocsr()
        self.eps = float(eps)
        super().__init__(dtype=np.result_type(self.base.dtype, np.float64), shape=base.shape)

    def _matvec(self, x):
        x = np.asarray(x).reshape(-1)
        return self.base @ x + self.eps * x.sum()

    def _rmatvec(self, x):  # symmetric
        return self._matvec(x)

    def toarray(self):
        return self.base.toarray() + self.eps


def graph_kernel(adjacency, eps=0.0):
    """Smoothing kernel of a graph: $K = D + A + \\varepsilon 1 1^\\top$.

    ``D`` is the degree matrix. With ``eps > 0`` every entry of ``K`` becomes
    strictly positive, and the kernel verifies our axioms for a Smoothing Operator.

    Parameters
    ----------
    adjacency : (N, N) sparse or dense matrix
        Symmetric non-negative adjacency (weights allowed).
    eps : float
        Dense regularization strength.

    Returns
    -------
    K : CSR matrix if ``eps == 0``, else a ``LinearOperator`` computing
        the matvec without densifying (use with :func:`sinkhornkernels.sinkhorn.sinkhorn`).
    """
    A = sparse.csr_matrix(adjacency)  # (N, N)
    if np.issubdtype(A.dtype, np.integer):
        A = A.astype(np.float64)
    degrees = np.asarray(A.sum(axis=1)).ravel()  # (N,)
    K = A + sparse.diags(degrees)  # (N, N)
    if eps == 0.0:
        return K.tocsr()
    return _SparsePlusConstant(K, eps)
