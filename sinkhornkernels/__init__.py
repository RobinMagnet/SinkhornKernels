"""sinkhornkernels — official implementation of
"Sinkhorn Normalization of Diffusion Kernels" (ICML 2026).

The top-level namespace exposes the numpy/scipy core only.
 GPU / learning components are in :mod:`sinkhornkernels.torch` (requires the ``[torch]``
extra) and are never imported implicitly.
"""

from . import grid, kernels, mesh, operators, spectral
from .kernels import (
    exponential_kernel,
    exponential_kernel_from_dist,
    gaussian_kernel,
    gaussian_kernel_from_sqdist,
    gmm_kernel,
    gmm_log_kernel,
    graph_kernel,
    knn_graph,
    knn_to_sparse_sqdist,
    squared_distances,
)
from .operators import (
    NormalizedKernel,
    exponential_diffusion,
    gaussian_diffusion,
    gmm_diffusion,
)
from .sinkhorn import (
    sinkhorn,
    sinkhorn_log,
    sinkhorn_log_kernel,
    sinkhorn_log_sparse,
)
from .spectral import diffusion_eigsh, gmm_effective_sigma2, laplacian_eigenvalues

__version__ = "0.1.0"

__all__ = [
    "NormalizedKernel",
    "diffusion_eigsh",
    "exponential_diffusion",
    "exponential_kernel",
    "exponential_kernel_from_dist",
    "gaussian_diffusion",
    "gaussian_kernel",
    "gaussian_kernel_from_sqdist",
    "gmm_diffusion",
    "gmm_effective_sigma2",
    "gmm_kernel",
    "gmm_log_kernel",
    "graph_kernel",
    "grid",
    "kernels",
    "knn_graph",
    "knn_to_sparse_sqdist",
    "laplacian_eigenvalues",
    "mesh",
    "operators",
    "sinkhorn",
    "sinkhorn_log",
    "sinkhorn_log_kernel",
    "sinkhorn_log_sparse",
    "spectral",
    "squared_distances",
]
