# SinkhornKernels

![](  https://robinmagnet.github.io/ICML26_Kernels/teaser.png)

Official implementation of the ICML 2026 paper
**“Sinkhorn Normalization of Diffusion Kernels”**
by Nathan Kessler, Robin Magnet and Jean Feydy.

Given any symmetric positive kernel $K$ and point masses $m$ (with $M = \mathrm{diag}(m)$),
the symmetric Sinkhorn normalization computes the unique positive diagonal $\Lambda$ such that

$$Q =\Lambda K M\Lambda$$

is a **diffusion operator**: $Q\mathbf{1} = \mathbf{1}$, self-adjoint for the mass-weighted
inner product, with spectrum in $[0, 1]$. Each Sinkhorn iteration costs one kernel
matvec, and 5–10 iterations suffice in practice. This naturally turns raw Gaussian kernels on point clouds, voxel grids, Gaussian mixtures or graphs into well-behaved heat-diffusion-like
operators at a fixed scale $\sigma$.

## Installation

```bash
git clone https://github.com/RobinMagnet/SinkhornKernels
cd SinkhornKernels
pip install -e .                # numpy/scipy core
pip install -e ".[torch]"       # + PyTorch backends and Q-DiffNet layers
pip install -e ".[keops]"       # + PyKeOps (O(N) memory GPU backend)
pip install -e ".[examples]"    # + notebook dependencies (pyvista, tetgen, scikit-learn, ...)
```

The top-level `sinkhornkernels` depends only on numpy / scipy / scikit-learn.
Torch implementation is in `sinkhornkernels.torch` and is never imported implicitly.

## Quickstart

```python
import numpy as np
from sinkhornkernels import gaussian_diffusion, diffusion_eigsh, laplacian_eigenvalues

points = np.random.rand(5000, 3)          # any point cloud
masses = np.full(5000, 1 / 5000)          # e.g. uniform weights or vertex areas

# Sinkhorn-normalized Gaussian diffusion operator (matrix-free, log-domain solver)
Q = gaussian_diffusion(points, sigma=0.05, masses=masses)
print(Q.marginal_error())                  # rows of Q sum to 1

smoothed = Q @ signal                      # mass-conserving smoothing
evals, evecs = diffusion_eigsh(Q, k=64)    # spectrum in (0, 1], mass-orthonormal modes
lambdas = laplacian_eigenvalues(evals, sigma=0.05)   # Laplacian-like eigenvalues
```

## Package overview

| Module | Contents |
|---|---|
| `sinkhornkernels.sinkhorn` | Sinkhorn solvers |
| `sinkhornkernels.kernels` | Gaussian & exponential kernels, anisotropic Gaussian-mixture kernel, graph kernel |
| `sinkhornkernels.operators` | `NormalizedKernel` with modes `sinkhorn` / `row` / `symmetric_one_step` / `none`; `gaussian_diffusion` / `exponential_diffusion` / `gmm_diffusion` |
| `sinkhornkernels.grid` | `GridGaussian`: separable-convolution kernel on voxel maskss |
| `sinkhornkernels.spectral` |eigenvalue conversions |
| `sinkhornkernels.mesh` | Reference FEM Laplacians |
| `sinkhornkernels.torch.sinkhorn` | Batched multi-scale log-domain Sinkhorn (dense / KeOps) |
| `sinkhornkernels.torch.diffusion` | GPU compatible diffusion with backends: dense, KeOps, flash-compatible **attention** |
| `sinkhornkernels.torch.nn` | **Q-DiffNet**: `KernelDiffusionLayer` with learned per-channel bandwidths, plugged into DiffusionNet; spectral / implicit-Euler baselines |

## Numerical stability

For exponentially-decaying kernels (Gaussian, exponential, GMM), prefer log-domain computation using `NormalizedKernel.from_log_kernel`.

## Examples


| Notebook | Contents |
|---|---|
| [`examples/01_pointcloud_quickstart.ipynb`](examples/01_pointcloud_quickstart.ipynb) | Kernel diffusion |
| [`examples/02_modalities_spectra.ipynb`](examples/02_modalities_spectra.ipynb) | Different shape representations |
| [`examples/03_qdiffnet_layers.ipynb`](examples/03_qdiffnet_layers.ipynb) | Q-DiffNet  |


## Q-DiffNet

```python
import torch
from sinkhornkernels.torch.nn import DiffusionNet, KernelDiffusionLayer

net = DiffusionNet(
    in_dim=128, out_dim=128, n_features=32, n_feats_per_scale=8, N_block=4,
    diffusion_layer=lambda: KernelDiffusionLayer(backend="keops", n_apply=2, sinkhorn_n_iter=10),
)
out = net(features, masses, points=points, gradX=gradX, gradY=gradY)
```


## Citation

```bibtex
@inproceedings{kessler2026sinkhorn,
  title     = {Sinkhorn Normalization of Diffusion Kernels},
  author    = {Kessler, Nathan and Magnet, Robin and Feydy, Jean},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
}
```
