"""Q-DiffNet: DiffusionNet with Sinkhorn-normalized kernel diffusion.

This module adapts DiffusionNet (Sharp et al., 2022) so that the spectral
heat diffusion can be replaced by a pluggable diffusion layer — in
particular :class:`KernelDiffusionLayer`, which applies the paper's
Sinkhorn-normalized Gaussian kernel directly from point positions, with a
*learned bandwidth per channel*. Baseline layers (spectral, implicit Euler)
are provided with the same interface.

Scale semantics: the learned per-channel parameter of
:class:`LearnedScaleDiffusion` is a Gaussian bandwidth $\\sigma$ for
:class:`KernelDiffusionLayer` (heat time $t \\approx \\sigma^2/2$), but a
diffusion *time* $t$ for :class:`SpectralDiffusionLayer` and
:class:`SpatialDiffusionLayer` — do not compare the raw values across layer
types.

Following the paper, the Sinkhorn potentials are computed under
``torch.no_grad()`` (10 iterations by default): gradients flow through the
kernel application, not through the normalization loop.
"""

import copy

import einops
import torch as th
import torch.nn as nn

from .diffusion import apply_gaussian
from .sinkhorn import sinkhorn_log


def to_basis(values, basis, massvec):
    """Project ``values`` (B, N, C) onto a mass-orthonormal ``basis`` (B, N, K)."""
    return th.matmul(basis.transpose(-2, -1), values * massvec.unsqueeze(-1))


def from_basis(values, basis):
    """Reconstruct from basis coefficients: ``basis @ values``."""
    return th.matmul(basis, values)


class KernelDiffusionLayer(nn.Module):
    """Multi-scale (normalized) Gaussian kernel diffusion from point positions.

    Applies $Q(\\sigma_p)^{\\texttt{n\\_apply}}$ to each channel block
    ``signals[:, p]``, where $Q = \\Lambda K_{\\sigma_p} M \\Lambda$ is the
    Sinkhorn normalization of the Gaussian kernel (recomputed on every
    forward pass with ``update_potentials=True``).

    Parameters
    ----------
    backend : {"keops", "dense", "attention"}
        Kernel application backend (see
        :func:`sinkhornkernels.torch.diffusion.apply_gaussian`).
        ``"attention"`` requires ``row_normalize=True`` and evaluates the
        normalized operator through its row-softmax form (exact at Sinkhorn
        convergence). No global SDPA flags are modified. Wrap the forward in
        ``torch.nn.attention.sdpa_kernel(...)`` to select attention kernels.
    sinkhorn : bool
        Whether to Sinkhorn-normalize the kernel (True, default) or use the
        raw / row-normalized kernel.
    sinkhorn_n_iter : int
        Sinkhorn iterations per forward pass (paper: 10). Runs under
        ``torch.no_grad()``.
    row_normalize : bool
        Row-softmax normalization in the application step.
    min_scale : float, optional
        Clamp the (learned) bandwidths from below, in pre-``rescaling`` units.
    n_apply : int
        Number of successive applications of the operator (paper: 2).
    rescaling : float
        Multiplicative factor applied to the input bandwidths.

    Forward
    -------
    forward(points, sigmas, signals, masses=None, update_potentials=True)
        points : (B, N, d); sigmas : (p,); signals : (B, p, N, C);
        masses : (B, N), optional. With ``update_potentials=False`` the cached
        potentials from a previous forward are reused (they are still computed
        if no cache exists yet). Returns (B, p, N, C).
    """

    def __init__(
        self,
        backend="keops",
        sinkhorn=True,
        sinkhorn_n_iter=10,
        row_normalize=False,
        min_scale=None,
        n_apply=1,
        rescaling=1.0,
    ):
        super().__init__()
        if backend not in ("keops", "dense", "attention"):
            raise ValueError(f"Unknown backend {backend!r}.")
        if backend == "attention" and not row_normalize:
            raise ValueError("backend='attention' requires row_normalize=True.")
        if not isinstance(n_apply, int) or n_apply < 1:
            raise ValueError(f"n_apply must be a positive integer, got {n_apply!r}.")

        self.backend = backend
        self.sinkhorn = sinkhorn
        self.sinkhorn_n_iter = sinkhorn_n_iter
        self.row_normalize = row_normalize
        self.min_scale = min_scale
        self.n_apply = n_apply
        self.rescaling = 1.0 if rescaling is None else rescaling

        # sinkhorn solver backend matching the application backend
        self._sinkhorn_backend = {"keops": "keops", "dense": "dense", "attention": "auto"}[backend]
        self.pot_cache = None

    def extra_repr(self):
        return (
            f"backend={self.backend}, sinkhorn={self.sinkhorn}, "
            f"sinkhorn_n_iter={self.sinkhorn_n_iter}, row_normalize={self.row_normalize}, "
            f"n_apply={self.n_apply}"
        )

    def forward(self, points, sigmas, signals, masses=None, update_potentials=True):
        if self.min_scale is not None:
            sigmas = th.clamp(sigmas, min=self.min_scale)
        sigmas = sigmas * self.rescaling

        if self.sinkhorn and (update_potentials or self.pot_cache is None):
            with th.no_grad():
                self.pot_cache = sinkhorn_log(
                    points,
                    sigmas,
                    masses=masses,
                    n_iter=self.sinkhorn_n_iter,
                    backend=self._sinkhorn_backend,
                )

        log_potentials = self.pot_cache if self.sinkhorn else None

        out = signals
        for _ in range(self.n_apply):
            out = apply_gaussian(
                points,
                sigmas,
                out,
                log_potentials=log_potentials,
                masses=masses,
                row_normalize=self.row_normalize,
                backend=self.backend,
            )
        return out


class SpectralDiffusionLayer(nn.Module):
    """Spectral heat diffusion baseline: $\\Phi e^{-\\lambda t_p} \\Phi^\\top M$.

    Here the per-channel scales are diffusion *times* $t_p$.

    Forward: ``forward(sigmas, signals, eigenvalues, eigenvectors, masses)``
    with sigmas (p,), signals (B, p, N, C), eigenvalues (B, K),
    eigenvectors (B, N, K), masses (B, N).
    """

    def __init__(self, k_diff=None):
        super().__init__()
        self.k_diff = k_diff

    def forward(self, sigmas, signals, eigenvalues, eigenvectors, masses):
        K = self.k_diff if self.k_diff is not None else eigenvalues.shape[-1]

        projected = th.einsum(
            "bnk, bpnq -> bpkq",
            eigenvectors[..., :K],
            masses.unsqueeze(1).unsqueeze(-1) * signals,
        )  # (B, p, K, C)

        log_factors = einops.rearrange(eigenvalues[..., :K], "b k -> b () k ()")
        log_factors = log_factors * einops.rearrange(sigmas, "p -> () p () ()")
        projected = th.exp(-log_factors) * projected

        return th.einsum("bnk, bpkq -> bpnq", eigenvectors[..., :K], projected)


class SpatialDiffusionLayer(nn.Module):
    """Implicit-Euler heat step baseline: solve $(M + t_p L) x = M f$.

    Densifies the Laplacian to (B, p, N, N) for a batched Cholesky solve —
    O(p N^2) memory; intended as a reference on small problems.

    Forward: ``forward(sigmas, signals, L, masses)`` with sigmas (p,) diffusion
    times, signals (B, p, N, C), L a (B, N, N) (sparse) Laplacian, masses (B, N).
    """

    def forward(self, sigmas, signals, L, masses):
        p = sigmas.shape[-1]
        L_dense = L.to_dense() if L.is_sparse else L
        system = einops.repeat(L_dense, "b n m -> b p n m", p=p)
        system = system * einops.rearrange(sigmas, "p -> () p () ()")
        system = system + th.diag_embed(masses).unsqueeze(1)  # (B, p, N, N)

        chol = th.linalg.cholesky(system)
        rhs = signals * einops.rearrange(masses, "b n -> b () n ()")
        return th.cholesky_solve(rhs, chol)


class LearnedScaleDiffusion(nn.Module):
    """Diffusion with one learned scale per channel.

    Wraps a diffusion layer and owns the per-channel scale parameter. In the
    spectral domain the diffusion acts as $f \\mapsto e^{-\\lambda t} f$.
    The learned parameter is a bandwidth $\\sigma$ for
    :class:`KernelDiffusionLayer` and a time $t$ for the spectral /
    spatial baselines ($t \\approx \\sigma^2 / 2$).

    Parameters
    ----------
    diffusion_layer : nn.Module
        An instance of :class:`KernelDiffusionLayer`,
        :class:`SpectralDiffusionLayer` or :class:`SpatialDiffusionLayer`.
    n_features : int
        Number of channels p (one scale each).
    """

    def __init__(self, diffusion_layer, n_features):
        super().__init__()
        if not isinstance(diffusion_layer, nn.Module):
            raise TypeError(
                "diffusion_layer must be an nn.Module instance "
                "(KernelDiffusionLayer, SpectralDiffusionLayer or SpatialDiffusionLayer); "
                f"got {type(diffusion_layer)!r}."
            )
        self.n_features = n_features
        self.diffusion_layer = diffusion_layer

        if isinstance(self.diffusion_layer, (SpectralDiffusionLayer, SpatialDiffusionLayer)):
            self.diffusion_time = nn.Parameter(th.rand(n_features))
        else:
            # bandwidths: keep away from 0 at initialization
            self.diffusion_time = nn.Parameter(0.1 + th.rand(n_features))

    def get_times(self):
        """Learned per-channel scales, shape (p,)."""
        return self.diffusion_time.data

    def forward(self, x, masses, evals=None, evecs=None, points=None, L=None):
        """
        Parameters
        ----------
        x : (B, p, N, C) tensor
            Per-channel signal blocks.
        masses : (B, N) tensor
        evals, evecs :
            Spectral data, for :class:`SpectralDiffusionLayer`.
        points : (B, N, d) tensor
            Point positions, for :class:`KernelDiffusionLayer`.
        L : (B, N, N) tensor
            Laplacian, for :class:`SpatialDiffusionLayer`.
        """
        # project scales to the positive half-space
        with th.no_grad():
            self.diffusion_time.data = th.clamp(self.diffusion_time, min=1e-8)

        if isinstance(self.diffusion_layer, KernelDiffusionLayer):
            return self.diffusion_layer(points, self.diffusion_time, x, masses=masses)
        if isinstance(self.diffusion_layer, SpatialDiffusionLayer):
            return self.diffusion_layer(self.diffusion_time, x, L, masses)
        if isinstance(self.diffusion_layer, SpectralDiffusionLayer):
            return self.diffusion_layer(self.diffusion_time, x, evals, evecs, masses)
        raise TypeError(f"Unsupported diffusion layer {type(self.diffusion_layer)!r}.")


class SpatialGradientFeatures(nn.Module):
    """Dot products between spatial gradient vectors (from DiffusionNet).

    Uses a learned complex-linear map; input (B, N, C, 2), output (B, N, C).
    """

    def __init__(
        self, n_features, with_gradient_rotations=True, activation=nn.Tanh, **activation_kwargs
    ):
        super().__init__()
        self.n_features = n_features
        self.with_gradient_rotations = with_gradient_rotations
        self.activation = (
            activation(**activation_kwargs) if activation is not None else nn.Identity()
        )

        if self.with_gradient_rotations:
            self.A_re = nn.Linear(n_features, n_features, bias=False)
            self.A_im = nn.Linear(n_features, n_features, bias=False)
        else:
            self.A = nn.Linear(n_features, n_features, bias=False)

    def forward(self, vectors):
        if self.with_gradient_rotations:
            real = self.A_re(vectors[..., 0]) - self.A_im(vectors[..., 1])
            imag = self.A_re(vectors[..., 1]) + self.A_im(vectors[..., 0])
        else:
            real = self.A(vectors[..., 0])
            imag = self.A(vectors[..., 1])

        dots = vectors[..., 0] * real + vectors[..., 1] * imag
        return self.activation(dots)


class MiniMLP(nn.Module):
    """Simple MLP with configurable hidden sizes (dropout before non-first layers)."""

    def __init__(
        self, layer_sizes, dropout=False, dropout_prob=0.5, activation=nn.ReLU, **activation_kwargs
    ):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            if dropout and i > 0:
                layers.append(nn.Dropout(p=dropout_prob))
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i + 2 < len(layer_sizes):
                layers.append(activation(**activation_kwargs))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class DiffusionNetBlock(nn.Module):
    """One DiffusionNet block: learned-scale diffusion, gradient features, MLP, skip.

    ``diffusion_layer`` must be an ``nn.Module`` instance; each block should
    own its own instance (handled by :class:`DiffusionNet`).
    """

    def __init__(
        self,
        n_features,
        mlp_hidden_dims,
        diffusion_layer,
        n_feats_per_scale=1,
        dropout=True,
        dropout_prob=0.5,
        with_gradient_features=True,
        with_gradient_rotations=True,
        mlp_activation=nn.ReLU,
        **mlp_activation_kwargs,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_feats_per_scale = n_feats_per_scale
        self.with_gradient_features = with_gradient_features

        self.diffusion = LearnedScaleDiffusion(diffusion_layer, n_features)

        n_feats_total = n_features * n_feats_per_scale
        mlp_in = 2 * n_feats_total
        if with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(
                n_feats_total, with_gradient_rotations=with_gradient_rotations
            )
            mlp_in += n_feats_total

        self.mlp = MiniMLP(
            [mlp_in] + list(mlp_hidden_dims) + [n_feats_total],
            dropout=dropout,
            dropout_prob=dropout_prob,
            activation=mlp_activation,
            **mlp_activation_kwargs,
        )

    def get_times(self):
        return self.diffusion.get_times()

    def forward(self, x_in, masses, evals, evecs, gradX, gradY, points, L):
        B = x_in.shape[0]

        x_blocks = einops.rearrange(
            x_in, "b n (p q) -> b p n q", p=self.n_features, q=self.n_feats_per_scale
        )
        x_diffuse = self.diffusion(x_blocks, masses, evals=evals, evecs=evecs, points=points, L=L)
        x_diffuse = einops.rearrange(x_diffuse, "b p n q -> b n (p q)")

        if self.with_gradient_features:
            # torch.mm has no batch support for sparse gradX/gradY: loop over B
            x_grads = []
            for b in range(B):
                x_gradX = th.mm(gradX[b], x_diffuse[b])
                x_gradY = th.mm(gradY[b], x_diffuse[b])
                x_grads.append(th.stack((x_gradX, x_gradY), dim=-1))
            x_grad_features = self.gradient_features(th.stack(x_grads, dim=0))
            combined = th.cat((x_in, x_diffuse, x_grad_features), dim=-1)
        else:
            combined = th.cat((x_in, x_diffuse), dim=-1)

        return x_in + self.mlp(combined)


class DiffusionNet(nn.Module):
    """DiffusionNet with pluggable diffusion (Q-DiffNet when kernel-based).

    Parameters
    ----------
    in_dim, out_dim : int
    n_features : int
        Number of diffusion channels p (paper: 32 for Q-DiffNet).
    n_feats_per_scale : int
        Features per channel (paper: 8 for Q-DiffNet).
    N_block : int
        Number of blocks.
    diffusion_layer : nn.Module or callable
        Either a template instance (deep-copied for each block, so every
        block owns independent state) or a zero-argument factory returning a
        fresh instance per block. E.g.
        ``lambda: KernelDiffusionLayer(backend="keops", n_apply=2)``.
    last_activation : callable, optional
    outputs_at : {"vertices", "edges", "faces", "global_mean"}
    smooth_output : bool
        Project the output onto the first ``k_smooth`` eigenvectors.
    k_smooth : int, required when ``smooth_output=True``.
    mlp_hidden_dims : list of int, optional
    dropout, dropout_prob, with_gradient_features, with_gradient_rotations,
    mlp_activation, **mlp_activation_kwargs :
        As in DiffusionNet.

    Forward
    -------
    forward(x_in, masses, evals=None, evecs=None, gradX=None, gradY=None,
            edges=None, faces=None, points=None, L=None)
        Shapes: x_in (B, N, in_dim) or (N, in_dim); masses (B, N); evals
        (B, K); evecs (B, N, K); gradX/gradY (B, N, N) sparse; points
        (B, N, d); L (B, N, N). Only the operators needed by the chosen
        diffusion layer are required.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        n_features=128,
        n_feats_per_scale=1,
        N_block=4,
        diffusion_layer=None,
        last_activation=None,
        outputs_at="vertices",
        smooth_output=False,
        k_smooth=None,
        mlp_hidden_dims=None,
        dropout=True,
        dropout_prob=0.5,
        with_gradient_features=True,
        with_gradient_rotations=True,
        mlp_activation=nn.ReLU,
        **mlp_activation_kwargs,
    ):
        super().__init__()
        if outputs_at not in ("vertices", "edges", "faces", "global_mean"):
            raise ValueError("invalid setting for outputs_at")
        if smooth_output and k_smooth is None:
            raise ValueError("smooth_output=True requires k_smooth.")
        if diffusion_layer is None:
            raise ValueError(
                "diffusion_layer is required: pass an instance (deep-copied per block) "
                "or a factory callable, e.g. lambda: KernelDiffusionLayer()."
            )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_features = n_features
        self.n_feats_per_scale = n_feats_per_scale
        self.N_block = N_block
        self.last_activation = last_activation
        self.outputs_at = outputs_at
        self.smooth_output = smooth_output
        self.k_smooth = k_smooth

        if mlp_hidden_dims is None:
            n_total = n_features * n_feats_per_scale
            mlp_hidden_dims = [n_total, n_total]

        if isinstance(diffusion_layer, nn.Module):
            make_layer = lambda: copy.deepcopy(diffusion_layer)  # noqa: E731
        elif callable(diffusion_layer):
            make_layer = diffusion_layer
        else:
            raise TypeError("diffusion_layer must be an nn.Module or a factory callable.")

        self.first_lin = nn.Linear(in_dim, n_features * n_feats_per_scale)
        self.last_lin = nn.Linear(n_features * n_feats_per_scale, out_dim)

        self.blocks = nn.ModuleList(
            DiffusionNetBlock(
                n_features=n_features,
                n_feats_per_scale=n_feats_per_scale,
                mlp_hidden_dims=mlp_hidden_dims,
                diffusion_layer=make_layer(),
                dropout=dropout,
                dropout_prob=dropout_prob,
                with_gradient_features=with_gradient_features,
                with_gradient_rotations=with_gradient_rotations,
                mlp_activation=mlp_activation,
                **mlp_activation_kwargs,
            )
            for _ in range(N_block)
        )

    def get_times(self):
        """Learned scales of all blocks, shape (N_block, p)."""
        return th.stack([block.get_times() for block in self.blocks], dim=0)

    def forward(
        self,
        x_in,
        masses,
        evals=None,
        evecs=None,
        gradX=None,
        gradY=None,
        edges=None,
        faces=None,
        points=None,
        L=None,
    ):
        if x_in.shape[-1] != self.in_dim:
            raise ValueError(
                f"DiffusionNet was constructed with in_dim={self.in_dim}, "
                f"but x_in has last dim={x_in.shape[-1]}"
            )

        appended_batch_dim = x_in.ndim == 2
        if appended_batch_dim:
            unsq = lambda t: None if t is None else t.unsqueeze(0)  # noqa: E731
            x_in, masses = x_in.unsqueeze(0), masses.unsqueeze(0)
            evals, evecs, gradX, gradY = map(unsq, (evals, evecs, gradX, gradY))
            edges, faces, points, L = map(unsq, (edges, faces, points, L))
        elif x_in.ndim != 3:
            raise ValueError("x_in should have shape (N, C) or (B, N, C)")

        x = self.first_lin(x_in)
        for block in self.blocks:
            x = block(x, masses, evals, evecs, gradX, gradY, points, L)
        x = self.last_lin(x)

        if self.smooth_output:
            basis = evecs[..., : self.k_smooth]
            x = from_basis(to_basis(x, basis, masses), basis)

        if self.outputs_at == "vertices":
            x_out = x
        elif self.outputs_at == "edges":
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 2)
            edges_gather = edges.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            x_out = th.gather(x_gather, 1, edges_gather).mean(dim=-1)
        elif self.outputs_at == "faces":
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
            faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            x_out = th.gather(x_gather, 1, faces_gather).mean(dim=-1)
        else:  # global_mean: mass-weighted, discretization-invariant
            x_out = (x * masses.unsqueeze(-1)).sum(dim=-2) / masses.sum(dim=-1, keepdim=True)

        if self.last_activation is not None:
            x_out = self.last_activation(x_out)

        return x_out.squeeze(0) if appended_batch_dim else x_out
