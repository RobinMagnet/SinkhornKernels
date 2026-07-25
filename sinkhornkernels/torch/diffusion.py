"""Applying (normalized) Gaussian kernels to signals — three backends.

Given points, bandwidths, Sinkhorn log-potentials ``f`` and masses ``m``,
these functions evaluate, per batch element and scale,

$$
    (Q s)_i = \\sum_j \\exp\\big({-\\|x_i-x_j\\|^2}/{2\\sigma_p^2}
                + f_i + f_j + \\log m_j\\big)\\, s_j,
$$

optionally row-softmax-normalized instead. Backends:

- ``"dense"``: explicit (B, p, N, N) log-kernel (reference; O(N^2) memory);
- ``"keops"``: PyKeOps lazy reduction (O(N) memory, GPU-friendly);
- ``"attention"``: ``scaled_dot_product_attention`` on augmented embeddings —
  the Gaussian scores become plain dot products, so Flash-attention kernels
  apply. Since attention always softmaxes, this backend is *row-normalized by
  construction*; at Sinkhorn convergence (``Q 1 = 1``) the row-softmax with
  ``bias = f`` coincides exactly with the normalized operator.
"""

from functools import lru_cache

import einops
import numpy as np
import torch as th
import torch.nn.functional as F
from scipy.sparse.linalg import LinearOperator

from .sinkhorn import _as_sigmas

_BACKENDS = ("dense", "keops", "attention")


def apply_gaussian(
    points,
    sigmas,
    signals,
    log_potentials=None,
    masses=None,
    row_normalize=False,
    backend="dense",
):
    """Apply the (normalized) Gaussian kernel to per-scale signals.

    Parameters
    ----------
    points : (B, N, d) tensor
    sigmas : (p,) tensor
    signals : (B, p, N, C) tensor
        One signal block per scale.
    log_potentials : (B, p, N) tensor, optional
        Sinkhorn log-potentials (from :func:`sinkhornkernels.torch.sinkhorn_log`).
    masses : (B, N) tensor, optional
    row_normalize : bool
        Row-softmax normalization instead of (or on top of) the potentials.
    backend : {"dense", "keops", "attention"}
        ``"attention"`` requires ``row_normalize=True`` (softmax is built in).

    Returns
    -------
    out : (B, p, N, C) tensor
    """
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; expected one of {_BACKENDS}.")
    if backend == "attention":
        if not row_normalize:
            raise ValueError(
                "backend='attention' is row-normalized by construction; "
                "call with row_normalize=True (with Sinkhorn potentials as bias, "
                "the result equals the normalized operator at convergence)."
            )
        return apply_gaussian_attention(points, sigmas, signals, bias=log_potentials, masses=masses)
    if backend == "keops":
        return apply_gaussian_keops(
            points,
            sigmas,
            signals,
            log_potentials=log_potentials,
            masses=masses,
            row_normalize=row_normalize,
        )
    return apply_gaussian_dense(
        points,
        sigmas,
        signals,
        log_potentials=log_potentials,
        masses=masses,
        row_normalize=row_normalize,
    )


def apply_gaussian_dense(
    points, sigmas, signals, log_potentials=None, masses=None, row_normalize=False
):
    """Dense reference implementation (materializes the (B, p, N, N) log-kernel)."""
    sigmas = _as_sigmas(sigmas, points)
    sqdist = th.cdist(points, points) ** 2  # (B, N, N)
    log_kernel = -sqdist.unsqueeze(1) / (2 * sigmas.view(1, -1, 1, 1) ** 2)  # (B, p, N, N)

    if masses is not None:
        log_kernel = log_kernel + einops.rearrange(th.log(masses), "b n -> b () () n")
    if log_potentials is not None:
        log_kernel = log_kernel + einops.rearrange(log_potentials, "b p n -> b p () n")
        log_kernel = log_kernel + einops.rearrange(log_potentials, "b p n -> b p n ()")

    weights = th.softmax(log_kernel, dim=-1) if row_normalize else th.exp(log_kernel)
    return th.einsum("bpnm, bpmq -> bpnq", weights, signals)


def _diffusion_formula(emb_dim, signal_dim, use_pot, use_masses, row_normalize):
    """KeOps reduction applying the (normalized) Gaussian kernel to a signal."""
    import pykeops.torch as pk

    s_j = pk.Vj(0, signal_dim)
    x_i = pk.Vi(1, emb_dim)
    y_j = pk.Vj(2, emb_dim)
    eps = pk.Pm(3, 1)  # 2 * sigma^2

    idx = 3
    mass_j = None
    if use_masses:
        idx += 1
        mass_j = pk.Vj(idx, 1)
    f_i = f_j = None
    if use_pot:
        f_i = pk.Vi(idx + 1, 1)
        f_j = pk.Vj(idx + 2, 1)

    log_kernel = -x_i.sqdist(y_j) / eps
    if use_pot:
        log_kernel = log_kernel + f_i + f_j
    if use_masses:
        log_kernel = log_kernel + mass_j.log()

    if row_normalize:
        return log_kernel.sumsoftmaxweight(s_j, axis=1)
    return (log_kernel.exp() * s_j).sum(axis=1)


def apply_gaussian_keops(
    points, sigmas, signals, log_potentials=None, masses=None, row_normalize=False
):
    """KeOps implementation — O(N) memory, scales folded into the batch dimension."""
    sigmas = _as_sigmas(sigmas, points)
    p = sigmas.shape[0]

    formula = _diffusion_formula(
        points.shape[-1],
        signals.shape[-1],
        use_pot=log_potentials is not None,
        use_masses=masses is not None,
        row_normalize=row_normalize,
    )

    points_par = einops.repeat(points, "b n d -> (b p) n d", p=p).contiguous()
    signals_par = einops.rearrange(signals, "b p n q -> (b p) n q").contiguous()
    eps_par = einops.repeat(2 * sigmas**2, "p -> (b p) ()", b=points.shape[0]).contiguous()

    args = [signals_par, points_par, points_par, eps_par]
    if masses is not None:
        args.append(einops.repeat(masses, "b n -> (b p) n", p=p).contiguous())
    if log_potentials is not None:
        f_par = einops.rearrange(log_potentials, "b p n -> (b p) n").contiguous()
        args += [f_par, f_par]

    out = formula(*args)  # (B*p, N, C)
    return einops.rearrange(out, "(b p) n q -> b p n q", p=p)


def apply_gaussian_attention(points, sigmas, signals, bias=None, masses=None, twodir=False):
    """Row-softmax Gaussian diffusion as scaled-dot-product attention.

    The Gaussian scores are expressed as dot products of augmented embeddings:
    with queries $\\tilde{x}_i = [x_i, 1]/\\sigma$ and keys
    $\\tilde{y}_j = [x_j, -\\|x_j\\|^2/2 + \\sigma^2 b_j
    + \\sigma^2 \\log m_j]/\\sigma$,

    $$
        \\tilde{x}_i^\\top \\tilde{y}_j = -\\|x_i - x_j\\|^2 / 2\\sigma^2
            + b_j + \\log m_j + \\text{const}_i,
    $$

    and the row constant cancels in the softmax. Runs through
    ``F.scaled_dot_product_attention(..., scale=1)``, so Flash / memory-
    efficient attention kernels can be used when available. No global SDPA
    backend flags are touched; select kernels externally with
    ``torch.nn.attention.sdpa_kernel`` if desired.

    Parameters
    ----------
    points : (B, N, d) tensor
    sigmas : (p,) tensor
    signals : (B, p, N, C) tensor
    bias : (B, p, N) tensor, optional
        Per-point additive scores — pass the Sinkhorn log-potentials to
        evaluate the normalized operator (exact at convergence).
    masses : (B, N) tensor, optional
    twodir : bool
        Use the symmetric (d+2)-dimensional augmentation carrying the bias on
        both sides. Equivalent after the softmax; useful when the raw scores
        matter.

    Returns
    -------
    out : (B, p, N, C) tensor
    """
    sigmas = _as_sigmas(sigmas, points)
    B, N, _ = points.shape

    sq_norm = points.square().sum(-1)  # (B, N)
    sig = sigmas.view(1, -1, 1, 1)  # (1, p, 1, 1)

    x = points.unsqueeze(1) / sig  # (B, p, N, d)
    ones = th.ones(B, sigmas.shape[0], N, 1, dtype=points.dtype, device=points.device)

    # key "constant slot": (-|y|^2/2 + sigma^2 * (bias + log m)) / sigma
    key_slot = -sq_norm.unsqueeze(1) / 2  # (B, p, N)
    if bias is not None:
        key_slot = key_slot + sig[..., 0] ** 2 * bias
    if masses is not None:
        key_slot = key_slot + sig[..., 0] ** 2 * th.log(masses).unsqueeze(1)
    key_slot = (key_slot / sig[..., 0]).unsqueeze(-1)  # (B, p, N, 1)

    if twodir:
        query_slot = -sq_norm.unsqueeze(1) / 2
        if bias is not None:
            query_slot = query_slot + sig[..., 0] ** 2 * bias
        query_slot = (query_slot / sig[..., 0]).unsqueeze(-1)
        queries = th.cat([x, ones / sig, query_slot], dim=-1)  # (B, p, N, d+2)
        keys = th.cat([x, key_slot, ones / sig], dim=-1)  # (B, p, N, d+2)
    else:
        queries = th.cat([x, ones / sig], dim=-1)  # (B, p, N, d+1)
        keys = th.cat([x, key_slot], dim=-1)  # (B, p, N, d+1)

    return F.scaled_dot_product_attention(
        queries.contiguous(), keys.contiguous(), signals.contiguous(), scale=1.0
    )


class KeopsGaussianKernel(LinearOperator):
    """numpy-oriented Gaussian kernel matvec evaluated with PyKeOps.

    A scipy ``LinearOperator`` whose matvec runs on GPU (when available)
    without ever materializing the (N, N) kernel — the bridge between the
    numpy core (:func:`sinkhornkernels.sinkhorn.sinkhorn`,
    :class:`sinkhornkernels.operators.NormalizedKernel`,
    :func:`sinkhornkernels.spectral.diffusion_eigsh`) and KeOps for large
    point clouds. numpy/torch conversion happens once per matvec, which can be
    expensive.

    Parameters
    ----------
    points : (N, d) array-like
    sigma : float
        Gaussian bandwidth.
    device : str or torch.device, optional
        Defaults to CUDA when available.
    """

    def __init__(self, points, sigma, device=None):
        if device is None:
            device = "cuda" if th.cuda.is_available() else "cpu"
        self.device = th.device(device)
        self.sigma = float(sigma)
        self._points = th.as_tensor(np.asarray(points), dtype=th.float64, device=self.device)
        N = self._points.shape[0]
        super().__init__(dtype=np.dtype(np.float64), shape=(N, N))

    def _matvec(self, x):
        from pykeops.torch import LazyTensor

        v = th.as_tensor(np.asarray(x).reshape(-1, 1), dtype=th.float64, device=self.device)
        x_i = LazyTensor(self._points[:, None, :])
        y_j = LazyTensor(self._points[None, :, :])
        log_kernel = -((x_i - y_j) ** 2).sum(-1) / (2 * self.sigma**2)
        with th.no_grad():
            out = log_kernel.exp() @ v  # (N, 1)
        return out.squeeze(-1).cpu().numpy()

    def _rmatvec(self, x):  # symmetric
        return self._matvec(x)
