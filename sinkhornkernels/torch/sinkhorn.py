"""Batched, multi-scale log-domain symmetric Sinkhorn (PyTorch / PyKeOps).

Torch counterpart of :mod:`sinkhornkernels.sinkhorn` for the Gaussian kernel,
with a batch dimension and a family of ``p`` bandwidths solved at once (as
used by the Q-DiffNet layers, where each channel owns its own scale):
``(B, N, d)`` points and ``(p,)`` sigmas produce ``(B, p, N)`` log-potentials.

The ``"keops"`` backend evaluates the logsumexp reduction with PyKeOps lazy
tensors — O(N) memory, no (N, N) matrix — folding the ``p`` scales into the
KeOps batch dimension.
"""

import einops
import torch as th


def _as_sigmas(sigmas, ref):
    sigmas = th.as_tensor(sigmas, dtype=ref.dtype, device=ref.device)
    return sigmas.reshape(1) if sigmas.ndim == 0 else sigmas


def sinkhorn_log(points, sigmas, masses=None, n_iter=10, backend="auto"):
    """Log-potentials of the symmetric Sinkhorn normalization, per scale.

    Solves, for every batch element and every bandwidth ``sigma_p``, the same
    fixed point as :func:`sinkhornkernels.sinkhorn.sinkhorn_log`:

    $$
        f \\leftarrow \\tfrac12 f - \\tfrac12 \\operatorname{logsumexp}_j
        \\big({-\\|x_i-x_j\\|^2}/{2\\sigma_p^2} + f_j + \\log m_j\\big).
    $$

    Parameters
    ----------
    points : (B, N, d) tensor
    sigmas : (p,) tensor
        Gaussian bandwidths.
    masses : (B, N) tensor, optional
    n_iter : int
        Number of updates (no early stopping in the batched solver; use
        :func:`marginal_error` to monitor convergence).
    backend : {"auto", "dense", "keops"}
        ``"auto"`` uses KeOps when importable, else the dense torch path.

    Returns
    -------
    log_potentials : (B, p, N) tensor
    """
    if backend == "auto":
        backend = "keops" if _has_keops() else "dense"
    if backend == "dense":
        return sinkhorn_log_dense(points, sigmas, masses=masses, n_iter=n_iter)
    if backend == "keops":
        return sinkhorn_log_keops(points, sigmas, masses=masses, n_iter=n_iter)
    raise ValueError(f"Unknown backend {backend!r}; expected 'auto', 'dense' or 'keops'.")


def _has_keops():
    try:
        import pykeops.torch  # noqa: F401

        return True
    except ImportError:
        return False


def sinkhorn_log_dense(points, sigmas, masses=None, n_iter=10):
    """Dense torch implementation of :func:`sinkhorn_log` (O(N^2) memory)."""
    sigmas = _as_sigmas(sigmas, points)
    sqdist = th.cdist(points, points) ** 2  # (B, N, N)
    return sinkhorn_log_dense_from_sqdist(sqdist, sigmas, masses=masses, n_iter=n_iter)


def sinkhorn_log_dense_from_sqdist(sqdist, sigmas, masses=None, n_iter=10):
    """:func:`sinkhorn_log_dense` from a precomputed (B, N, N) squared-distance matrix."""
    sigmas = _as_sigmas(sigmas, sqdist)
    D = -sqdist.unsqueeze(1) / (2 * sigmas.view(1, -1, 1, 1) ** 2)  # (B, p, N, N)

    B, p, N = D.shape[0], D.shape[1], D.shape[-1]
    f = th.zeros(B, p, N, dtype=D.dtype, device=D.device)
    log_m = None if masses is None else th.log(masses)  # (B, N)

    for _ in range(n_iter):
        if log_m is None:
            lse = th.logsumexp(D + f.unsqueeze(-2), dim=-1)  # (B, p, N)
        else:
            lse = th.logsumexp(D + (f + log_m.unsqueeze(1)).unsqueeze(-2), dim=-1)
        f = 0.5 * f - 0.5 * lse

    return f


def _sinkhorn_formula(emb_dim, use_masses):
    """KeOps reduction: lse_i = logsumexp_j(-|x_i-x_j|^2/eps + f_j [+ log m_j])."""
    import pykeops.torch as pk

    x_i = pk.Vi(0, emb_dim)
    y_j = pk.Vj(1, emb_dim)
    eps = pk.Pm(2, 1)  # 2 * sigma^2
    f_j = pk.Vj(3, 1)
    mass_j = pk.Vj(4, 1) if use_masses else None

    scores = -x_i.sqdist(y_j) / eps + f_j
    return scores.logsumexp(axis=1, weight=mass_j)


def _sanity_formula(emb_dim, use_masses):
    """KeOps reduction: log of the row sums of the normalized kernel."""
    import pykeops.torch as pk

    x_i = pk.Vi(0, emb_dim)
    y_j = pk.Vj(1, emb_dim)
    eps = pk.Pm(2, 1)
    f_j = pk.Vj(3, 1)
    f_i = pk.Vi(4, 1)
    mass_j = pk.Vj(5, 1) if use_masses else None

    scores = -x_i.sqdist(y_j) / eps + f_i + f_j
    return scores.logsumexp(axis=1, weight=mass_j)


def _parallel_args(points, sigmas, masses):
    """Fold the p scales into the KeOps batch dimension."""
    B, p = points.shape[0], sigmas.shape[0]
    points_par = einops.repeat(points, "b n d -> (b p) n d", p=p).contiguous()
    eps_par = einops.repeat(2 * sigmas**2, "p -> (b p) ()", b=B).contiguous()
    masses_par = (
        None if masses is None else einops.repeat(masses, "b n -> (b p) n", p=p).contiguous()
    )
    return points_par, eps_par, masses_par


def sinkhorn_log_keops(points, sigmas, masses=None, n_iter=10):
    """KeOps implementation of :func:`sinkhorn_log` (O(N) memory)."""
    sigmas = _as_sigmas(sigmas, points)
    B, N = points.shape[0], points.shape[1]
    p = sigmas.shape[0]

    formula = _sinkhorn_formula(points.shape[-1], masses is not None)
    points_par, eps_par, masses_par = _parallel_args(points, sigmas, masses)

    f_par = th.zeros(B * p, N, dtype=points.dtype, device=points.device)
    for _ in range(n_iter):
        args = [points_par, points_par, eps_par, f_par]
        if masses_par is not None:
            args.append(masses_par)
        lse = formula(*args).squeeze(-1)  # (B*p, N)
        f_par = 0.5 * f_par - 0.5 * lse

    return einops.rearrange(f_par, "(b p) n -> b p n", p=p)


def marginal_error(points, sigmas, log_potentials, masses=None, backend="auto"):
    """L1(mu)-averaged violation of the unit row-sum constraint, per scale.

    Parameters
    ----------
    points : (B, N, d) tensor
    sigmas : (p,) tensor
    log_potentials : (B, p, N) tensor
        As returned by :func:`sinkhorn_log`.
    masses : (B, N) tensor, optional
    backend : {"auto", "dense", "keops"}

    Returns
    -------
    err : (B, p) tensor
    """
    if backend == "auto":
        backend = "keops" if _has_keops() else "dense"
    sigmas = _as_sigmas(sigmas, points)
    p = sigmas.shape[0]

    if backend == "dense":
        sqdist = th.cdist(points, points) ** 2
        D = -sqdist.unsqueeze(1) / (2 * sigmas.view(1, -1, 1, 1) ** 2)
        fj = log_potentials
        if masses is not None:
            fj = fj + th.log(masses).unsqueeze(1)
        lse = th.logsumexp(D + fj.unsqueeze(-2), dim=-1)  # (B, p, N)
    elif backend == "keops":
        formula = _sanity_formula(points.shape[-1], masses is not None)
        points_par, eps_par, masses_par = _parallel_args(points, sigmas, masses)
        f_par = einops.rearrange(log_potentials, "b p n -> (b p) n").contiguous()
        args = [points_par, points_par, eps_par, f_par, f_par]
        if masses_par is not None:
            args.append(masses_par)
        # note: all arguments share the (B*p) batch dimension — the potentials
        # and masses must be the repeated "parallel" versions
        lse = einops.rearrange(formula(*args).squeeze(-1), "(b p) n -> b p n", p=p)
        lse = lse - log_potentials  # sanity formula already includes f_i
    else:
        raise ValueError(f"Unknown backend {backend!r}; expected 'auto', 'dense' or 'keops'.")

    rowsums = th.exp(log_potentials + lse)  # (B, p, N)
    dev = th.abs(rowsums - 1.0)
    if masses is None:
        return dev.mean(dim=-1)
    w = masses.unsqueeze(1)
    return (dev * w).sum(dim=-1) / w.sum(dim=-1)
