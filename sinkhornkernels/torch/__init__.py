"""GPU / learning side of sinkhornkernels (PyTorch, optional pykeops).

Requires the ``[torch]`` extra: ``pip install sinkhornkernels[torch]``.
pykeops is only needed for the ``"keops"`` backends and is imported lazily.
"""

try:
    import einops  # noqa: F401
    import torch  # noqa: F401
except ImportError as e:
    raise ImportError(
        "sinkhornkernels.torch requires torch and einops; "
        "install them with `pip install sinkhornkernels[torch]`."
    ) from e

from . import diffusion, nn, sinkhorn
from .diffusion import apply_gaussian
from .sinkhorn import sinkhorn_log

__all__ = ["apply_gaussian", "diffusion", "nn", "sinkhorn", "sinkhorn_log"]
