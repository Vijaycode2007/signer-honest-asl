"""ASL fingerspelling recognition."""

from __future__ import annotations

from functools import cache

__version__ = "0.1.0"


@cache
def _register_safe_globals() -> None:
    """Allow the numpy types our checkpoints contain."""
    import numpy as np
    import numpy.dtypes as nd
    import torch

    allow = [np.core.multiarray._reconstruct, np.ndarray, np.dtype]
    allow += [getattr(nd, n) for n in dir(nd) if n.endswith("DType")]
    torch.serialization.add_safe_globals(allow)


def safe_load(path, map_location="cpu"):
    """`torch.load` with weights_only=True.

    The server can download weights from SLR_WEIGHTS_URL, so loading must not
    run arbitrary pickle code. Numpy arrays are allowed for the scaler.
    """
    import torch

    _register_safe_globals()
    return torch.load(path, map_location=map_location, weights_only=True)
