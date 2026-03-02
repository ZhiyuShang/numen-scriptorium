from __future__ import annotations

import os
import random

import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Set process-wide random seed for reproducibility.

    Notes:
    - Full bitwise reproducibility across different hardware/drivers is not guaranteed.
    - deterministic=True may reduce performance.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    try:
        import numpy as np  # optional runtime dependency

        np.random.seed(seed)
    except Exception:
        pass

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
