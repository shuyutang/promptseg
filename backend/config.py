"""Runtime configuration, read once from the environment at import time.

Every knob is an environment variable so a deployment can be retuned without
touching code, and so the test suite can run the whole API against a stub model.
The documented list lives in ``docs/setup.md``.
"""
from __future__ import annotations
import os


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable.

    Args:
        name: Variable name.
        default: Value to use when unset or unparseable. A typo in a limit
            should not stop the server from starting.

    Returns:
        The parsed value, or ``default``.
    """
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class Settings:
    """Process-wide settings, resolved from the environment.

    Attributes:
        model_id: Hugging Face checkpoint to load. Any SAM2-compatible model
            works, e.g. ``facebook/sam2.1-hiera-small`` (faster) or
            ``wanglab/MedSAM2`` (medical fine-tune). From ``SAM2_MODEL_ID``.
        device: Preferred torch device, from ``SAM2_DEVICE``. The runner falls
            back to CPU on its own if CUDA is unavailable.
        stub: True when ``SAM2_STUB=1``: run the whole API against a
            prompt-responsive fake model, with no weights and no GPU.
        max_workspaces: How many workspaces (folders of images) stay resident,
            from ``SAM2_MAX_WORKSPACES``. Eviction is per workspace, never per
            image, so a folder cannot lose slices mid-annotation.
        max_files: Upper bound on one workspace, from ``SAM2_MAX_FILES``, so a
            mis-picked folder cannot exhaust RAM.
        max_embeddings: How many per-frame image embeddings to cache, from
            ``SAM2_MAX_EMBEDDINGS``. These are the memory hogs (~17 MB each).
        default_threshold: Probability above which a pixel is foreground, used
            when a request does not say.
    """

    model_id: str = os.environ.get("SAM2_MODEL_ID", "facebook/sam2.1-hiera-base-plus")
    device: str = os.environ.get("SAM2_DEVICE", "cuda")
    stub: bool = os.environ.get("SAM2_STUB", "0") == "1"

    max_workspaces: int = _env_int("SAM2_MAX_WORKSPACES", 4)
    max_files: int = _env_int("SAM2_MAX_FILES", 500)
    max_embeddings: int = _env_int("SAM2_MAX_EMBEDDINGS", 24)

    default_threshold: float = 0.5


settings = Settings()
"""The single settings instance every module imports."""
