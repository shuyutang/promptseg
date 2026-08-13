from __future__ import annotations
import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class Settings:
    # Model. SAM2_MODEL_ID can point at any SAM2-compatible checkpoint on the Hub,
    # e.g. facebook/sam2.1-hiera-small (faster) or wanglab/MedSAM2 (medical fine-tune).
    model_id: str = os.environ.get("SAM2_MODEL_ID", "facebook/sam2.1-hiera-base-plus")
    device: str = os.environ.get("SAM2_DEVICE", "cuda")
    # Set SAM2_STUB=1 to run the whole API without downloading weights (CI, laptops).
    stub: bool = os.environ.get("SAM2_STUB", "0") == "1"

    # How many workspaces (folders of images) to keep resident. Eviction is per
    # workspace, never per image, so a folder cannot lose slices mid-annotation.
    max_workspaces: int = _env_int("SAM2_MAX_WORKSPACES", 4)
    # Upper bound on one workspace, so a mis-picked folder cannot exhaust RAM.
    max_files: int = _env_int("SAM2_MAX_FILES", 500)
    # How many per-frame image embeddings to keep. These are the memory hogs.
    max_embeddings: int = _env_int("SAM2_MAX_EMBEDDINGS", 24)

    default_threshold: float = 0.5


settings = Settings()
