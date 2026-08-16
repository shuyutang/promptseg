"""Real-model checks. Skipped unless SAM2_STUB=0 and CUDA is present.

These assert what only a real checkpoint can show: plausible masks, candidates
ranked by score, prompts that actually steer the result, and the embedding cache
paying for itself.

Run with:  SAM2_STUB=0 pytest tests/test_sam2_gpu.py -q -s
"""
from __future__ import annotations
import os
import time

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SAM2_STUB", "1") != "0", reason="set SAM2_STUB=0 to run real-model tests"
)


@pytest.fixture(scope="module")
def runner():
    """Load the real model once for the module.

    Returns:
        Sam2Runner: With weights on the GPU.

    Raises:
        Skipped: If CUDA is unavailable.
    """
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from models.sam2_runner import Sam2Runner
    return Sam2Runner(stub=False)


@pytest.fixture(scope="module")
def mr_rgb():
    """Render a real CT frame the way the app would.

    Returns:
        numpy.ndarray: HxWx3 uint8, exactly what the model is given in production.
    """
    import pydicom
    from pydicom.data import get_testdata_file
    from dicom.io import frame_rgb, load_single
    from pathlib import Path
    img = load_single(Path(get_testdata_file("CT_small.dcm")).read_bytes())
    return frame_rgb(img, 0)


def test_produces_plausible_mask(runner, mr_rgb):
    """A centre click returns a mask that is neither empty nor the whole frame.

    Args:
        runner: The real-model runner.
        mr_rgb: The rendered frame.
    """
    h, w = mr_rgb.shape[:2]
    res = runner.segment("t:0:0:0", mr_rgb, {"points": [[w // 2, h // 2, 1]]}, 0.5, 0)
    assert res.mask.shape == (h, w)
    assert res.mask.dtype == np.uint8
    assert 0 < res.mask.sum() < h * w, "mask should be neither empty nor the whole frame"
    assert 0.0 <= res.score <= 1.0
    assert res.num_candidates == 3


def test_candidates_are_ranked_by_score(runner, mr_rgb):
    """mask_index 0 is the best-scoring candidate, which is what the UI assumes.

    Args:
        runner: The real-model runner.
        mr_rgb: The rendered frame.
    """
    h, w = mr_rgb.shape[:2]
    p = {"points": [[w // 2, h // 2, 1]]}
    scores = [runner.segment("t:0:0:0", mr_rgb, p, 0.5, i).score for i in range(3)]
    assert scores == sorted(scores, reverse=True), f"index 0 must be best: {scores}"


def test_negative_point_shrinks_mask(runner, mr_rgb):
    """An exclude point reaches the model rather than being dropped in translation.

    Args:
        runner: The real-model runner.
        mr_rgb: The rendered frame.
    """
    h, w = mr_rgb.shape[:2]
    a = runner.segment("t:0:0:0", mr_rgb, {"points": [[w // 2, h // 2, 1]]}, 0.5, 0)
    b = runner.segment("t:0:0:0", mr_rgb, {
        "points": [[w // 2, h // 2, 1], [w // 2, max(h // 2 - 12, 0), 0]]}, 0.5, 0)
    assert b.mask.sum() != a.mask.sum(), "an exclude point must change the result"


def test_embedding_cache_makes_edits_fast(runner, mr_rgb):
    """The encoder/decoder split pays off: a cached edit beats a cold encode.

    Prints both timings, which is where the numbers in ``docs/architecture.md``
    come from.

    Args:
        runner: The real-model runner.
        mr_rgb: The rendered frame.
    """
    h, w = mr_rgb.shape[:2]
    key = "bench:0:0:0"
    runner.segment(key, mr_rgb, {"points": [[w // 2, h // 2, 1]]}, 0.5, 0)  # warm

    t0 = time.perf_counter()
    for i in range(10):
        runner.segment(key, mr_rgb, {"points": [[w // 2 + i, h // 2, 1]]}, 0.5, 0)
    cached = (time.perf_counter() - t0) / 10

    runner.drop_image("bench")
    t0 = time.perf_counter()
    runner.segment("bench:0:0:0", mr_rgb, {"points": [[w // 2, h // 2, 1]]}, 0.5, 0)
    cold = time.perf_counter() - t0

    print(f"\ncached edit {cached*1000:.1f}ms | cold (encode+decode) {cold*1000:.1f}ms")
    assert cached < cold, "cached edits must beat a cold encode"


def test_box_prompt(runner, mr_rgb):
    """A box prompt on its own produces a mask.

    Args:
        runner: The real-model runner.
        mr_rgb: The rendered frame.
    """
    h, w = mr_rgb.shape[:2]
    res = runner.segment("t:0:0:0", mr_rgb, {"boxes": [[w // 4, h // 4, 3 * w // 4, 3 * h // 4]]}, 0.5, 0)
    assert res.mask.sum() > 0
