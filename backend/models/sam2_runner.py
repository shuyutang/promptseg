from __future__ import annotations
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch

from config import settings


@dataclass
class SegmentationResult:
    mask: np.ndarray  # HxW uint8, 0/1
    score: float      # model's predicted IoU for the chosen candidate
    num_candidates: int


class _EmbeddingCache:
    """LRU over per-frame image embeddings, which are the expensive artifact.

    SAM2's encoder returns a list of multi-scale feature maps (~17MB fp32 for a
    1024x1024 input), so this is capped rather than unbounded.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, capacity)
        self._d: OrderedDict[str, object] = OrderedDict()

    def get(self, key: str):
        if key not in self._d:
            return None
        self._d.move_to_end(key)
        return self._d[key]

    def put(self, key: str, value) -> None:
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.capacity:
            self._d.popitem(last=False)

    def drop_prefix(self, prefix: str) -> None:
        for k in [k for k in self._d if k.startswith(prefix)]:
            del self._d[k]

    def __len__(self) -> int:
        return len(self._d)


class _StubModel:
    """Prompt-responsive stand-in used when SAM2_STUB=1.

    Unlike a fixed blob, this actually reacts to the prompts, so the annotation,
    labelling and export paths can be tested end-to-end without weights or a GPU.
    """

    def segment(self, rgb: np.ndarray, prompts: dict, threshold: float, mask_index: int):
        h, w = rgb.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        mask = np.zeros((h, w), dtype=bool)

        radius = max(4.0, 0.12 * min(h, w) * (1.0 + 0.5 * mask_index))
        for x, y, lab in prompts.get("points", []):
            blob = ((xx - x) ** 2 + (yy - y) ** 2) <= radius**2
            if lab == 1:
                mask |= blob
            else:
                mask &= ~blob
        for x1, y1, x2, y2 in prompts.get("boxes", []):
            box = (xx >= min(x1, x2)) & (xx <= max(x1, x2)) & (yy >= min(y1, y2)) & (yy <= max(y1, y2))
            mask |= box
        return SegmentationResult(mask.astype(np.uint8), 1.0, 3)


class Sam2Runner:
    """Wraps SAM2 with an encoder/decoder split.

    The image encoder runs once per (image, frame) and is cached; each prompt
    edit only re-runs the lightweight mask decoder. Measured on an RTX 4090:
    ~160ms encode vs ~3ms decode.
    """

    def __init__(self, model_id: str | None = None, device: str | None = None,
                 stub: bool | None = None) -> None:
        self.model_id = model_id or settings.model_id
        self.stub = settings.stub if stub is None else stub
        self._lock = threading.Lock()
        self._cache = _EmbeddingCache(settings.max_embeddings)

        if self.stub:
            self.device = torch.device("cpu")
            self._impl = _StubModel()
            self.model = None
            self.processor = None
            return

        want = device or settings.device
        self.device = torch.device("cuda" if want == "cuda" and torch.cuda.is_available() else "cpu")

        from transformers import Sam2Model, Sam2Processor

        self.processor = Sam2Processor.from_pretrained(self.model_id)
        self.model = Sam2Model.from_pretrained(self.model_id).to(self.device).eval()
        self._impl = None

    # ---- embeddings -------------------------------------------------

    @torch.inference_mode()
    def _embed(self, rgb: np.ndarray):
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        return self.model.get_image_embeddings(inputs["pixel_values"])

    def embeddings_for(self, key: str, rgb: np.ndarray):
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        emb = self._embed(rgb)
        self._cache.put(key, emb)
        return emb

    def drop_image(self, image_id: str) -> None:
        self._cache.drop_prefix(f"{image_id}:")

    # ---- inference --------------------------------------------------

    def segment(self, key: str, rgb: np.ndarray, prompts: dict,
                threshold: float = 0.5, mask_index: int = 0) -> SegmentationResult:
        """rgb: HxWx3 uint8. prompts: {"points": [[x,y,label]], "boxes": [[x1,y1,x2,y2]]}.

        Candidates are ranked by the model's predicted IoU, so mask_index=0 is
        always the best-scoring one.
        """
        points = prompts.get("points") or []
        boxes = prompts.get("boxes") or []
        if not points and not boxes:
            raise ValueError("At least one point or box prompt is required.")

        if self.stub:
            return self._impl.segment(rgb, {"points": points, "boxes": boxes}, threshold, mask_index)

        with self._lock:
            return self._segment_locked(key, rgb, points, boxes, threshold, mask_index)

    @torch.inference_mode()
    def _segment_locked(self, key, rgb, points, boxes, threshold, mask_index) -> SegmentationResult:
        h, w = rgb.shape[:2]
        emb = self.embeddings_for(key, rgb)

        kwargs = {}
        if points:
            # (batch, object, point, 2) / (batch, object, point)
            kwargs["input_points"] = [[[[int(x), int(y)] for x, y, _ in points]]]
            kwargs["input_labels"] = [[[1 if int(l) == 1 else 0 for _, _, l in points]]]
        if boxes:
            kwargs["input_boxes"] = [[[int(v) for v in b] for b in boxes]]

        enc = self.processor(images=rgb, return_tensors="pt", **kwargs).to(self.device)
        call = {k: enc[k] for k in ("input_points", "input_labels", "input_boxes") if k in enc}

        out = self.model(image_embeddings=emb, multimask_output=True, **call)

        # Upscale logits to native resolution, then apply the caller's threshold.
        logits = self.processor.post_process_masks(
            out.pred_masks, enc["original_sizes"], binarize=False
        )[0]
        probs = torch.sigmoid(logits.float())
        if probs.dim() == 4:      # (obj, candidate, H, W) -> collapse object dim
            probs = probs[0]
        scores = out.iou_scores.flatten().float()

        n = probs.shape[0]
        order = torch.argsort(scores[:n], descending=True)
        idx = int(order[min(max(mask_index, 0), n - 1)])

        mask = (probs[idx] >= threshold).to(torch.uint8).cpu().numpy()
        if mask.shape != (h, w):  # defensive; post_process should already match
            mask = mask[:h, :w]
        return SegmentationResult(mask, float(scores[idx]), int(n))
