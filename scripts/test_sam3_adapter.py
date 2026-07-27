"""
test_sam3_adapter.py
====================
Offline contract test for heatmaps/sam3_adapter.py -- no GPU, no checkpoint,
no `sam3` package needed. Runs in seconds on CPU in any env.

It builds a FAKE SAM3-shaped model and drives the REAL consumer code through
it (refine_box_by_iou_grad, _predict_single_box, _predict_boxes,
best_of_n_multimask, stability_score). That is the point: the adapter's job
is to satisfy a contract those functions impose, and this checks the contract
directly instead of re-asserting the adapter's own internals.

The fake model deliberately uses NON-SAM2 names (`tracker.prompt_enc`,
`tracker.dec`) and a non-1024 input resolution (1008), so that component
DISCOVERY -- not a lucky attribute-name match -- is what has to work.

What this can and cannot catch
------------------------------
CAN : broken discovery, wrong feature-dict normalisation, wrong return
      shapes/dtypes from predict(), a coordinate round-trip that does not
      land back in this repo's SAM1-1024 frame, a severed autograd graph.
CANNOT: whether the real SAM3 matches the fake's shape at all. That is what
      scripts/probe_sam3_api.py is for -- run it on the server first.

Usage:
    python scripts/test_sam3_adapter.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling import


def _install_stubs() -> None:
    """Both of these ARE installed in the sam3 env on the server (see
    scripts/setup_repo.sh::setup_sam3_env); stub them only so this test also
    runs on a bare checkout, since neither is exercised by what it tests."""
    if "loguru" not in sys.modules:
        try:
            import loguru  # noqa: F401
        except ImportError:
            mod = types.ModuleType("loguru")

            class _Silent:
                def __getattr__(self, _):
                    return lambda *a, **k: None
            mod.logger = _Silent()
            sys.modules["loguru"] = mod

    try:
        import segment_anything  # noqa: F401
    except ImportError:
        # Verbatim reimplementation of SAM1's
        # ResizeLongestSide.get_preprocess_shape -- the only thing
        # heatmaps/defend_critical_shifts.py imports from that package.
        sa = types.ModuleType("segment_anything")
        utils = types.ModuleType("segment_anything.utils")
        transforms = types.ModuleType("segment_anything.utils.transforms")

        class ResizeLongestSide:
            @staticmethod
            def get_preprocess_shape(oldh, oldw, long_side_length):
                scale = long_side_length * 1.0 / max(oldh, oldw)
                return (int(oldh * scale + 0.5), int(oldw * scale + 0.5))

        transforms.ResizeLongestSide = ResizeLongestSide
        utils.transforms = transforms
        sa.utils = utils
        sys.modules["segment_anything"] = sa
        sys.modules["segment_anything.utils"] = utils
        sys.modules["segment_anything.utils.transforms"] = transforms


# ---------------------------------------------------------------------------
# the fake SAM3
# ---------------------------------------------------------------------------

EMB = 32
RES = 1008          # deliberately NOT 1024
LOW = 64


class Sam3PromptEncoder(nn.Module):
    """Differentiable box -> sparse embedding, like SAM's
    PositionEmbeddingRandom (normalise -> projection -> sin/cos)."""

    def __init__(self):
        super().__init__()
        self.dense = nn.Parameter(torch.randn(1, EMB, LOW // 4, LOW // 4),
                                  requires_grad=False)
        self.proj = nn.Parameter(torch.ones(4, EMB) * 3.0, requires_grad=False)

    def get_dense_pe(self):
        return self.dense

    def forward(self, points=None, boxes=None, masks=None):
        coords, _labels = points
        x = coords.reshape(1, -1) / RES
        sparse = torch.stack([torch.sin(x @ self.proj), torch.cos(x @ self.proj)], dim=1)
        return sparse, self.dense.expand(1, EMB, LOW // 4, LOW // 4)


class Sam3MaskDecoder(nn.Module):
    """SAM2-style signature: extra repeat_image / high_res_features kwargs and
    a 4-tuple return (the adapter must match these BY SIGNATURE)."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(EMB, 4)

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                dense_prompt_embeddings, multimask_output, repeat_image=False,
                high_res_features=None):
        assert high_res_features is not None and len(high_res_features) == 2, \
            f"expected 2 high-res levels, got {high_res_features}"
        iou_all = torch.sigmoid(self.head(sparse_prompt_embeddings.mean(dim=1)))
        masks = image_embeddings.mean(1, keepdim=True).expand(1, 4, LOW, LOW).clone()
        masks = masks * iou_all[..., None, None]
        if multimask_output:
            return masks[:, 1:], iou_all[:, 1:], None, None
        return masks[:, :1], iou_all[:, :1], None, None


class Sam3Tracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_enc = Sam3PromptEncoder()
        self.dec = Sam3MaskDecoder()


class Sam3ImageModel(nn.Module):
    image_size = RES

    def __init__(self):
        super().__init__()
        self.tracker = Sam3Tracker()

    def forward_image(self, batch):
        b = batch.shape[0]
        # SAM2-style backbone output: the LAST fpn level is the image embed,
        # so the adapter must drop it from high_res_feats.
        return {"vision_features": torch.randn(b, EMB, LOW, LOW),
                "backbone_fpn": [torch.randn(b, EMB, LOW * 4, LOW * 4),
                                 torch.randn(b, EMB, LOW * 2, LOW * 2),
                                 torch.randn(b, EMB, LOW, LOW)]}


# ---------------------------------------------------------------------------

def main() -> int:
    _install_stubs()
    from heatmaps.comp_hw_smoothed import get_original_size
    from heatmaps.defend_critical_shifts import (
        _predict_single_box, boxes_to_original, original_to_1024,
    )
    from heatmaps.sam3_adapter import SAM3ImagePredictor
    from refine_box_iou_grad import (
        _predict_boxes, best_of_n_multimask, refine_box_by_iou_grad, stability_score,
    )

    torch.manual_seed(0)
    device = torch.device("cpu")
    model = Sam3ImageModel().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("== discovery ==")
    predictor = SAM3ImagePredictor(model, native=None, device=device)
    print(f"  {predictor}")
    print(f"  prompt encoder = {type(predictor._prompt_encoder).__name__}")
    print(f"  mask decoder   = {type(predictor._mask_decoder).__name__} "
          f"extra kwargs {predictor._decoder_extra}")
    assert isinstance(predictor._prompt_encoder, Sam3PromptEncoder)
    assert isinstance(predictor._mask_decoder, Sam3MaskDecoder)
    assert predictor._decoder_extra == {"repeat_image": False}
    assert predictor._transforms.resolution == RES

    print("\n== backend detection (how refine_box_by_iou_grad branches) ==")
    assert not hasattr(predictor, "predict_torch"), \
        "must NOT expose predict_torch -- that is how is_sam2 is decided"
    assert not hasattr(predictor, "interm_features"), \
        "must NOT expose interm_features -- that would select the SAM-HQ branch"
    assert model.sam_prompt_encoder is predictor._prompt_encoder
    assert model.sam_mask_decoder is predictor._mask_decoder
    ids = [id(p) for p in model.parameters()]
    assert len(ids) == len(set(ids)), "aliasing must not duplicate parameters"
    print("  is_sam2 branch selected, model.sam_* aliases present, params not duplicated")

    print("\n== set_image / feature harvesting ==")
    H, W = 480, 640
    img = np.random.default_rng(0).integers(0, 255, (H, W, 3)).astype(np.uint8)
    predictor.set_image(img)
    assert get_original_size(predictor) == (H, W)
    embed = predictor._features["image_embed"]
    hires = predictor._features["high_res_feats"]
    print(f"  image_embed {tuple(embed.shape)}  high_res {[tuple(f.shape) for f in hires]}")
    assert embed.shape == (1, EMB, LOW, LOW)
    assert len(hires) == 2, "last backbone_fpn level IS image_embed, must be dropped"

    print("\n== predict() surface ==")
    box = np.array([100.0, 80.0, 400.0, 380.0], dtype=np.float32)
    for mm in (False, True):
        m, s, lr = predictor.predict(point_coords=None, point_labels=None, box=box,
                                     multimask_output=mm, return_logits=False)
        print(f"  multimask={mm}: masks {m.shape} {m.dtype}  scores {np.round(s, 3)}")
        assert m.shape == ((3 if mm else 1), H, W) and m.dtype == bool
        assert s.shape == ((3,) if mm else (1,))
        assert lr.shape[0] == (3 if mm else 1)
    m, _, _ = predictor.predict(point_coords=None, point_labels=None, box=box,
                                multimask_output=False, return_logits=True)
    assert m.dtype != bool, "return_logits=True must return logits, not booleans"

    print("\n== coordinate round trip (1024 frame <-> original pixels) ==")
    box_1024 = original_to_1024(box[None].astype(np.float64), (H, W))[0]
    back = original_to_1024(boxes_to_original(box_1024[None], (H, W))[None][0], (H, W))[0]
    print(f"  {np.round(box, 1)} -> 1024 {np.round(box_1024, 1)} -> {np.round(back, 1)}")
    assert np.allclose(back, box_1024, atol=1.0)

    print("\n== INTEGRATION: refine_box_by_iou_grad, unmodified ==")
    gt = torch.zeros(H, W, dtype=torch.bool)
    gt[100:380, 120:420] = True
    final_box, traj = refine_box_by_iou_grad(box_1024, predictor, device,
                                             steps=5, lr=2.0, multimask=True,
                                             gt_tensor=gt)
    for t in traj:
        print(f"  step {t['step']} head {t['head']} pred {t['pred_score']:.4f} "
              f"true {t['true_iou']:.4f} box {np.round(t['box'], 1)}")
    assert len(traj) == 6
    assert all({"step", "head", "pred_score", "true_iou", "box"} <= set(t) for t in traj)
    assert traj[-1]["pred_score"] > traj[0]["pred_score"], \
        "gradient ascent did not increase the predicted IoU"
    assert not np.allclose(traj[0]["box"], traj[-1]["box"]), "the box never moved"
    # the returned box must be back in THIS repo's SAM1-1024 frame, NOT left
    # in SAM3's 1008 model frame -- everything downstream assumes 1024.
    assert final_box.shape == (4,) and 0 <= float(final_box.min())
    assert float(final_box.max()) <= 1100, f"box left in the wrong frame: {final_box}"
    print(f"  final box (1024 frame) {np.round(final_box.numpy(), 1)}")

    print("\n== INTEGRATION: the other backend-specific call sites ==")
    mask = _predict_single_box(torch.tensor(box_1024, dtype=torch.float32),
                               predictor, device, boxes_already_transformed=True)
    assert tuple(mask.shape) == (H, W) and mask.dtype == torch.bool
    print(f"  _predict_single_box     -> {tuple(mask.shape)} {mask.dtype}")

    boxes_t = torch.tensor(np.stack([box_1024, box_1024 * 0.95]), dtype=torch.float32)
    masks, scores = _predict_boxes(boxes_t, predictor, multimask=True)
    assert masks.shape[:2] == (2, 3) and scores.shape == (2, 3)
    print(f"  _predict_boxes          -> {tuple(masks.shape)} {tuple(scores.shape)}")

    bmask, bscore, _bbox, bhead = best_of_n_multimask(
        box_1024, predictor, device, Y=4, sigma=0.05, sigma_center=0.03,
        perturb_mode="size", seed=42)
    assert tuple(bmask.shape) == (H, W)
    print(f"  best_of_n_multimask     -> score {bscore:.4f} head {bhead}")

    stab, _ = stability_score(box_1024, predictor, device, use_mm=True, head=0,
                              M=3, sigma_s=0.04, seed=1)
    assert 0.0 <= stab <= 1.0
    print(f"  stability_score         -> {stab:.4f}")

    print("\nALL CHECKS PASSED")
    print("NOTE: this proves the adapter satisfies the repo's contract, NOT that\n"
          "      the real SAM3 matches this fake. Run scripts/probe_sam3_api.py\n"
          "      on the server and reconcile before trusting any SAM3 number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
