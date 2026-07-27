"""
sam3_adapter.py
===============
Make facebookresearch/sam3 look like the predictor the rest of this repo
already speaks.

Why an adapter at all
---------------------
Every backend so far was reached by adding a branch to
scripts/refine_box_iou_grad.py::refine_box_by_iou_grad (SAM1, SAM-HQ, then
SAM2.1/SAM-HQ2 -- see its docstring). SAM3 is a bigger jump: it is a
concept-segmentation model (text / exemplar prompts, DETR-style decoder)
whose public entry point is a processor, not SamPredictor. Adding a fourth
branch to every backend-specific site (refine_box_by_iou_grad,
_predict_single_box, _predict_boxes, best_of_n_multimask,
_predict_smoothed_box, _predict_all_heads, animate_grad_refine._mask_for)
would mean seven more code paths to keep in sync.

Instead this module presents SAM3 through the SAM2ImagePredictor-shaped
surface those call sites already handle, so NOT ONE of them changes:

    predictor.predict(point_coords=, point_labels=, box=, multimask_output=,
                      return_logits=)      -> (masks, scores, low_res) numpy
    predictor.set_image(rgb_uint8_hwc)
    predictor._features["image_embed"]     -> (1, C, h, w)
    predictor._features["high_res_feats"]  -> list of (1, C, h, w)
    predictor._orig_hw                     -> [(H, W)]        (get_original_size)
    predictor._transforms                  -> .resolution, .transform_boxes,
                                              .postprocess_masks
    predictor.mask_threshold
    predictor.model.sam_prompt_encoder / .sam_mask_decoder
    NO .predict_torch  -> refine_box_by_iou_grad takes its is_sam2 branch
    NO .interm_features -> ... and not its is_samhq branch

predict() is deliberately implemented on top of the SAME prompt-encoder ->
mask-decoder call the gradient path uses, rather than delegating to SAM3's
own predict(). Two reasons: it removes any dependency on SAM3's public
predict() signature, and -- more importantly -- it guarantees the undefended
/ best_of_n baselines and the gradient trajectory are measured through one
identical code path, which is the whole point of the comparison.

What is UNVERIFIED here
-----------------------
SAM3's internals were not available when this was written. Three things are
discovered at runtime rather than hardcoded, and every one of them is
reported by scripts/probe_sam3_api.py -- run that FIRST and reconcile:

  1. WHICH submodules are the prompt encoder / mask decoder  (probe Q3).
     Discovered by class name + required-attribute duck typing below.
  2. HOW the image embedding is obtained          (probe Q2 "cached state").
     Harvested from SAM3's own predictor if it caches a SAM2-style dict,
     else by calling a discovered image-encoder entry point.
  3. The input RESOLUTION and pixel NORMALISATION (probe Q3 "public
     non-module attributes": image_size / img_size / pixel_mean...).

If discovery fails it raises with the exact probe section to look at -- it
never silently guesses, because a silently wrong coordinate frame produces
plausible-looking IoU numbers that are entirely meaningless (the same class
of bug as the torch-1.13.1-on-H100 trap in session_handoff.txt section 2).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

# ImageNet statistics -- SAM1/SAM2 both normalise with these. VERIFY against
# the probe's dump of the model's pixel_mean/pixel_std (or its image
# processor); a wrong normalisation degrades masks subtly rather than loudly.
_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_STD = (0.229, 0.224, 0.225)

# attribute names to try, in order, when hunting for a scalar on the model
_RESOLUTION_ATTRS = ("image_size", "img_size", "input_size", "image_resolution")
_PROMPT_ENCODER_ATTRS = ("sam_prompt_encoder", "prompt_encoder")
_MASK_DECODER_ATTRS = ("sam_mask_decoder", "mask_decoder")


# ---------------------------------------------------------------------------
# building SAM3, and reaching its OWN SAM2-shaped predictor
#
# Confirmed on sam3 0.1.4 by scripts/probe_sam3_api.py (results/
# sam3_api_probe.txt), reading the installed source -- not from memory:
#
#   sam3.model.sam1_task_predictor.SAM3InteractiveImagePredictor is a port of
#   SAM2ImagePredictor. Its set_image() sets EXACTLY the state this repo's
#   SAM2 code path reads:
#       self._orig_hw   = [image.shape[:2]]
#       self._features  = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
#   and its predict() has this repo's exact signature
#       (point_coords, point_labels, box, mask_input, multimask_output,
#        return_logits, normalize_coords) -> 3 numpy arrays.
#   Sam3Processor.set_image() dereferences
#       model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0
#   which is what proves the inner model carries the SAM2 attribute names.
#
# So for SAM3 the SAM3ImagePredictor class further down is the FALLBACK; the
# preferred path is SAM3's own predictor, which needs no guessed
# preprocessing, resolution or normalisation at all.
# ---------------------------------------------------------------------------

_BPE_GLOB = "bpe_simple_vocab*.txt.gz"

# The interactive predictor hangs off the image model under one of these.
_INTERACTIVE_PREDICTOR_ATTRS = (
    "inst_interactive_predictor", "inst_predictor", "interactive_predictor",
)


def find_bpe_path() -> str:
    """Locate SAM3's BPE vocabulary.

    sam3 0.1.4's build_sam3_image_model() defaults bpe_path to
        os.path.join(os.path.dirname(__file__), "..", "assets", <bpe>)
    i.e. dirname(sam3/model_builder.py)/../assets/ -- which is correct in a
    repo checkout (repo/sam3/ + repo/assets/) but escapes the INSTALLED
    package, resolving to site-packages/assets/ and raising FileNotFoundError.
    Pass bpe_path explicitly to work around it.
    """
    import sam3

    root = Path(sam3.__file__).resolve().parent
    for base in (root, root.parent):
        hits = sorted(base.glob(f"assets/{_BPE_GLOB}")) or sorted(base.glob(_BPE_GLOB))
        if hits:
            return str(hits[0])
    hits = sorted(root.rglob(_BPE_GLOB))
    if hits:
        return str(hits[0])
    raise FileNotFoundError(
        f"SAM3: no {_BPE_GLOB} found under {root} or {root.parent}. sam3 "
        f"{getattr(sam3, '__version__', '?')} looks for it at "
        f"<package>/../assets/, which does not exist for an installed wheel "
        f"-- and here it did not ship inside the package either.\n"
        f"Fetch it from the source repo and pass the path through:\n"
        f"    curl -L -o /tmp/bpe_simple_vocab_16e6.txt.gz \\\n"
        f"      https://raw.githubusercontent.com/facebookresearch/sam3/main/"
        f"assets/bpe_simple_vocab_16e6.txt.gz\n"
        f"then set the BREPS_SAM3_BPE_PATH environment variable to it."
    )


def build_sam3_model(checkpoint: str | None = None, device="cuda",
                     enable_inst_interactivity: bool = True, **extra):
    """Call sam3.model_builder.build_sam3_image_model with the two settings
    this repo needs.

    enable_inst_interactivity=True is REQUIRED and is NOT the default: it is
    what makes the builder construct
        SAM3InteractiveImagePredictor(build_tracker(...))
    and attach it to the model. Without it SAM3 exposes only the concept
    (text / exemplar) path and there is no box-promptable predictor at all.

    checkpoint=None is fine -- the builder's load_from_HF default pulls the
    gated checkpoint from Hugging Face (needs `hf auth login`).
    """
    import os

    from sam3.model_builder import build_sam3_image_model

    bpe_path = os.environ.get("BREPS_SAM3_BPE_PATH") or find_bpe_path()
    kwargs = dict(bpe_path=bpe_path, device=str(device),
                  enable_inst_interactivity=enable_inst_interactivity, **extra)
    if checkpoint:
        kwargs["checkpoint_path"] = checkpoint
    return build_sam3_image_model(**kwargs)


def get_interactive_predictor(model):
    """SAM3's own SAM2-shaped predictor, or None if the model was built
    without enable_inst_interactivity."""
    for attr in _INTERACTIVE_PREDICTOR_ATTRS:
        pred = getattr(model, attr, None)
        if pred is not None and hasattr(pred, "set_image") and hasattr(pred, "predict"):
            return pred
    return None


def adopt_native_predictor(predictor):
    """Make SAM3's own predictor satisfy this repo's contract, in place.

    Only fills gaps -- it never overrides anything SAM3 already provides.
    Raises (rather than papering over) when something load-bearing is
    missing, naming the probe section that shows the truth.
    """
    problems: list[str] = []

    # Branch selection in refine_box_by_iou_grad is by ABSENCE of these.
    for forbidden, branch in (("predict_torch", "SAM1"), ("interm_features", "SAM-HQ")):
        if hasattr(predictor, forbidden):
            problems.append(
                f"predictor has .{forbidden}, which would make "
                f"refine_box_by_iou_grad take its {branch} branch instead of "
                f"the SAM2-style one SAM3 needs"
            )

    tr = getattr(predictor, "_transforms", None)
    if tr is None:
        problems.append("predictor has no ._transforms")
    else:
        for attr in ("resolution", "transform_boxes", "postprocess_masks"):
            if not hasattr(tr, attr):
                problems.append(f"predictor._transforms has no .{attr}")

    model = getattr(predictor, "model", None)
    if model is None:
        problems.append("predictor has no .model")
    else:
        # SAM2 spelling is what refine_box_by_iou_grad reads; alias if needed.
        if not hasattr(model, "sam_prompt_encoder"):
            _, pe = _named_attr(model, _PROMPT_ENCODER_ATTRS)
            if pe is None:
                _, pe = _find_module(model, "promptencoder", required=("get_dense_pe",))
            if pe is None:
                problems.append("predictor.model has no prompt encoder")
            else:
                model.sam_prompt_encoder = pe
        if not hasattr(model, "sam_mask_decoder"):
            _, md = _named_attr(model, _MASK_DECODER_ATTRS)
            if md is None:
                _, md = _find_module(model, "maskdecoder")
            if md is None:
                problems.append("predictor.model has no mask decoder")
            else:
                model.sam_mask_decoder = md

    if not hasattr(predictor, "mask_threshold"):
        # SAM2ImagePredictor keeps it on the predictor and defaults to 0.0
        # (masks are logits). Only set it if SAM3 dropped the attribute.
        predictor.mask_threshold = 0.0

    if problems:
        raise NotImplementedError(
            "SAM3's own predictor does not satisfy this repo's contract:\n  - "
            + "\n  - ".join(problems)
            + "\n\nRun `python scripts/probe_sam3_api.py --checkpoint_path <ckpt>` "
            "and read 'Q3 SAM2-branch compatibility checklist'. If SAM3 has "
            "genuinely diverged, fall back to wrapping it in "
            "heatmaps.sam3_adapter.SAM3ImagePredictor, which synthesises the "
            "missing pieces."
        )
    return predictor


# ---------------------------------------------------------------------------
# component discovery
# ---------------------------------------------------------------------------

def _named_attr(obj: Any, names: tuple[str, ...]):
    """First present attribute out of *names*, or (None, None)."""
    for n in names:
        val = getattr(obj, n, None)
        if val is not None:
            return n, val
    return None, None


def _find_module(model: torch.nn.Module, class_fragment: str,
                 required: tuple[str, ...] = ()) -> tuple[Optional[str], Any]:
    """Deepest-first search for a submodule whose class name contains
    *class_fragment* (case-insensitive) and that has every attribute in
    *required*. Deepest-first because SAM3 nests its SAM-style tracker under
    a wrapper that may re-expose the same names."""
    best: tuple[Optional[str], Any] = (None, None)
    for name, child in model.named_modules():
        if class_fragment not in type(child).__name__.lower():
            continue
        if any(not hasattr(child, r) for r in required):
            continue
        if best[0] is None or name.count(".") > best[0].count("."):
            best = (name, child)
    return best


def discover_components(model: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Locate SAM3's (prompt_encoder, mask_decoder).

    Tries the SAM2 / SAM1 attribute names first (SAM3's tracker is
    SAM2-derived, so it may well carry them verbatim), then falls back to a
    duck-typed class-name search over the whole module tree.
    """
    _, pe = _named_attr(model, _PROMPT_ENCODER_ATTRS)
    _, md = _named_attr(model, _MASK_DECODER_ATTRS)

    if pe is None:
        _, pe = _find_module(model, "promptencoder", required=("get_dense_pe",))
    if md is None:
        _, md = _find_module(model, "maskdecoder")

    missing = [n for n, v in (("prompt encoder", pe), ("mask decoder", md)) if v is None]
    if missing:
        raise NotImplementedError(
            f"SAM3 adapter: could not locate the {' and '.join(missing)} inside "
            f"{type(model).__name__}. Run\n"
            f"    python scripts/probe_sam3_api.py --checkpoint_path <ckpt>\n"
            f"and read section 'Q3 model structure' -- it prints the full module "
            f"tree and every candidate class. Then either pass the components "
            f"explicitly to SAM3ImagePredictor(prompt_encoder=..., mask_decoder=...) "
            f"or extend the name lists at the top of heatmaps/sam3_adapter.py.\n"
            f"If the tree has no SAM-style prompt/mask pair at all, SAM3 prompts "
            f"boxes through its DETR decoder instead and the gradient defence "
            f"needs a genuinely different objective -- see probe section Q4."
        )
    if not hasattr(pe, "get_dense_pe"):
        raise NotImplementedError(
            f"SAM3 adapter: prompt encoder {type(pe).__name__} has no "
            f"get_dense_pe(); refine_box_by_iou_grad needs the dense positional "
            f"encoding to call the mask decoder. See probe section Q3."
        )
    return pe, md


def _decoder_call_kwargs(mask_decoder: torch.nn.Module) -> dict:
    """Extra kwargs this particular decoder's forward() demands, beyond the
    five every SAM decoder takes. Mirrors the per-backend special-casing
    already in refine_box_by_iou_grad (SAM2's repeat_image/high_res_features,
    SAM-HQ's hq_token_only) but derived from the signature instead of a
    hardcoded branch, since SAM3's is unknown."""
    try:
        params = inspect.signature(type(mask_decoder).forward).parameters
    except (TypeError, ValueError):
        return {}
    extra = {}
    if "repeat_image" in params:
        extra["repeat_image"] = False
    if "hq_token_only" in params:
        extra["hq_token_only"] = False
    return extra


# ---------------------------------------------------------------------------
# transforms: SAM2's convention (plain resize to a square, no letterbox pad)
# ---------------------------------------------------------------------------

class SAM3Transforms:
    """Reimplementation of sam2.utils.transforms.SAM2Transforms' three methods
    this repo uses. Written out rather than imported because the `sam2`
    package is not installed in the `sam3` conda env (and SAM3 may use a
    different input resolution anyway).

    Coordinate convention, identical to SAM2's: boxes arrive in ORIGINAL
    pixels, are divided by (W, H) and multiplied by `resolution`. This repo's
    own SAM1-style 1024 frame is converted to/from original pixels by
    heatmaps.defend_critical_shifts.{boxes_to_original,original_to_1024} --
    refine_box_by_iou_grad already does exactly that around the is_sam2
    branch, so nothing here has to know about it.
    """

    def __init__(self, resolution: int, mask_threshold: float = 0.0,
                 mean=_DEFAULT_MEAN, std=_DEFAULT_STD):
        self.resolution = int(resolution)
        self.mask_threshold = float(mask_threshold)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def preprocess(self, image_rgb: np.ndarray, device) -> torch.Tensor:
        """(H, W, 3) uint8 RGB -> (1, 3, resolution, resolution) normalised."""
        t = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float() / 255.0
        t = F.interpolate(t.unsqueeze(0), size=(self.resolution, self.resolution),
                          mode="bilinear", align_corners=False)
        t = (t - self.mean.to(t)) / self.std.to(t)
        return t.to(device)

    def transform_coords(self, coords: torch.Tensor, normalize: bool = False,
                         orig_hw=None) -> torch.Tensor:
        if normalize:
            h, w = orig_hw
            coords = torch.stack([coords[..., 0] / w, coords[..., 1] / h], dim=-1)
        return coords * self.resolution

    def transform_boxes(self, boxes: torch.Tensor, normalize: bool = False,
                        orig_hw=None) -> torch.Tensor:
        return self.transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw)

    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """Low-res mask logits -> original image resolution."""
        return F.interpolate(masks.float(), size=tuple(orig_hw),
                             mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# feature harvesting
# ---------------------------------------------------------------------------

def _as_feature_dict(raw: Any) -> Optional[dict]:
    """Normalise whatever an encoder returned into
    {"image_embed": (1,C,h,w), "high_res_feats": [(1,C,h,w), ...]}."""
    if isinstance(raw, dict):
        embed = raw.get("image_embed", raw.get("vision_features",
                        raw.get("image_embeddings", raw.get("last_hidden_state"))))
        if embed is None:
            return None
        hires = raw.get("high_res_feats", raw.get("backbone_fpn", []))
        if hires and not isinstance(hires, (list, tuple)):
            hires = [hires]
        # SAM2's backbone_fpn holds ALL pyramid levels, the last being the one
        # that becomes image_embed; the decoder wants only the higher-res ones.
        if "high_res_feats" not in raw and len(hires) > 1:
            hires = list(hires)[:-1]
        return {"image_embed": embed, "high_res_feats": list(hires)}
    if torch.is_tensor(raw):
        return {"image_embed": raw, "high_res_feats": []}
    return None


def harvest_features(source: Any) -> Optional[dict]:
    """Pull a SAM2-style feature dict off an object that has just encoded an
    image (SAM3's own predictor, or a raw encoder output)."""
    feats = _as_feature_dict(source)
    if feats is not None:
        return feats
    for attr in ("_features", "features", "image_embed", "_image_embed"):
        got = getattr(source, attr, None)
        if got is None:
            continue
        feats = _as_feature_dict(got)
        if feats is not None:
            return feats
    return None


# ---------------------------------------------------------------------------
# the predictor
# ---------------------------------------------------------------------------

class SAM3ImagePredictor:
    """SAM2ImagePredictor-shaped view over a SAM3 image model.

    Parameters
    ----------
    model : the object returned by SAM3's image-model builder.
    native : SAM3's own predictor/processor, if one exists. When given, its
        set_image() does the image encoding and we harvest the cached
        features from it -- far safer than re-deriving SAM3's preprocessing.
        When absent, the model's own encoder entry point is called directly.
    resolution / mask_threshold / prompt_encoder / mask_decoder : escape
        hatches to override discovery once the probe has told you the truth.
    """

    # deliberately NOT defining predict_torch / interm_features: their
    # absence is how refine_box_by_iou_grad selects its SAM2-style branch.

    def __init__(self, model, native=None, resolution: int | None = None,
                 mask_threshold: float = 0.0, prompt_encoder=None,
                 mask_decoder=None, device=None):
        self.model = model
        self._native = native
        self.mask_threshold = float(mask_threshold)

        if prompt_encoder is None or mask_decoder is None:
            found_pe, found_md = discover_components(model)
            prompt_encoder = prompt_encoder or found_pe
            mask_decoder = mask_decoder or found_md
        self._prompt_encoder = prompt_encoder
        self._mask_decoder = mask_decoder
        self._decoder_extra = _decoder_call_kwargs(mask_decoder)

        # refine_box_by_iou_grad reads these OFF THE MODEL (model.sam_prompt_encoder
        # / model.sam_mask_decoder, the SAM2 spelling). Alias them if SAM3 uses
        # other names -- assigning an existing nn.Module under a second name
        # shares parameters, it does not copy them (nn.Module.parameters()
        # de-duplicates by default).
        if isinstance(model, torch.nn.Module):
            if not hasattr(model, "sam_prompt_encoder"):
                model.sam_prompt_encoder = prompt_encoder
            if not hasattr(model, "sam_mask_decoder"):
                model.sam_mask_decoder = mask_decoder

        if resolution is None:
            _, resolution = _named_attr(model, _RESOLUTION_ATTRS)
            if isinstance(resolution, (tuple, list)):
                resolution = resolution[0]
        if resolution is None:
            raise NotImplementedError(
                "SAM3 adapter: could not determine the model's input resolution "
                f"(tried {_RESOLUTION_ATTRS} on {type(model).__name__}). Run "
                "scripts/probe_sam3_api.py and read 'Q3 -> public non-module "
                "attributes', then pass resolution=<N> explicitly. Getting this "
                "wrong silently misplaces every box prompt."
            )
        self._transforms = SAM3Transforms(int(resolution), mask_threshold)

        self.device = device or next(model.parameters()).device
        self._features: dict | None = None
        self._orig_hw: list[tuple[int, int]] = []
        self._is_image_set = False

    # -- image encoding ----------------------------------------------------

    def set_image(self, image_rgb: np.ndarray) -> None:
        """image_rgb: (H, W, 3) uint8 RGB, exactly what
        heatmaps.defend_critical_shifts._prepare_image passes."""
        h, w = image_rgb.shape[:2]
        self._orig_hw = [(h, w)]

        feats = None
        if self._native is not None:
            self._native.set_image(image_rgb)
            feats = harvest_features(self._native)

        if feats is None:
            feats = self._encode_directly(image_rgb)

        if feats is None:
            raise NotImplementedError(
                "SAM3 adapter: encoded the image but could not find the image "
                "embedding afterwards. Run scripts/probe_sam3_api.py and read "
                "'Q2 -> cached state after set_image' -- it lists exactly what "
                "SAM3 caches and under which names; add those names to "
                "heatmaps.sam3_adapter._as_feature_dict."
            )

        embed = feats["image_embed"]
        if embed.dim() == 3:
            embed = embed.unsqueeze(0)
        feats["image_embed"] = embed
        feats["high_res_feats"] = [
            f.unsqueeze(0) if f.dim() == 3 else f for f in feats["high_res_feats"]
        ]
        self._features = feats
        self._is_image_set = True

    def _encode_directly(self, image_rgb: np.ndarray) -> Optional[dict]:
        """No native predictor: call the model's own encoder entry point."""
        batch = self._transforms.preprocess(image_rgb, self.device)
        for attr in ("forward_image", "encode_image", "get_image_embedding",
                     "image_encoder", "vision_encoder", "encoder"):
            fn = getattr(self.model, attr, None)
            if fn is None:
                continue
            with torch.no_grad():
                out = fn(batch)
            feats = _as_feature_dict(out)
            if feats is not None:
                return feats
        return None

    # -- prediction --------------------------------------------------------

    def _decode(self, box_model_frame: torch.Tensor, multimask_output: bool):
        """box in the model's own input frame -> (low_res_logits, iou_pred).

        The single place the adapter touches SAM3's decoder; the gradient
        path in refine_box_by_iou_grad reproduces this same call (it needs
        the graph, so it cannot go through predict())."""
        # SAM2 convention: a box is two corner points labelled 2 / 3 rather
        # than a boxes= kwarg. Fall back to boxes= if the encoder takes it.
        pe_params = inspect.signature(type(self._prompt_encoder).forward).parameters
        if "boxes" in pe_params and "points" not in pe_params:
            sparse, dense = self._prompt_encoder(boxes=box_model_frame[None, :], masks=None)
        else:
            coords = box_model_frame.reshape(1, 2, 2)
            labels = torch.tensor([[2, 3]], dtype=torch.int, device=box_model_frame.device)
            sparse, dense = self._prompt_encoder(points=(coords, labels), boxes=None, masks=None)

        kwargs = dict(
            image_embeddings=self._features["image_embed"],
            image_pe=self._prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask_output,
            **self._decoder_extra,
        )
        if self._features["high_res_feats"]:
            try:
                out = self._mask_decoder(**kwargs,
                                         high_res_features=self._features["high_res_feats"])
            except TypeError:
                out = self._mask_decoder(**kwargs)
        else:
            out = self._mask_decoder(**kwargs)
        return out[0], out[1]

    def predict(self, point_coords=None, point_labels=None, box=None,
                mask_input=None, multimask_output: bool = True,
                return_logits: bool = False, normalize_coords: bool = True):
        """Signature-compatible with SAM2ImagePredictor.predict for the subset
        this repo uses (box prompts only). Returns numpy
        (masks (K,H,W), scores (K,), low_res_logits (K,h,w))."""
        if not self._is_image_set:
            raise RuntimeError("call set_image() before predict()")
        if box is None:
            raise NotImplementedError(
                "SAM3 adapter: only box prompts are implemented -- this repo's "
                "defences prompt SAM with boxes only (see "
                "heatmaps/defend_critical_shifts.py)."
            )
        if point_coords is not None or point_labels is not None or mask_input is not None:
            raise NotImplementedError("SAM3 adapter: point/mask prompts not implemented")

        orig_hw = self._orig_hw[0]
        box_t = torch.as_tensor(np.asarray(box, dtype=np.float32).reshape(1, 4),
                                dtype=torch.float32, device=self.device)
        box_model = self._transforms.transform_boxes(
            box_t, normalize=True, orig_hw=orig_hw).reshape(4)

        with torch.no_grad():
            low_res, iou_pred = self._decode(box_model, multimask_output)
            full = self._transforms.postprocess_masks(low_res, orig_hw)

        masks = full[0]                                   # (K, H, W) logits
        scores = iou_pred[0]                              # (K,)
        if not return_logits:
            masks = masks > self.mask_threshold
        return (masks.cpu().numpy(),
                scores.float().cpu().numpy(),
                low_res[0].float().cpu().numpy())

    # get_original_size() prefers .original_size, so expose only _orig_hw
    # (SAM2's spelling) and let it take the SAM2 path.
    def __repr__(self):
        return (f"SAM3ImagePredictor(model={type(self.model).__name__}, "
                f"resolution={self._transforms.resolution}, "
                f"native={type(self._native).__name__ if self._native else None})")
