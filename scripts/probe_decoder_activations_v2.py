"""
probe_decoder_activations_v2.py
===============================
Where does a *critical shift* actually break SAM -- with the measurement
artefacts of v1 (`probe_decoder_activations.py`) removed.

A critical-shift case is a pair of spatially adjacent box prompts
(best_box, bad_box) (<=10 px apart, same image) where best_box gives a good
mask and bad_box a broken one (see scripts/find_critical_shifts.py). The image
embedding is identical inside a pair, so ALL divergence is born in the prompt
encoder and travels through the mask decoder.

WHAT IS FIXED RELATIVE TO v1
----------------------------
1. GROUNDING. v1 trusted `best_iou`/`bad_iou` from the JSON, which were
   produced by a different pipeline under bf16 autocast
   (heatmaps/comp_hw_smoothed.py). Here every pair is RE-PREDICTED under the
   exact numerics used for capture, and kept only if the drop reproduces
   (vs GT when --masks_dir is given, otherwise via mask disagreement).
   Non-reproducing pairs are dropped and reported in --out_repro_csv.

2. NOISE FLOOR. A `noise` baseline runs the SAME best_box twice, so every
   curve can be read against the level of pure run-to-run nondeterminism.

3. HONEST CONTROL. The control neighbour is a uniformly random direction in
   R^4 rescaled to EXACTLY the L-inf displacement of best->bad (v1 used
   `rng.integers(-d, d+1, size=4)`, whose expected max-|shift| is ~0.85*d, so
   its control was systematically weaker than the attack). Directions too
   close to the true one are rejected (--ctrl_max_cos), and a control whose
   own mask is broken is discarded rather than averaged in.

4. COMPARABLE METRICS. `rms_l2` (= ||d||_2 / sqrt(numel)) is plotted instead of
   raw L2, whose v1 peak was just the layer's element count. `rel_l2` is
   plotted together with its denominator ||A_best||, so a rise cannot be
   confused with a shrinking baseline.

5. GAIN, NOT LEVEL. A local amplification panel, gain = rel_l2(L)/rel_l2(L-1),
   computed per case along the previous layer OF THE SAME BRANCH. The culprit
   layer is where gain peaks, not where the level peaks.

6. GEOMETRY. ||a-b||^2 = (||a||-||b||)^2 + 2||a|| ||b|| (1-cos) is reported as
   its two parts: cosine distance (representation rotated) and log norm ratio
   (activation blew up / collapsed). v1's single L2 number conflated them.

7. BRANCH SPLIT. Every tensor is tagged token-branch (5 output tokens + box
   tokens) or image-branch (64x64 image tokens / upscaled maps), so "the query
   tokens drifted" is distinguishable from "the feature map was rewritten".
   The tag is derived from the tensor's real shape, which automatically
   handles the swapped q/k in `cross_attn_image_to_token`.

8. STATISTICS. v1 averaged over 235 pairs that are really 47 images x 5
   near-duplicate boxes, and drew mean +/- std bands clipped at 0 on a
   right-skewed quantity. Here: cases are sampled at random (v1 took the first
   N of a list sorted by iou_drop, i.e. the most extreme cases), aggregated
   per image first and then across images, reported as median + IQR on a log
   axis. critical-vs-control is compared PAIRWISE within a case
   (log2 ratio, image-level bootstrap CI + Wilcoxon), not as a ratio of means.

9. CAUSALITY (--patch). Activation patching: re-run bad_box while replacing
   one captured tensor with its best_box value, and measure IoU(patched,
   best). Large ||d|| in a layer does not mean that layer decides the mask;
   this measures whether it does. Patching `prompt_encoder[sparse_embeddings]`
   is a complete cut and must restore IoU ~= 1.0 -- it is the built-in
   self-test of the patching machinery.

LAYER SET
---------
Hooked, in forward execution order: the prompt_encoder root (sparse/dense
embeddings), every USED leaf of the mask decoder, the tuple-returning cut
points `mask_decoder.transformer.layers.N -> (queries, keys)` and
`mask_decoder.transformer -> (queries, keys)`, and the mask_decoder root
(masks, iou_pred). Excluded: `pe_layer` (fires only via get_dense_pe(), box
independent) and `output_hypernetworks_mlps.1..3` (run, but discarded when
multimask_output=False). Plain container modules are excluded as duplicates of
their last child.

Example
-------
    CUDA_VISIBLE_DEVICES=3 python scripts/probe_decoder_activations_v2.py \
        --critical_shifts critical_shifts.json \
        --images_dir /.../FOR_TEST/images \
        --masks_dir  /.../FOR_TEST/masks \
        --checkpoint_path /.../sam_vit_b_01ec64.pth \
        --model_name SAM --model_type vit_b \
        --limit 100 --max_per_image 2 --control --patch --patch_limit 20 \
        --out_dir exp_res/probe_v2
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import cv2
from tqdm import tqdm

from heatmaps.comp_hw_smoothed import load_model, get_original_size
from heatmaps.defend_critical_shifts import _find_file, _predict_single_box, _prepare_image
from segment_anything.utils.transforms import ResizeLongestSide

EPS = 1e-12


# ===========================================================================
# 1. layer naming / taxonomy
# ===========================================================================

# Tuple-returning modules, labelled with the authors' own `return` variable
# names (segment_anything/modeling/{prompt_encoder,mask_decoder,transformer}.py).
_TUPLE_OUT_NAMES = {
    "prompt_encoder": ["sparse_embeddings", "dense_embeddings"],
    "mask_decoder": ["masks", "iou_pred"],
    "mask_decoder.transformer": ["queries", "keys"],
}
_TWOWAY_LAYER_RE = re.compile(r"^mask_decoder\.transformer\.layers\.\d+$")

# Containers that are NOT duplicates of their last child: they return a tuple
# carrying the full (token, image) state, i.e. real cut points of the network.
_CUT_CONTAINER_RE = re.compile(r"^mask_decoder\.transformer(\.layers\.\d+)?$")

# Modules that run but whose output never reaches the returned mask:
#   pe_layer            -- fires only from prompt_encoder.get_dense_pe(); the
#                          box path uses forward_with_coords(), invisible to a
#                          forward hook. Box independent -> divergence == 0.
#   hypernetworks 1..3  -- multimask_output=False keeps mask_slice=slice(0,1).
_UNUSED_RE = re.compile(
    r"(?:^|\.)pe_layer(?:$|\.)|(?:^|\.)output_hypernetworks_mlps\.[123](?:$|\.)"
)


def _tuple_out_names(full: str) -> list[str] | None:
    if full in _TUPLE_OUT_NAMES:
        return _TUPLE_OUT_NAMES[full]
    if _TWOWAY_LAYER_RE.match(full):
        return ["queries", "keys"]
    return None


def _branch_of_shape(shape) -> str:
    """token branch (5 output tokens + box tokens) vs image branch (64x64 image
    tokens and everything upscaled from them).

    Derived from the real tensor shape, not from the module name -- which is
    what makes it correct for `cross_attn_image_to_token`, where the authors
    call attention with swapped arguments (q=keys, k=queries), so that block's
    `q_proj` processes the IMAGE side and its `k_proj` the TOKEN side.
    """
    if len(shape) >= 4:          # (B, C, H, W) conv maps, mask logits
        return "image"
    if len(shape) == 3:          # (B, N, C) sequence
        return "image" if shape[1] >= 1024 else "token"
    return "token"               # (B, C) heads


def _block_of(layer: str) -> str:
    """Coarse block, used for x-axis shading and for reading the plot."""
    if layer.startswith("prompt_encoder"):
        return "prompt_enc"
    m = re.search(r"transformer\.layers\.(\d+)", layer)
    if m:
        return f"twoway.{m.group(1)}"
    if "final_attn_token_to_image" in layer or "norm_final_attn" in layer:
        return "final_attn"
    if layer.startswith("mask_decoder.transformer["):
        return "transf_out"
    if "output_upscaling" in layer:
        return "upscaling"
    if "output_hypernetworks" in layer:
        return "hypernet"
    if "iou_prediction_head" in layer:
        return "iou_head"
    if layer.startswith("mask_decoder["):
        return "outputs"
    return "other"


# ===========================================================================
# 2. capture / patch machinery
# ===========================================================================

_STATE = {"mode": "off"}                                # off | capture | patch
_CAPTURED: "OrderedDict[str, torch.Tensor]" = OrderedDict()   # key -> CPU fp32, SHAPED
_CALLS: "Counter[str]" = Counter()                      # multi-fire detector
_PATCH: "dict[str, torch.Tensor]" = {}                  # key -> replacement
_PATCH_HITS: "Counter[str]" = Counter()


def _elem_key(name: str, i: int, out_names: list[str] | None) -> str:
    if out_names is not None and i < len(out_names):
        return f"{name}[{out_names[i]}]"
    return f"{name}.out{i}"


def _hook(module, inputs, out, _name: str, _on: list[str] | None):
    """One hook serving both modes. In `patch` mode it RETURNS a replacement
    output, which is how torch lets a forward hook rewrite a module's result."""
    mode = _STATE["mode"]
    if mode == "off":
        return None

    # items: (position-in-tuple or None, key, tensor)
    if torch.is_tensor(out):
        items = [(None, _name, out)]
    elif isinstance(out, (tuple, list)):
        items = [(i, _elem_key(_name, i, _on), o)
                 for i, o in enumerate(out) if torch.is_tensor(o)]
    else:
        return None

    if mode == "capture":
        _CALLS[_name] += 1
        for _, key, t in items:
            _CAPTURED[key] = t.detach().float().cpu()
        return None

    # ---- patch ----
    repl = {}
    for pos, key, t in items:
        src = _PATCH.get(key)
        if src is not None:
            repl[pos] = src.to(device=t.device, dtype=t.dtype).reshape(t.shape)
            _PATCH_HITS[key] += 1
    if not repl:
        return None
    if torch.is_tensor(out):
        return repl[None]
    new = list(out)
    for pos, t in repl.items():
        new[pos] = t
    return tuple(new) if isinstance(out, tuple) else new


def _get_submodules(model):
    """prompt encoder / mask decoder, across SAM1 and SAM2 attribute naming."""
    pe = getattr(model, "prompt_encoder", None) or getattr(model, "sam_prompt_encoder", None)
    md = getattr(model, "mask_decoder", None) or getattr(model, "sam_mask_decoder", None)
    if pe is None or md is None:
        raise SystemExit(
            "Could not find prompt_encoder / mask_decoder on the model. "
            f"Available: {[a for a in dir(model) if not a.startswith('_')][:40]}"
        )
    return pe, md


def register_hooks(model) -> list:
    hooks = []
    pe, md = _get_submodules(model)
    for group, root in (("prompt_encoder", pe), ("mask_decoder", md)):
        for name, mod in root.named_modules():
            full = f"{group}.{name}" if name else group
            if name and _UNUSED_RE.search(name):
                continue
            is_root = (name == "")
            is_leaf = next(mod.children(), None) is None
            is_cut = bool(_CUT_CONTAINER_RE.match(full))
            if not (is_root or is_leaf or is_cut):
                continue
            hooks.append(mod.register_forward_hook(
                lambda m, i, o, _n=full, _on=_tuple_out_names(full): _hook(m, i, o, _n, _on)
            ))
    return hooks


def capture_for_box(box_1024, predictor, device, autocast_dtype):
    """Run one box; return (activation snapshot, predicted binary mask)."""
    _CAPTURED.clear()
    _CALLS.clear()
    _STATE["mode"] = "capture"
    try:
        mask = _run_box(box_1024, predictor, device, autocast_dtype)
    finally:
        _STATE["mode"] = "off"
    multi = [k for k, v in _CALLS.items() if v > 1]
    return OrderedDict(_CAPTURED), mask, multi


def patched_mask_for_box(box_1024, predictor, device, autocast_dtype, patch: dict):
    """Run one box with `patch` (key -> tensor) injected. Returns (mask, n_hits)."""
    _PATCH.clear()
    _PATCH.update(patch)
    _PATCH_HITS.clear()
    _STATE["mode"] = "patch"
    try:
        mask = _run_box(box_1024, predictor, device, autocast_dtype)
    finally:
        _STATE["mode"] = "off"
        _PATCH.clear()
    return mask, sum(_PATCH_HITS.values())


def _run_box(box_1024, predictor, device, autocast_dtype):
    box_t = torch.as_tensor(box_1024, dtype=torch.float32)
    if autocast_dtype is not None and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            return _predict_single_box(box_t, predictor, device,
                                       boxes_already_transformed=True)
    return _predict_single_box(box_t, predictor, device,
                               boxes_already_transformed=True)


# ===========================================================================
# 3. metrics
# ===========================================================================

def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.bool().reshape(-1)
    b = b.bool().reshape(-1)
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return float(inter) / float(union) if union else 1.0


def _pair_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Divergence of one layer between a reference run `a` and a run `b`.

    Returns the size-comparable magnitudes plus the geometric decomposition
        ||a-b||^2 = (||a||-||b||)^2 + 2||a|| ||b|| (1 - cos)
    so a rotation of the representation is never reported as a blow-up.
    """
    af = a.reshape(-1).double()
    bf = b.reshape(-1).double()
    n = af.numel()
    d = af - bf
    raw = float(torch.linalg.vector_norm(d).item())
    na = float(torch.linalg.vector_norm(af).item())
    nb = float(torch.linalg.vector_norm(bf).item())
    cos = float((af @ bf).item() / (na * nb + EPS))
    cos = max(-1.0, min(1.0, cos))
    return {
        "raw_l2": raw,
        "rms_l2": raw / math.sqrt(n),
        "rel_l2": raw / (na + EPS),
        "l1_rel": float(d.abs().sum().item()) / (float(af.abs().sum().item()) + EPS),
        "cos_dist": 1.0 - cos,
        "log_norm_ratio": math.log((nb + EPS) / (na + EPS)),
        "ref_norm": na,
        "ref_rms": na / math.sqrt(n),
        "n_elem": n,
    }


def divergence(A_ref, A_other) -> dict:
    out = {}
    for layer, a in A_ref.items():
        b = A_other.get(layer)
        if b is None or b.shape != a.shape:
            continue
        out[layer] = _pair_metrics(a, b)
    return out


# ===========================================================================
# 4. control box
# ===========================================================================

def make_control_box(best_box, bad_box, rng, frame_hw, max_cos=0.9, tries=200):
    """Random neighbour of best_box at EXACTLY the same L-inf displacement as
    best->bad, in a direction that is not (nearly) the adversarial one.

    v1 drew `rng.integers(-d, d+1, size=4)`, whose expected max-|shift| is only
    ~0.85*d -- its control was a weaker perturbation, not just a different one,
    which inflates any critical/control contrast.
    """
    best = np.asarray(best_box, dtype=np.float64)
    bad = np.asarray(bad_box, dtype=np.float64)
    delta = bad - best
    d_inf = float(np.abs(delta).max())
    if d_inf <= 0:
        return None, d_inf, float("nan")
    u = delta / (np.linalg.norm(delta) + EPS)
    h, w = frame_hw
    for _ in range(tries):
        g = rng.normal(size=4)
        m = float(np.abs(g).max())
        if m <= 0:
            continue
        v = g / m * d_inf                      # exact L-inf match
        cos = float(v @ u / (np.linalg.norm(v) + EPS))
        if abs(cos) > max_cos:
            continue
        ctrl = best + v
        if not (ctrl[2] - ctrl[0] > 2 and ctrl[3] - ctrl[1] > 2):
            continue
        if ctrl[0] < 0 or ctrl[1] < 0 or ctrl[2] > w or ctrl[3] > h:
            continue
        return ctrl.tolist(), d_inf, cos
    return None, d_inf, float("nan")


def frame_size_1024(orig_hw, target_length=1024):
    """Extent of SAM's internal 1024-frame for an image of size orig_hw."""
    return ResizeLongestSide.get_preprocess_shape(orig_hw[0], orig_hw[1], target_length)


# ===========================================================================
# 5. case sampling
# ===========================================================================

def sample_cases(shifts, limit, max_per_image, seed):
    """Random (seeded) sample, optionally capped per image.

    v1 used `shifts[:limit]` on a JSON that find_critical_shifts.py writes
    SORTED BY iou_drop DESCENDING -- i.e. the most extreme cases only, over a
    handful of images. That is a biased, low-diversity subsample.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(shifts))
    per_image = Counter()
    picked = []
    for i in idx:
        c = shifts[int(i)]
        if max_per_image and per_image[c["image_name"]] >= max_per_image:
            continue
        per_image[c["image_name"]] += 1
        picked.append(dict(c, case_uid=int(i)))
        if limit and len(picked) >= limit:
            break
    return picked


# ===========================================================================
# 6. aggregation
# ===========================================================================

_METRIC_COLS = ["rms_l2", "rel_l2", "raw_l2", "l1_rel", "cos_dist",
                "log_norm_ratio", "ref_rms", "ref_norm", "gain"]


def add_gain(df: pd.DataFrame) -> pd.DataFrame:
    """Local amplification gain = rel_l2(L) / rel_l2(previous layer of the SAME
    branch, same case). The x axis is hook-firing order, not network depth --
    q/k/v of one block are parallel and residual paths skip layers -- so the
    previous point on the plot is not generally the input of the next. Chaining
    within a branch is the closest defensible approximation, and the gain panel
    should be read as "where amplification concentrates", not as an exact
    layer-to-layer Jacobian."""
    df = df.sort_values(["case_uid", "pair_type", "branch", "order"]).copy()
    g = df.groupby(["case_uid", "pair_type", "branch"], sort=False)["rel_l2"]
    prev = g.shift(1)
    gain = df["rel_l2"] / prev
    gain[prev.isna() | (prev < 1e-9)] = np.nan
    df["gain"] = gain
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Two-stage aggregation: median within an image first, then across images.

    The 235 pairs in critical_shifts.json are 47 images x up to 5 boxes that
    differ by 1-2 px, so pairs are not independent observations; pooling them
    directly (v1) shrinks the spread by roughly sqrt(5).
    """
    keys = ["pair_type", "layer"]
    per_img = (df.groupby(keys + ["image_name", "order", "group", "branch", "block"],
                          dropna=False)[_METRIC_COLS]
                 .median().reset_index())

    def _q(p):
        return lambda s: s.quantile(p)

    agg = (per_img.groupby(keys + ["order", "group", "branch", "block"], dropna=False)
                  .agg(**{
                      **{f"{c}_med": (c, "median") for c in _METRIC_COLS},
                      **{f"{c}_q25": (c, _q(0.25)) for c in _METRIC_COLS},
                      **{f"{c}_q75": (c, _q(0.75)) for c in _METRIC_COLS},
                      "n_images": ("rms_l2", "size"),
                  }).reset_index())
    n_cases = df.groupby(keys)["case_uid"].nunique().rename("n_cases")
    return agg.merge(n_cases, on=keys, how="left").sort_values("order")


def _bootstrap_ci(vals, n_boot=2000, seed=0, alpha=0.05):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    b = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


def paired_specificity(df: pd.DataFrame, seed=0) -> pd.DataFrame:
    """critical vs control compared WITHIN each case, on log2(rel_crit/rel_ctrl),
    then medianed per image and tested across images.

    v1 printed mean(critical)/mean(control) -- a ratio of means over a skewed
    quantity, computed on unpaired aggregates, with no test.
    """
    piv = df.pivot_table(index=["case_uid", "image_name", "layer", "order", "branch", "block"],
                         columns="pair_type", values="rel_l2", aggfunc="median")
    if "critical" not in piv.columns or "control" not in piv.columns:
        return pd.DataFrame()
    piv = piv.dropna(subset=["critical", "control"]).reset_index()
    piv = piv[(piv["critical"] > 0) & (piv["control"] > 0)]
    piv["log2_ratio"] = np.log2(piv["critical"] / piv["control"])

    per_img = (piv.groupby(["layer", "order", "branch", "block", "image_name"])["log2_ratio"]
                  .median().reset_index())

    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None

    rows = []
    for (layer, order, branch, block), sub in per_img.groupby(["layer", "order", "branch", "block"]):
        v = sub["log2_ratio"].to_numpy()
        lo, hi = _bootstrap_ci(v, seed=seed)
        p = float("nan")
        if wilcoxon is not None and np.isfinite(v).sum() >= 6 and np.any(v != 0):
            try:
                p = float(wilcoxon(v, alternative="two-sided").pvalue)
            except Exception:
                pass
        rows.append({"layer": layer, "order": order, "branch": branch, "block": block,
                     "n_images": int(np.isfinite(v).sum()),
                     "log2_ratio_med": float(np.median(v)),
                     "log2_ratio_mean": float(np.mean(v)),
                     "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": p})
    return pd.DataFrame(rows).sort_values("order")


# ===========================================================================
# 7. plotting
# ===========================================================================

_STYLE = {
    "critical": dict(color="#d62728", marker="o", zorder=3),
    "control":  dict(color="#555555", marker="s", zorder=2),
    "noise":    dict(color="#2ca02c", marker=".", zorder=1),
}
_BRANCH_COLOR = {"token": "#1f3d99", "image": "#b35900"}


def _axis_layers(agg: pd.DataFrame) -> pd.DataFrame:
    return (agg.sort_values("order")
               .drop_duplicates("order")[["order", "layer", "branch", "block"]]
               .reset_index(drop=True))


def _decorate(ax, layers, ticks=False, shade=True):
    ax.set_xlim(layers["order"].min() - 0.6, layers["order"].max() + 0.6)
    ax.grid(axis="y", ls=":", alpha=0.35)
    if shade:
        blocks = layers["block"].to_numpy()
        orders = layers["order"].to_numpy()
        start = 0
        band = 0
        for i in range(1, len(blocks) + 1):
            if i == len(blocks) or blocks[i] != blocks[start]:
                if band % 2 == 1:
                    ax.axvspan(orders[start] - 0.5, orders[i - 1] + 0.5,
                               color="#000000", alpha=0.04, lw=0, zorder=0)
                start, band = i, band + 1
    ax.set_xticks(layers["order"])
    if ticks:
        ax.set_xticklabels(layers["layer"], rotation=90, fontsize=5.5)
        for lbl, br in zip(ax.get_xticklabels(), layers["branch"]):
            lbl.set_color(_BRANCH_COLOR.get(br, "#000000"))
    else:
        ax.set_xticklabels([])


def _band(ax, agg, metric, logy=True, types=None):
    """Median line + IQR band per pair type. Divergences are right-skewed with
    heavy tails, so v1's mean +/- std (clipped at 0) both overstated symmetry
    and understated the tail.

    Exact zeros are expected and meaningful here: `prompt_encoder[dense_embeddings]`
    is box independent, and the noise baseline is 0 whenever the kernels are
    deterministic. A log axis would silently drop those points, so they are
    clipped to a visible floor and the clipping is annotated instead."""
    series = []
    for pt in (types or ["critical", "control", "noise"]):
        sub = agg[agg["pair_type"] == pt].sort_values("order")
        if sub.empty:
            continue
        series.append((pt, sub["order"].to_numpy(),
                       sub[f"{metric}_med"].to_numpy(float),
                       sub[f"{metric}_q25"].to_numpy(float),
                       sub[f"{metric}_q75"].to_numpy(float)))
    if not series:
        return []

    floor = None
    n_zero = 0
    if logy:
        pos = np.concatenate([a[np.isfinite(a) & (a > 0)]
                              for _, _, m, q1, q3 in series for a in (m, q1, q3)] or [np.array([1.0])])
        floor = float(pos.min()) * 0.3 if pos.size else 1e-12
        n_zero = int(sum(int(np.sum(np.isfinite(a) & (a <= 0)))
                         for _, _, m, q1, q3 in series for a in (m,)))

    for pt, x, med, q1, q3 in series:
        if floor is not None:
            med, q1, q3 = (np.where(np.isfinite(a) & (a <= 0), floor, a) for a in (med, q1, q3))
        st = _STYLE.get(pt, {})
        ax.plot(x, med, label=pt, lw=1.3, ms=3.0, **st)
        ax.fill_between(x, q1, q3, color=st.get("color", "#888"), alpha=0.15, lw=0, zorder=1)

    if logy:
        ax.set_yscale("log")
        if n_zero:
            ax.axhline(floor, color="k", lw=0.6, ls="-", alpha=0.35)
            ax.text(0.995, 0.02, f"{n_zero} exact zero(s) drawn on the floor line",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color="#444")
    return [s[0] for s in series]


def plot_divergence(agg: pd.DataFrame, out_path: Path, title_extra: str):
    layers = _axis_layers(agg)
    n = len(layers)
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(max(12, 0.26 * n), 16))

    # -- panel 0: per-element RMS (size-comparable across layers) -------------
    ax = axes[0]
    _band(ax, agg, "rms_l2")
    ax.set_ylabel("rms_l2  ||$\\Delta$|| / $\\sqrt{numel}$")
    ax.set_title("Divergence per element (median, IQR band) — comparable across layers, "
                 "unlike raw L2 which just tracks layer size", pad=34)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92,
              title="line=median, band=IQR over images")

    # -- panel 1: relative L2 + its denominator ------------------------------
    ax = axes[1]
    _band(ax, agg, "rel_l2")
    ax.set_ylabel("rel_l2  ||$\\Delta$|| / ||A_best||")
    ref = agg[agg["pair_type"] == "critical"].sort_values("order")
    tw = ax.twinx()
    tw.plot(ref["order"], ref["ref_rms_med"], color="#7f7f7f", ls="--", lw=1.0, alpha=0.8)
    tw.set_yscale("log")
    tw.set_ylabel("||A_best|| / $\\sqrt{numel}$  (denominator, dashed)", color="#7f7f7f")
    tw.tick_params(axis="y", colors="#7f7f7f")
    ax.set_title("Relative L2 shown WITH its denominator — a rise in the ratio caused by a "
                 "shrinking baseline is visible here, not hidden")
    ax.legend(loc="upper left", fontsize=8)

    # -- panel 2: local gain -------------------------------------------------
    ax = axes[2]
    _band(ax, agg, "gain")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("gain = rel_l2(L) / rel_l2(L-1)")
    ax.set_title("Local amplification within a branch — the culprit layer is where gain "
                 "peaks (>1), not where the level peaks")
    ax.legend(loc="upper left", fontsize=8)

    # -- panel 3: geometry ---------------------------------------------------
    ax = axes[3]
    _band(ax, agg, "cos_dist")
    ax.set_ylabel("1 - cos(A_best, A_other)   (rotation)")
    tw = ax.twinx()
    for pt in ["critical", "control"]:
        sub = agg[agg["pair_type"] == pt].sort_values("order")
        if sub.empty:
            continue
        tw.plot(sub["order"], sub["log_norm_ratio_med"],
                color=_STYLE[pt]["color"], ls=":", lw=1.1, alpha=0.85)
    tw.axhline(0.0, color="k", lw=0.6, ls="--", alpha=0.4)
    tw.set_ylabel("log(||A_other|| / ||A_best||)  (dotted: scale change)")
    ax.set_title("Geometry split:  ||a-b||² = (||a||-||b||)² + 2||a||·||b||·(1-cos).  "
                 "Solid = rotation, dotted = scale — L2 alone conflates them")
    ax.legend(loc="upper left", fontsize=8)

    for i, ax in enumerate(axes):
        _decorate(ax, layers, ticks=(i == len(axes) - 1))

    # block labels along the top, staggered on two rows so short blocks
    # (transf_out, iou_head, outputs) stay readable next to long ones
    top = axes[0]
    blocks = layers["block"].to_numpy()
    orders = layers["order"].to_numpy()
    start = 0
    row = 0
    for i in range(1, len(blocks) + 1):
        if i == len(blocks) or blocks[i] != blocks[start]:
            mid = 0.5 * (orders[start] + orders[i - 1])
            top.text(mid, 1.012 + 0.030 * (row % 2), blocks[start],
                     transform=top.get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=7, color="#444")
            top.axvline(orders[start] - 0.5, color="#999", lw=0.5, ls="-", alpha=0.5)
            start, row = i, row + 1

    handles = [Line2D([], [], color=c, lw=3, label=f"{b}-branch (tick colour)")
               for b, c in _BRANCH_COLOR.items()]
    fig.legend(handles=handles, loc="lower left", fontsize=8, ncol=2,
               bbox_to_anchor=(0.01, 0.0))
    fig.suptitle(f"SAM activation divergence, verified critical shifts {title_extra}",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0.015, 1, 0.985])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved divergence plot -> {out_path}")


def plot_causal(spec_df: pd.DataFrame, patch_df: pd.DataFrame,
                agg: pd.DataFrame, out_path: Path, restore_thresh: float,
                baseline_iou: float | None):
    panels = int(not spec_df.empty) + int(patch_df is not None and not patch_df.empty)
    if panels == 0:
        return
    layers = _axis_layers(agg)
    fig, axes = plt.subplots(panels, 1, sharex=True,
                             figsize=(max(12, 0.26 * len(layers)), 5.5 * panels))
    axes = np.atleast_1d(axes)
    k = 0

    if patch_df is not None and not patch_df.empty:
        ax = axes[k]; k += 1
        p = patch_df.sort_values("order")
        ax.plot(p["order"], p["iou_to_best_med"], color="#1f77b4", marker="o", ms=3, lw=1.3,
                label="median IoU(patched, best)")
        ax.fill_between(p["order"], p["iou_to_best_q25"], p["iou_to_best_q75"],
                        color="#1f77b4", alpha=0.15, lw=0)
        if baseline_iou is not None:
            ax.axhline(baseline_iou, color="#d62728", ls="--", lw=1.0,
                       label=f"unpatched IoU(bad, best) = {baseline_iou:.3f}")
        ax.axhline(1.0, color="k", ls=":", lw=0.8, alpha=0.5)
        ax.set_ylabel("IoU(patched, best)")
        ax.set_ylim(-0.02, 1.05)
        tw = ax.twinx()
        tw.plot(p["order"], p["restored_frac"], color="#2ca02c", lw=1.0, alpha=0.8)
        tw.set_ylabel(f"fraction restored (IoU > {restore_thresh})", color="#2ca02c")
        tw.tick_params(axis="y", colors="#2ca02c")
        tw.set_ylim(-0.02, 1.05)
        ax.set_title("Activation patching: run bad_box, inject ONE best_box tensor. "
                     "This measures where the divergence is DECISIVE, which a norm plot "
                     "cannot. prompt_encoder[sparse_embeddings] is a full cut → must read ~1.0")
        ax.legend(loc="lower right", fontsize=8)

    if not spec_df.empty:
        ax = axes[k]; k += 1
        s = spec_df.sort_values("order")
        yerr = np.vstack([s["log2_ratio_med"] - s["ci_lo"], s["ci_hi"] - s["log2_ratio_med"]])
        yerr = np.clip(yerr, 0, None)
        sig = s["wilcoxon_p"].to_numpy() < 0.05
        ax.errorbar(s["order"], s["log2_ratio_med"], yerr=yerr, fmt="o", ms=3, lw=1.0,
                    elinewidth=0.8, color="#8c1a8c", ecolor="#8c1a8c", alpha=0.9,
                    label="median log2(rel_crit / rel_ctrl), 95% bootstrap CI")
        if sig.any():
            ax.plot(s["order"].to_numpy()[sig], s["log2_ratio_med"].to_numpy()[sig],
                    "o", ms=6, mfc="none", mec="#8c1a8c", mew=1.2,
                    label="Wilcoxon p < 0.05 (uncorrected)")
        ax.axhline(0.0, color="k", ls="--", lw=0.8, alpha=0.6)
        ax.set_ylabel("log2 ratio  (>0: adversarial direction diverges more)")
        ax.set_title("Specificity, computed PAIRWISE inside each case then across images — "
                     "not a ratio of two means")
        ax.legend(loc="upper left", fontsize=8)

    for i, ax in enumerate(axes):
        _decorate(ax, layers, ticks=(i == len(axes) - 1))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved causal / specificity plot -> {out_path}")


# ===========================================================================
# 8. main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--critical_shifts", default="critical_shifts.json")
    p.add_argument("--images_dir", required=True)
    p.add_argument("--masks_dir", default=None,
                   help="GT masks. Strongly recommended: without it the drop can only be "
                        "verified as mask disagreement, not against ground truth.")
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--model_name", default="SAM",
                   choices=["SAM", "SAM2.1", "SAM-HQ", "SAM-HQ2", "SAM3"])
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument("--limit", type=int, default=100, help="0 = all")
    p.add_argument("--max_per_image", type=int, default=2,
                   help="cap near-duplicate cases per image (0 = no cap)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--autocast", default="bf16", choices=["bf16", "fp16", "none"],
                   help="bf16 matches the pipeline that produced critical_shifts.json; "
                        "verification and capture always share this setting")

    p.add_argument("--control", action="store_true", default=False)
    p.add_argument("--ctrl_max_cos", type=float, default=0.9,
                   help="reject control directions this aligned with best->bad")
    p.add_argument("--no_noise_floor", dest="noise_floor", action="store_false", default=True)

    p.add_argument("--min_best_iou", type=float, default=0.85,
                   help="GT mode: best_box must still be this good")
    p.add_argument("--min_drop", type=float, default=0.75,
                   help="GT mode: IoU(best)-IoU(bad) must still be this large")
    p.add_argument("--ctrl_min_iou", type=float, default=0.85,
                   help="GT mode: a control whose own mask is worse than this is discarded")
    p.add_argument("--max_agree", type=float, default=0.5,
                   help="no-GT mode: IoU(best_mask, bad_mask) must be below this")
    p.add_argument("--ctrl_min_agree", type=float, default=0.85,
                   help="no-GT mode: IoU(best_mask, ctrl_mask) must be above this")

    p.add_argument("--patch", action="store_true", default=False)
    p.add_argument("--patch_limit", type=int, default=20, help="cases used for patching")
    p.add_argument("--patch_layers", default=None, help="regex to restrict patched layers")
    p.add_argument("--restore_thresh", type=float, default=0.9)

    p.add_argument("--out_dir", default="exp_res/probe_v2")
    p.add_argument("--tag", default="", help="suffix for all output filenames")
    return p.parse_args()


def main():
    args = parse_args()
    from heatmaps.env_dispatch import maybe_dispatch_to_env
    maybe_dispatch_to_env(args.model_name, __file__)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ac = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.autocast]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    with open(args.critical_shifts) as f:
        shifts = json.load(f)
    cases = sample_cases(shifts, args.limit, args.max_per_image, args.seed)
    print(f"Sampled {len(cases)} cases (random, seed={args.seed}, "
          f"max_per_image={args.max_per_image or 'inf'}) out of {len(shifts)} in "
          f"{args.critical_shifts}")
    if args.masks_dir is None:
        print("[warn] no --masks_dir: the drop is verified only as mask DISAGREEMENT "
              "(IoU(best_mask, bad_mask) <= --max_agree), not against ground truth.")

    predictor = load_model(model_name=args.model_name, model_type=args.model_type,
                           checkpoint=args.checkpoint_path, device=device)
    model = getattr(predictor, "model", predictor)
    hooks = register_hooks(model)
    print(f"Hooked {len(hooks)} modules (used leaves + transformer cut points + roots).")

    by_image = defaultdict(list)
    for c in cases:
        by_image[c["image_name"]].append(c)

    rng = np.random.default_rng(args.seed)
    # Canonical x-axis position per layer, assigned from the FULL capture
    # snapshot (hook firing order). Deriving it per case from the filtered
    # divergence dict would silently shift every layer after a dropped one.
    layer_order: dict[str, int] = {}

    def _order_of(layer: str) -> int:
        if layer not in layer_order:
            layer_order[layer] = len(layer_order)
        return layer_order[layer]

    case_rows: list[dict] = []
    repro_rows: list[dict] = []
    patch_rows: list[dict] = []
    unpatched_ious: list[float] = []
    n_patch_done = 0
    warned_multi = False
    decomp_checked = False

    pbar = tqdm(total=len(cases), desc="cases")
    try:
        for image_name, img_cases in by_image.items():
            image_path = _find_file(args.images_dir, image_name)
            if image_path is None:
                print(f"[warn] image not found: {image_name} ({len(img_cases)} cases)")
                pbar.update(len(img_cases))
                continue
            try:
                H, W = _prepare_image(str(image_path), predictor)
            except Exception as e:
                print(f"[warn] encode failed for {image_name}: {e}")
                pbar.update(len(img_cases))
                continue

            gt = None
            if args.masks_dir:
                mp = _find_file(args.masks_dir, image_name)
                if mp is None:
                    print(f"[warn] GT mask not found: {image_name}; cases will be dropped")
                else:
                    m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                    if m is not None:
                        m = (m > 0)
                        if m.shape[0] > 1024 and m.shape[1] > 1024:
                            m = cv2.resize(m.astype(np.uint8), (1024, 1024),
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                        gt = torch.from_numpy(m)

            frame_hw = frame_size_1024(get_original_size(predictor))

            for case in img_cases:
                pbar.update(1)
                rec = {"image_name": image_name, "case_uid": case["case_uid"],
                       "json_best_iou": case["best_iou"], "json_bad_iou": case["bad_iou"],
                       "json_drop": case["iou_drop"]}
                try:
                    A_best, m_best, multi = capture_for_box(case["best_box"], predictor, device, ac)
                    if multi and not warned_multi:
                        print(f"[warn] modules fired more than once in one pass "
                              f"(their capture is the LAST call): {multi[:5]}")
                        warned_multi = True
                    A_bad, m_bad, _ = capture_for_box(case["bad_box"], predictor, device, ac)
                except Exception as e:
                    print(f"[warn] forward failed for {image_name}: {e}")
                    rec.update(kept=False, reason="forward_failed")
                    repro_rows.append(rec)
                    continue

                # ---- fix 1: does the critical shift reproduce HERE? ----------
                if gt is not None:
                    iou_best, iou_bad = _iou(m_best, gt), _iou(m_bad, gt)
                    rec.update(iou_best=iou_best, iou_bad=iou_bad, drop=iou_best - iou_bad,
                               mode="gt")
                    ok = (iou_best >= args.min_best_iou
                          and (iou_best - iou_bad) >= args.min_drop)
                else:
                    agree = _iou(m_best, m_bad)
                    rec.update(agree_best_bad=agree, mode="agreement")
                    ok = agree <= args.max_agree
                rec["reproduced"] = bool(ok)
                if not ok:
                    rec.update(kept=False, reason="not_reproduced")
                    repro_rows.append(rec)
                    continue

                pairs = {"critical": divergence(A_best, A_bad)}

                # ---- fix 2: noise floor -------------------------------------
                if args.noise_floor:
                    try:
                        A_best2, m_best2, _ = capture_for_box(case["best_box"], predictor, device, ac)
                        pairs["noise"] = divergence(A_best, A_best2)
                        rec["noise_mask_iou"] = _iou(m_best, m_best2)
                        del A_best2
                    except Exception as e:
                        print(f"[warn] noise-floor run failed: {e}")

                # ---- fix 3: honest, validated control -----------------------
                if args.control:
                    ctrl_box, d_inf, cos = make_control_box(
                        case["best_box"], case["bad_box"], rng, frame_hw, args.ctrl_max_cos)
                    rec.update(ctrl_dinf=d_inf, ctrl_cos_with_attack=cos)
                    if ctrl_box is None:
                        rec["ctrl_status"] = "no_valid_direction"
                    else:
                        try:
                            A_ctrl, m_ctrl, _ = capture_for_box(ctrl_box, predictor, device, ac)
                            if gt is not None:
                                iou_ctrl = _iou(m_ctrl, gt)
                                ctrl_ok = iou_ctrl >= args.ctrl_min_iou
                            else:
                                iou_ctrl = _iou(m_best, m_ctrl)
                                ctrl_ok = iou_ctrl >= args.ctrl_min_agree
                            rec.update(ctrl_iou=iou_ctrl,
                                       ctrl_status="ok" if ctrl_ok else "control_also_broken")
                            if ctrl_ok:
                                pairs["control"] = divergence(A_best, A_ctrl)
                            del A_ctrl
                        except Exception as e:
                            rec["ctrl_status"] = f"failed: {e}"

                # ---- record per-layer metrics -------------------------------
                for layer in A_best:
                    _order_of(layer)
                for ptype, dv in pairs.items():
                    for layer, mt in dv.items():
                        shape = A_best[layer].shape
                        row = {"image_name": image_name, "case_uid": case["case_uid"],
                               "pair_type": ptype, "layer": layer, "order": _order_of(layer),
                               "group": ("prompt_encoder" if layer.startswith("prompt_encoder")
                                         else "mask_decoder"),
                               "branch": _branch_of_shape(shape),
                               "block": _block_of(layer)}
                        row.update(mt)
                        case_rows.append(row)
                        if not decomp_checked and mt["ref_norm"] > 1e-6:
                            na = mt["ref_norm"]
                            nb = na * math.exp(mt["log_norm_ratio"])
                            lhs = mt["raw_l2"] ** 2
                            rhs = (na - nb) ** 2 + 2 * na * nb * mt["cos_dist"]
                            print(f"[check] ||a-b||^2 decomposition: {lhs:.6g} vs {rhs:.6g} "
                                  f"(rel err {abs(lhs - rhs) / (lhs + EPS):.2e})")
                            decomp_checked = True

                # ---- fix 7: activation patching ------------------------------
                if args.patch and n_patch_done < args.patch_limit:
                    base = _iou(m_bad, m_best)
                    unpatched_ious.append(base)
                    keys = set(A_best.keys())
                    if args.patch_layers:
                        pr = re.compile(args.patch_layers)
                        keys = {k for k in keys if pr.search(k)}
                    for key in A_best.keys():
                        if key not in keys:
                            continue
                        try:
                            m_p, hits = patched_mask_for_box(
                                case["bad_box"], predictor, device, ac, {key: A_best[key]})
                        except Exception as e:
                            print(f"[warn] patch failed on {key}: {e}")
                            continue
                        patch_rows.append({
                            "image_name": image_name, "case_uid": case["case_uid"],
                            "layer": key, "order": _order_of(key),
                            "branch": _branch_of_shape(A_best[key].shape),
                            "block": _block_of(key), "hits": hits,
                            "iou_to_best": _iou(m_p, m_best),
                            "iou_to_gt": _iou(m_p, gt) if gt is not None else float("nan"),
                            "baseline_iou_bad_to_best": base,
                        })
                    n_patch_done += 1

                rec.update(kept=True, reason="ok")
                repro_rows.append(rec)
                del A_best, A_bad
    except KeyboardInterrupt:
        print("\n[interrupted] writing partial results...")
    finally:
        pbar.close()
        for h in hooks:
            h.remove()
        _STATE["mode"] = "off"

    # ---------------- reporting ------------------------------------------
    repro = pd.DataFrame(repro_rows)
    repro_csv = out_dir / f"reproduction{tag}.csv"
    repro.to_csv(repro_csv, index=False)
    n_kept = int(repro["kept"].sum()) if "kept" in repro else 0
    print(f"\nVerification: {n_kept}/{len(repro)} cases reproduced and kept "
          f"-> {repro_csv}")
    if "reason" in repro:
        for r, c in repro["reason"].value_counts().items():
            print(f"    {r}: {c}")
    if "ctrl_status" in repro:
        for r, c in repro["ctrl_status"].value_counts().items():
            print(f"    control {r}: {c}")

    if not case_rows:
        raise SystemExit("No verified cases -- nothing to plot. Check thresholds / paths.")

    df = pd.DataFrame(case_rows)
    df = add_gain(df)
    case_csv = out_dir / f"per_case{tag}.csv"
    df.to_csv(case_csv, index=False)
    print(f"Saved per-case metrics -> {case_csv}  ({len(df)} rows)")

    agg = aggregate(df)
    layer_csv = out_dir / f"per_layer{tag}.csv"
    agg.to_csv(layer_csv, index=False)
    n_img = df["image_name"].nunique()
    n_case = df["case_uid"].nunique()
    print(f"Saved per-layer aggregates -> {layer_csv}  "
          f"({n_case} cases over {n_img} images; aggregated per image first)")

    spec = paired_specificity(df, seed=args.seed) if args.control else pd.DataFrame()
    if not spec.empty:
        spec_csv = out_dir / f"specificity{tag}.csv"
        spec.to_csv(spec_csv, index=False)
        print(f"Saved paired specificity -> {spec_csv}")

    patch_agg = None
    baseline_iou = float(np.median(unpatched_ious)) if unpatched_ious else None
    if patch_rows:
        pdf = pd.DataFrame(patch_rows)
        pdf.to_csv(out_dir / f"patching_per_case{tag}.csv", index=False)
        per_img = (pdf.groupby(["layer", "order", "branch", "block", "image_name"])
                      .agg(iou_to_best=("iou_to_best", "median"),
                           restored=("iou_to_best", lambda s: float((s > args.restore_thresh).mean())),
                           hits=("hits", "min")).reset_index())
        patch_agg = (per_img.groupby(["layer", "order", "branch", "block"])
                     .agg(iou_to_best_med=("iou_to_best", "median"),
                          iou_to_best_q25=("iou_to_best", lambda s: s.quantile(0.25)),
                          iou_to_best_q75=("iou_to_best", lambda s: s.quantile(0.75)),
                          restored_frac=("restored", "mean"),
                          min_hits=("hits", "min"),
                          n_images=("iou_to_best", "size")).reset_index().sort_values("order"))
        patch_agg.to_csv(out_dir / f"patching_per_layer{tag}.csv", index=False)
        print(f"Saved patching results -> {out_dir / f'patching_per_layer{tag}.csv'}")
        dead = patch_agg[patch_agg["min_hits"] == 0]["layer"].tolist()
        if dead:
            print(f"[warn] patch never fired for {len(dead)} layer(s), e.g. {dead[:5]}")
        sanity = patch_agg[patch_agg["layer"] == "prompt_encoder[sparse_embeddings]"]
        if not sanity.empty:
            v = float(sanity["iou_to_best_med"].iloc[0])
            verdict = "OK" if v > 0.99 else "FAILED -- patching machinery is not sound"
            print(f"[self-test] patching prompt_encoder[sparse_embeddings] (a full cut) "
                  f"restores IoU = {v:.4f}  [{verdict}]")

    # ---------------- plots ----------------------------------------------
    extra = f"(n={n_case} cases / {n_img} images, autocast={args.autocast})"
    plot_divergence(agg, out_dir / f"divergence{tag}.png", extra)
    plot_causal(spec, patch_agg, agg, out_dir / f"causal{tag}.png",
                args.restore_thresh, baseline_iou)

    # ---------------- text summary ---------------------------------------
    crit = agg[agg["pair_type"] == "critical"]
    noise = agg[agg["pair_type"] == "noise"].set_index("layer")["rms_l2_med"]
    print("\nTop layers by local GAIN (where amplification concentrates):")
    for _, r in crit.sort_values("gain_med", ascending=False).head(8).iterrows():
        nf = float(noise.get(r["layer"], float("nan")))
        snr = r["rms_l2_med"] / nf if nf and np.isfinite(nf) and nf > 0 else float("inf")
        print(f"  gain x{r['gain_med']:.2f}  rel_l2 {r['rel_l2_med']:.4f}  "
              f"rms/noise {snr:8.1f}  [{r['branch']:5s}] {r['layer']}")
    if not spec.empty:
        print("\nTop layers by paired specificity (log2 critical/control, CI excludes 0):")
        s = spec[(spec["ci_lo"] > 0)].sort_values("log2_ratio_med", ascending=False).head(8)
        for _, r in s.iterrows():
            print(f"  x{2 ** r['log2_ratio_med']:.2f}  CI [{2 ** r['ci_lo']:.2f}, "
                  f"{2 ** r['ci_hi']:.2f}]  p={r['wilcoxon_p']:.1e}  {r['layer']}")
    if patch_agg is not None and not patch_agg.empty:
        print(f"\nTop layers by CAUSAL restoration (unpatched IoU(bad,best) = "
              f"{baseline_iou:.3f}):")
        pa = patch_agg[patch_agg["layer"] != "prompt_encoder[sparse_embeddings]"]
        for _, r in pa.sort_values("iou_to_best_med", ascending=False).head(8).iterrows():
            print(f"  IoU {r['iou_to_best_med']:.3f}  restored {r['restored_frac']:.0%}  "
                  f"[{r['branch']:5s}] {r['layer']}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
