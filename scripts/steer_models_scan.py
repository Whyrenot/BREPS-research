"""
steer_models_scan.py
====================
Where in the mask decoder does one fixed steering vector repair the boxes
people drew -- for SAM-1, SAM-2.1 and SAM-3 alike?

steer_sam_decoder.py answered this for SAM-1 on critical shifts and found
L2.norm2; steer_user_study.py then estimated the vector on user-study boxes.
This script runs the whole site scan on the user study, for any of the three
models, with the same code. SAM-2.1 and SAM-3 (its interactive "SAM 1 task"
predictor, Sam3TrackerPredictor) keep SAM-1's mask decoder: a two-way
transformer of depth 2 whose blocks have the same norm1-norm4, so every site of
the SAM-1 scan exists there. What differs is read off the loaded model, never
assumed (meta.json "layout"):

  tokens   SAM-1: 7 = IoU | mask 0-3 | 2 box corners. SAM-2.1 / SAM-3 put an
           object-score token first (pred_obj_scores) and the predictor passes
           the box as two points with labels 2, 3 and boxes=None, so the prompt
           encoder appends a 'not a point' pad token: 9 = obj | IoU | mask 0-3 |
           2 corners | pad
  output   multimask_output=False is mask token 0 on SAM-1. On SAM-2.1 /
           SAM-3 the decoder falls back to the best of masks 1-3 when token 0's
           mask is unstable (dynamic_multimask_via_stability, set by build_sam2
           and by SAM-3's build_tracker). That is the model as shipped and what
           the reference csvs measured, so it is kept and the head used is
           recorded per box
  frame    SAM-1 1024 with letterbox padding; SAM-2.1 1024 and SAM-3 1008 by a
           plain resize to a square; image grid 64 x 64 / 64 x 64 / 72 x 72;
           SAM-2.1 / SAM-3 also add the backbone's high-res features in the
           upscaling

PROTOCOL  (as steer_user_study.py; STEERING_USER_STUDY_HANDOFF.md)
--------
  data     user study FOR_TEST, --use m, the --datasets subsets (50 images x 25
           annotations each); box = the tight bbox of the user's mask
           (refine_box_iou_grad.build_user_cases), GT = masks/<stem> > 0
  split    the 50 COCO stems, steer_user_study.split_halves(--split_seed): half A
           fits everything, half B and the other subsets are test sets
  groups   bottom50 / top50 inside each image by the IoU of multimask=False
           (12 / 13), for both output modes; reporting only
  vector   per site and token group, difference in means inside each image,
           mean(top50 states) - mean(bottom50 states), averaged over images.
           On half A the vector of the other --folds image folds of A (out of
           fold); on the test sets the vector of all of A, frozen
  methods  dm      h' = h + alpha * v at the site, every alpha of --alphas
           rand    a fixed random direction with v's norm per token (per box
                   for the image stream), the same alphas: the control
           oracle  activation patching: every box gets the site's tensor (or its
                   token rows) of the image's best box -- the box with the
                   highest multimask=False IoU among the 25: the ceiling of a
                   fixed vector at that site
  modes    every forward is scored twice, from the same decoder pass:
           mm0  multimask_output=False (the model's single output, as shipped)
           mm1  multimask_output=True, head = argmax of predicted IoU over masks
                1-3 (head-select). The token stream is the same in both modes:
                the mode only picks which output becomes the mask

SITES (forward order; ids = node ids of graph.json)
-----
  prompt_tok    the prompt encoder's sparse embeddings (corners, + pad)
  L{l}.norm1-3  token stream of two-way block l after each LayerNorm, steered
                per token group all / mask0 / mask123 / prompt (every token its
                own vector)
  L{l}.norm4    image stream after image-to-token attention: one 256-d vector
                (the box-pooled DM) added to the cells whose centre is inside
                the box
  final_norm    token stream after the final attention: all / mask0 / mask123
                (prompt rows are not read after it)
  hyper         the outputs of all mask tokens' hypernetwork MLPs (n_mask x 32),
                the masks' linear readout

PROBE (per site, the whole tensor; box-pooled for the image stream)
  L2 logistic regression with balanced classes and standardised features
  (sklearn's objective, solved with LBFGS in torch so it also runs in the sam3
  env), top50 vs bottom50: out of fold on A, the all-A probe on every test set;
  pooled AUC and the mean AUC inside images

CHECKS (checks.json; the ones marked * stop the run)
------
  * zero dose: a zero vector at every site (first image of every subset) and at
    L2.norm2 (every image) reproduces the unsteered logits bit for bit
  * own forward: the decoder pass used here (predict_masks + the module's own
    head choice) equals mask_decoder.forward() bit for bit, both modes
  * hooks: every hooked module fires exactly once per forward
  * repro: multimask=False IoU on half A matches --reference_csv
    (undefended_iou): at most --repro_max_frac boxes off by > --repro_tol and
    the mean within --repro_mean_tol. A wrong checkpoint fails here, before
    the expensive pass
    public API: the predictor's own predict / predict_torch gives the same
    masks on the same batch, and one box per call (the box frame; what is left
    is batch-dependent numerics) (reported); mm1 vs the csv's headsel_iou (reported)
    TF32: sam3 turns TF32 matmuls on at import, and masks then depend on the
    batch; they are switched off for every model (--keep_tf32 keeps them)
    baseline repeat: half A re-encoded for pass 2 gives the same IoUs
    oracle cut: patching prompt_tok reproduces the best box's IoU on every box
    graph: every decoder module that fires is drawn in graph.json
  * split: A and B disjoint, 25 + 25; 12 / 13 groups on every image

OUTPUTS (--out_dir; steer_models_report.py turns them into tables + HTML)
-------
  per_case_mm0.csv.gz, per_case_mm1.csv.gz
                one row per box: dataset, image_name, kind, user, attempt, half,
                fold, group, base_iou (mm0: undef, mm1: head-select), base_head,
                undef_iou, best_iou (the image's best box in this mode),
                ref_iou (the reference csv), probe_<site> (probe scores), and
                per config iou_c<cid>, head_c<cid>
  configs.csv   cid, site, kind, group, method, alpha, dose = ||alpha v|| / mean ||h||
  vectors.npz   dm and rand per site from all of A (+ per fold)
  vector_stats.csv  per site x group: ||v||, mean ||h||, cosine between fold vectors
  probe.csv     per site x dataset: AUC, AUC inside images
  states_A.npz  the A states of every site (float16), for later vector variants
  split.json, graph.json, meta.json, checks.json

Example
-------
    # SAM-1 (pishugin-breps)
    CUDA_VISIBLE_DEVICES=0 python scripts/steer_models_scan.py --model_name SAM \\
        --root /.../user_study/FOR_TEST \\
        --checkpoint_path /.../MODEL_CHECKPOINTS/SAM/sam_vit_b_01ec64.pth \\
        --reference_csv results/samfull_m.csv --out_dir exp_res/steer_scan_SAM

    # SAM-2.1 base_plus (results/sam21full_m.csv is hiera_large: no reference)
    CUDA_VISIBLE_DEVICES=0 python scripts/steer_models_scan.py --model_name SAM2.1 \\
        --root /.../FOR_TEST --checkpoint_path /.../MODEL_CHECKPOINTS/SAM2.1/sam2.1_hiera_base_plus.pt \\
        --out_dir exp_res/steer_scan_SAM2.1

    # SAM-3: re-execs itself into the sam3 env (heatmaps/env_dispatch.py)
    CUDA_VISIBLE_DEVICES=0 python scripts/steer_models_scan.py --model_name SAM3 \\
        --root /.../FOR_TEST --checkpoint_path "" \\
        --reference_csv results/sam3full_m.csv --out_dir exp_res/steer_scan_SAM3

    # smoke: 2 images per half / subset, 2 alphas
    ... --limit_images 2 --alphas 1,4 --out_dir exp_res/steer_scan_smoke
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import sys
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heatmaps.env_dispatch import maybe_dispatch_to_env  # noqa: E402

EPS = 1e-12
MODES = (0, 1)                       # multimask_output False / True
TOKEN_GROUPS = ("all", "mask0", "mask123", "prompt")
FINAL_GROUPS = ("all", "mask0", "mask123")
METHODS = ("dm", "rand")


class CheckFailed(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# The model's decoder, called the way its own predictor calls it
# ---------------------------------------------------------------------------

class Decoder:
    """Prompt encoder + mask decoder of SAM-1 (SamPredictor) or SAM-2.1 / SAM-3
    (SAM2ImagePredictor-shaped), with every output token kept, so one pass
    serves both output modes."""

    def __init__(self, predictor):
        self.p = predictor
        self.model = predictor.model
        self.sam1 = hasattr(predictor, "predict_torch")
        if self.sam1:
            self.pe_path, self.md_path = "prompt_encoder", "mask_decoder"
            self.frame = float(self.model.image_encoder.img_size)
            self.thr = float(self.model.mask_threshold)
        else:
            self.pe_path, self.md_path = "sam_prompt_encoder", "sam_mask_decoder"
            self.frame = float(predictor._transforms.resolution)
            self.thr = float(predictor.mask_threshold)
        self.pe = self.model.get_submodule(self.pe_path)
        self.md = self.model.get_submodule(self.md_path)
        self.device = next(self.md.parameters()).device
        self.emb_hw = tuple(int(x) for x in self.pe.image_embedding_size)
        self.depth = len(self.md.transformer.layers)
        self.s = 1 if getattr(self.md, "pred_obj_scores", False) else 0
        self.n_mask = int(self.md.num_mask_tokens)
        self.dynamic = bool(getattr(self.md, "dynamic_multimask_via_stability", False))
        self.stab_delta = float(getattr(self.md, "dynamic_multimask_stability_delta", math.nan))
        self.stab_thresh = float(getattr(self.md, "dynamic_multimask_stability_thresh", math.nan))
        self.training = bool(self.md.training)
        # sam2 keeps the flag only on the head (MLP.sigmoid_output)
        self.iou_sigmoid = bool(getattr(self.md, "iou_prediction_use_sigmoid",
                                        getattr(self.md.iou_prediction_head, "sigmoid_output", False)))
        self._pm_kw = set(inspect.signature(self.md.predict_masks).parameters)
        self._fw_kw = set(inspect.signature(self.md.forward).parameters)
        with torch.inference_mode():
            sparse, _ = self._encode(torch.tensor([[10.0, 10.0, 100.0, 120.0]], device=self.device))
        self.n_sparse = int(sparse.shape[1])
        self.T = self.s + 1 + self.n_mask + self.n_sparse
        n0 = self.s + 1
        self.rows = OrderedDict([("all", list(range(self.T))), ("mask0", [n0]),
                                 ("mask123", list(range(n0 + 1, n0 + self.n_mask))),
                                 ("prompt", list(range(n0 + self.n_mask, self.T)))])
        self.token_names = ((["obj"] if self.s else []) + ["iou"] + [f"mask{i}" for i in range(self.n_mask)]
                            + ["corner_tl", "corner_br"] + ["pad"] * (self.n_sparse - 2))

    def layout(self) -> dict:
        return {"family": "sam1" if self.sam1 else "sam2", "predictor": type(self.p).__name__,
                "decoder": type(self.md).__name__, "prompt_encoder_path": self.pe_path,
                "mask_decoder_path": self.md_path, "frame": self.frame, "emb_hw": list(self.emb_hw),
                "low_res_hw": [4 * self.emb_hw[0], 4 * self.emb_hw[1]], "depth": self.depth,
                "n_tokens": self.T, "token_names": self.token_names, "obj_score_token": bool(self.s),
                "n_mask_tokens": self.n_mask, "n_sparse": self.n_sparse, "rows": self.rows,
                "dynamic_multimask_via_stability": self.dynamic, "stability_delta": self.stab_delta,
                "stability_thresh": self.stab_thresh, "decoder_training": self.training,
                "mask_threshold": self.thr,
                "iou_prediction_use_sigmoid": self.iou_sigmoid,
                "use_high_res_features": bool(getattr(self.md, "use_high_res_features", False))}

    def model_boxes(self, boxes_1024: np.ndarray) -> torch.Tensor:
        """This repo's SAM-1 1024 frame -> the model's input frame. SAM-2.1 /
        SAM-3: back to original pixels (boxes_to_original, the integer user
        bbox) and through the predictor's own transform, as
        defend_critical_shifts._predict_single_box -> predictor.predict does."""
        if self.sam1:
            return torch.as_tensor(np.asarray(boxes_1024, np.float32), device=self.device)
        from heatmaps.comp_hw_smoothed import get_original_size
        from heatmaps.defend_critical_shifts import boxes_to_original
        hw = get_original_size(self.p)
        orig = boxes_to_original(np.asarray(boxes_1024, np.float64), hw)
        b = torch.as_tensor(orig, dtype=torch.float, device=self.device)
        return self.p._transforms.transform_boxes(b, normalize=True, orig_hw=hw).reshape(-1, 4)

    def original_boxes(self, boxes_1024: np.ndarray) -> np.ndarray:
        from heatmaps.comp_hw_smoothed import get_original_size
        from heatmaps.defend_critical_shifts import boxes_to_original
        return boxes_to_original(np.asarray(boxes_1024, np.float64), get_original_size(self.p))

    def _encode(self, b: torch.Tensor):
        if self.sam1:
            return self.pe(points=None, boxes=b, masks=None)
        labels = torch.tensor([[2, 3]], dtype=torch.int, device=b.device).repeat(b.shape[0], 1)
        return self.pe(points=(b.reshape(-1, 2, 2), labels), boxes=None, masks=None)

    def _kwargs(self, b: torch.Tensor, allowed: set) -> dict:
        sparse, dense = self._encode(b)
        if self.sam1:
            feats, extra = self.p.features, {}
        else:
            f = self.p._features
            feats = f["image_embed"][-1].unsqueeze(0)
            extra = {"repeat_image": b.shape[0] > 1,
                     "high_res_features": [x[-1].unsqueeze(0) for x in f["high_res_feats"]]}
        kw = dict(image_embeddings=feats, image_pe=self.pe.get_dense_pe(),
                  sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense)
        kw.update({k: v for k, v in extra.items() if k in allowed})
        return kw

    def decode(self, b: torch.Tensor):
        """-> low-res logits of every mask token (N, n_mask, h, w) and their
        predicted IoU (N, n_mask): MaskDecoder.forward before its slicing."""
        out = self.md.predict_masks(**self._kwargs(b, self._pm_kw))
        return out[0], out[1]

    def official(self, b: torch.Tensor, multimask: bool) -> torch.Tensor:
        """MaskDecoder.forward()'s own low-res logits (N, 1 or 3, h, w)."""
        kw = self._kwargs(b, self._fw_kw)
        return self.md(**kw, multimask_output=multimask)[0]

    def pick(self, low_all: torch.Tensor, iou_all: torch.Tensor, multimask: bool):
        """The mask forward() returns, and its token index. multimask: argmax
        of predicted IoU over tokens 1..; single: token 0, or with the
        stability fallback the same argmax where token 0's mask is unstable
        (MaskDecoder._dynamic_multimask_via_stability)."""
        ar = torch.arange(low_all.shape[0], device=low_all.device)
        best = iou_all[:, 1:].argmax(1) + 1
        if multimask:
            head = best
        elif self.dynamic and not self.md.training:
            stable = self.md._get_stability_scores(low_all[:, 0]) >= self.stab_thresh
            head = torch.where(stable, torch.zeros_like(best), best)
        else:
            head = torch.zeros_like(best)
        return low_all[ar, head], head

    def masks(self, low: torch.Tensor) -> torch.Tensor:
        """(N, h, w) low-res logits -> bool masks at the original resolution."""
        if self.sam1:
            up = self.model.postprocess_masks(low[:, None], self.p.input_size, self.p.original_size)
        else:
            from heatmaps.comp_hw_smoothed import get_original_size
            up = self.p._transforms.postprocess_masks(low[:, None], get_original_size(self.p))
        return (up > self.thr)[:, 0]

    def public(self, boxes_1024: np.ndarray, multimask: bool) -> torch.Tensor:
        """The predictor's own public call on the same boxes -> bool masks (N, H, W)."""
        if self.sam1:
            b = self.model_boxes(boxes_1024)
            m, s, _ = self.p.predict_torch(None, None, boxes=b, multimask_output=multimask)
            if multimask:
                m = m[torch.arange(len(m)), s.argmax(1)]
            else:
                m = m[:, 0]
            return m.bool()
        m, s, _ = self.p.predict(point_coords=None, point_labels=None, box=self.original_boxes(boxes_1024),
                                 multimask_output=multimask, return_logits=False)
        m, s = np.asarray(m), np.asarray(s)
        if m.ndim == 3:                       # a single box
            m, s = m[None], s[None]
        m = m[np.arange(len(m)), s.argmax(1)] if multimask else m[:, 0]
        return torch.from_numpy(m > 0)


# ---------------------------------------------------------------------------
# Sites, hooks, edits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Site:
    id: str
    paths: tuple       # module paths under predictor.model whose outputs are steered / recorded
    kind: str          # sparse | tok | img | vec
    groups: tuple


def build_sites(dec: Decoder) -> list[Site]:
    md = dec.md_path
    out = [Site("prompt_tok", (dec.pe_path,), "sparse", ("prompt",))]
    for li in range(dec.depth):
        p = f"{md}.transformer.layers.{li}"
        for k in (1, 2, 3):
            out.append(Site(f"L{li + 1}.norm{k}", (f"{p}.norm{k}",), "tok", TOKEN_GROUPS))
        out.append(Site(f"L{li + 1}.norm4", (f"{p}.norm4",), "img", ("box",)))
    out.append(Site("final_norm", (f"{md}.transformer.norm_final_attn",), "tok", FINAL_GROUPS))
    out.append(Site("hyper", tuple(f"{md}.output_hypernetworks_mlps.{i}" for i in range(dec.n_mask)), "vec", ("all",)))
    return out


def site_rows(dec: Decoder, site: Site, group: str):
    """Token rows a group steers, or None for the whole tensor."""
    if site.kind == "tok" and group != "all":
        return dec.rows[group]
    return None


class Ctx:
    def __init__(self, runner: "Runner", boxes: torch.Tensor, sl: slice):
        self.r, self.sl, self.boxes = runner, sl, boxes[sl]
        self._cells = None

    @property
    def cells(self):
        if self._cells is None:
            self._cells = self.r.cells(self.boxes)
        return self._cells


class Runner:
    """Decoder passes on one image's boxes, in fixed chunks, with forward hooks
    that edit (steer) and / or record sites."""

    def __init__(self, dec: Decoder, sites, batch: int):
        self.d, self.batch = dec, batch
        self.sites = OrderedDict((s.id, s) for s in sites)
        self.mods = {p: dec.model.get_submodule(p) for s in sites for p in s.paths}
        self.fired = Counter()
        self.chunks = 0

    @torch.inference_mode()
    def forward(self, b: torch.Tensor, edits=None, record=None):
        """b (N, 4) in the model's frame -> (low_all (N, n_mask, h, w), iou_all
        (N, n_mask)). edits {site: fn(t, ctx, i) -> t}, record {site: fn(t, ctx, i)}
        (i = which of the site's modules; record sees the edited tensor)."""
        edits, record = edits or {}, record or {}
        holder, handles = {}, []
        for sid in dict.fromkeys([*edits, *record]):
            site = self.sites[sid]
            for i, path in enumerate(site.paths):
                handles.append(self.mods[path].register_forward_hook(
                    self._hook(site, i, path, edits.get(sid), record.get(sid), holder)))
        lows, ious = [], []
        before = Counter(self.fired)
        try:
            for s0 in range(0, len(b), self.batch):
                sl = slice(s0, min(s0 + self.batch, len(b)))
                holder["ctx"] = Ctx(self, b, sl)
                low, iou = self.d.decode(b[sl])
                lows.append(low)
                ious.append(iou)
        finally:
            for h in handles:
                h.remove()
        n = math.ceil(len(b) / self.batch)
        self.chunks += n
        for sid in dict.fromkeys([*edits, *record]):
            for path in self.sites[sid].paths:
                if self.fired[path] - before[path] != n:
                    raise CheckFailed(f"hook {path} fired {self.fired[path] - before[path]} times in {n} forwards")
        return torch.cat(lows), torch.cat(ious)

    def _hook(self, site, i, path, edit, rec, holder):
        tup = site.kind == "sparse"

        def fn(_mod, _inp, out):
            self.fired[path] += 1
            t = out[0] if tup else out
            ctx = holder["ctx"]
            if edit is not None:
                t = edit(t, ctx, i)
            if rec is not None:
                rec(t, ctx, i)
            if edit is None:
                return None
            return (t, *out[1:]) if tup else t
        return fn

    def cells(self, boxes: torch.Tensor) -> torch.Tensor:
        """(B, h*w) float: 1 on the image-embedding cells whose centre lies in
        the box (model frame); a box with no centre inside gets the cell under
        its centre (steer_sam_decoder.Runner.cells)."""
        eh, ew = self.d.emb_hw
        sy, sx = self.d.frame / eh, self.d.frame / ew
        cy = (torch.arange(eh, device=boxes.device) + 0.5) * sy
        cx = (torch.arange(ew, device=boxes.device) + 0.5) * sx
        x0, y0, x1, y1 = boxes.unbind(1)
        inx = (cx[None] >= x0[:, None]) & (cx[None] <= x1[:, None])
        iny = (cy[None] >= y0[:, None]) & (cy[None] <= y1[:, None])
        m = iny[:, :, None] & inx[:, None, :]
        empty = (~m.flatten(1).any(1)).nonzero().squeeze(1)
        if empty.numel():
            ci = ((y0 + y1)[empty] / 2 / sy).long().clamp(0, eh - 1)
            cj = ((x0 + x1)[empty] / 2 / sx).long().clamp(0, ew - 1)
            m[empty, ci, cj] = True
        return m.flatten(1).float()


class Recorder:
    """Raw site tensors of every box of one forward (on the device)."""

    def __init__(self, sites):
        self.buf = {s.id: {} for s in sites}

    def fns(self):
        return {sid: (lambda t, ctx, i, _s=sid: self.buf[_s].setdefault(i, []).append(t.detach().clone()))
                for sid in self.buf}

    def raw(self, sid: str) -> torch.Tensor:
        """(N, ...); a site of several modules (hyper) stacked on dim 1."""
        parts = [torch.cat(self.buf[sid][i]) for i in sorted(self.buf[sid])]
        return parts[0] if len(parts) == 1 else torch.stack(parts, dim=1)


def feature(site: Site, raw: torch.Tensor, cells: torch.Tensor) -> np.ndarray:
    """What vectors and probes are built from (float32, CPU): the tensor as it
    is, the image stream pooled over the box's cells."""
    if site.kind == "img":
        return ((raw * cells[..., None]).sum(1) / cells.sum(1, keepdim=True)).float().cpu().numpy()
    return raw.float().cpu().numpy()


def edit_rows(W):
    return lambda t, ctx, i: t + W.to(t.dtype)


def edit_vec(W):
    return lambda t, ctx, i: t + W[i].to(t.dtype)


def edit_cells(w):
    return lambda t, ctx, i: t + (ctx.cells[..., None] * w).to(t.dtype)


def edit_patch(site: Site, D: torch.Tensor, rows):
    """Oracle: the best box's tensor (or its token rows) for every box."""
    def fn(t, ctx, i):
        d = D[i] if site.kind == "vec" else D
        if rows is None:
            return d.expand_as(t).clone()
        t = t.clone()
        t[:, rows] = d[rows]
        return t
    return fn


def zero_edit(t, ctx, i):
    return t + torch.zeros_like(t)


@dataclass(frozen=True)
class Cfg:
    cid: int
    site: str
    group: str
    method: str        # dm | rand | oracle
    alpha: float


def build_configs(sites, alphas) -> list[Cfg]:
    cfgs: list[Cfg] = []
    for s in sites:
        for g in s.groups:
            for m in METHODS:
                for a in alphas:
                    cfgs.append(Cfg(len(cfgs), s.id, g, m, float(a)))
        for g in s.groups:
            cfgs.append(Cfg(len(cfgs), s.id, g, "oracle", math.nan))
    return cfgs


def make_edit(cfg: Cfg, site: Site, dec: Decoder, V: dict, donors: dict):
    rows = site_rows(dec, site, cfg.group)
    if cfg.method == "oracle":
        return edit_patch(site, donors[site.id], rows)
    W = cfg.alpha * V[site.id][cfg.method]
    if site.kind == "img":
        return edit_cells(W)
    if site.kind == "vec":
        return edit_vec(W)
    if rows is not None:
        keep = torch.zeros(W.shape[0], 1, device=W.device, dtype=W.dtype)
        keep[rows] = 1
        W = W * keep
    return edit_rows(W)


# ---------------------------------------------------------------------------
# Vectors and probes
# ---------------------------------------------------------------------------

def random_directions(sites, feats: dict, seed: int) -> dict:
    """One fixed random direction per site: unit norm per token row (tokens,
    corners, hypernet rows), unit norm overall for the image stream."""
    R = {}
    for i, s in enumerate(sites):
        u = np.random.default_rng([seed, 101, i]).standard_normal(feats[s.id].shape[1:]).astype(np.float32)
        u /= np.linalg.norm(u, axis=-1, keepdims=True) if s.kind != "img" else np.linalg.norm(u)
        R[s.id] = u
    return R


def image_dms(X: np.ndarray, df: pd.DataFrame):
    """Per image mean(top50) - mean(bottom50) -> (stems, D (n_img, ...), boxes used)."""
    stems, D, used = [], [], 0
    img, grp = df.image_name.to_numpy(), df.group.to_numpy()
    for stem in sorted(set(img)):
        idx = np.where(img == stem)[0]
        top, bot = idx[grp[idx] == "top50"], idx[grp[idx] == "bottom50"]
        if len(top) == 0 or len(bot) == 0:
            raise CheckFailed(f"{stem}: an empty half ({len(bot)} bottom50, {len(top)} top50)")
        stems.append(stem)
        D.append(X[top].mean(0) - X[bot].mean(0))
        used += len(top) + len(bot)
    return stems, np.stack(D).astype(np.float32), used


def scaled_rand(site: Site, u: np.ndarray, dm: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(dm, axis=-1, keepdims=True) if site.kind != "img" else np.linalg.norm(dm)
    return (u * n).astype(np.float32)


def build_vectors(sites, feats: dict, df: pd.DataFrame, fold_of: dict, k: int, R: dict):
    """-> V {"all" | fold: {site: {"dm", "rand"}}}, the per-image DMs, boxes used."""
    V = {"all": {}, **{f: {} for f in range(k)}}
    per_img, used = {}, None
    for s in sites:
        stems, D, n = image_dms(feats[s.id], df)
        used = n
        per_img[s.id] = (stems, D)
        fold = np.array([fold_of[x] for x in stems])
        for key, sel in [("all", np.ones(len(stems), bool)), *[(f, fold != f) for f in range(k)]]:
            dm = D[sel].mean(0)
            V[key][s.id] = {"dm": dm.astype(np.float32), "rand": scaled_rand(s, R[s.id], dm)}
    return V, per_img, used


def vectors_to_device(V: dict, device) -> dict:
    return {sid: {m: torch.as_tensor(v, device=device) for m, v in d.items()} for sid, d in V.items()}


def vectors_sha256(V: dict) -> str:
    h = hashlib.sha256()
    for s in sorted(V):
        for m in sorted(V[s]):
            h.update(f"{s}/{m}".encode())
            h.update(np.ascontiguousarray(V[s][m], dtype=np.float32).tobytes())
    return h.hexdigest()


def vector_stats(sites, dec: Decoder, V: dict, feats: dict, k: int) -> pd.DataFrame:
    rows = []
    for s in sites:
        X = feats[s.id]
        for g in s.groups:
            r = site_rows(dec, s, g)
            sel = (lambda a: a[..., r, :]) if r is not None else (lambda a: a)
            hn = float(np.linalg.norm(sel(X).reshape(len(X), -1), axis=1).mean())
            v = sel(V["all"][s.id]["dm"]).ravel()
            fv = [sel(V[f][s.id]["dm"]).ravel() for f in range(k)]
            cos = [float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), EPS))
                   for i, a in enumerate(fv) for b in fv[i + 1:]]
            rows.append({"site": s.id, "kind": s.kind, "group": g, "dim": int(v.size),
                         "v_norm": float(np.linalg.norm(v)), "h_norm": hn,
                         "dose_alpha1": float(np.linalg.norm(v) / max(hn, EPS)),
                         "fold_cos_mean": float(np.mean(cos)) if cos else math.nan,
                         "fold_cos_min": float(np.min(cos)) if cos else math.nan})
    return pd.DataFrame(rows)


def config_table(cfgs, sites, dec: Decoder, V: dict, feats: dict) -> pd.DataFrame:
    """dose = ||alpha v|| / mean ||h|| over the A boxes, on the steered rows
    (box-pooled for the image stream)."""
    by = {s.id: s for s in sites}
    rows = []
    for c in cfgs:
        s = by[c.site]
        dose = math.nan
        if c.method != "oracle":
            r = site_rows(dec, s, c.group)
            X, v = feats[s.id], V["all"][s.id][c.method]
            if r is not None:
                X, v = X[:, r], v[r]
            hn = np.linalg.norm(X.reshape(len(X), -1), axis=1).mean()
            dose = float(c.alpha * np.linalg.norm(v) / max(hn, EPS))
        rows.append({"cid": c.cid, "site": c.site, "kind": s.kind, "group": c.group, "method": c.method,
                     "alpha": c.alpha, "dose": dose})
    return pd.DataFrame(rows)


def fit_logreg(X: np.ndarray, y: np.ndarray, C: float, iters: int):
    """L2 logistic regression with balanced class weights on standardised
    features -- sklearn's objective 0.5 ||w||^2 + C sum_i s_i log(1 + e^(-y_i f_i)),
    LBFGS in float64. -> a scoring function."""
    with torch.inference_mode(False), torch.enable_grad():
        Xt = torch.as_tensor(X, dtype=torch.float64)
        mu, sd = Xt.mean(0), Xt.std(0).clamp_min(1e-8)
        Z = (Xt - mu) / sd
        yy = torch.as_tensor(y, dtype=torch.float64) * 2 - 1
        n1, n0 = float((yy > 0).sum()), float((yy < 0).sum())
        sw = torch.where(yy > 0, torch.tensor(len(yy) / (2 * n1)), torch.tensor(len(yy) / (2 * n0))).double()
        w = torch.zeros(Z.shape[1] + 1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], lr=1.0, max_iter=iters, tolerance_grad=1e-9, tolerance_change=1e-12,
                                history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            f = Z @ w[:-1] + w[-1]
            loss = 0.5 * (w[:-1] ** 2).sum() + C * (sw * F.softplus(-yy * f)).sum()
            loss.backward()
            return loss
        opt.step(closure)
        wd = w.detach()

    def score(Xn: np.ndarray) -> np.ndarray:
        Zn = (torch.as_tensor(Xn, dtype=torch.float64) - mu) / sd
        return (Zn @ wd[:-1] + wd[-1]).numpy()
    return score


def auc(y: np.ndarray, s: np.ndarray) -> float:
    y, s = np.asarray(y, bool), np.asarray(s, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return math.nan
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_within(y: np.ndarray, s: np.ndarray, img: np.ndarray) -> float:
    vals = [auc(y[img == k], s[img == k]) for k in np.unique(img)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else math.nan


def probe_X(site: Site, X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1).astype(np.float64)


# ---------------------------------------------------------------------------
# Graph of the model's decoder (graph.json, drawn by steer_models_report.py)
# ---------------------------------------------------------------------------

LANES = [{"id": "S", "label": "token sub-layers"}, {"id": "T", "label": "tokens (residual stream)"},
         {"id": "X", "label": "cross-attention"}, {"id": "I", "label": "image embedding (residual stream)"}]
BLOCK_PARTS = ("self_attn", "norm1", "cross_attn_token_to_image", "norm2", "mlp", "norm3",
               "cross_attn_image_to_token", "norm4")


def graph_spec(dec: Decoder) -> dict:
    """The prompt encoder + mask decoder as a dataflow graph, node ids = site
    ids (activation_flow_graph.build_spec's layout, extended by what the
    loaded model has: object-score token, pad token, high-res features, the
    stability fallback). `key` = the module path a node stands for."""
    nodes, edges, groups = [], [], []

    def node(nid, label, lane, row, code, kind="var", key="", note=""):
        nodes.append({"id": nid, "label": label, "lane": lane, "row": row, "code": code, "kind": kind,
                      "key": key, "note": note})

    def edge(src, dst, role, label=""):
        edges.append({"id": f"{src}>{dst}", "src": src, "dst": dst, "role": role, "label": label})

    S, Tl, X, I = 0, 1, 2, 3
    pe, md, sam2 = dec.pe_path, dec.md_path, not dec.sam1
    eh, ew = dec.emb_hw
    T, s, n = dec.T, dec.s, dec.n_mask
    node("box", "box prompt", S, 0,
         "SamPredictor.predict_torch(boxes=...)  [1024 frame, xyxy]" if dec.sam1 else
         f"predictor.predict(box=...) -> 2 points, labels 2, 3, boxes=None  [{int(dec.frame)} frame]")
    node("img_emb", "image embedding", I, 0,
         "image_encoder(...) -> predictor.features" if dec.sam1 else
         "backbone -> predictor._features['image_embed'] (+ no_mem_embed)", kind="const",
         note=f"{eh} x {ew} x 256; the same for every box of the image.")
    node("img_pe", "image positional encoding", X, 0, f"{pe}.get_dense_pe()", kind="const", key=f"{pe}.pe_layer")
    node("corner_pe", "positional encoding (corners)", S, 1, f"{pe}.pe_layer.forward_with_coords(corners + 0.5)")
    node("corner_emb", "learned corner embeddings", Tl, 1, f"{pe}.point_embeddings[2], [3]", kind="const")
    if dec.n_sparse > 2:
        node("pad_emb", "'not a point' (pad)", Tl, 2, f"{pe}.not_a_point_embed  (pad token: boxes=None)",
             kind="const", note="The box comes in as two points, so the prompt encoder pads one 'not a point' token.")
    node("prompt_tok", "prompt tokens", S, 2, f"{pe}(...) -> sparse_embeddings ({dec.n_sparse} x 256)", key=pe)
    node("no_mask", "'no mask' embedding", X, 2, f"{pe}.no_mask_embed -> dense_embeddings", kind="const")
    node("out_tok", "output tokens", S, 3, ("obj_score_token, " if s else "") + "iou_token, mask_tokens",
         kind="const")
    node("tokens", "output tokens + prompt tokens", Tl, 3,
         f"{md}.predict_masks: tokens = cat(output_tokens, sparse)  ({T} tokens: " + ", ".join(dec.token_names) + ")",
         note="Also re-added before every attention (the prompt-token re-add is not drawn).")
    node("src", "image embedding + 'no mask'", I, 3, f"{md}.predict_masks: src = image_embeddings + dense",
         kind="const")
    edge("box", "corner_pe", "flow", "+0.5, / frame, Fourier PE")
    edge("corner_pe", "prompt_tok", "flow", "+")
    edge("corner_emb", "prompt_tok", "flow", "+")
    if dec.n_sparse > 2:
        edge("pad_emb", "prompt_tok", "flow", "concat")
    edge("prompt_tok", "tokens", "flow", "concat")
    edge("out_tok", "tokens", "flow", "concat")
    edge("img_emb", "src", "flow", "+")
    edge("no_mask", "src", "flow", "+")
    groups.append({"label": "prompt encoder", "code": pe, "row0": 0, "row1": 2, "lane0": 0, "lane1": 1})

    q_prev, k_prev = "tokens", "src"
    for li in range(dec.depth):
        r0 = 4 + 8 * li
        p = f"{md}.transformer.layers.{li}"
        nn_ = f"L{li + 1}."
        first = li == 0
        node(nn_ + "self", "self attn.", S, r0, f"{p}.self_attn", key=f"{p}.self_attn",
             note="Layer 1 (skip_first_layer_pe): q = k = v = tokens, no residual." if first else "")
        node(nn_ + "norm1", "norm (no residual)" if first else "add & norm", Tl, r0 + 1, f"{p}.norm1", key=f"{p}.norm1")
        node(nn_ + "t2i", "token to image attn.", X, r0 + 2, f"{p}.cross_attn_token_to_image",
             key=f"{p}.cross_attn_token_to_image")
        node(nn_ + "norm2", "add & norm", Tl, r0 + 3, f"{p}.norm2", key=f"{p}.norm2")
        node(nn_ + "mlp", "mlp", S, r0 + 4, f"{p}.mlp", key=f"{p}.mlp")
        node(nn_ + "norm3", "add & norm", Tl, r0 + 5, f"{p}.norm3", key=f"{p}.norm3")
        node(nn_ + "i2t", "image to token attn.", X, r0 + 6, f"{p}.cross_attn_image_to_token",
             key=f"{p}.cross_attn_image_to_token")
        node(nn_ + "norm4", "add & norm", I, r0 + 7, f"{p}.norm4  (updated image embedding, {eh * ew} x 256)",
             key=f"{p}.norm4")
        if first:
            edge(q_prev, nn_ + "self", "flow", "q = k = v")
            edge(nn_ + "self", nn_ + "norm1", "flow", "no residual")
        else:
            edge(q_prev, nn_ + "self", "flow", "q, k (+PE), v")
            edge(q_prev, nn_ + "norm1", "res", "residual")
            edge(nn_ + "self", nn_ + "norm1", "flow", "+")
        edge(nn_ + "norm1", nn_ + "t2i", "flow", "q")
        edge(k_prev, nn_ + "t2i", "flow", "k, v")
        edge("img_pe", nn_ + "t2i", "pe", "image PE (k)")
        edge(nn_ + "norm1", nn_ + "norm2", "res", "residual")
        edge(nn_ + "t2i", nn_ + "norm2", "flow", "+")
        edge(nn_ + "norm2", nn_ + "mlp", "flow", "")
        edge(nn_ + "norm2", nn_ + "norm3", "res", "residual")
        edge(nn_ + "mlp", nn_ + "norm3", "flow", "+")
        edge(nn_ + "norm3", nn_ + "i2t", "flow", "k, v (tokens)")
        edge(k_prev, nn_ + "i2t", "flow", "q (image)")
        edge("img_pe", nn_ + "i2t", "pe", "image PE (q)")
        edge(k_prev, nn_ + "norm4", "res", "residual")
        edge(nn_ + "i2t", nn_ + "norm4", "flow", "+")
        groups.append({"label": f"decoder layer {li + 1}", "code": p, "row0": r0, "row1": r0 + 7,
                       "lane0": 0, "lane1": 3})
        q_prev, k_prev = nn_ + "norm3", nn_ + "norm4"

    rf = 4 + 8 * dec.depth
    pf = f"{md}.transformer"
    node("final_t2i", "token to image attn.", X, rf, f"{pf}.final_attn_token_to_image",
         key=f"{pf}.final_attn_token_to_image")
    node("final_norm", "add & norm", Tl, rf + 1, f"{pf}.norm_final_attn", key=f"{pf}.norm_final_attn")
    edge(q_prev, "final_t2i", "flow", "q")
    edge(k_prev, "final_t2i", "flow", "k, v")
    edge("img_pe", "final_t2i", "pe", "image PE (k)")
    edge(q_prev, "final_norm", "res", "residual")
    edge("final_t2i", "final_norm", "flow", "+")
    groups.append({"label": "final attention", "code": f"{pf}.final_attn_token_to_image / norm_final_attn",
                   "row0": rf, "row1": rf + 1, "lane0": 0, "lane1": 3})

    ro = rf + 2
    lh, lw = 4 * eh, 4 * ew
    node("iou_tok", "IoU output token", S, ro, f"hs[:, {s}]")
    node("mask_tok", f"mask output tokens 0–{n - 1}", Tl, ro, f"hs[:, {s + 1}:{s + 1 + n}]")
    if sam2:
        node("hires", "high-res features", I, ro, "backbone FPN levels 0, 1 -> conv_s0 / conv_s1 (in set_image)",
             kind="const", note="feat_s1 is added after the first transposed conv, feat_s0 after the second.")
    node("upscaled", "2x conv. trans.", I, ro + (1 if sam2 else 0),
         f"{md}.output_upscaling" + ("  (+ feat_s1, + feat_s0)" if sam2 else ""), key=f"{md}.output_upscaling")
    node("iou_scores", "IoU scores", S, ro + 1, f"{md}.iou_prediction_head -> iou_pred ({n})"
         + (", sigmoid" if dec.iou_sigmoid else ""),
         key=f"{md}.iou_prediction_head")
    node("hyper", f"{n} MLPs -> dynamic classifiers", Tl, ro + 1, f"{md}.output_hypernetworks_mlps.0–{n - 1}",
         key=f"{md}.output_hypernetworks_mlps")
    if s:
        node("obj_score", "object score", S, ro + 2, f"{md}.pred_obj_score_head(hs[:, 0])",
             key=f"{md}.pred_obj_score_head", note="Computed, but the image predictor does not use it.")
    node("low_res", "point-wise product", X, ro + 2, f"hyper_in @ upscaled -> masks ({n} x {lh} x {lw} logits)")
    fall = (" unless its mask is unstable (stability < %g: best of 1–%d by IoU)" % (dec.stab_thresh, n - 1)
            if dec.dynamic else "")
    node("select", "mask choice", X, ro + 3, f"multimask=False: token 0{fall}; multimask=True: argmax IoU of 1–{n - 1}")
    node("mask_logits", "upscaled mask logits", X, ro + 4,
         "Sam.postprocess_masks -> original size" if dec.sam1 else "predictor._transforms.postprocess_masks -> original size")
    node("mask", "mask", X, ro + 5, f"logits > {dec.thr:g}")
    edge("final_norm", "iou_tok", "flow", f"token {s}")
    edge("final_norm", "mask_tok", "flow", f"tokens {s + 1}–{s + n}")
    if s:
        edge("final_norm", "obj_score", "flow", "token 0")
    edge(k_prev, "upscaled", "flow", "image embedding")
    if sam2:
        edge("hires", "upscaled", "flow", "+")
    edge("iou_tok", "iou_scores", "flow", "")
    edge("mask_tok", "hyper", "flow", "")
    edge("hyper", "low_res", "flow", "classifier weights")
    edge("upscaled", "low_res", "flow", "upscaled embedding")
    edge("low_res", "select", "flow", "")
    edge("iou_scores", "select", "flow", "argmax IoU")
    edge("select", "mask_logits", "flow", "")
    edge("mask_logits", "mask", "flow", "> thr")
    groups.append({"label": "output heads", "code": md, "row0": ro, "row1": ro + 5, "lane0": 0, "lane1": 3})

    ids = [x["id"] for x in nodes]
    assert len(ids) == len(set(ids)), "duplicate node ids"
    assert all(e["src"] in ids and e["dst"] in ids for e in edges), "edge to an unknown node"
    return {"nodes": nodes, "edges": edges, "groups": groups, "lanes": LANES}


def graph_coverage(dec: Decoder, spec: dict, b: torch.Tensor) -> dict:
    """Every module under the prompt encoder and the mask decoder that fires in
    a decoder pass must be a drawn node, inside one, or a container of one."""
    keys = [n["key"] for n in spec["nodes"] if n["key"]]
    fired, handles = [], []
    for root in (dec.pe_path, dec.md_path):
        for name, mod in dec.model.get_submodule(root).named_modules():
            path = f"{root}.{name}" if name else root
            handles.append(mod.register_forward_hook(lambda m, i, o, _p=path: fired.append(_p)))
    try:
        with torch.inference_mode():
            dec.decode(b)
    finally:
        for h in handles:
            h.remove()
    fired = list(dict.fromkeys(fired))

    def covered(p):
        return any(p == k or p.startswith(k + ".") or k.startswith(p + ".") for k in keys)
    missing = [k for k in keys if k not in fired and not any(f.startswith(k + ".") for f in fired)]
    return {"fired": len(fired), "undrawn": [p for p in fired if not covered(p)], "drawn_not_fired": missing,
            "ok": not [p for p in fired if not covered(p)] and not missing}


# ---------------------------------------------------------------------------
# Per image
# ---------------------------------------------------------------------------

def load_item(dec: Decoder, root: Path, dataset: str, half: str, stem: str, recs, fold: int):
    import steer_user_study as SUS
    cases, boxes, gt = SUS.load_image_cases(dec.p, root, stem, recs[stem], dec.device)
    rows = [{"dataset": dataset, "image_name": stem, "kind": c["kind"], "user": c["user"],
             "attempt": c["attempt"], "half": half, "fold": fold} for c in cases]
    return {"stem": stem, "dataset": dataset, "half": half, "fold": fold, "rows": rows,
            "boxes": boxes, "b": dec.model_boxes(boxes), "gt": gt,
            "best_box": cases[0]["best_box"] if cases else None}       # the GT's tight box, 1024 frame


def score_modes(dec: Decoder, low_all, iou_all, gt):
    out = {}
    for mm in MODES:
        low, head = dec.pick(low_all, iou_all, bool(mm))
        out[mm] = (batch_iou(dec.masks(low), gt).cpu().numpy(), head.cpu().numpy().astype(np.int8))
    return out


def batch_iou(m: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """heatmaps/comp_hw.py batch_iou_torch: inter / (union + 1e-6)."""
    gt = gt[None] if gt.dim() == 2 else gt
    inter = (m & gt).flatten(1).sum(1).float()
    union = (m | gt).flatten(1).sum(1).float()
    return inter / (union + 1e-6)


def check_zero(runner: Runner, b, low0, iou0, site_ids, checks: dict, where: str):
    z = checks["zero_dose"]
    for sid in site_ids:
        low, iou = runner.forward(b, edits={sid: zero_edit})
        z["forwards"] += 1
        if torch.equal(low, low0) and torch.equal(iou, iou0):
            z["bit_exact"] += 1
            continue
        z["ok"] = False
        z["failures"].append(f"{where}/{sid}")
        raise CheckFailed(f"zero dose at {where}/{sid}")


def check_official(dec: Decoder, item: dict, low_all, iou_all, base: dict, checks: dict):
    """Own decoder pass vs MaskDecoder.forward (stop on any difference) and
    vs the predictor's public call (reported)."""
    c = checks["own_forward"]
    with torch.inference_mode():
        for mm in MODES:
            low, head = dec.pick(low_all, iou_all, bool(mm))
            off = dec.official(item["b"], bool(mm))
            ref = off[torch.arange(len(off)), head - 1] if mm else off[:, 0]
            same = torch.equal(low, ref)
            c["compared"] += 1
            c["bit_exact"] += int(same)
            if not same:
                c["ok"] = False
                c["failures"].append(f"{item['dataset']}/{item['stem']}/mm{mm}: max |d| "
                                     f"{(low - ref).abs().max().item():.3g}")
                raise CheckFailed(f"own decoder pass != MaskDecoder.forward at {item['stem']} mm{mm}")
            p = checks["public_api"]
            try:
                pub = dec.public(item["boxes"], bool(mm)).to(item["gt"].device)
            except Exception as e:          # reported, not fatal: own_forward above is the hard check
                p["where"].append(f"{item['dataset']}/{item['stem']}/mm{mm}: {type(e).__name__}: {e}")
                continue
            iou_pub = batch_iou(pub, item["gt"]).cpu().numpy()
            d = float(np.abs(iou_pub - base[mm][0]).max())
            p["compared"] += len(iou_pub)
            p["max_abs_iou_diff"] = max(p["max_abs_iou_diff"], d)
            p["n_over_1e-4"] += int((np.abs(iou_pub - base[mm][0]) > 1e-4).sum())
            if d > 1e-4:
                p["where"].append(f"{item['dataset']}/{item['stem']}/mm{mm}: {d:.3g}")
            # one box per call, as the reference csvs were made: the box frame
            # must match exactly; what is left is batch-dependent numerics (TF32)
            q = checks["public_single"]
            try:
                one = torch.stack([dec.public(item["boxes"][j:j + 1], bool(mm))[0] for j in range(len(item["boxes"]))])
            except Exception as e:
                q["where"].append(f"{item['dataset']}/{item['stem']}/mm{mm}: {type(e).__name__}: {e}")
                continue
            dd = np.abs(batch_iou(one.to(item["gt"].device), item["gt"]).cpu().numpy() - base[mm][0])
            q["compared"] += len(dd)
            q["max_abs_iou_diff"] = max(q["max_abs_iou_diff"], float(dd.max()))
            q["n_over_1e-3"] += int((dd > 1e-3).sum())
            if dd.max() > 1e-3:
                q["where"].append(f"{item['dataset']}/{item['stem']}/mm{mm}: {dd.max():.3g}")


def base_pass(runner: Runner, dec: Decoder, sites, item: dict, checks: dict, full_checks: bool):
    """Unsteered forward with every site recorded -> IoU / head per mode, the
    features, the best box's tensors (oracle donors, on CPU)."""
    rec = Recorder(sites)
    low0, iou0 = runner.forward(item["b"], record=rec.fns())
    base = score_modes(dec, low0, iou0, item["gt"])
    where = f"{item['dataset']}/{item['stem']}"
    if full_checks:
        check_official(dec, item, low0, iou0, base, checks)
        check_zero(runner, item["b"], low0, iou0, [s.id for s in sites], checks, where)
    else:
        check_zero(runner, item["b"], low0, iou0, ["L2.norm2"], checks, where)
    cells = runner.cells(item["b"])
    best = int(np.argmax(base[0][0]))
    feats, donors = {}, {}
    for s in sites:
        raw = rec.raw(s.id)
        feats[s.id] = feature(s, raw, cells)
        donors[s.id] = raw[best].detach().cpu()
    item.update(base=base, best=best, low0=low0, iou0=iou0)
    return feats, donors


def steer_pass(runner: Runner, dec: Decoder, sites, item: dict, cfgs, V: dict, donors: dict):
    """Every config on the image's boxes -> {mode: (iou (N, n_cfg), head (N, n_cfg))}."""
    by = {s.id: s for s in sites}
    Dd = {sid: t.to(dec.device) for sid, t in donors.items()}
    N = len(item["rows"])
    out = {mm: (np.zeros((N, len(cfgs)), np.float32), np.zeros((N, len(cfgs)), np.int8)) for mm in MODES}
    for c in cfgs:
        low_all, iou_all = runner.forward(item["b"], edits={c.site: make_edit(c, by[c.site], dec, V, Dd)})
        for mm, (iou, head) in score_modes(dec, low_all, iou_all, item["gt"]).items():
            out[mm][0][:, c.cid] = iou
            out[mm][1][:, c.cid] = head
    return out


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def read_reference(path: str, use: str) -> pd.DataFrame | None:
    import selector_data as SD
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"--reference_csv not found: {p}")
    ref = pd.read_csv(p, dtype={"image_name": str, "user": str, "attempt": str, "kind": str})
    ref = ref[ref.kind.isin(list(use))] if "kind" in ref.columns else ref
    ref[SD.CASE_KEYS] = ref[SD.CASE_KEYS].fillna("")
    return ref


def repro(df: pd.DataFrame, ref: pd.DataFrame | None, col: str, ref_col: str, tol: float) -> dict:
    import selector_data as SD
    if ref is None or ref_col not in ref.columns:
        return {"available": False}
    m = df.merge(ref[SD.CASE_KEYS + [ref_col]], on=SD.CASE_KEYS, how="left")
    have = m[ref_col].notna()
    d = (m.loc[have, col] - m.loc[have, ref_col]).abs()
    return {"available": True, "column": ref_col, "boxes": int(len(m)), "matched": int(have.sum()),
            "max_abs": float(d.max()) if len(d) else math.nan, "n_over_tol": int((d > tol).sum()),
            "frac_over_tol": float((d > tol).mean()) if len(d) else math.nan, "tol": tol,
            "mean_ours": float(m.loc[have, col].mean()) if have.any() else math.nan,
            "mean_ref": float(m.loc[have, ref_col].mean()) if have.any() else math.nan}


def ref_values(df: pd.DataFrame, ref: pd.DataFrame | None, ref_col: str) -> np.ndarray:
    import selector_data as SD
    if ref is None or ref_col not in ref.columns:
        return np.full(len(df), np.nan)
    m = df[SD.CASE_KEYS].merge(ref[SD.CASE_KEYS + [ref_col]], on=SD.CASE_KEYS, how="left")
    return m[ref_col].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def package_versions() -> dict:
    out = {"torch": torch.__version__}
    for name in ("segment_anything", "sam2", "sam3"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "installed")
        except Exception:
            pass
    return out


def run(args) -> int:
    import selector_data as SD
    import steer_user_study as SUS
    from heatmaps.comp_hw_smoothed import load_model
    from steer_sam_decoder import assign_folds

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    device = torch.device(args.device)
    datasets = list(dict.fromkeys(["COCO", *args.datasets]))
    checks = {"zero_dose": {"forwards": 0, "bit_exact": 0, "ok": True, "failures": []},
              "own_forward": {"compared": 0, "bit_exact": 0, "ok": True, "failures": []},
              "public_api": {"compared": 0, "max_abs_iou_diff": 0.0, "n_over_1e-4": 0, "where": []},
              "public_single": {"compared": 0, "max_abs_iou_diff": 0.0, "n_over_1e-3": 0, "where": []}}

    predictor = load_model(model_name=args.model_name, model_type=args.model_type,
                           checkpoint=args.checkpoint_path, device=device)
    for prm in predictor.model.parameters():
        prm.requires_grad_(False)
    # sam3.model_builder turns TF32 matmuls on at import (Ampere and newer), and
    # a box's mask then depends on the batch it is decoded in. Every model runs
    # with fp32 matmuls, as SAM-1 / SAM-2.1 do in pishugin-breps (heatmaps/
    # comp_hw.load_sam3_model does the same, commit 7829071); cuDNN keeps torch's default.
    tf32 = {"matmul_after_load": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn": bool(torch.backends.cudnn.allow_tf32)}
    if not args.keep_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
    tf32["matmul_run"] = bool(torch.backends.cuda.matmul.allow_tf32)
    checks["tf32"] = tf32
    dec = Decoder(predictor)
    if dec.training:
        raise CheckFailed("mask decoder is in training mode: the stability fallback would be off")
    sites = build_sites(dec)
    runner = Runner(dec, sites, args.batch)
    cfgs = build_configs(sites, args.alphas)
    spec = graph_spec(dec)
    print(f"{args.model_name}: {dec.layout()['decoder']}, {dec.T} tokens {dec.token_names}, grid {dec.emb_hw}, "
          f"frame {dec.frame:g}, fallback {dec.dynamic} (thr {dec.stab_thresh:g}); {len(sites)} sites, "
          f"{len(cfgs)} configs; TF32 matmul {tf32['matmul_after_load']} after load -> {tf32['matmul_run']}, "
          f"cuDNN {tf32['cudnn']}", flush=True)

    root = Path(args.root)
    recs = SUS.index_study(root, args.use, datasets)
    for d in datasets:
        if not recs[d]:
            raise SystemExit(f"no user masks for {d} under {root}")
    A, B = SUS.split_halves(recs["COCO"].keys(), args.split_seed)
    split_ok = not set(A) & set(B) and len(A) == len(B) == len(recs["COCO"]) // 2
    checks["split"] = {"A": len(A), "B": len(B), "disjoint": not set(A) & set(B), "ok": split_ok}
    if not split_ok:
        raise CheckFailed(f"split: {checks['split']}")
    (out / "split.json").write_text(json.dumps({"seed": args.split_seed, "A": A, "B": B}, indent=1), encoding="utf-8")
    fold_of = assign_folds(A, args.folds, args.seed)
    lim = (lambda xs: list(xs)[:args.limit_images]) if args.limit_images > 0 else list
    A_run = lim(A)
    ref = read_reference(args.reference_csv, args.use)

    # --- pass 1: half A, unsteered, every site recorded --------------------------
    feats_A, items_A, cov = {s.id: [] for s in sites}, [], None
    t0 = time.time()
    for i, stem in enumerate(A_run):
        item = load_item(dec, root, "COCO", "A", stem, recs["COCO"], fold_of[stem])
        if cov is None:
            cov = graph_coverage(dec, spec, item["b"])
            checks["graph"] = cov
            print(f"graph: {cov['fired']} modules fired, undrawn {cov['undrawn']}, drawn but silent "
                  f"{cov['drawn_not_fired']}", flush=True)
        f, _ = base_pass(runner, dec, sites, item, checks, full_checks=i == 0)
        for sid in feats_A:
            feats_A[sid].append(f[sid])
        item["base1"] = {mm: v[0].copy() for mm, v in item["base"].items()}
        for k in ("low0", "iou0", "b", "gt"):
            item.pop(k)
        items_A.append(item)
        print(f"[A 1/2] [{i + 1}/{len(A_run)}] {stem}: {len(item['rows'])} boxes, undef "
              f"{item['base'][0][0].mean():.4f}, head-select {item['base'][1][0].mean():.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    feats_A = {sid: np.concatenate(v) for sid, v in feats_A.items()}
    df_A = pd.DataFrame([r for it in items_A for r in it["rows"]])
    df_A["undef_iou"] = np.concatenate([it["base"][0][0] for it in items_A])
    df_A["group"] = SUS.assign_groups(df_A).to_numpy()
    checks["repro_A"] = repro(df_A, ref, "undef_iou", "undefended_iou", args.repro_tol)
    r = checks["repro_A"]
    if r.get("available"):
        print(f"repro A vs {args.reference_csv}: max |d| {r['max_abs']:.3g}, {r['n_over_tol']}/{r['matched']} boxes "
              f"> {args.repro_tol:g}; mean ours {r['mean_ours']:.4f} vs ref {r['mean_ref']:.4f}", flush=True)
        r["mean_tol"], r["max_frac"] = args.repro_mean_tol, args.repro_max_frac
        r["ok"] = bool(r["matched"] == r["boxes"] and r["frac_over_tol"] <= args.repro_max_frac
                       and abs(r["mean_ours"] - r["mean_ref"]) <= args.repro_mean_tol)
        if not r["ok"] and not args.allow_repro_fail:
            (out / "checks.json").write_text(json.dumps(checks, indent=1, default=_json_default), encoding="utf-8")
            raise CheckFailed("multimask=False IoU on half A does not match the reference csv "
                              f"({r['n_over_tol']}/{r['matched']} boxes off, mean {r['mean_ours']:.4f} vs "
                              f"{r['mean_ref']:.4f}): wrong checkpoint or frame? --allow_repro_fail to go on anyway")

    # --- vectors, doses, probes ---------------------------------------------------
    R = random_directions(sites, feats_A, args.seed)
    V, per_img, used = build_vectors(sites, feats_A, df_A, fold_of, args.folds, R)
    checks["vector_boxes"] = {"used": used, "expected": len(df_A), "images": len(A_run),
                              "ok": used == len(df_A)}
    checks["vector_sha256"] = vectors_sha256(V["all"])
    cfg_df = config_table(cfgs, sites, dec, V, feats_A)
    cfg_df.to_csv(out / "configs.csv", index=False)
    vector_stats(sites, dec, V, feats_A, args.folds).to_csv(out / "vector_stats.csv", index=False)
    npz = {}
    for key, d in V.items():
        for sid, mv in d.items():
            for m, v in mv.items():
                npz[f"{key}/{sid}/{m}"] = v
    np.savez_compressed(out / "vectors.npz", **npz)
    np.savez_compressed(out / "states_A.npz", **{sid: x.astype(np.float16) for sid, x in feats_A.items()},
                        **{f"dimg/{sid}": d for sid, (_, d) in per_img.items()})

    y_A = (df_A.group.to_numpy() == "top50")
    fold_row = df_A.fold.to_numpy()
    probes, probe_scores = {}, {"A": {}}
    for s in sites:
        X = probe_X(s, feats_A[s.id])
        sc = np.full(len(X), np.nan)
        for k in range(args.folds):
            tr, te = fold_row != k, fold_row == k
            if te.any() and len(np.unique(y_A[tr])) == 2:
                sc[te] = fit_logreg(X[tr], y_A[tr], args.probe_C, args.probe_iters)(X[te])
        probe_scores["A"][s.id] = sc
        probes[s.id] = fit_logreg(X, y_A, args.probe_C, args.probe_iters)
    print(f"vectors + probes ({time.time() - t0:.0f}s)", flush=True)

    # --- pass 2: steering on A (fold vectors) and the test sets (all-A vectors) ----
    Vdev = {key: vectors_to_device(d, device) for key, d in V.items()}
    results = []          # (rows df, {mm: (base_iou, base_head, best_iou)}, steer {mm: (iou, head)}, probe dict)
    checks["baseline_repeat"] = {"images": 0, "max_abs_iou_diff": 0.0}
    plan = [("COCO", "A", stem) for stem in A_run] + [("COCO", "B", stem) for stem in lim(B)]
    for d in datasets[1:]:
        plan += [(d, "-", stem) for stem in lim(recs[d].keys())]
    first_of = {}
    t0 = time.time()
    for i, (d, half, stem) in enumerate(plan):
        key = f"{d}_{half}" if d == "COCO" else d
        fold = fold_of[stem] if half == "A" else -1
        item = load_item(dec, root, d, half, stem, recs[d], fold)
        first = key not in first_of
        first_of.setdefault(key, stem)
        feats, donors = base_pass(runner, dec, sites, item, checks, full_checks=first and half != "A")
        if half == "A":
            prev = next(it for it in items_A if it["stem"] == stem)["base1"]
            dd = max(float(np.abs(prev[mm] - item["base"][mm][0]).max()) for mm in MODES)
            checks["baseline_repeat"]["images"] += 1
            checks["baseline_repeat"]["max_abs_iou_diff"] = max(checks["baseline_repeat"]["max_abs_iou_diff"], dd)
            pr = {}
        else:
            pr = {s.id: probes[s.id](probe_X(s, feats[s.id])) for s in sites}
        st = steer_pass(runner, dec, sites, item, cfgs, Vdev[fold if half == "A" else "all"], donors)
        base = {mm: (item["base"][mm][0], item["base"][mm][1], np.full(len(item["rows"]), item["base"][mm][0][item["best"]]))
                for mm in MODES}
        results.append((item["rows"], base, st, pr, half == "A"))
        a1 = next((c.cid for c in cfgs if c.site == "L2.norm2" and c.group == "all" and c.method == "dm"
                   and math.isclose(c.alpha, 1.0)), None)
        msg = f"undef {base[0][0].mean():.4f}, head-select {base[1][0].mean():.4f}"
        if a1 is not None:
            msg += f", L2.norm2 dm a=1: {st[0][0][:, a1].mean():.4f} / {st[1][0][:, a1].mean():.4f}"
        print(f"[2/2 {key}] [{i + 1}/{len(plan)}] {stem}: {msg} ({time.time() - t0:.0f}s)", flush=True)
        del item, feats, donors

    # --- tables per mode ------------------------------------------------------------
    rows = pd.DataFrame([r for res in results for r in res[0]])
    rows["undef_iou"] = np.concatenate([res[1][0][0] for res in results])
    rows["group"] = SUS.assign_groups(rows).to_numpy()
    gcount = rows.groupby(["image_name", "group"]).size().unstack(fill_value=0)
    ok_12_13 = bool(((gcount.get("bottom50", 0) == 12) & (gcount.get("top50", 0) == 13)).all())
    checks["groups"] = {"images": int(len(gcount)), "all_12_13": ok_12_13, "ok": ok_12_13}
    # the A probe scores are out of fold; the others came from the all-A probe
    probe_cols = {}
    for s in sites:
        vals, ia = [], 0
        for res in results:
            n = len(res[0])
            if res[4]:
                vals.append(probe_scores["A"][s.id][ia:ia + n])
                ia += n
            else:
                vals.append(res[3][s.id])
        probe_cols[f"probe_{s.id}"] = np.concatenate(vals)
    if not checks["groups"]["ok"]:
        raise CheckFailed(f"groups are not 12 / 13 on every image: {gcount.head()}")
    ref_cols = {0: "undefended_iou", 1: "headsel_iou"}
    for mm in MODES:
        df = rows.copy()
        df["base_iou"] = np.concatenate([res[1][mm][0] for res in results])
        df["base_head"] = np.concatenate([res[1][mm][1] for res in results])
        df["best_iou"] = np.concatenate([res[1][mm][2] for res in results])
        df["ref_iou"] = ref_values(df, ref, ref_cols[mm])
        for k, v in probe_cols.items():
            df[k] = v
        iou = np.concatenate([res[2][mm][0] for res in results])
        head = np.concatenate([res[2][mm][1] for res in results])
        wide = pd.concat([pd.DataFrame(iou, columns=[f"iou_c{c.cid}" for c in cfgs]),
                          pd.DataFrame(head, columns=[f"head_c{c.cid}" for c in cfgs])], axis=1)
        df = pd.concat([df.reset_index(drop=True), wide], axis=1)
        df.to_csv(out / f"per_case_mm{mm}.csv.gz", index=False, float_format="%.6f")
        checks[f"repro_mm{mm}"] = repro(df.assign(_v=df.base_iou), ref, "_v", ref_cols[mm], args.repro_tol)
        if mm == 0:
            oc = next(c.cid for c in cfgs if c.site == "prompt_tok" and c.method == "oracle")
            checks["oracle_cut_prompt_tok"] = {"max_abs_iou_vs_best": float(np.abs(df[f"iou_c{oc}"] - df.best_iou).max())}

    # --- probe table ------------------------------------------------------------------
    prow = []
    img = rows.image_name.to_numpy()
    keyc = np.where(rows.dataset == "COCO", "COCO_" + rows.half.astype(str), rows.dataset)
    y = rows.group.to_numpy() == "top50"
    for s in sites:
        sc = probe_cols[f"probe_{s.id}"]
        for k in dict.fromkeys(keyc):
            sel = keyc == k
            prow.append({"site": s.id, "dataset": k, "n": int(sel.sum()), "dim": int(probe_X(s, feats_A[s.id][:1]).shape[1]),
                         "auc": auc(y[sel], sc[sel]), "auc_within": auc_within(y[sel], sc[sel], img[sel]),
                         "oof": k == "COCO_A"})
    pd.DataFrame(prow).to_csv(out / "probe.csv", index=False)

    meta = {"script": "steer_models_scan.py", "created": time.strftime("%Y-%m-%d %H:%M"),
            "model_name": args.model_name, "checkpoint": args.checkpoint_path, "device": str(device),
            "versions": package_versions(), "layout": dec.layout(), "args": vars(args),
            "sites": [{"id": s.id, "paths": list(s.paths), "kind": s.kind, "groups": list(s.groups)} for s in sites],
            "datasets": list(dict.fromkeys(keyc)), "images": rows.groupby(keyc).image_name.nunique().to_dict(),
            "boxes": rows.groupby(keyc).size().to_dict(), "folds": args.folds, "alphas": list(args.alphas),
            "n_configs": len(cfgs), "seconds": round(time.time() - t_start)}
    (out / "graph.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=_json_default), encoding="utf-8")
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=_json_default), encoding="utf-8")
    print(f"done in {time.time() - t_start:.0f}s -> {out}; checks: zero dose "
          f"{checks['zero_dose']['bit_exact']}/{checks['zero_dose']['forwards']}, own forward "
          f"{checks['own_forward']['bit_exact']}/{checks['own_forward']['compared']}, public API max |dIoU| "
          f"{checks['public_api']['max_abs_iou_diff']:.3g} (one box per call "
          f"{checks['public_single']['max_abs_iou_diff']:.3g}), baseline repeat "
          f"{checks['baseline_repeat']['max_abs_iou_diff']:.3g}, oracle cut "
          f"{checks['oracle_cut_prompt_tok']['max_abs_iou_vs_best']:.3g}", flush=True)
    return 0


def _floats(s: str) -> list:
    return [float(x) for x in s.split(",") if x.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", required=True, choices=["SAM", "SAM2.1", "SAM3"])
    p.add_argument("--checkpoint_path", required=True,
                   help="SAM: sam_vit_b_01ec64.pth; SAM2.1: the .pt of the reference csv; SAM3: '' = HF cache")
    p.add_argument("--model_type", default="vit_b", help="SAM-1 only")
    p.add_argument("--root", required=True, help="user study FOR_TEST (images/, masks/, user_masks/)")
    p.add_argument("--use", default="m", help="user mask kinds: m (mobile), p, or pm")
    p.add_argument("--datasets", nargs="*", default=["Berkeley", "GrabCut"],
                   help="test subsets besides COCO B (steer_user_study.DATASETS)")
    p.add_argument("--reference_csv", default="", help="per-case results of the same model, e.g. results/sam21full_m.csv")
    p.add_argument("--alphas", type=_floats, default=_floats("0.5,1,2,4,8,12,16"))
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--folds", type=int, default=5, help="image folds of half A (out-of-fold vectors and probes)")
    p.add_argument("--seed", type=int, default=0, help="folds and random directions")
    p.add_argument("--batch", type=int, default=64, help="boxes per decoder pass (one image = one pass)")
    p.add_argument("--probe_C", type=float, default=1.0, help="inverse L2 strength of the probes")
    p.add_argument("--probe_iters", type=int, default=200)
    p.add_argument("--repro_tol", type=float, default=1e-3, help="|IoU - reference| counted as a mismatch")
    p.add_argument("--repro_max_frac", type=float, default=0.05, help="max share of mismatched boxes on half A")
    p.add_argument("--repro_mean_tol", type=float, default=2e-3, help="max |mean IoU - reference mean| on half A")
    p.add_argument("--keep_tf32", action="store_true",
                   help="leave TF32 matmuls as the model loader set them (sam3 turns them on)")
    p.add_argument("--allow_repro_fail", action="store_true", help="record a failed repro check and go on")
    p.add_argument("--limit_images", type=int, default=0, help="per half / subset, for a smoke run; 0 = all")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # SAM-3 lives in its own conda env; re-execs there and exits (no-op otherwise)
    maybe_dispatch_to_env(args.model_name, __file__)
    try:
        return run(args)
    except CheckFailed as e:
        print(f"[check failed] {e}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
