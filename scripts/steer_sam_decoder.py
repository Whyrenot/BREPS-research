"""
steer_sam_decoder.py
====================
Can ONE fixed vector, added inside SAM-1's mask decoder, repair a critical box
shift -- and from which layer on does that work?

A critical pair (best_box, bad_box) from find_critical_shifts.py is two box
prompts a few pixels apart on the same image: best gives a good mask, bad a
broken one. activation_flow_graph.py located the break by activation patching:
re-run bad with one tensor taken from best's forward. That repair uses the
answer -- best's activation for this very image and box. Steering asks how much
of it survives when the replacement is a single vector, estimated on OTHER
images, frozen, and added to every box alike. This is a diagnostic: it applies
the vector to every box, with no gate deciding which boxes need it.

The two steering types of GuideVLA ("Steering Vision-Language-Action Models for
Out-of-Distribution Generalization"), with SAM in place of the VLA:

  residual steering     h~ = h + alpha * v at one decoder site and token group
                          mean    v = E_good[h]
                          dm      v = E_good[h] - E_bad[h]     (difference in means)
  action-head steering  z~ = z + alpha * s * d,  d = E_good[z] - E_bad[z]  (eq. 1),
                        s = +1 (dm) or -1 (dm_rev), on the decoder's output
                        interface, in two forms:
                          hyper    z = the 32-d output of the mask token's
                                   hypernetwork MLP, the mask's linear readout:
                                   logit(x, y) = <z, upscaled(x, y)>
                          low_res  z = the 256x256 mask logits. Every pixel is a
                                   binary decision (GuideVLA's binary-gripper
                                   case), and pixels are comparable across images
                                   only in the box's own frame: d is a G x G logit
                                   field over the box grown by a margin, estimated
                                   in that frame and stretched onto each box

good = best_box and the benign controls (the attack's shift rearranged across
the four box sides, mask still good: activation_flow_graph.control_candidates);
bad = bad_box.

SITES (forward order; ids are the node ids of activation_flow_graph's graph)
-----
  prompt_tok    prompt_encoder[sparse_embeddings], the 2 box-corner tokens
  L{l}.norm1-3  token stream of two-way block l after each LayerNorm, 7 tokens
                (IoU | mask 0 | masks 1-3 | 2 prompt), steered per token group
                all / mask0 / mask123 / prompt, every token with its own vector
  L{l}.norm4    image stream after image-to-token attention (4096 cells x 256):
                one 256-d vector, pooled over and added to the cells whose
                centre lies inside the box
  final_norm    token stream after the final attention; only mask token 0
                reaches the mask (multimask_output=False), so only mask0
  hyper, low_res  the action head, above

PROTOCOL
--------
1. collect    source dataset: re-predict every pair, keep it only if both IoUs
              reproduce the JSON (--repro_tol); find up to --n_controls benign
              controls; draw --dense_per_image jittered boxes around the GT box
              (the boxes a user would draw); record every site.
2. probe      where good vs bad is readable (GuideVLA Fig. 2A / A1), per site and
              token group: AUC of an L2 logistic regression with per-fold
              standardisation, cross-validated over image folds, bad vs all
              good and bad vs controls only (the same shift size); off-support
              fraction: a state is off support when its nearest good state of
              ANOTHER image is farther than the 95th percentile of that distance
              among the good states.
3. sweep      every source image is steered with the vectors of the folds that
              do not contain it. residual {mean, dm, dm_rev, dm_rand} x site x
              group x --alphas; action head {dm, dm_rev, dm_rand, const} x
              --head_alphas. Controls: dm_rand = a fixed random direction with
              dm's norm (per token); dm_rev = -dm; const (low_res) = one shift on
              every pixel, i.e. a threshold change. Per site also:
                oracle  activation patching, bad run with the tensor of the pair's
                        own best: the ceiling a fixed vector is compared with;
                noise   readout sensitivity, a random eps*||h|| perturbation
                        (--sens_eps), GuideVLA's "0.10||h|| changes 1.9%".
4. calibrate  GuideVLA sets the dose by the share of in-distribution decisions
              it changes, not by alpha. Here change = |m XOR m0| / |m OR m0| on
              the good boxes. For every (site, method) and budget b (--budgets),
              the (group, alpha) with the best restore among those with change
              <= b is chosen on the OTHER folds and scored on the held-out fold.
5. transfer   the configs chosen on the whole source dataset, with vectors of the
              whole source dataset, frozen, on every --transfer dataset; the
              source probes and the source support are scored there too.

METRICS (per pair, mean per image, mean over images; CI = image bootstrap)
-------
  restore     (IoU_bad' - IoU_bad) / (IoU_best - IoU_bad): the share of the drop
              repaired -- activation_flow_graph's patching formula, so
              steering / oracle = the share of the per-box repair that a fixed
              vector achieves
  fixed       restore >= 0.5
  d_good, chg_good   IoU change and mask change on best + controls (the harm)
  d_dense, chg_dense, lt05_delta   the same on the dense boxes, and the change
              in the share of dense boxes with IoU < 0.5

CHECKS (checks.json; a, b and c abort the run)
------
  a. zero dose: adding a zero vector at any site reproduces the unsteered
     forward bit for bit (GuideVLA's zero-dose arm);
  b. hook = trace: steering a site through a forward hook gives the same mask
     logits as activation_flow_graph.trace_forward with that tensor replaced
     by its steered value, for every steered site and group;
  c. no-op: steering final_norm's IoU / mask 1-3 / prompt tokens leaves the
     mask logits unchanged;
  d. reported: the oracle at the cuts prompt_tok and low_res restores 1.0; an
     oracle forward leaves every box but the bad ones unchanged; the baselines
     of pass 1 and pass 2 agree.

OUTPUTS (--out_dir)
-------
  configs.csv            every config: site, group, method, alpha, dose = ||alpha v|| / ||h||
  repro.csv              every pair of every dataset: kept / dropped, and why
  probe_layers.csv       AUC and off-support per site x group (+ @<transfer> columns)
  <ds>_pairs.csv.gz      per pair x config: steered IoUs and mask changes
  <ds>_images.csv        per image x config: the metrics
  <ds>_grid.csv          per config: mean over images, 95% bootstrap CI (the full grid)
  <ds>_calibrated.csv    per site x method x budget: the chosen configs and their metrics
  oracle.csv, sensitivity.csv, vectors.npz (the source-wide, frozen vectors)
  checks.json, meta.json
  layer_probe.png        AUC / off-support / readout sensitivity per site
  steer_layers_<ds>.png  restore at a matched change budget, per site and method, + oracle
  pareto_<ds>.png        restore vs change on the good boxes over alpha, per site

Example
-------
    CUDA_VISIBLE_DEVICES=0 python scripts/steer_sam_decoder.py \\
        --critical_shifts critical_shifts_coco.json \\
        --images_dir /.../datasets/COCO_MVal/img --masks_dir /.../datasets/COCO_MVal/gt \\
        --transfer Berkeley GrabCut --datasets_root /.../datasets \\
        --checkpoint_path /.../MODEL_CHECKPOINTS/SAM/sam_vit_b_01ec64.pth \\
        --out_dir exp_res/steer_sam

    # no data needed: hooks, trace agreement, box frames and oracle on random weights
    python scripts/steer_sam_decoder.py --self_test --device cpu
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from activation_flow_graph import (  # noqa: E402  -- the verified SAM-1 trace and its data loading
    IMAGE_EXTS, TOKEN_GROUPS, control_candidates, find_by_stem, load_gt, load_image_rgb,
    load_predictor, trace_forward,
)

EPS = 1e-12

# name: (critical-shifts JSON, image dir, GT dir, GT extension) under a dataset
# root, as isegm/data/datasets/*.py read them. GT = gray > 0, as in
# heatmaps/comp_hw.py, which produced the JSONs' IoUs.
DATASETS = {
    "COCO_MVal": ("critical_shifts_coco.json", "img", "gt", ".png"),
    "Berkeley": ("critical_shifts_berkeley.json", "images", "masks", ".png"),
    "GrabCut": ("critical_shifts.json", "data_GT", "boundary_GT", ".bmp"),
}
# tried after the table's own dirs (heatmaps/draw_heatmaps.py: images/ or data_GT/, masks/ or boundary_GT/)
IMAGE_SUBDIRS = ("images", "img", "data_GT")
MASK_SUBDIRS = ("masks", "gt", "boundary_GT")

ROLES = ("best", "bad", "ctrl", "dense")
TOK_ROWS = {g: sl for g, (_, sl) in TOKEN_GROUPS.items()}        # iou, mask0, mask123, prompt
PROBE_GROUPS = ("all", "iou", "mask0", "mask123", "prompt")
RES_METHODS = ("mean", "dm", "dm_rev", "dm_rand")
HEAD_METHODS = {"vec": ("dm", "dm_rev", "dm_rand"), "logits": ("dm", "dm_rev", "dm_rand", "const")}
PAIR_METRICS = ["restore", "fixed", "d_bad", "d_good", "chg_good", "chg_bad"]
DENSE_METRICS = ["d_dense", "chg_dense", "lt05_delta"]
CI_METRICS = ["restore", "d_good", "chg_good", "d_dense"]


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Site:
    id: str          # node id in activation_flow_graph's graph
    module: str      # the module whose output is steered (named_modules path)
    kind: str        # sparse | tok | img | vec | logits
    trace_key: str   # the same tensor in activation_flow_graph.trace_forward


def build_sites(depth: int) -> list[Site]:
    sites = [Site("prompt_tok", "prompt_encoder", "sparse", "prompt_encoder[sparse_embeddings]")]
    for li in range(depth):
        p = f"mask_decoder.transformer.layers.{li}"
        for k in (1, 2, 3):
            sites.append(Site(f"L{li + 1}.norm{k}", f"{p}.norm{k}", "tok", f"{p}.norm{k}"))
        sites.append(Site(f"L{li + 1}.norm4", f"{p}.norm4", "img", f"{p}.norm4"))
    sites.append(Site("final_norm", "mask_decoder.transformer.norm_final_attn", "tok",
                      "mask_decoder.transformer.norm_final_attn"))
    sites.append(Site("hyper", "mask_decoder.output_hypernetworks_mlps.0", "vec",
                      "mask_decoder.output_hypernetworks_mlps.0"))
    sites.append(Site("low_res", "mask_decoder", "logits", "mask_decoder[masks]"))
    return sites


def steer_groups(site: Site, groups) -> list[str]:
    if site.kind == "sparse":
        return ["prompt"]
    if site.kind == "tok":
        return ["mask0"] if site.id == "final_norm" else list(groups)
    return ["box"] if site.kind == "img" else ["-"]


def oracle_groups(site: Site, groups) -> list[str]:
    """The whole tensor first (the node's patching restore), then token groups."""
    if site.kind == "sparse":
        return ["prompt"]
    if site.kind == "tok":
        return ["mask0"] if site.id == "final_norm" else list(dict.fromkeys(["all", *groups]))
    return ["all"] if site.kind == "img" else ["-"]


def noise_group(site: Site) -> str:
    return {"sparse": "prompt", "img": "all", "vec": "-", "logits": "-"}.get(
        site.kind, "mask0" if site.id == "final_norm" else "all")


def probe_groups(site: Site) -> list[str]:
    if site.kind == "sparse":
        return ["prompt"]
    if site.kind == "tok":
        return list(PROBE_GROUPS)
    return ["box"] if site.kind == "img" else ["-"]


def row_index(site: Site, group: str):
    """Token rows of a group (LongTensor), or None for the whole tensor."""
    if site.kind != "tok" or group == "all":
        return None
    sl = TOK_ROWS[group]
    return torch.arange(sl.start, sl.stop)


def row_mask(site: Site, group: str, n_rows: int) -> np.ndarray:
    m = np.zeros(n_rows, dtype=bool)
    rows = row_index(site, group)
    m[rows.numpy() if rows is not None else slice(None)] = True
    return m


# ---------------------------------------------------------------------------
# Forward with hooks: steer and record decoder sites on a list of boxes
# ---------------------------------------------------------------------------

class Ctx:
    """What the hooks of one predict_torch call need: which boxes of the list
    are in this chunk, and their geometry (computed on first use)."""

    def __init__(self, runner: "Runner", boxes: torch.Tensor, sl: slice):
        self.r, self.sl, self.boxes = runner, sl, boxes[sl]
        self._cells = self._fgrid = None
        self._igrid = {}

    @property
    def cells(self):
        if self._cells is None:
            self._cells = self.r.cells(self.boxes)
        return self._cells

    @property
    def fgrid(self):
        if self._fgrid is None:
            self._fgrid = self.r.field_grid(self.boxes)
        return self._fgrid

    def igrid(self, hw):
        hw = tuple(hw)
        if hw not in self._igrid:
            self._igrid[hw] = self.r.inverse_grid(self.boxes, hw)
        return self._igrid[hw]


class Runner:
    """SamPredictor.predict_torch on a list of boxes, in fixed chunks, with
    forward hooks that edit (steer) and/or record decoder sites. The chunking
    depends only on the list, so every forward of one list is computed by the
    same kernels on the same batch shapes."""

    def __init__(self, predictor, sites, batch: int, field_grid: int, field_margin: float):
        self.p = predictor
        self.sam = predictor.model
        self.device = predictor.device
        self.sites = OrderedDict((s.id, s) for s in sites)
        mods = dict(self.sam.named_modules())
        self.mods = {s.id: mods[s.module] for s in sites}
        self.batch = batch
        self.G, self.margin = field_grid, field_margin
        self.frame = float(self.sam.image_encoder.img_size)                 # 1024
        self.emb_hw = tuple(self.sam.prompt_encoder.image_embedding_size)   # 64 x 64

    @torch.inference_mode()
    def forward(self, boxes: torch.Tensor, edits=None, record=None):
        """boxes (N, 4) in SAM's 1024 frame -> (masks (N, H, W) bool, low-res
        logits (N, h, w)). edits {site: fn(t, ctx) -> t}, record {site: fn(t, ctx)};
        t is the site's tensor (element 0 of the tuple that prompt_encoder and
        mask_decoder return); record sees it after the edit."""
        edits, record = edits or {}, record or {}
        holder = {}
        handles = [self.mods[sid].register_forward_hook(self._hook(sid, edits.get(sid), record.get(sid), holder))
                   for sid in dict.fromkeys([*edits, *record])]
        masks, lows = [], []
        try:
            for s in range(0, len(boxes), self.batch):
                sl = slice(s, min(s + self.batch, len(boxes)))
                holder["ctx"] = Ctx(self, boxes, sl)
                m, _, low = self.p.predict_torch(None, None, boxes=boxes[sl], multimask_output=False)
                masks.append(m[:, 0])
                lows.append(low[:, 0])
        finally:
            for h in handles:
                h.remove()
        return torch.cat(masks), torch.cat(lows)

    def _hook(self, sid, edit, rec, holder):
        tup = self.sites[sid].kind in ("sparse", "logits")

        def fn(_mod, _inp, out):
            t = out[0] if tup else out
            ctx = holder["ctx"]
            if edit is not None:
                t = edit(t, ctx)
            if rec is not None:
                rec(t, ctx)
            if edit is None:
                return None
            return (t, *out[1:]) if tup else t
        return fn

    # --- geometry -------------------------------------------------------------

    def cells(self, boxes: torch.Tensor) -> torch.Tensor:
        """(B, h*w) float: 1 on the image-embedding cells whose centre lies inside
        the box; a box that contains no centre gets the cell under its centre."""
        eh, ew = self.emb_hw
        sy, sx = self.frame / eh, self.frame / ew
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

    def region(self, boxes: torch.Tensor):
        """The box frame: the box grown by --field_margin of its size on every side."""
        x0, y0, x1, y1 = boxes.unbind(1)
        w, h = (x1 - x0).clamp(min=1.0), (y1 - y0).clamp(min=1.0)
        return x0 - self.margin * w, y0 - self.margin * h, x1 + self.margin * w, y1 + self.margin * h

    def field_grid(self, boxes: torch.Tensor):
        """Sampling grid (B, G, G, 2) of the box frame on the low-res logit map
        (grid_sample, align_corners=False: -1 / +1 are the edges of the 1024
        frame), and its validity (B, G, G): 1 inside the unpadded image."""
        X0, Y0, X1, Y1 = self.region(boxes)
        B, G = boxes.shape[0], self.G
        u = (torch.arange(G, device=boxes.device) + 0.5) / G
        X = X0[:, None] + u[None] * (X1 - X0)[:, None]
        Y = Y0[:, None] + u[None] * (Y1 - Y0)[:, None]
        half = self.frame / 2
        grid = torch.stack([(X / half - 1)[:, None, :].expand(B, G, G),
                            (Y / half - 1)[:, :, None].expand(B, G, G)], dim=-1)
        ih, iw = self.p.input_size
        valid = ((X >= 0) & (X < iw))[:, None, :] & ((Y >= 0) & (Y < ih))[:, :, None]
        return grid, valid.float()

    def inverse_grid(self, boxes: torch.Tensor, hw) -> torch.Tensor:
        """(B, h, w, 2): where every low-res logit pixel lies in its box frame
        ([-1, 1] inside the frame), to stretch a box-frame field onto the box."""
        h, w = hw
        X0, Y0, X1, Y1 = self.region(boxes)
        px = (torch.arange(w, device=boxes.device) + 0.5) * (self.frame / w)
        py = (torch.arange(h, device=boxes.device) + 0.5) * (self.frame / h)
        sx = (px[None] - X0[:, None]) / (X1 - X0)[:, None] * 2 - 1
        sy = (py[None] - Y0[:, None]) / (Y1 - Y0)[:, None] * 2 - 1
        B = boxes.shape[0]
        return torch.stack([sx[:, None, :].expand(B, h, w), sy[:, :, None].expand(B, h, w)], dim=-1)


def feature_of(site: Site, t: torch.Tensor, ctx: Ctx) -> torch.Tensor:
    """The per-box feature vectors and probes are built from (float32):
    tokens as they are, the image stream pooled over the box's cells, the
    logits as a box-frame field stacked with its validity (B, 2, G, G)."""
    if site.kind in ("sparse", "tok", "vec"):
        return t.float()
    if site.kind == "img":
        c = ctx.cells
        return (t * c[..., None]).sum(1) / c.sum(1, keepdim=True)
    grid, valid = ctx.fgrid
    f = F.grid_sample(t.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]
    return torch.stack([f * valid, valid], dim=1)


class Collector:
    """Records the feature of every site, for every box of a forward."""

    def __init__(self, runner: Runner):
        self.r = runner
        self.out = {sid: [] for sid in runner.sites}

    def fns(self):
        return {sid: (lambda t, ctx, _s=site: self.out[_s.id].append(feature_of(_s, t, ctx).cpu()))
                for sid, site in self.r.sites.items()}

    def result(self) -> dict:
        return {sid: torch.cat(v).numpy() for sid, v in self.out.items()}


def donor_recorder(store: dict, sid: str, keep: torch.Tensor):
    """Keeps the raw tensor of the boxes flagged in keep (the best boxes)."""
    def fn(t, ctx):
        k = keep[ctx.sl]
        if k.any():
            store[sid].append(t[k].clone())
    return fn


# ---------------------------------------------------------------------------
# Edits
# ---------------------------------------------------------------------------

def edit_add(w: torch.Tensor | float):
    """Tokens (W: T x C), hyper (32), or a constant logit shift."""
    return lambda t, ctx: t + w


def edit_add_cells(v: torch.Tensor):
    """Image stream: v added to the cells inside each box."""
    return lambda t, ctx: t + ctx.cells[..., None] * v


def edit_add_field(D: torch.Tensor):
    """Logits: the box-frame field D (1, 1, G, G) stretched onto each box."""
    def fn(t, ctx):
        delta = F.grid_sample(D.expand(t.shape[0], -1, -1, -1), ctx.igrid(t.shape[-2:]),
                              mode="bilinear", padding_mode="zeros", align_corners=False)
        return t + delta
    return fn


def edit_noise(site: Site, eps: float, rows, seed: int):
    """Readout sensitivity: a random direction of norm eps * ||h||, per token or
    image cell, or per box for hyper and the logits."""
    def fn(t, ctx):
        g = torch.Generator(device=t.device)
        g.manual_seed(seed + ctx.sl.start)
        u = torch.randn(t.shape, generator=g, device=t.device, dtype=t.dtype)
        dims = (-1,) if site.kind in ("sparse", "tok", "img") else tuple(range(1, t.dim()))
        u = u / torch.linalg.vector_norm(u, dim=dims, keepdim=True).clamp_min(EPS)
        delta = eps * torch.linalg.vector_norm(t, dim=dims, keepdim=True) * u
        if rows is not None:
            keep = torch.zeros(t.shape[1], 1, device=t.device, dtype=t.dtype)
            keep[rows.to(t.device)] = 1
            delta = delta * keep
        return t + delta
    return fn


def edit_oracle(rows, donors: torch.Tensor, donor_of: torch.Tensor):
    """Activation patching: every bad box gets the tensor (or the token rows) of
    its pair's best box; the other boxes are left alone."""
    def fn(t, ctx):
        di = donor_of[ctx.sl]
        sel = (di >= 0).nonzero().squeeze(1)
        if sel.numel() == 0:
            return t
        t = t.clone()
        D = donors[di[sel]]
        if rows is None:
            t[sel] = D
        else:
            r = rows.to(t.device)
            t[sel[:, None], r[None, :]] = D[:, r]
        return t
    return fn


def zero_edit(t, ctx):
    return t + torch.zeros_like(t)


# ---------------------------------------------------------------------------
# Steering vectors
# ---------------------------------------------------------------------------

def random_directions(sites, feats: dict, seed: int) -> dict:
    """One fixed random direction per site (unit norm per token row for token
    sites), shared by all folds; dm_rand rescales it to dm's norm."""
    R = {}
    for i, s in enumerate(sites):
        shape = feats[s.id].shape[2:] if s.kind == "logits" else feats[s.id].shape[1:]
        u = np.random.default_rng([seed, 101, i]).standard_normal(shape).astype(np.float32)
        u /= (np.linalg.norm(u, axis=-1, keepdims=True) if s.kind in ("sparse", "tok") else np.linalg.norm(u))
        R[s.id] = u
    return R


def build_vectors(feats: dict, good: np.ndarray, bad: np.ndarray, sites, R: dict) -> dict:
    """{site: {method: array}} from the features of the training boxes."""
    V = {}
    for s in sites:
        X = feats[s.id]
        if s.kind == "logits":
            f, w = X[:, 0], X[:, 1]
            wg, wb = w[good].sum(0), w[bad].sum(0)
            mg = f[good].sum(0) / np.maximum(wg, 1e-9)
            mb = f[bad].sum(0) / np.maximum(wb, 1e-9)
            valid = (wg > 0) & (wb > 0)
            mg, mb = mg * valid, mb * valid
        else:
            mg, mb = X[good].mean(0), X[bad].mean(0)
        dm = (mg - mb).astype(np.float32)
        if s.kind in ("sparse", "tok"):
            rnd = R[s.id] * np.linalg.norm(dm, axis=-1, keepdims=True)
        else:
            rnd = R[s.id] * np.linalg.norm(dm)
        d = {"mean": mg.astype(np.float32), "dm": dm, "dm_rev": -dm, "dm_rand": rnd.astype(np.float32)}
        if s.kind == "logits":
            v = dm[valid]
            d["const"] = float(np.sign(v.mean()) * np.sqrt((v ** 2).mean())) if v.size else 0.0
        V[s.id] = d
    return V


def vectors_to_device(V: dict, device) -> dict:
    return {s: {m: (torch.as_tensor(v, device=device) if isinstance(v, np.ndarray) else float(v))
                for m, v in d.items()} for s, d in V.items()}


def vectors_sha256(V: dict) -> str:
    h = hashlib.sha256()
    for s in sorted(V):
        for m in sorted(V[s]):
            h.update(f"{s}/{m}".encode())
            h.update(np.ascontiguousarray(np.asarray(V[s][m], dtype=np.float32)).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cfg:
    cid: int
    kind: str        # steer | oracle | noise
    site: str
    group: str
    method: str      # mean | dm | dm_rev | dm_rand | const | oracle | noise
    alpha: float     # steering strength; eps for noise; nan for oracle


def build_configs(sites, args) -> list[Cfg]:
    cfgs: list[Cfg] = []

    def add(kind, site, group, method, alpha):
        cfgs.append(Cfg(len(cfgs), kind, site, group, method, float(alpha)))

    for s in sites:
        head = s.kind in HEAD_METHODS
        for group in steer_groups(s, args.groups):
            for m in (HEAD_METHODS[s.kind] if head else RES_METHODS):
                for a in (args.head_alphas if head else args.alphas):
                    add("steer", s.id, group, m, a)
        for group in oracle_groups(s, args.groups):
            add("oracle", s.id, group, "oracle", math.nan)
        for e in args.sens_eps:
            add("noise", s.id, noise_group(s), "noise", e)
    return cfgs


def make_edit(cfg: Cfg, site: Site, Vt: dict | None, donors=None, donor_of=None, seed: int = 0):
    rows = row_index(site, cfg.group)
    if cfg.kind == "oracle":
        return edit_oracle(rows, donors[site.id], donor_of)
    if cfg.kind == "noise":
        return edit_noise(site, cfg.alpha, rows, seed)
    v, a = Vt[site.id][cfg.method], cfg.alpha
    if site.kind in ("sparse", "tok"):
        keep = torch.zeros(v.shape[0], 1, device=v.device, dtype=v.dtype)
        keep[rows.to(v.device) if rows is not None else slice(None)] = 1
        return edit_add(a * v * keep)
    if site.kind == "img":
        return edit_add_cells(a * v)
    if site.kind == "vec":
        return edit_add(a * v)
    if cfg.method == "const":
        return edit_add(a * v)
    return edit_add_field(a * v[None, None])


def config_table(cfgs, sites, V: dict, feats: dict, good: np.ndarray) -> pd.DataFrame:
    """configs.csv: dose = ||alpha v|| / ||h||, h = the mean norm of the good
    states at the site and group (box-pooled for the image stream, the
    box-frame field for the logits)."""
    by = {s.id: s for s in sites}
    rows = []
    for c in cfgs:
        s = by[c.site]
        dose = math.nan
        if c.kind == "steer":
            X = feats[s.id][good]
            if s.kind in ("sparse", "tok"):
                m = row_mask(s, c.group, X.shape[1])
                hn = np.linalg.norm(X[:, m].reshape(len(X), -1), axis=1).mean()
                vn = np.linalg.norm(V[s.id][c.method][m])
            elif s.kind == "logits":
                hn = np.linalg.norm(X[:, 0].reshape(len(X), -1), axis=1).mean()
                v = V[s.id][c.method]
                vn = abs(v) * math.sqrt(X[:, 1].mean(0).sum()) if c.method == "const" else np.linalg.norm(v)
            else:
                hn = np.linalg.norm(X.reshape(len(X), -1), axis=1).mean()
                vn = np.linalg.norm(V[s.id][c.method])
            dose = float(c.alpha * vn / max(hn, EPS))
        elif c.kind == "noise":
            dose = c.alpha
        rows.append({"cid": c.cid, "kind": c.kind, "site": c.site, "kind_site": s.kind, "group": c.group,
                     "method": c.method, "alpha": c.alpha, "dose": dose,
                     "whole": c.kind == "oracle" and c.group == oracle_groups(s, ["all"])[0]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def batch_iou(m: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """heatmaps/comp_hw.py batch_iou_torch: inter / (union + 1e-6)."""
    gt = gt[None] if gt.dim() == 2 else gt
    inter = (m & gt).flatten(1).sum(1).float()
    union = (m | gt).flatten(1).sum(1).float()
    return inter / (union + 1e-6)


def mask_change(m: torch.Tensor, m0: torch.Tensor) -> torch.Tensor:
    """|m XOR m0| / |m OR m0|: 1 - IoU(m, m0), and 0 when both are empty."""
    x = (m ^ m0).flatten(1).sum(1).float()
    u = (m | m0).flatten(1).sum(1).float()
    return x / u.clamp_min(1.0)


def pair_values(x: np.ndarray, item: dict):
    """Per-box values -> per pair (best, bad, mean over the pair's controls)."""
    ci = item["ci"]
    valid = ci >= 0
    xc = np.where(valid, x[np.maximum(ci, 0)], 0.0).sum(1)
    n = valid.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ctrl = np.where(n > 0, xc / np.maximum(n, 1), np.nan)
    return x[item["bi"]], x[item["ri"]], ctrl


class Store:
    """Per pair x config and per image x config results of one dataset."""

    def __init__(self):
        self.pairs, self.dense = [], []

    def add(self, item: dict, cid: int, iou: np.ndarray, chg: np.ndarray):
        b, r, c = pair_values(iou, item)
        cb, cr, cc = pair_values(chg, item)
        ds, im = item["dataset"], item["image"]
        for k, pid in enumerate(item["pids"]):
            self.pairs.append((ds, im, pid, cid, r[k], b[k], c[k], cr[k], cb[k], cc[k]))
        di = item["di"]
        if len(di):
            d = iou[di]
            self.dense.append((ds, im, cid, float(d.mean()), float((d < 0.5).mean()), float(chg[di].mean())))

    def frames(self):
        pairs = pd.DataFrame(self.pairs, columns=["dataset", "image", "pid", "cid", "iou_bad", "iou_best", "iou_ctrl",
                                                  "chg_bad", "chg_best", "chg_ctrl"])
        dense = pd.DataFrame(self.dense, columns=["dataset", "image", "cid", "iou_dense", "lt05_dense", "chg_dense"])
        return pairs, dense


# ---------------------------------------------------------------------------
# Data: datasets, pairs, controls, dense boxes
# ---------------------------------------------------------------------------

@dataclass
class DataSpec:
    name: str
    json: Path
    images_dir: Path
    masks_dir: Path
    mask_ext: str


def _resolve_json(p) -> Path:
    p = Path(p)
    for q in (p, _HERE.parent / p):
        if q.is_file():
            return q
    raise FileNotFoundError(f"critical-shifts JSON not found: {p}")


def resolve_datasets(args):
    js, _, gsub, gext = DATASETS.get(args.source_name, (None, None, "gt", ".png"))
    if js is None and not args.critical_shifts:
        raise SystemExit(f"--critical_shifts is required for a source not in {list(DATASETS)}")
    images = Path(args.images_dir)
    masks = Path(args.masks_dir) if args.masks_dir else images.parent / gsub
    src = DataSpec(args.source_name, _resolve_json(args.critical_shifts or js), images, masks, args.mask_ext or gext)
    roots = [Path(r) for r in args.datasets_root] if args.datasets_root else [images.parent.parent]
    targets, missing = [], []
    for name in args.transfer:
        if name == src.name:
            continue
        js, isub, gsub, gext = DATASETS[name]
        for r in roots:
            d = r / name
            im = next((d / s for s in (isub, *IMAGE_SUBDIRS) if (d / s).is_dir()), None)
            gt = next((d / s for s in (gsub, *MASK_SUBDIRS) if (d / s).is_dir()), None)
            if im is not None and gt is not None:
                targets.append(DataSpec(name, _resolve_json(js), im, gt, gext))
                print(f"[{name}] images {im}, GT {gt}", flush=True)
                break
        else:
            missing.append(name)
            print(f"[warn] transfer dataset {name}: no image + GT dirs under "
                  f"{[str(r / name) for r in roots]} -- skipped", flush=True)
    return src, targets, missing


def find_mask(masks_dir: Path, stem: str, ext: str) -> Path | None:
    p = masks_dir / (stem + ext)
    return p if p.is_file() else find_by_stem(masks_dir, stem, IMAGE_EXTS)


def find_controls(runner: Runner, best, bad, frame_wh, gt, rng, args):
    """activation_flow_graph's controls: the candidates in the same random order,
    the first --n_controls whose mask keeps IoU >= --ctrl_min_iou; tested in
    chunks of 32 instead of one by one."""
    gen = control_candidates(best, bad, frame_wh, rng, args.ctrl_max_cos)
    found, tried = [], 0
    while len(found) < args.n_controls:
        n = 32 if not args.ctrl_tries else min(32, args.ctrl_tries - tried)
        chunk = list(itertools.islice(gen, max(n, 0)))
        if not chunk:
            break
        tried += len(chunk)
        m, _ = runner.forward(torch.as_tensor(np.stack([c for c, _ in chunk]), device=runner.device))
        for (c, cos), v in zip(chunk, batch_iou(m, gt).tolist()):
            if v >= args.ctrl_min_iou and len(found) < args.n_controls:
                found.append((c, cos, v))
    return found, tried


def dense_boxes(gt_np: np.ndarray, n: int, jitter: float, rng) -> np.ndarray:
    """n boxes around the GT box (comp_hw.get_bbox_from_mask), each side moved
    by a uniform integer of up to jitter * the box size: boxes a user might draw."""
    if n <= 0 or not gt_np.any():
        return np.zeros((0, 4), np.float32)
    rows, cols = np.any(gt_np, axis=1), np.any(gt_np, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    H, W = gt_np.shape
    jx, jy = int(round(jitter * (x1 - x0))), int(round(jitter * (y1 - y0)))
    out = []
    for _ in range(20 * n):
        dx = rng.integers(-jx, jx + 1, size=2)
        dy = rng.integers(-jy, jy + 1, size=2)
        b = np.array([np.clip(x0 + dx[0], 0, W - 1), np.clip(y0 + dy[0], 0, H - 1),
                      np.clip(x1 + dx[1], 0, W - 1), np.clip(y1 + dy[1], 0, H - 1)], np.float32)
        if b[2] > b[0] and b[3] > b[1]:
            out.append(b)
            if len(out) == n:
                break
    return np.stack(out) if out else np.zeros((0, 4), np.float32)


def _name_seed(name: str) -> int:
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)


def prepare_image(runner: Runner, ds: DataSpec, name: str, cases, args, sink: dict):
    """Encode one image, re-predict and filter its pairs, find controls, draw
    the dense boxes, run the baseline once and record every site. Returns
    (item, features, meta, gt) or None when no pair is kept."""
    p, dev = runner.p, runner.device
    img_path = find_by_stem(ds.images_dir, name, IMAGE_EXTS)
    gt_path = find_mask(ds.masks_dir, name, ds.mask_ext)
    if img_path is None or gt_path is None:
        why = f"image not found in {ds.images_dir}" if img_path is None else f"mask not found in {ds.masks_dir}"
        for pid, c in cases:
            sink["repro"].append({"dataset": ds.name, "image": name, "pid": pid, "status": "dropped",
                                  "reason": why, "json_best_iou": c["best_iou"], "json_bad_iou": c["bad_iou"]})
        return None
    img, gt_np = load_image_rgb(img_path), load_gt(gt_path)
    with torch.inference_mode():
        p.set_image(img)
    if gt_np.shape != tuple(p.original_size):
        raise RuntimeError(f"{ds.name}/{name}: mask {gt_np.shape} != image {tuple(p.original_size)}")
    gt = torch.from_numpy(gt_np).to(dev)
    in_h, in_w = p.input_size

    best = np.stack([np.asarray(c["best_box"], np.float32) for _, c in cases])
    bad = np.stack([np.asarray(c["bad_box"], np.float32) for _, c in cases])
    m, _ = runner.forward(torch.as_tensor(np.concatenate([best, bad]), device=dev))
    iou = batch_iou(m, gt).cpu().numpy()
    ib, ir = iou[:len(cases)], iou[len(cases):]
    kept = []
    for j, (pid, c) in enumerate(cases):
        r = {"dataset": ds.name, "image": name, "pid": pid, "json_best_iou": c["best_iou"],
             "json_bad_iou": c["bad_iou"], "iou_best": float(ib[j]), "iou_bad": float(ir[j]),
             "linf_shift": float(np.abs(bad[j] - best[j]).max())}
        d_b, d_r = abs(ib[j] - c["best_iou"]), abs(ir[j] - c["bad_iou"])
        if d_b >= args.repro_tol or d_r >= args.repro_tol:
            r.update(status="dropped", reason=f"not reproduced: |dIoU| best {d_b:.4f}, bad {d_r:.4f} "
                                              f"(tol {args.repro_tol})")
        elif ib[j] - ir[j] <= 1e-6:
            r.update(status="dropped", reason="no IoU drop")
        else:
            ctrls, tried = find_controls(runner, best[j], bad[j], (in_w, in_h), gt,
                                         np.random.default_rng([args.seed, pid]), args)
            r.update(status="kept", reason="", n_ctrl=len(ctrls), ctrl_tried=tried,
                     ctrl_ious=";".join(f"{x[2]:.4f}" for x in ctrls))
            kept.append((pid, c, best[j], bad[j], ctrls))
        sink["repro"].append(r)
    if not kept:
        return None

    dense = dense_boxes(gt_np, args.dense_per_image, args.dense_jitter,
                        np.random.default_rng([args.seed, 2, _name_seed(f"{ds.name}/{name}")]))
    if len(dense):
        dense = p.transform.apply_boxes(dense, p.original_size).astype(np.float32)
    boxes, role, pair = [], [], []
    for k, (_, _, b, d, ctrls) in enumerate(kept):
        boxes += [b, d, *[x[0] for x in ctrls]]
        role += [0, 1] + [2] * len(ctrls)
        pair += [k] * (2 + len(ctrls))
    boxes += list(dense)
    role += [3] * len(dense)
    pair += [-1] * len(dense)
    boxes = np.stack(boxes).astype(np.float32)
    role, pair = np.asarray(role, np.int8), np.asarray(pair, np.int64)

    coll = Collector(runner)
    m0, _ = runner.forward(torch.as_tensor(boxes, device=dev), record=coll.fns())
    iou0 = batch_iou(m0, gt).cpu().numpy()
    P = len(kept)
    K = max(1, max(len(x[4]) for x in kept))
    ci = np.full((P, K), -1, np.int64)
    for i in np.where(role == 2)[0]:
        k = pair[i]
        ci[k, int((ci[k] >= 0).sum())] = i
    item = {"dataset": ds.name, "image": name, "img_path": str(img_path), "gt_path": str(gt_path),
            "boxes": boxes, "role": role, "pair": pair, "pids": [x[0] for x in kept],
            "bi": np.where(role == 0)[0], "ri": np.where(role == 1)[0], "ci": ci, "di": np.where(role == 3)[0],
            "iou0": iou0, "fold": -1}
    b0, r0, c0 = pair_values(iou0, item)
    for k, (pid, c, _, _, ctrls) in enumerate(kept):
        sink["base"].append({"dataset": ds.name, "image": name, "pid": pid, "iou_best0": float(b0[k]),
                             "iou_bad0": float(r0[k]), "iou_ctrl0": float(c0[k]), "n_ctrl": len(ctrls)})
    if len(item["di"]):
        d0 = iou0[item["di"]]
        sink["base_dense"].append({"dataset": ds.name, "image": name, "iou_dense0": float(d0.mean()),
                                   "lt05_dense0": float((d0 < 0.5).mean()), "n_dense": len(d0)})
    sel = role != 3
    feats = {sid: v[sel] for sid, v in coll.result().items()}
    meta = pd.DataFrame({"dataset": ds.name, "image": name, "pid": [kept[k][0] for k in pair[sel]],
                         "role": [ROLES[r] for r in role[sel]]})
    return item, feats, meta, gt


def collect(runner: Runner, ds: DataSpec, args, sink: dict, on_item=None):
    """prepare_image over a dataset; on_item(item, gt, index) runs while the
    image is still set on the predictor."""
    cases = json.loads(ds.json.read_text())
    by_img: OrderedDict[str, list] = OrderedDict()
    for pid, c in enumerate(cases):
        by_img.setdefault(c["image_name"], []).append((pid, c))
    names = list(by_img)[: args.limit_images] if args.limit_images > 0 else list(by_img)
    print(f"[{ds.name}] {len(cases)} pairs over {len(by_img)} images in {ds.json.name}; using {len(names)} images",
          flush=True)
    items, feats, metas = [], {sid: [] for sid in runner.sites}, []
    t0 = time.time()
    for ii, name in enumerate(names):
        res = prepare_image(runner, ds, name, by_img[name], args, sink)
        if res is None:
            print(f"[{ds.name}] [{ii + 1}/{len(names)}] {name}: no pair kept", flush=True)
            continue
        item, f, meta, gt = res
        items.append(item)
        for sid in feats:
            feats[sid].append(f[sid])
        metas.append(meta)
        msg = ""
        if on_item is not None:
            msg = on_item(item, gt, len(items) - 1) or ""
        print(f"[{ds.name}] [{ii + 1}/{len(names)}] {name}: {len(item['pids'])}/{len(by_img[name])} pairs, "
              f"{len(item['boxes'])} boxes{msg} ({time.time() - t0:.0f}s)", flush=True)
    if not items:
        return [], {}, pd.DataFrame(columns=["dataset", "image", "pid", "role"])
    return items, {sid: np.concatenate(v) for sid, v in feats.items()}, pd.concat(metas, ignore_index=True)


def assign_folds(names, k: int, seed: int) -> dict:
    names = sorted(names)
    order = np.random.default_rng([seed, 5]).permutation(len(names))
    return {names[j]: int(i % k) for i, j in enumerate(order)}


# ---------------------------------------------------------------------------
# Checks and the per-image sweep
# ---------------------------------------------------------------------------

def rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        return math.inf
    if torch.equal(a, b):
        return 0.0
    if a.dtype == torch.bool:
        return float((a != b).float().mean())
    return float((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(EPS))


def new_checks() -> dict:
    return {"zero_dose": {"forwards": 0, "bit_exact": 0, "failures": []},
            "noop": {"forwards": 0, "max_rel": 0.0, "failures": []},
            "trace_vs_hook": {"targets": 0, "max_rel_low_res": 0.0, "mask_mismatch": 0.0, "where": "",
                              "failures": []},
            "baseline_repeat": {"images": 0, "max_abs_iou_diff": 0.0},
            "oracle_other_boxes_max_change": 0.0}


def check_zero_and_noop(runner: Runner, boxes, m0, low0, Vt, checks: dict, tol: float):
    """a. a zero vector at every site reproduces the baseline bit for bit;
    c. steering final_norm's unused tokens leaves the logits unchanged."""
    z = checks["zero_dose"]
    for sid in runner.sites:
        m, low = runner.forward(boxes, edits={sid: zero_edit})
        z["forwards"] += 1
        if torch.equal(m, m0) and torch.equal(low, low0):
            z["bit_exact"] += 1
        else:
            z["failures"].append(f"{sid}: mask rel {rel_diff(m, m0):.3g}, logits rel {rel_diff(low, low0):.3g}")
    n = checks["noop"]
    site = runner.sites["final_norm"]
    for group in ("iou", "mask123", "prompt"):
        cfg = Cfg(-1, "steer", "final_norm", group, "dm_rand", 4.0)
        _, low = runner.forward(boxes, edits={"final_norm": make_edit(cfg, site, Vt)})
        d = rel_diff(low, low0)
        n["forwards"] += 1
        n["max_rel"] = max(n["max_rel"], d)
        if d > tol:
            n["failures"].append(f"final_norm#{group}: logits rel {d:.3g}")
    if z["failures"] or n["failures"]:
        raise RuntimeError("zero-dose / no-op check failed:\n  " + "\n  ".join(z["failures"] + n["failures"]))


def check_trace(runner: Runner, box: np.ndarray, Vt: dict, groups, checks: dict, tol: float, alpha: float = 1.0):
    """b. a steered hook forward == trace_forward with the site's tensor
    replaced by its steered value, for every steered site and group."""
    c = checks["trace_vs_hook"]
    b = torch.as_tensor(box[None], device=runner.device)
    T0 = trace_forward(runner.p, box)
    for site in runner.sites.values():
        methods = ["dm", "const"] if site.kind == "logits" else ["dm"]
        for group in steer_groups(site, groups):
            for method in methods:
                edit = make_edit(Cfg(-1, "steer", site.id, group, method, alpha), site, Vt)
                m_h, low_h = runner.forward(b, edits={site.id: edit})
                donor = edit(T0[site.trace_key].clone(), Ctx(runner, b, slice(0, 1)))
                T1 = trace_forward(runner.p, box, patch={site.trace_key: (donor, None)})
                d = rel_diff(low_h, T1["mask_decoder[masks]"][:, 0])
                dm = rel_diff(m_h, T1["mask"][:, 0])
                c["targets"] += 1
                if d > c["max_rel_low_res"]:
                    c["max_rel_low_res"], c["where"] = d, f"{site.id}#{group}/{method}"
                c["mask_mismatch"] = max(c["mask_mismatch"], dm)
                if d > tol or dm > 0:
                    c["failures"].append(f"{site.id}#{group}/{method}: logits rel {d:.3g}, mask {dm:.3g}")
    if c["failures"]:
        raise RuntimeError("hook steering does not match the trace:\n  " + "\n  ".join(c["failures"]))


def evaluate_image(runner: Runner, item: dict, gt: torch.Tensor, cfgs, Vt: dict, args, checks: dict,
                   store: Store, do_checks: bool, trace: bool) -> str:
    """Baseline (+ donors for the oracle), the checks, then every config."""
    dev = runner.device
    boxes = torch.as_tensor(item["boxes"], device=dev)
    role = torch.as_tensor(item["role"], device=dev)
    oracle_sites = list(dict.fromkeys(c.site for c in cfgs if c.kind == "oracle"))
    store_d = {s: [] for s in oracle_sites}
    keep = role == 0
    m0, low0 = runner.forward(boxes, record={s: donor_recorder(store_d, s, keep) for s in oracle_sites})
    iou0 = batch_iou(m0, gt)
    br = checks["baseline_repeat"]
    br["images"] += 1
    br["max_abs_iou_diff"] = max(br["max_abs_iou_diff"], float(np.abs(iou0.cpu().numpy() - item["iou0"]).max()))
    donors = {s: torch.cat(v) for s, v in store_d.items()}
    donor_of = torch.full((len(boxes),), -1, dtype=torch.long, device=dev)
    donor_of[torch.as_tensor(item["ri"], device=dev)] = torch.arange(len(item["ri"]), device=dev)
    not_bad = role != 1
    if do_checks:
        check_zero_and_noop(runner, boxes, m0, low0, Vt, checks, args.trace_tol)
    if trace:
        check_trace(runner, item["boxes"][item["bi"][0]], Vt, args.groups, checks, args.trace_tol)
    seed0 = args.seed * 1_000_003 + _name_seed(f"{item['dataset']}/{item['image']}") % 1_000_000
    for cfg in cfgs:
        edit = make_edit(cfg, runner.sites[cfg.site], Vt, donors, donor_of, seed=seed0 + 7919 * cfg.cid)
        m, _ = runner.forward(boxes, edits={cfg.site: edit})
        chg = mask_change(m, m0)
        if cfg.kind == "oracle":
            checks["oracle_other_boxes_max_change"] = max(checks["oracle_other_boxes_max_change"],
                                                          float(chg[not_bad].max()))
        store.add(item, cfg.cid, batch_iou(m, gt).cpu().numpy(), chg.cpu().numpy())
    return f", {len(cfgs)} configs"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def image_table(pairs: pd.DataFrame, base: pd.DataFrame, dense: pd.DataFrame, base_dense: pd.DataFrame,
                folds: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["dataset", "image", "pid"]
    df = pairs.merge(base[key + ["iou_best0", "iou_bad0", "iou_ctrl0", "n_ctrl"]], on=key, how="left")
    den = df.iou_best0 - df.iou_bad0
    df["restore"] = (df.iou_bad - df.iou_bad0) / den.where(den > 1e-6)
    df["fixed"] = (df.restore >= 0.5).astype(float).where(df.restore.notna())
    df["d_bad"] = df.iou_bad - df.iou_bad0
    n = df.n_ctrl.astype(float)
    df["d_good"] = (df.iou_best - df.iou_best0 + n * (df.iou_ctrl - df.iou_ctrl0).fillna(0.0)) / (1 + n)
    df["chg_good"] = (df.chg_best + n * df.chg_ctrl.fillna(0.0)) / (1 + n)
    img = df.groupby(["dataset", "image", "cid"], sort=False)[PAIR_METRICS].mean().reset_index()
    if len(dense):
        dd = dense.merge(base_dense, on=["dataset", "image"], how="left")
        dd["d_dense"] = dd.iou_dense - dd.iou_dense0
        dd["lt05_delta"] = dd.lt05_dense - dd.lt05_dense0
        img = img.merge(dd[["dataset", "image", "cid", *DENSE_METRICS]], on=["dataset", "image", "cid"], how="left")
    else:
        for m in DENSE_METRICS:
            img[m] = np.nan
    img["fold"] = img.image.map(folds).fillna(-1).astype(int) if folds else -1
    return img, df


def boot_ci(M: np.ndarray, n_boot: int, seed: int, chunk: int = 64):
    """95% bootstrap CI over the columns (images) of every row of M (NaN = missing)."""
    lo, hi = np.full(len(M), np.nan), np.full(len(M), np.nan)
    if M.size == 0 or n_boot <= 0:
        return lo, hi
    idx = np.random.default_rng(seed).integers(0, M.shape[1], size=(n_boot, M.shape[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for s in range(0, len(M), chunk):
            S = np.nanmean(M[s:s + chunk][:, idx], axis=2)
            lo[s:s + chunk], hi[s:s + chunk] = np.nanpercentile(S, [2.5, 97.5], axis=1)
    return lo, hi


def grid_table(img: pd.DataFrame, cfg_df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    g = img.groupby("cid")[PAIR_METRICS + DENSE_METRICS].mean()
    g["n_images"] = img.groupby("cid").image.nunique()
    for m in CI_METRICS:
        M = img.pivot_table(index="cid", columns="image", values=m, aggfunc="mean")
        lo, hi = boot_ci(M.to_numpy(float), n_boot, seed)
        g[m + "_lo"] = pd.Series(lo, index=M.index)
        g[m + "_hi"] = pd.Series(hi, index=M.index)
    return cfg_df.merge(g.reset_index(), on="cid", how="inner")


def choose(img: pd.DataFrame, budget: float, alpha_of: pd.Series):
    """The config with the best mean restore among those whose mean change on
    the good boxes stays within the budget (ties -> the smaller alpha)."""
    g = img.groupby("cid")[["restore", "chg_good"]].mean()
    g = g[g.chg_good <= budget].dropna(subset=["restore"])
    if g.empty:
        return None
    g = g.assign(alpha=alpha_of.reindex(g.index).to_numpy()).sort_values(["restore", "alpha"],
                                                                         ascending=[False, True])
    return int(g.index[0])


def _describe(cfg_df: pd.DataFrame, cid) -> str:
    if cid is None:
        return "-"
    r = cfg_df.loc[cid]
    return f"{r.group}/a={r.alpha:g}"


def _summary(T: pd.DataFrame, n_boot: int, seed: int) -> dict:
    r = {m: float(T[m].mean()) if len(T) else math.nan for m in PAIR_METRICS + DENSE_METRICS}
    r["n_images"] = int(T.image.nunique()) if len(T) else 0
    for m in ("restore", "chg_good"):
        lo, hi = boot_ci(T[m].to_numpy(float)[None], n_boot, seed) if len(T) else ([math.nan], [math.nan])
        r[m + "_lo"], r[m + "_hi"] = float(lo[0]), float(hi[0])
    return r


def calibrate_oof(img: pd.DataFrame, cfg_df: pd.DataFrame, budgets, n_boot: int, seed: int) -> pd.DataFrame:
    """Per (site, method, budget): (group, alpha) chosen on the other folds,
    scored on the held-out fold; every image is scored exactly once."""
    cfg_i = cfg_df.set_index("cid")
    rows = []
    steer = cfg_df[cfg_df.kind == "steer"]
    for (site, method), sub in steer.groupby(["site", "method"], sort=False):
        I = img[img.cid.isin(sub.cid)]
        for b in budgets:
            picks, test = [], []
            for k in sorted(I.fold.unique()):
                cid = choose(I[I.fold != k], b, cfg_i.alpha)
                picks.append(f"f{k}:{_describe(cfg_i, cid)}")
                if cid is not None:
                    test.append(I[(I.fold == k) & (I.cid == cid)])
            T = pd.concat(test) if test else I.iloc[:0]
            rows.append({"site": site, "method": method, "budget": b, **_summary(T, n_boot, seed),
                         "chosen": " ".join(picks)})
    return pd.DataFrame(rows)


def select_whole(img: pd.DataFrame, cfg_df: pd.DataFrame, budgets) -> dict:
    """(site, method, budget) -> cid chosen on all source images (frozen for transfer)."""
    cfg_i = cfg_df.set_index("cid")
    picks = {}
    for (site, method), sub in cfg_df[cfg_df.kind == "steer"].groupby(["site", "method"], sort=False):
        I = img[img.cid.isin(sub.cid)]
        for b in budgets:
            picks[(site, method, b)] = choose(I, b, cfg_i.alpha)
    return picks


def calibrate_frozen(img: pd.DataFrame, cfg_df: pd.DataFrame, picks: dict, src_img: pd.DataFrame,
                     n_boot: int, seed: int) -> pd.DataFrame:
    cfg_i = cfg_df.set_index("cid")
    rows = []
    for (site, method, b), cid in picks.items():
        T = img[img.cid == cid] if cid is not None else img.iloc[:0]
        src_chg = float(src_img[src_img.cid == cid].chg_good.mean()) if cid is not None else math.nan
        rows.append({"site": site, "method": method, "budget": b, **_summary(T, n_boot, seed),
                     "chosen": _describe(cfg_i, cid), "cid": cid, "source_chg_good": src_chg})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Probes (layer analysis)
# ---------------------------------------------------------------------------

def probe_matrix(site: Site, group: str, feats: dict) -> np.ndarray:
    X = feats[site.id]
    if site.kind in ("sparse", "tok"):
        return X[:, row_mask(site, group, X.shape[1])].reshape(len(X), -1)
    if site.kind == "logits":
        return F.adaptive_avg_pool2d(torch.from_numpy(X[:, :1]), 16).flatten(1).numpy()
    return X.reshape(len(X), -1)


def _auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else math.nan


def _nn_dist(X: np.ndarray, A: np.ndarray, same=None) -> np.ndarray:
    """Distance from every row of X to its nearest row of A (same[i, j] -> excluded)."""
    out = np.empty(len(X))
    At = torch.from_numpy(A.astype(np.float32))
    for s in range(0, len(X), 1024):
        D = torch.cdist(torch.from_numpy(X[s:s + 1024].astype(np.float32)), At)
        if same is not None:
            D[torch.from_numpy(same[s:s + 1024])] = math.inf
        out[s:s + 1024] = D.min(1).values.numpy()
    return out


def run_probes(sites, feats: dict, meta: pd.DataFrame, fold_row: np.ndarray, targets: dict, C: float) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def clf():
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))

    y = (meta.role == "bad").to_numpy().astype(int)
    good = y == 0
    img_code = pd.factorize(meta.image)[0]
    same = img_code[:, None] == img_code[good][None, :]
    rows = []
    for s in sites:
        for g in probe_groups(s):
            X = probe_matrix(s, g, feats).astype(np.float64)
            score = np.full(len(X), np.nan)
            for k in np.unique(fold_row):
                tr, te = fold_row != k, fold_row == k
                if len(np.unique(y[tr])) < 2:
                    continue
                score[te] = clf().fit(X[tr], y[tr]).decision_function(X[te])
            r = {"site": s.id, "group": g, "dim": X.shape[1], "n_good": int(good.sum()), "n_bad": int((~good).sum()),
                 "auc_all": _auc(y, score)}
            for other in ("ctrl", "best"):
                m = meta.role.isin(["bad", other]).to_numpy()
                r[f"auc_vs_{other}"] = _auc(y[m], score[m])
            nn = _nn_dist(X, X[good], same)
            tau = float(np.quantile(nn[good], 0.95))
            r["support_tau"] = tau
            for role in ("bad", "ctrl", "best"):
                m = (meta.role == role).to_numpy()
                r[f"off_{role}"] = float((nn[m] > tau).mean()) if m.any() else math.nan
            if targets:
                full = clf().fit(X, y)
                for name, (tf, tm) in targets.items():
                    Xt = probe_matrix(s, g, tf).astype(np.float64)
                    yt = (tm.role == "bad").to_numpy().astype(int)
                    st = full.decision_function(Xt)
                    r[f"auc_all@{name}"] = _auc(yt, st)
                    m = tm.role.isin(["bad", "ctrl"]).to_numpy()
                    r[f"auc_vs_ctrl@{name}"] = _auc(yt[m], st[m])
                    nt = _nn_dist(Xt, X[good])
                    r[f"off_good@{name}"] = float((nt[yt == 0] > tau).mean()) if (yt == 0).any() else math.nan
                    r[f"off_bad@{name}"] = float((nt[yt == 1] > tau).mean()) if (yt == 1).any() else math.nan
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
METHOD_COLOR = {"dm": SERIES[0], "mean": SERIES[1], "dm_rand": SERIES[2], "dm_rev": SERIES[3], "const": SERIES[4]}
GROUP_COLOR = {"all": SERIES[0], "mask0": SERIES[1], "mask123": SERIES[2], "prompt": SERIES[3], "iou": SERIES[4]}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.titlecolor": INK, "legend.frameon": False})
    return plt


def _style(ax, sites, ticks):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xticks(range(len(sites)))
    ax.set_xticklabels(sites if ticks else [], rotation=45, ha="right")
    ax.set_xlim(-0.5, len(sites) - 0.5)


def _line(ax, x, y, color, label, dashed=False):
    y = np.asarray(y, float)
    ax.plot(x, y, color=color, lw=2, ls="--" if dashed else "-", marker="o", ms=5,
            markeredgecolor=SURFACE, markeredgewidth=1, label=label)


def _handle(color, label, dashed=False):
    from matplotlib.lines import Line2D
    return Line2D([], [], color=color, lw=2, ls="--" if dashed else "-", marker="o", ms=5,
                  markeredgecolor=SURFACE, markeredgewidth=1, label=label)


def _legend_right(ax, handles=None, title=None):
    """Legend outside the plot area, right of the axes: never on top of data."""
    kw = {"handles": handles} if handles is not None else {}
    ax.legend(**kw, title=title, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, title_fontsize=8,
              borderaxespad=0)


def plot_layer_probe(probe: pd.DataFrame, sens: pd.DataFrame, sites, path: Path, title: str):
    plt = _plt()
    ids = [s.id for s in sites]
    kind = {s.id: s.kind for s in sites}
    fig, axes = plt.subplots(3, 1, figsize=(10, 9.5), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    P = probe.set_index(["site", "group"])

    def value(site, line, col):
        k = kind[site]
        if k == "tok":
            key = (site, line)
        elif k == "sparse":
            key = (site, "prompt") if line in ("all", "prompt") else None
        else:
            key = (site, "box" if k == "img" else "-") if line == "all" else None
        return float(P.loc[key, col]) if key is not None and key in P.index else math.nan

    for ax, col, ylab in ((axes[0], "auc_vs_ctrl", "AUC, bad vs controls"),
                          (axes[1], "off_bad", "off-support share of bad states")):
        for line in PROBE_GROUPS:
            _line(ax, range(len(ids)), [value(s, line, col) for s in ids], GROUP_COLOR[line], line)
        ax.set_ylabel(ylab)
        _style(ax, ids, False)
        _legend_right(ax, title="token group")
    axes[0].axhline(0.5, color=MUTED, lw=1, ls=":")
    axes[1].axhline(0.05, color=MUTED, lw=1, ls=":")
    axes[0].set_title("Where good vs bad is readable (probe cross-validated over image folds)", loc="left")
    axes[1].set_title("Bad states outside the support of good states of other images (dotted: 5% by construction)",
                      loc="left")
    ax = axes[2]
    S = sens.set_index(["site", "alpha"])
    for i, eps in enumerate(sorted(sens.alpha.unique())):
        y = [float(S.loc[(s, eps), "chg_good"]) if (s, eps) in S.index else math.nan for s in ids]
        _line(ax, range(len(ids)), y, SERIES[i], f"eps = {eps:g}")
        last = max((j for j, v in enumerate(y) if np.isfinite(v)), default=None)
        if last is not None:
            ax.annotate(f"eps {eps:g}", (last, y[last]), xytext=(6, 0), textcoords="offset points",
                        color=INK2, fontsize=8, va="center")
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("mask change on good boxes")
    ax.set_title("Readout sensitivity: random perturbation of norm eps * ||h|| at the site", loc="left")
    _style(ax, ids, True)
    _legend_right(ax)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_steer_layers(cal: pd.DataFrame, oracle: pd.DataFrame, sites, budgets, path: Path, title: str):
    plt = _plt()
    ids = [s.id for s in sites]
    fig, axes = plt.subplots(len(budgets), 1, figsize=(10, 3.2 * len(budgets) + 0.6), sharex=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    O = oracle.set_index("site").restore if len(oracle) else pd.Series(dtype=float)
    present = [m for m in ("dm", "mean", "dm_rand", "dm_rev", "const") if (cal.method == m).any()]
    for ax, b in zip(axes[:, 0], budgets):
        C = cal[cal.budget == b].set_index(["site", "method"])
        _line(ax, range(len(ids)), [float(O.get(s, math.nan)) for s in ids], INK2, "oracle (patching)", dashed=True)
        for method in present:
            y = [float(C.loc[(s, method), "restore"]) if (s, method) in C.index else math.nan for s in ids]
            _line(ax, range(len(ids)), y, METHOD_COLOR[method], method)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_ylabel("restore")
        ax.set_title(f"change on good boxes <= {b:g}", loc="left", fontsize=9)
        _style(ax, ids, ax is axes[-1, 0])
    _legend_right(axes[0, 0], [_handle(INK2, "oracle (patching)", True)]
                  + [_handle(METHOD_COLOR[m], m) for m in present], title="method")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(grid: pd.DataFrame, picks: dict, sites, budget: float, path: Path, title: str):
    plt = _plt()
    from matplotlib.ticker import NullFormatter
    ids = [s.id for s in sites]
    ncol = 4
    nrow = math.ceil(len(ids) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.9 * nrow + 0.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    G = grid[grid.kind == "steer"]
    present = [m for m in ("dm", "mean", "dm_rand", "dm_rev", "const") if (G.method == m).any()]
    for ax, sid in zip(axes.flat, ids):
        sub = G[G.site == sid]
        cid = picks.get((sid, "dm", budget))
        group = grid.set_index("cid").group.get(cid) if cid is not None else sub.group.iloc[0]
        for method in ("dm", "mean", "dm_rand", "dm_rev", "const"):
            r = sub[(sub.method == method) & (sub.group == group)].sort_values("alpha")
            if len(r):
                ax.plot(r.chg_good.clip(lower=1e-4), r.restore, color=METHOD_COLOR[method], lw=2, marker="o", ms=4,
                        markeredgecolor=SURFACE, markeredgewidth=1, label=method)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.axhline(0, color=AXIS, lw=1)
        ax.axvline(budget, color=MUTED, lw=1, ls=":")
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"{sid}  [{group}]", loc="left", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(ids):]:
        ax.axis("off")
    fig.legend(handles=[_handle(METHOD_COLOR[m], m) for m in present], loc="upper right", ncol=len(present),
               fontsize=8)
    fig.supxlabel("mask change on good boxes (log)", color=INK2, fontsize=9)
    fig.supylabel("restore", color=INK2, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def _analyse(name, store, sink, cfg_df, folds, args):
    pairs, dense = store.frames()
    base = pd.DataFrame(sink["base"])
    base_dense = pd.DataFrame(sink["base_dense"]) if sink["base_dense"] else pd.DataFrame(
        columns=["dataset", "image", "iou_dense0", "lt05_dense0", "n_dense"])
    img, per_pair = image_table(pairs, base[base.dataset == name], dense,
                                base_dense[base_dense.dataset == name], folds)
    grid = grid_table(img, cfg_df, args.n_boot, args.seed)
    return per_pair, img, grid


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def _print_layers(name, cal, oracle, sites, budget):
    C = cal[cal.budget == budget].set_index(["site", "method"])
    O = oracle.set_index("site").restore if len(oracle) else pd.Series(dtype=float)
    print(f"\n[{name}] restore at change on good boxes <= {budget:g}   (oracle = activation patching)")
    print(f"  {'site':<12}{'oracle':>8}{'dm':>8}{'mean':>8}{'dm_rand':>9}{'dm_rev':>8}{'const':>8}   dm chosen")
    for s in sites:
        vals = [C.loc[(s.id, m), "restore"] if (s.id, m) in C.index else math.nan
                for m in ("dm", "mean", "dm_rand", "dm_rev", "const")]
        ch = C.loc[(s.id, "dm"), "chosen"] if (s.id, "dm") in C.index else ""
        print(f"  {s.id:<12}{O.get(s.id, math.nan):>8.3f}" + "".join(f"{v:>8.3f}" if i != 2 else f"{v:>9.3f}"
                                                                  for i, v in enumerate(vals)) + f"   {ch}")


def run(args) -> int:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    t0 = time.time()
    src, targets, missing = resolve_datasets(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = load_predictor(args.checkpoint_path, args.model_type, device)
    depth = len(predictor.model.mask_decoder.transformer.layers)
    sites = build_sites(depth)
    runner = Runner(predictor, sites, args.batch, args.field_grid, args.field_margin)
    cfgs = build_configs(sites, args)
    by_cid = {c.cid: c for c in cfgs}
    checks = new_checks()
    sink = {"repro": [], "base": [], "base_dense": []}
    print(f"{len(cfgs)} configs: {sum(c.kind == 'steer' for c in cfgs)} steering, "
          f"{sum(c.kind == 'oracle' for c in cfgs)} oracle, {sum(c.kind == 'noise' for c in cfgs)} noise", flush=True)

    # 1. collect on the source dataset
    items, feats, meta = collect(runner, src, args, sink)
    if len(items) < 2:
        pd.DataFrame(sink["repro"]).to_csv(out_dir / "repro.csv", index=False)
        print(f"{len(items)} source image(s) with a kept pair, need >= 2 for out-of-fold vectors -- "
              f"see repro.csv", file=sys.stderr)
        return 1
    folds = assign_folds([it["image"] for it in items], min(args.folds, len(items)), args.seed)
    for it in items:
        it["fold"] = folds[it["image"]]
    fold_row = meta.image.map(folds).to_numpy()
    good = (meta.role != "bad").to_numpy()
    bad = ~good
    R = random_directions(sites, feats, args.seed)
    V_fold = {k: build_vectors(feats, good & (fold_row != k), bad & (fold_row != k), sites, R)
              for k in sorted(set(folds.values()))}
    V_all = build_vectors(feats, good, bad, sites, R)
    checks["vectors_sha256"] = {"all": vectors_sha256(V_all), **{f"fold{k}": vectors_sha256(v)
                                                                 for k, v in V_fold.items()}}
    np.savez_compressed(out_dir / "vectors.npz", **{f"{s}/{m}": np.asarray(v, np.float32)
                                                   for s, d in V_all.items() for m, v in d.items()})
    cfg_df = config_table(cfgs, sites, V_all, feats, good)
    cfg_df.to_csv(out_dir / "configs.csv", index=False)

    # 3. sweep on the source dataset, out of fold
    store = Store()
    Vt_fold = {k: vectors_to_device(v, device) for k, v in V_fold.items()}
    print(f"[{src.name}] sweep: {len(items)} images x {len(cfgs)} configs, vectors out of fold "
          f"({len(V_fold)} folds)", flush=True)
    t1 = time.time()
    for i, it in enumerate(items):
        with torch.inference_mode():
            predictor.set_image(load_image_rgb(Path(it["img_path"])))
        gt = torch.from_numpy(load_gt(Path(it["gt_path"]))).to(device)
        evaluate_image(runner, it, gt, cfgs, Vt_fold[it["fold"]], args, checks, store,
                       do_checks=i < args.check_images, trace=i == 0)
        print(f"[{src.name}] sweep [{i + 1}/{len(items)}] {it['image']} ({time.time() - t1:.0f}s)", flush=True)
    pairs, img, grid = _analyse(src.name, store, sink, cfg_df, folds, args)
    cal = calibrate_oof(img, cfg_df, args.budgets, args.n_boot, args.seed)
    picks = select_whole(img, cfg_df, args.budgets)
    outputs = {src.name: (pairs, img, grid, cal)}

    # 5. transfer: frozen configs and source-wide vectors
    t_cids = sorted({c for c in picks.values() if c is not None} | set(cfg_df[cfg_df.kind == "oracle"].cid))
    t_cfgs = [by_cid[c] for c in t_cids]
    Vt_all = vectors_to_device(V_all, device)
    tgt_feats = {}
    for ds in targets:
        st = Store()

        def on_item(item, gt, idx, _st=st):
            return evaluate_image(runner, item, gt, t_cfgs, Vt_all, args, checks, _st,
                                  do_checks=idx < args.check_images, trace=False)
        t_items, t_feats, t_meta = collect(runner, ds, args, sink, on_item=on_item)
        if not t_items:
            print(f"[{ds.name}] no pair kept -- skipped", flush=True)
            continue
        tgt_feats[ds.name] = (t_feats, t_meta)
        p_t, img_t, grid_t = _analyse(ds.name, st, sink, cfg_df, {}, args)
        cal_t = calibrate_frozen(img_t, cfg_df, picks, img, args.n_boot, args.seed)
        outputs[ds.name] = (p_t, img_t, grid_t, cal_t)

    # 2. probes (computed last: the transfer columns need the target features)
    print("probes ...", flush=True)
    probe = run_probes(sites, feats, meta, fold_row, tgt_feats, args.probe_C)
    probe.to_csv(out_dir / "probe_layers.csv", index=False)

    # --- write ----------------------------------------------------------------
    pd.DataFrame(sink["repro"]).to_csv(out_dir / "repro.csv", index=False)
    oracle_rows, sens = [], None
    for name, (p_, img_, grid_, cal_) in outputs.items():
        p_.to_csv(out_dir / f"{name}_pairs.csv.gz", index=False)
        img_.to_csv(out_dir / f"{name}_images.csv", index=False)
        grid_.to_csv(out_dir / f"{name}_grid.csv", index=False)
        cal_.to_csv(out_dir / f"{name}_calibrated.csv", index=False)
        orc = grid_[grid_.kind == "oracle"].assign(dataset=name)
        oracle_rows.append(orc)
        whole = orc[orc.whole]
        plot_steer_layers(cal_, whole, sites, args.budgets, out_dir / f"steer_layers_{name}.png",
                          f"SAM {args.model_type} steering on {name}: share of the IoU drop repaired, per site "
                          f"({'out of fold' if name == src.name else 'frozen from ' + src.name})")
        _print_layers(name, cal_, whole, sites, args.budgets[len(args.budgets) // 2])
    oracle = pd.concat(oracle_rows, ignore_index=True)
    oracle.to_csv(out_dir / "oracle.csv", index=False)
    sens = grid[grid.kind == "noise"]
    sens.to_csv(out_dir / "sensitivity.csv", index=False)
    mid = args.budgets[len(args.budgets) // 2]
    plot_layer_probe(probe, sens, sites, out_dir / "layer_probe.png",
                     f"SAM {args.model_type} on {src.name}: layer analysis of critical shifts")
    plot_pareto(grid, picks, sites, mid, out_dir / f"pareto_{src.name}.png",
                f"SAM {args.model_type} on {src.name}: restore vs harm over alpha (out of fold vectors; "
                f"group = dm's pick at budget {mid:g})")

    whole_src = oracle[(oracle.dataset == src.name) & oracle.whole].set_index("site").restore
    checks["oracle_cuts"] = {s: float(whole_src.get(s, math.nan)) for s in ("prompt_tok", "low_res")}
    checks["repro"] = (pd.DataFrame(sink["repro"]).groupby(["dataset", "status"]).size()
                       .unstack(fill_value=0).to_dict(orient="index"))
    (out_dir / "checks.json").write_text(json.dumps(checks, indent=2, default=_json_default))
    meta_out = {"model": f"SAM {args.model_type}", "checkpoint": Path(args.checkpoint_path).name,
                "source": src.name, "source_json": str(src.json), "transfer": [d.name for d in targets],
                "transfer_missing": missing, "n_configs": len(cfgs), "n_transfer_configs": len(t_cfgs),
                "images": {n: int(o[1].image.nunique()) for n, o in outputs.items()},
                "pairs": {n: int(o[0].pid.nunique()) for n, o in outputs.items()},
                "folds": len(V_fold), "device": torch.cuda.get_device_name(device) if device.type == "cuda"
                else str(device), "torch": torch.__version__, "args": vars(args),
                "minutes": round((time.time() - t0) / 60, 1), "created": time.strftime("%Y-%m-%d %H:%M")}
    (out_dir / "meta.json").write_text(json.dumps(meta_out, indent=2, default=_json_default))
    z, tv = checks["zero_dose"], checks["trace_vs_hook"]
    print(f"\nchecks: zero dose {z['bit_exact']}/{z['forwards']} bit-exact; hook vs trace {tv['targets']} targets, "
          f"worst logits rel {tv['max_rel_low_res']:.3g}; no-op max rel {checks['noop']['max_rel']:.3g}; "
          f"baseline repeat max |dIoU| {checks['baseline_repeat']['max_abs_iou_diff']:.3g}; "
          f"oracle cuts {checks['oracle_cuts']}; oracle other boxes max change "
          f"{checks['oracle_other_boxes_max_change']:.3g}")
    print(f"wrote {out_dir} ({(time.time() - t0) / 60:.1f} min)")
    return 0


# ---------------------------------------------------------------------------
# Self-test (no data needed)
# ---------------------------------------------------------------------------

def self_test(args) -> int:
    device = torch.device(args.device)
    torch.manual_seed(0)
    print(f"self-test on {device}, weights: {args.checkpoint_path or 'random init'}")
    predictor = load_predictor(args.checkpoint_path or None, args.model_type, device)
    rng = np.random.default_rng(0)
    with torch.inference_mode():
        predictor.set_image(rng.integers(0, 256, size=(427, 640, 3), dtype=np.uint8))
    sites = build_sites(len(predictor.model.mask_decoder.transformer.layers))
    runner = Runner(predictor, sites, batch=3, field_grid=args.field_grid, field_margin=args.field_margin)
    boxes_np = np.array([[100.0, 80.0, 700.0, 600.0], [109.6, 70.4, 700.0, 600.0], [33.6, 25.6, 766.4, 300.8],
                         [512.5, 300.25, 530.75, 320.0], [0.0, 0.0, 1023.0, 682.0]], np.float32)
    boxes = torch.as_tensor(boxes_np, device=device)
    ok = True

    coll = Collector(runner)
    m0, low0 = runner.forward(boxes, record=coll.fns())
    feats = coll.result()
    good = np.array([True, False, True, True, False])
    V = build_vectors(feats, good, ~good, sites, random_directions(sites, feats, 0))
    Vt = vectors_to_device(V, device)
    checks = new_checks()
    try:
        check_zero_and_noop(runner, boxes, m0, low0, Vt, checks, args.trace_tol)
        print(f"a/c. zero dose {checks['zero_dose']['bit_exact']}/{checks['zero_dose']['forwards']} bit-exact; "
              f"no-op max rel {checks['noop']['max_rel']:.3g}")
    except RuntimeError as e:
        ok = False
        print("FAIL", e)
    try:
        check_trace(runner, boxes_np[0], Vt, ["all", "mask0", "mask123", "prompt"], checks, args.trace_tol)
        tv = checks["trace_vs_hook"]
        print(f"b. hook vs trace: {tv['targets']} targets, worst logits rel {tv['max_rel_low_res']:.3g} "
              f"({tv['where'] or '-'}), mask mismatch {tv['mask_mismatch']:.3g}")
    except RuntimeError as e:
        ok = False
        print("FAIL", e)

    # every steering config changes the logits (an edit that is silently ignored would not)
    cfgs = build_configs(sites, argparse.Namespace(groups=["all", "mask0", "mask123", "prompt"], alphas=[1.0],
                                                   head_alphas=[1.0], sens_eps=[0.5]))
    dead = []
    for c in cfgs:
        if c.kind == "steer":
            _, low = runner.forward(boxes, edits={c.site: make_edit(c, runner.sites[c.site], Vt)})
            if torch.equal(low, low0):
                dead.append(f"{c.site}#{c.group}/{c.method}")
    print(f"steering configs without effect: {dead or 'none'}")
    if dead:
        ok = False

    # box frame: a linear logit map is sampled exactly, a linear field is stretched exactly
    b1 = torch.as_tensor([[200.0, 150.0, 520.0, 410.0]], device=device)
    h = w = 256
    px = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * (1024 / w)
    lin = px[None, None, None, :].expand(1, 1, h, w).contiguous()
    grid, valid = runner.field_grid(b1)
    f = F.grid_sample(lin, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0]
    X0, _, X1, _ = runner.region(b1)
    u = (torch.arange(runner.G, device=device) + 0.5) / runner.G
    want = (X0 + u * (X1 - X0))[None, :].expand_as(f)
    inner = (want > 4) & (want < 1020)
    e_fwd = float((f - want)[inner].abs().max())
    G = runner.G
    D = ((torch.arange(G, device=device, dtype=torch.float32) + 0.5) / G * 2 - 1)[None, None, None, :].expand(1, 1, G, G)
    delta = edit_add_field(D.contiguous())(torch.zeros(1, 1, h, w, device=device), Ctx(runner, b1, slice(0, 1)))[0, 0]
    sx = runner.inverse_grid(b1, (h, w))[0, ..., 0]
    sy = runner.inverse_grid(b1, (h, w))[0, ..., 1]
    inside = (sx.abs() < 1 - 1.5 / G) & (sy.abs() < 1 - 1.5 / G)
    outside = (sx.abs() > 1 + 1.5 / G) | (sy.abs() > 1 + 1.5 / G)
    e_inv = float((delta - sx)[inside].abs().max())
    e_out = float(delta[outside].abs().max())
    print(f"box frame: sample err {e_fwd:.2g} px, stretch err {e_inv:.2g}, outside the frame {e_out:.2g}; "
          f"valid share {float(valid.mean()):.2f}")
    if e_fwd > 1e-2 or e_inv > 1e-4 or e_out > 0:
        ok = False
        print("FAIL box-frame geometry")

    # embedding cells: 2x2 cells, and a box without a centre falls back to one cell
    c = runner.cells(torch.as_tensor([[0.0, 0.0, 32.0, 32.0], [100.0, 100.0, 101.0, 101.0]], device=device))
    n_cells = c.sum(1).tolist()
    idx = int(c[1].nonzero()[0, 0])
    print(f"cells: {n_cells} (want [4, 1]), fallback cell {divmod(idx, runner.emb_hw[1])} (want (6, 6))")
    if n_cells != [4.0, 1.0] or divmod(idx, runner.emb_hw[1]) != (6, 6):
        ok = False
        print("FAIL embedding cells")

    # oracle: box 0 = best, box 1 = bad of one pair; at a cut the bad box gets best's mask exactly,
    # and no other box changes at any site
    role = torch.as_tensor([0, 1, 2, 2, 3], device=device)
    store = {s.id: [] for s in sites}
    m0, _ = runner.forward(boxes, record={s.id: donor_recorder(store, s.id, role == 0) for s in sites})
    donors = {k: torch.cat(v) for k, v in store.items()}
    donor_of = torch.as_tensor([-1, 0, -1, -1, -1], device=device)
    other, cut_err = 0.0, {}
    for s in sites:
        for g in oracle_groups(s, ["all", "mask0", "mask123", "prompt"]):
            m, _ = runner.forward(boxes, edits={s.id: make_edit(Cfg(-1, "oracle", s.id, g, "oracle", math.nan), s,
                                                                Vt, donors, donor_of)})
            other = max(other, float(mask_change(m, m0)[role != 1].max()))
            if s.id in ("prompt_tok", "low_res"):
                cut_err[s.id] = float(mask_change(m[1:2], m0[0:1])[0])
    print(f"oracle: other boxes max change {other:.3g}; cut -> donor mask change {cut_err}")
    if other > 0 or any(v > 0 for v in cut_err.values()):
        ok = False
        print("FAIL oracle")

    # noise at every site: changes the output, reproducibly, with norm eps * ||h|| per row / box
    bad_noise = []
    for c in (c for c in cfgs if c.kind == "noise"):
        s = runner.sites[c.site]
        rows = row_index(s, c.group)
        _, l1 = runner.forward(boxes, edits={s.id: edit_noise(s, c.alpha, rows, 123)})
        _, l2 = runner.forward(boxes, edits={s.id: edit_noise(s, c.alpha, rows, 123)})
        t = feats[s.id][:1] if s.kind != "logits" else low0[:1, None]
        t = torch.as_tensor(t, device=device)
        dims = (-1,) if s.kind in ("sparse", "tok", "img") else tuple(range(1, t.dim()))
        if s.kind == "img":                      # the recorded feature is pooled: test on a raw-shaped tensor
            t = torch.randn(1, 4096, 256, device=device)
        d = edit_noise(s, c.alpha, rows, 7)(t, Ctx(runner, boxes, slice(0, 1))) - t
        ratio = torch.linalg.vector_norm(d, dim=dims) / torch.linalg.vector_norm(t, dim=dims)
        ratio = ratio[ratio > 0]
        if torch.equal(l1, low0) or not torch.equal(l1, l2) or float((ratio - c.alpha).abs().max()) > 1e-4:
            bad_noise.append(f"{c.site}#{c.group}@{c.alpha:g}")
    print(f"noise: {sum(c.kind == 'noise' for c in cfgs)} configs, failing: {bad_noise or 'none'}")
    if bad_noise:
        ok = False
        print("FAIL noise")
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source_name", default="COCO_MVal", help=f"source dataset, one of {list(DATASETS)}")
    p.add_argument("--critical_shifts", default=None, help="source JSON (default: the source's entry in DATASETS)")
    p.add_argument("--images_dir", help="source images, e.g. COCO_MVal/img")
    p.add_argument("--masks_dir", default=None, help="source GT (default: <images_dir>/../gt)")
    p.add_argument("--mask_ext", default=None, help="source GT extension (default: from DATASETS)")
    p.add_argument("--transfer", nargs="*", default=["Berkeley", "GrabCut"], choices=list(DATASETS),
                   help="target datasets, found as <root>/<name>/<images>, <root>/<name>/<gt>; none = no transfer")
    p.add_argument("--datasets_root", nargs="*", default=None,
                   help="roots tried in order for --transfer (default: <images_dir>/../..)")
    p.add_argument("--checkpoint_path", default="", help="sam_vit_b_01ec64.pth")
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=64, help="boxes per predict_torch call")
    p.add_argument("--repro_tol", type=float, default=0.02, help="max |IoU recomputed - IoU in JSON|")
    p.add_argument("--n_controls", type=int, default=4)
    p.add_argument("--ctrl_min_iou", type=float, default=0.85)
    p.add_argument("--ctrl_max_cos", type=float, default=0.9)
    p.add_argument("--ctrl_tries", type=int, default=0, help="0 = all rearrangements of the attack shift")
    p.add_argument("--dense_per_image", type=int, default=32, help="jittered GT boxes per image (harm check)")
    p.add_argument("--dense_jitter", type=float, default=0.1, help="max side shift, share of the GT box size")
    p.add_argument("--folds", type=int, default=5, help="image folds for the out-of-fold vectors and probes")
    p.add_argument("--groups", type=lambda s: s.split(","), default=["all", "mask0", "mask123", "prompt"],
                   help="token groups steered on the token stream")
    p.add_argument("--alphas", type=_floats, default=_floats("0.03125,0.0625,0.125,0.25,0.5,1,2,4,8"),
                   help="residual steering strengths (dm at 1 moves the bad mean onto the good mean)")
    p.add_argument("--head_alphas", type=_floats,
                   default=_floats("0.0078125,0.015625,0.03125,0.0625,0.125,0.25,0.5,1,2"),
                   help="action-head steering strengths")
    p.add_argument("--budgets", type=_floats, default=_floats("0.01,0.02,0.05"),
                   help="max mask change on the good boxes, for the calibrated view")
    p.add_argument("--sens_eps", type=_floats, default=_floats("0.1,0.5"), help="readout-sensitivity eps")
    p.add_argument("--field_grid", type=int, default=64, help="G of the box-frame logit field")
    p.add_argument("--field_margin", type=float, default=0.25, help="box-frame margin, share of the box size")
    p.add_argument("--probe_C", type=float, default=1.0, help="inverse L2 strength of the probes")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--check_images", type=int, default=2, help="images per dataset with the zero-dose/no-op checks")
    p.add_argument("--trace_tol", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit_images", type=int, default=0, help="per dataset, 0 = all")
    p.add_argument("--out_dir", default="exp_res/steer_sam")
    p.add_argument("--self_test", action="store_true",
                   help="hooks, trace agreement, box frames and oracle on a random image (random weights)")
    args = p.parse_args(argv)
    if not args.self_test and not (args.images_dir and args.checkpoint_path):
        p.error("--images_dir and --checkpoint_path are required (or use --self_test)")
    unknown = [g for g in args.groups if g not in ("all", *TOK_ROWS)]
    if unknown:
        p.error(f"unknown token groups {unknown}; known: all, {', '.join(TOK_ROWS)}")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
