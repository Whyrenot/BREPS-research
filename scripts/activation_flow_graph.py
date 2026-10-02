"""
activation_flow_graph.py
========================
Where along SAM-1's inference does a critical box shift break the mask?

A critical pair (best_box, bad_box) from find_critical_shifts.py is two box
prompts a few pixels apart on the same image: best gives a good mask, bad a
broken one. For every pair this script runs SAM on best, bad and K control
boxes, and draws the prompt encoder + mask decoder as a directed dataflow graph
(one self-contained HTML file):

  node colour  ratio = rel(best, bad) / rel(best, control)
               rel(a, b) = ||A_b - A_a||_2 / ||A_a||_2 on the node's tensor
  edge colour  gain along the edge, rel(dst) / rel(src); by default the
               specific gain gain(bad) / gain(control) = ratio(dst) / ratio(src)

A control box is shifted from best_box by the same L-inf distance as bad_box,
in a random direction (cosine with the attack direction <= --ctrl_max_cos),
and is kept only if its mask is still good (IoU with GT >= --ctrl_min_iou).
ratio ~ 1 means the node reacts to the critical shift like to any shift of
that size; the layer that breaks the mask is where ratio jumps, i.e. where the
specific gain on the incoming edges is >> 1.

Node labels use the terms of the paper (Kirillov et al., "Segment Anything",
ICCV 2023, Appendix A); the exact attribute path in facebookresearch/
segment-anything is shown next to each node.

WHY A TRACE AND NOT ONLY HOOKS
------------------------------
Forward hooks only see module outputs. The decoder's dataflow also runs through
things that are not modules: residual sums, the prompt tokens re-added before
every attention, the image PE added to the image side, the first layer's
self-attention that has NO residual (skip_first_layer_pe), the q/k swap in
image-to-token attention, the token slicing and the mask-token selection of
multimask_output=False. So the decoder is re-executed step by step here
(trace_forward, a line-by-line copy of segment_anything's forward()s calling
SAM's own submodules), and that trace is verified against SAM itself:

  1. every module of prompt_encoder / mask_decoder that fires during the
     official SamPredictor.predict_torch() is hooked; it must fire exactly
     once, its output must equal the traced tensor of the same name, and every
     traced module must have fired (nothing missed, nothing invented);
  2. the traced binary mask equals the official one;
  3. running best_box twice gives identical activations (noise floor = 0);
  4. box-independent tensors (image embedding, 'no mask' embedding, image PE,
     learned tokens, layer-1 k/v of token-to-image attention, ...) diverge by
     exactly 0;
  5. the JSON boxes lie on the integer pixel grid of the original image after
     dividing by SAM's resize factor, i.e. they really are in SAM's 1024-frame.
Any failure of 1-2 aborts the run: a graph built from activations SAM did not
use is worse than no graph. 3-5 are reported in the HTML.

GROUNDING IN THE JSON
---------------------
critical_shifts_coco.json was produced from heatmaps/comp_hw.py tables
(fp32, no autocast, SamPredictor.predict_torch with multimask_output=False,
mask = logits > 0 at the original resolution, IoU = inter / (union + 1e-6),
boxes stored after ResizeLongestSide, i.e. in SAM's 1024-frame). That pipeline
is reproduced here exactly; a pair is used only if the recomputed IoU of both
boxes is within --repro_tol of the JSON. Dropped pairs are listed in the
report. This script imports nothing from the earlier probe_decoder_activations
scripts or from heatmaps/.

Outputs (in --out_dir)
----------------------
  activation_graph.html   the interactive graph (open in a browser)
  per_pair_keys.csv       rel(best,bad), rel(best,control), ratio per traced tensor
  per_pair_edges.csv      gains per edge
  repro.csv               JSON vs recomputed IoU for every pair, kept/dropped
  checks.json             results of checks 1-5

Example
-------
    CUDA_VISIBLE_DEVICES=0 python scripts/activation_flow_graph.py \\
        --critical_shifts critical_shifts_coco.json \\
        --images_dir /.../datasets/COCO_MVal/img \\
        --masks_dir  /.../datasets/COCO_MVal/gt \\
        --checkpoint_path /.../MODEL_CHECKPOINTS/SAM/sam_vit_b_01ec64.pth \\
        --out_dir exp_res/activation_graph

    # no data needed: verifies the trace on random weights and a random image
    python scripts/activation_flow_graph.py --self_test --device cpu
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-12
ZERO = 1e-9            # rel below this = the tensor does not depend on the box
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG")

# Element names of the tuples returned by official forward()s, taken from
# their `return` statements in segment_anything/modeling/*.py.
TUPLE_ITEMS = {
    "PromptEncoder": ("sparse_embeddings", "dense_embeddings"),
    "MaskDecoder": ("masks", "iou_pred"),
    "TwoWayTransformer": ("queries", "keys"),
    "TwoWayAttentionBlock": ("queries", "keys"),
}

# Token layout of the decoder's token stream for a single box prompt:
# [IoU token | 4 mask tokens | 2 box-corner prompt tokens].
TOKEN_GROUPS = OrderedDict([
    ("iou", ("IoU output token", slice(0, 1))),
    ("mask0", ("mask output token #0 (the one used)", slice(1, 2))),
    ("mask123", ("mask output tokens #1-3 (discarded)", slice(2, 5))),
    ("prompt", ("prompt tokens (box corners)", slice(5, 7))),
])
N_TOKENS = 7


# ---------------------------------------------------------------------------
# Data loading -- mirrors heatmaps/comp_hw.py (the producer of the JSON's CSVs)
# ---------------------------------------------------------------------------

def load_predictor(checkpoint: str | None, model_type: str, device: torch.device):
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=checkpoint or None)
    for p in sam.parameters():
        p.requires_grad_(False)
    sam.to(device)
    sam.eval()
    return SamPredictor(sam)


def load_image_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"cannot read image {path}")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if img.shape[0] > 1024 and img.shape[1] > 1024:      # comp_hw.prepare_input
        img = cv2.resize(img, (1024, 1024))
    return img


def load_gt(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"cannot read mask {path}")
    if m.shape[0] > 1024 and m.shape[1] > 1024:           # comp_hw.evaluate_mask
        m = cv2.resize(m, (1024, 1024))
    return m > 0


def find_by_stem(directory: Path, stem: str, exts) -> Path | None:
    for e in exts:
        p = directory / (stem + e)
        if p.is_file():
            return p
    return None


def mask_iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """heatmaps/comp_hw.py batch_iou_torch for one mask."""
    inter = (pred & gt).sum().float()
    union = (pred | gt).sum().float()
    return float(inter / (union + 1e-6))


# ---------------------------------------------------------------------------
# The trace: segment_anything's box -> mask path, unrolled
# ---------------------------------------------------------------------------

def _attention(attn, q, k, v, rec, name):
    """segment_anything/modeling/transformer.py  Attention.forward."""
    q = rec(f"{name}.q_proj", attn.q_proj(q))
    k = rec(f"{name}.k_proj", attn.k_proj(k))
    v = rec(f"{name}.v_proj", attn.v_proj(v))
    q = attn._separate_heads(q, attn.num_heads)
    k = attn._separate_heads(k, attn.num_heads)
    v = attn._separate_heads(v, attn.num_heads)
    _, _, _, c_per_head = q.shape
    a = q @ k.permute(0, 1, 3, 2)
    a = a / math.sqrt(c_per_head)
    a = rec(f"{name}.softmax", torch.softmax(a, dim=-1))
    out = a @ v
    out = attn._recombine_heads(out)
    out = rec(f"{name}.out_proj", attn.out_proj(out))
    return rec(name, out)


def _mlp(mlp, x, rec, name):
    """segment_anything/modeling/mask_decoder.py  MLP.forward (records pre-ReLU)."""
    for i, layer in enumerate(mlp.layers):
        y = rec(f"{name}.layers.{i}", layer(x))
        x = F.relu(y) if i < mlp.num_layers - 1 else y
    if mlp.sigmoid_output:
        x = F.sigmoid(x)
    return rec(name, x)


@torch.inference_mode()
def trace_forward(predictor, box_1024) -> "OrderedDict[str, torch.Tensor]":
    """Every tensor on the path box -> mask, keyed by the official module path
    where the tensor is a module output, and by a descriptive key otherwise.

    Mirrors SamPredictor.predict_torch(None, None, boxes=box[None],
    multimask_output=False) on the image set with predictor.set_image().
    """
    sam = predictor.model
    pe, dec = sam.prompt_encoder, sam.mask_decoder
    T: OrderedDict[str, torch.Tensor] = OrderedDict()

    def rec(key, t):
        if key in T:
            raise KeyError(f"trace key recorded twice: {key}")
        T[key] = t
        return t

    box = torch.as_tensor(np.asarray(box_1024, dtype=np.float32), device=predictor.device)[None]
    rec("box", box)

    # --- PromptEncoder.forward(points=None, boxes=box, masks=None) ---
    bs = box.shape[0]
    sparse = torch.empty((bs, 0, pe.embed_dim), device=pe._get_device())
    boxes = box + 0.5                                   # _embed_boxes
    coords = boxes.reshape(-1, 2, 2)
    corner = pe.pe_layer.forward_with_coords(coords, pe.input_image_size)
    rec("prompt_encoder.corner_pe", corner.clone())
    rec("prompt_encoder.corner_learned",
        torch.cat([pe.point_embeddings[2].weight, pe.point_embeddings[3].weight], dim=0)[None])
    corner[:, 0, :] += pe.point_embeddings[2].weight
    corner[:, 1, :] += pe.point_embeddings[3].weight
    sparse = torch.cat([sparse, corner], dim=1)
    dense = pe.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
        bs, -1, pe.image_embedding_size[0], pe.image_embedding_size[1]
    )
    rec("prompt_encoder[sparse_embeddings]", sparse)
    rec("prompt_encoder[dense_embeddings]", dense)

    # --- predict_torch: image_embeddings=self.features, image_pe=get_dense_pe() ---
    image_embeddings = rec("image_embedding", predictor.features)
    image_pe = rec("prompt_encoder.pe_layer", pe.pe_layer(pe.image_embedding_size)).unsqueeze(0)

    # --- MaskDecoder.predict_masks ---
    output_tokens = torch.cat([dec.iou_token.weight, dec.mask_tokens.weight], dim=0)
    output_tokens = output_tokens.unsqueeze(0).expand(sparse.size(0), -1, -1)
    rec("mask_decoder.output_tokens", output_tokens)
    tokens = rec("mask_decoder.tokens", torch.cat((output_tokens, sparse), dim=1))
    src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
    src = rec("mask_decoder.src", src + dense)
    pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
    b, c, h, w = src.shape

    # --- TwoWayTransformer.forward(src, pos_src, tokens) ---
    tr = dec.transformer
    image_embedding = src.flatten(2).permute(0, 2, 1)
    image_pe_flat = pos_src.flatten(2).permute(0, 2, 1)
    point_embedding = tokens
    queries, keys = point_embedding, image_embedding
    for li, layer in enumerate(tr.layers):
        # --- TwoWayAttentionBlock.forward(queries, keys, query_pe=point_embedding, key_pe=image_pe) ---
        p = f"mask_decoder.transformer.layers.{li}"
        query_pe, key_pe = point_embedding, image_pe_flat
        if layer.skip_first_layer_pe:
            queries = _attention(layer.self_attn, queries, queries, queries, rec, f"{p}.self_attn")
        else:
            q = queries + query_pe
            attn_out = _attention(layer.self_attn, q, q, queries, rec, f"{p}.self_attn")
            queries = queries + attn_out
        queries = rec(f"{p}.norm1", layer.norm1(queries))

        q = queries + query_pe
        k = keys + key_pe
        attn_out = _attention(layer.cross_attn_token_to_image, q, k, keys, rec,
                              f"{p}.cross_attn_token_to_image")
        queries = queries + attn_out
        queries = rec(f"{p}.norm2", layer.norm2(queries))

        h1 = rec(f"{p}.mlp.lin1", layer.mlp.lin1(queries))       # MLPBlock.forward
        a1 = rec(f"{p}.mlp.act", layer.mlp.act(h1))
        mlp_out = rec(f"{p}.mlp.lin2", layer.mlp.lin2(a1))
        rec(f"{p}.mlp", mlp_out)
        queries = queries + mlp_out
        queries = rec(f"{p}.norm3", layer.norm3(queries))

        q = queries + query_pe
        k = keys + key_pe
        # official call: cross_attn_image_to_token(q=k, k=q, v=queries)
        attn_out = _attention(layer.cross_attn_image_to_token, k, q, queries, rec,
                              f"{p}.cross_attn_image_to_token")
        keys = keys + attn_out
        keys = rec(f"{p}.norm4", layer.norm4(keys))
        rec(f"{p}[queries]", queries)
        rec(f"{p}[keys]", keys)

    q = queries + point_embedding
    k = keys + image_pe_flat
    attn_out = _attention(tr.final_attn_token_to_image, q, k, keys, rec,
                          "mask_decoder.transformer.final_attn_token_to_image")
    queries = queries + attn_out
    queries = rec("mask_decoder.transformer.norm_final_attn", tr.norm_final_attn(queries))
    rec("mask_decoder.transformer[queries]", queries)
    rec("mask_decoder.transformer[keys]", keys)
    hs, src = queries, keys

    iou_token_out = rec("mask_decoder.iou_token_out", hs[:, 0, :])
    mask_tokens_out = rec("mask_decoder.mask_tokens_out", hs[:, 1:(1 + dec.num_mask_tokens), :])
    rec("mask_decoder.mask_tokens_out[0]", mask_tokens_out[:, 0, :])

    src = src.transpose(1, 2).view(b, c, h, w)
    x = src
    for i, m in enumerate(dec.output_upscaling):
        x = rec(f"mask_decoder.output_upscaling.{i}", m(x))
    upscaled = rec("mask_decoder.output_upscaling", x)
    hyper_in_list = [
        _mlp(dec.output_hypernetworks_mlps[i], mask_tokens_out[:, i, :], rec,
             f"mask_decoder.output_hypernetworks_mlps.{i}")
        for i in range(dec.num_mask_tokens)
    ]
    hyper_in = torch.stack(hyper_in_list, dim=1)
    b, c, h, w = upscaled.shape
    masks = (hyper_in @ upscaled.view(b, c, h * w)).view(b, -1, h, w)
    rec("mask_decoder.masks_all", masks)
    iou_pred = _mlp(dec.iou_prediction_head, iou_token_out, rec, "mask_decoder.iou_prediction_head")

    # --- MaskDecoder.forward, multimask_output=False -> slice(0, 1) ---
    low_res = rec("mask_decoder[masks]", masks[:, slice(0, 1), :, :])
    rec("mask_decoder[iou_pred]", iou_pred[:, slice(0, 1)])

    # --- SamPredictor.predict_torch: postprocess + threshold ---
    up = sam.postprocess_masks(low_res, predictor.input_size, predictor.original_size)
    rec("postprocess_masks", up)
    rec("mask", up > sam.mask_threshold)
    return T


# ---------------------------------------------------------------------------
# Verification of the trace against SAM's own forward
# ---------------------------------------------------------------------------

class ModuleRecorder:
    """Forward hooks on every submodule of SAM; records each call's output."""

    def __init__(self, model):
        self.model = model
        self.calls: dict[str, list] = {}
        self._handles = []

    def __enter__(self):
        for name, m in self.model.named_modules():
            if name:
                self._handles.append(m.register_forward_hook(self._hook(name)))
        return self

    def _hook(self, name):
        def fn(_mod, _inp, out):
            outs = out if isinstance(out, tuple) else (out,)
            self.calls.setdefault(name, []).append(tuple(o.detach().clone() for o in outs))
        return fn

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _cmp(a: torch.Tensor, b: torch.Tensor) -> tuple[float, bool]:
    """(relative max abs difference, bit-exact) of trace tensor a vs official b."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return math.inf, False
    if torch.equal(a, b):
        return 0.0, True
    if a.dtype == torch.bool:
        return float((a != b).float().mean()), False
    d = (a.double() - b.double()).abs().max().item()
    s = b.double().abs().max().item()
    return d / max(s, EPS), False


class CheckLog:
    def __init__(self):
        self.runs = 0
        self.exact_runs = 0
        self.tensors = 0
        self.worst = 0.0
        self.worst_where = ""
        self.fired: list[str] | None = None

    def as_dict(self):
        return {
            "runs": self.runs, "bit_exact_runs": self.exact_runs,
            "tensors_compared": self.tensors, "worst_rel_diff": self.worst,
            "worst_where": self.worst_where, "modules_fired": self.fired or [],
        }


def compare_with_official(sam, calls, T, official_mask, tol):
    names = dict(sam.named_modules())
    errs, worst, where, n, n_exact = [], 0.0, "", 0, 0
    for name, outs in calls.items():
        if len(outs) != 1:
            errs.append(f"{name}: fired {len(outs)}x in one prediction (expected once)")
            continue
        items = TUPLE_ITEMS.get(type(names[name]).__name__)
        keys = [f"{name}[{it}]" for it in items] if items else [name]
        if len(keys) != len(outs[0]):
            errs.append(f"{name}: returned {len(outs[0])} tensors, expected {len(keys)}")
            continue
        for key, got in zip(keys, outs[0]):
            if key not in T:
                errs.append(f"{key}: ran in SAM's forward but is missing from the trace")
                continue
            d, exact = _cmp(T[key], got)
            n += 1
            n_exact += exact
            if d > worst:
                worst, where = d, key
            if d > tol:
                errs.append(f"{key}: trace differs from SAM's own output (rel {d:.3g})")
    for key in T:
        base = key.split("[")[0]
        if base in names and base not in calls:
            errs.append(f"{key}: traced, but module '{base}' never ran in SAM's forward")
    d, exact = _cmp(T["mask"], official_mask)
    n += 1
    n_exact += exact
    if d > worst:
        worst, where = d, "mask"
    if d > tol:
        errs.append(f"mask: traced binary mask differs from SamPredictor's (fraction {d:.3g})")
    return errs, worst, where, n, n_exact


def trace_and_verify(predictor, box_1024, log: CheckLog, tol: float):
    """Official prediction (hooked) + trace; raises if they disagree."""
    sam = predictor.model
    box_t = torch.as_tensor(np.asarray(box_1024, dtype=np.float32), device=predictor.device)[None]
    with ModuleRecorder(sam) as recorder, torch.inference_mode():
        masks, iou_pred, _ = predictor.predict_torch(None, None, boxes=box_t, multimask_output=False)
    T = trace_forward(predictor, box_1024)
    errs, worst, where, n, n_exact = compare_with_official(sam, recorder.calls, T, masks, tol)
    log.runs += 1
    log.tensors += n
    log.exact_runs += int(n_exact == n)
    if worst > log.worst:
        log.worst, log.worst_where = worst, where
    if log.fired is None:
        log.fired = sorted(recorder.calls)
    if errs:
        raise RuntimeError("trace does not reproduce SAM's forward:\n  " + "\n  ".join(errs[:30]))
    return T, masks[0, 0], float(iou_pred[0, 0])


@torch.inference_mode()
def predict_official(predictor, box_1024):
    box_t = torch.as_tensor(np.asarray(box_1024, dtype=np.float32), device=predictor.device)[None]
    masks, iou_pred, _ = predictor.predict_torch(None, None, boxes=box_t, multimask_output=False)
    return masks[0, 0], float(iou_pred[0, 0])


# Tensors that cannot depend on the box: their divergence must be exactly 0.
def box_independent_keys(depth: int) -> list[str]:
    keys = [
        "image_embedding", "prompt_encoder[dense_embeddings]", "prompt_encoder.pe_layer",
        "prompt_encoder.corner_learned", "mask_decoder.output_tokens", "mask_decoder.src",
        # layer 1: the image side is untouched until image-to-token attention
        "mask_decoder.transformer.layers.0.cross_attn_token_to_image.k_proj",
        "mask_decoder.transformer.layers.0.cross_attn_token_to_image.v_proj",
        "mask_decoder.transformer.layers.0.cross_attn_image_to_token.q_proj",
    ]
    return keys if depth >= 1 else []


# ---------------------------------------------------------------------------
# Divergence metrics
# ---------------------------------------------------------------------------

@torch.inference_mode()
def rel_divergence(T_ref, T_x) -> dict[str, float]:
    """rel = ||x - ref||_2 / ||ref||_2 per traced tensor, plus per token group."""
    out = {}
    for key, a in T_ref.items():
        bx = T_x[key]
        a64, b64 = a.double(), bx.double()
        num = torch.linalg.vector_norm(b64 - a64).item()
        den = torch.linalg.vector_norm(a64).item()
        out[key] = num / den if den > EPS else (0.0 if num <= EPS else math.inf)
        if a.dim() == 3 and a.shape[1] == N_TOKENS:
            for gid, (_, sl) in TOKEN_GROUPS.items():
                ag, bg = a64[:, sl], b64[:, sl]
                num = torch.linalg.vector_norm(bg - ag).item()
                den = torch.linalg.vector_norm(ag).item()
                out[f"{key}#{gid}"] = num / den if den > EPS else (0.0 if num <= EPS else math.inf)
    return out


# ---------------------------------------------------------------------------
# Graph specification (paper terms, official code paths)
# ---------------------------------------------------------------------------

LANES = [
    {"id": "S", "label": "token sub-layers"},
    {"id": "T", "label": "tokens (residual stream)"},
    {"id": "X", "label": "cross-attention"},
    {"id": "I", "label": "image embedding (residual stream)"},
]

Q_DECODER_STEPS = {
    "self": "(1) self-attention on the tokens",
    "t2i": "(2) cross-attention from tokens (as queries) to the image embedding",
    "mlp": "(3) a point-wise MLP updates each token",
    "i2t": "(4) cross-attention from the image embedding (as queries) to tokens",
    "norm": "Each self/cross-attention and MLP has a residual connection, layer normalization",
}
Q_READD = ("the entire original prompt tokens (including their positional encodings) are "
           "re-added to the updated tokens whenever they participate in an attention layer")
Q_IMG_PE = "positional encodings are added to the image embedding whenever they participate in an attention layer"


def _attn_internals(path, kind):
    if kind == "self":
        roles = ("q <- tokens (+ re-added prompt tokens from layer 2)", "k <- same as q",
                 "v <- tokens", "attention weights, 7 x 7 tokens")
    elif kind == "t2i":
        roles = ("q <- tokens + re-added prompt tokens", "k <- image embedding + image PE",
                 "v <- image embedding", "attention weights, 7 tokens x 4096 positions")
    else:
        roles = ("q <- image embedding + image PE  (code passes q=k)",
                 "k <- tokens + re-added prompt tokens  (code passes k=q)",
                 "v <- tokens", "attention weights, 4096 positions x 7 tokens")
    return [
        [f"{path}.q_proj", f"q_proj  {roles[0]}"],
        [f"{path}.k_proj", f"k_proj  {roles[1]}"],
        [f"{path}.v_proj", f"v_proj  {roles[2]}"],
        [f"{path}.softmax", f"softmax  {roles[3]}"],
        [f"{path}.out_proj", "out_proj"],
    ]


def build_spec(depth: int) -> dict:
    nodes, edges, groups = [], [], []

    def node(nid, label, key, lane, row, code, kind="var", quote="", internals=(), tok=False, note=""):
        nodes.append({"id": nid, "label": label, "key": key, "lane": lane, "row": row,
                      "code": code, "kind": kind, "quote": quote,
                      "internals": [list(x) for x in internals], "tok": tok, "note": note})

    def edge(src, dst, role, label=""):
        edges.append({"id": f"{src}>{dst}", "src": src, "dst": dst, "role": role, "label": label})

    S, Tl, X, I = 0, 1, 2, 3
    q_box = ("A box is represented by an embedding pair: (1) the positional encoding of its "
             "top-left corner summed with a learned embedding representing 'top-left corner' and "
             "(2) the same structure but using a learned embedding indicating 'bottom-right corner'.")

    # --- inputs and prompt encoder -------------------------------------
    node("box", "box prompt", "box", S, 0, "SamPredictor.predict_torch(boxes=...)  [1024-frame xyxy]",
         note="rel = ||box_x - box_best|| / ||box_best|| on the 4 coordinates: the size of the input perturbation.")
    node("img_emb", "image embedding", "image_embedding", I, 0, "image_encoder(...) -> predictor.features",
         kind="const", note="Same image inside a pair: identical by construction.")
    node("corner_pe", "positional encoding (corners)", "prompt_encoder.corner_pe", S, 1,
         "prompt_encoder.pe_layer.forward_with_coords(box + 0.5)", quote=q_box)
    node("corner_emb", "learned corner embeddings", "prompt_encoder.corner_learned", Tl, 1,
         "prompt_encoder.point_embeddings[2], [3]", kind="const", quote=q_box)
    node("no_mask", "'no mask' embedding", "prompt_encoder[dense_embeddings]", X, 2,
         "prompt_encoder.no_mask_embed -> dense_embeddings", kind="const",
         quote="If there is no mask prompt, a learned embedding representing 'no mask' is added to each image embedding location.")
    node("prompt_tok", "prompt tokens", "prompt_encoder[sparse_embeddings]", S, 2,
         "prompt_encoder(...) -> sparse_embeddings", quote=q_box)
    node("img_pe", "image positional encoding", "prompt_encoder.pe_layer", X, 0,
         "prompt_encoder.get_dense_pe()", kind="const", quote=Q_IMG_PE)
    node("out_tok", "output tokens", "mask_decoder.output_tokens", S, 3,
         "mask_decoder.iou_token, mask_decoder.mask_tokens", kind="const",
         quote="we first insert into the set of prompt embeddings a learned output token embedding "
               "that will be used at the decoder's output, analogous to the [class] token")
    node("tokens", "output tokens + prompt tokens", "mask_decoder.tokens", Tl, 3,
         "mask_decoder.predict_masks: tokens = cat(output_tokens, sparse_prompt_embeddings)", tok=True,
         note="This tensor is also re-added before every attention (dashed 'PE re-add' edges).")
    node("src", "image embedding + 'no mask'", "mask_decoder.src", I, 3,
         "mask_decoder.predict_masks: src = image_embeddings + dense_prompt_embeddings", kind="const")
    edge("box", "corner_pe", "flow", "+0.5, /1024, random Fourier PE")
    edge("corner_pe", "prompt_tok", "flow", "+")
    edge("corner_emb", "prompt_tok", "flow", "+")
    edge("prompt_tok", "tokens", "flow", "concat")
    edge("out_tok", "tokens", "flow", "concat")
    edge("img_emb", "src", "flow", "+")
    edge("no_mask", "src", "flow", "+")
    groups.append({"label": "prompt encoder", "code": "sam.prompt_encoder", "row0": 0, "row1": 2,
                   "lane0": 0, "lane1": 1})

    # --- two-way transformer layers -------------------------------------
    q_prev, k_prev = "tokens", "src"
    for li in range(depth):
        r0 = 4 + 8 * li
        p = f"mask_decoder.transformer.layers.{li}"
        n = f"L{li + 1}."
        first = li == 0
        node(n + "self", "self attn.", f"{p}.self_attn", S, r0, f"{p}.self_attn",
             quote=Q_DECODER_STEPS["self"], internals=_attn_internals(f"{p}.self_attn", "self"), tok=True,
             note=("Layer 1 (skip_first_layer_pe=True): q = k = v = tokens, no PE re-add." if first else ""))
        node(n + "norm1", "norm (no residual)" if first else "add & norm", f"{p}.norm1", Tl, r0 + 1,
             f"{p}.norm1", quote=Q_DECODER_STEPS["norm"], tok=True,
             note=("Layer 1: queries = norm1(self_attn(...)); the self-attention REPLACES the tokens, "
                   "there is no residual sum." if first else ""))
        node(n + "t2i", "token to image attn.", f"{p}.cross_attn_token_to_image", X, r0 + 2,
             f"{p}.cross_attn_token_to_image", quote=Q_DECODER_STEPS["t2i"],
             internals=_attn_internals(f"{p}.cross_attn_token_to_image", "t2i"), tok=True)
        node(n + "norm2", "add & norm", f"{p}.norm2", Tl, r0 + 3, f"{p}.norm2",
             quote=Q_DECODER_STEPS["norm"], tok=True)
        node(n + "mlp", "mlp", f"{p}.mlp", S, r0 + 4, f"{p}.mlp  (lin1 -> ReLU -> lin2)",
             quote=Q_DECODER_STEPS["mlp"], tok=True,
             internals=[[f"{p}.mlp.lin1", "lin1  256 -> 2048"], [f"{p}.mlp.act", "act (ReLU)"],
                        [f"{p}.mlp.lin2", "lin2  2048 -> 256"]])
        node(n + "norm3", "add & norm", f"{p}.norm3", Tl, r0 + 5, f"{p}.norm3",
             quote=Q_DECODER_STEPS["norm"], tok=True)
        node(n + "i2t", "image to token attn.", f"{p}.cross_attn_image_to_token", X, r0 + 6,
             f"{p}.cross_attn_image_to_token", quote=Q_DECODER_STEPS["i2t"],
             internals=_attn_internals(f"{p}.cross_attn_image_to_token", "i2t"))
        node(n + "norm4", "add & norm", f"{p}.norm4", I, r0 + 7, f"{p}.norm4  (updated image embedding)",
             quote=Q_DECODER_STEPS["norm"])

        if first:
            edge(q_prev, n + "self", "flow", "q = k = v (no PE)")
            edge(n + "self", n + "norm1", "flow", "no residual")
        else:
            edge(q_prev, n + "self", "flow", "q, k (+PE), v")
            edge("tokens", n + "self", "pe", "re-added prompt tokens (q, k)")
            edge(q_prev, n + "norm1", "res", "residual")
            edge(n + "self", n + "norm1", "flow", "+")
        edge(n + "norm1", n + "t2i", "flow", "q")
        edge("tokens", n + "t2i", "pe", "re-added prompt tokens (q)")
        edge(k_prev, n + "t2i", "flow", "k, v")
        edge("img_pe", n + "t2i", "pe", "image PE (k)")
        edge(n + "norm1", n + "norm2", "res", "residual")
        edge(n + "t2i", n + "norm2", "flow", "+")
        edge(n + "norm2", n + "mlp", "flow", "")
        edge(n + "norm2", n + "norm3", "res", "residual")
        edge(n + "mlp", n + "norm3", "flow", "+")
        edge(n + "norm3", n + "i2t", "flow", "k, v (tokens)")
        edge("tokens", n + "i2t", "pe", "re-added prompt tokens (k)")
        edge(k_prev, n + "i2t", "flow", "q (image)")
        edge("img_pe", n + "i2t", "pe", "image PE (q)")
        edge(k_prev, n + "norm4", "res", "residual")
        edge(n + "i2t", n + "norm4", "flow", "+")
        groups.append({"label": f"decoder layer {li + 1}", "code": p,
                       "row0": r0, "row1": r0 + 7, "lane0": 0, "lane1": 3})
        q_prev, k_prev = n + "norm3", n + "norm4"

    # --- final attention -------------------------------------------------
    rf = 4 + 8 * depth
    pf = "mask_decoder.transformer"
    node("final_t2i", "token to image attn.", f"{pf}.final_attn_token_to_image", X, rf,
         f"{pf}.final_attn_token_to_image",
         quote="the tokens attend once more to the image embedding",
         internals=_attn_internals(f"{pf}.final_attn_token_to_image", "t2i"), tok=True)
    node("final_norm", "add & norm", f"{pf}.norm_final_attn", Tl, rf + 1, f"{pf}.norm_final_attn", tok=True)
    edge(q_prev, "final_t2i", "flow", "q")
    edge("tokens", "final_t2i", "pe", "re-added prompt tokens (q)")
    edge(k_prev, "final_t2i", "flow", "k, v")
    edge("img_pe", "final_t2i", "pe", "image PE (k)")
    edge(q_prev, "final_norm", "res", "residual")
    edge("final_t2i", "final_norm", "flow", "+")
    groups.append({"label": "final attention", "code": f"{pf}.final_attn_token_to_image / norm_final_attn",
                   "row0": rf, "row1": rf + 1, "lane0": 0, "lane1": 3})

    # --- output heads ----------------------------------------------------
    ro = rf + 2
    md = "mask_decoder"
    node("iou_tok", "IoU output token", f"{md}.iou_token_out", S, ro, "hs[:, 0, :]  (iou_token_out)")
    node("mask_tok", "mask output token", f"{md}.mask_tokens_out[0]", Tl, ro,
         "hs[:, 1, :]  (mask_tokens_out[:, 0]; tokens 1-3 are discarded with multimask_output=False)")
    node("upscaled", "2x conv. trans.", f"{md}.output_upscaling", I, ro,
         f"{md}.output_upscaling  (ConvT, LayerNorm2d, GELU, ConvT, GELU)",
         quote="we upsample the updated image embedding by 4x with two transposed convolutional layers",
         internals=[[f"{md}.output_upscaling.0", "0  ConvTranspose2d 256 -> 64"],
                    [f"{md}.output_upscaling.1", "1  LayerNorm2d"],
                    [f"{md}.output_upscaling.2", "2  GELU"],
                    [f"{md}.output_upscaling.3", "3  ConvTranspose2d 64 -> 32"],
                    [f"{md}.output_upscaling.4", "4  GELU"]])
    node("iou_scores", "IoU scores", f"{md}[iou_pred]", S, ro + 1, f"{md}.iou_prediction_head -> iou_pred[:, 0]",
         quote="we add a small head (operating on an additional output token) that estimates the IoU "
               "between each predicted mask and the object it covers",
         internals=[[f"{md}.iou_prediction_head.layers.0", "layers.0 (pre-ReLU)"],
                    [f"{md}.iou_prediction_head.layers.1", "layers.1 (pre-ReLU)"],
                    [f"{md}.iou_prediction_head.layers.2", "layers.2: 4 scores, #0 used"]])
    node("hyper", "MLP -> dynamic classifier", f"{md}.output_hypernetworks_mlps.0", Tl, ro + 1,
         f"{md}.output_hypernetworks_mlps.0",
         quote="pass the updated output token embedding to a small 3-layer MLP that outputs a vector "
               "matching the channel dimension of the upscaled image embedding",
         internals=[[f"{md}.output_hypernetworks_mlps.0.layers.0", "layers.0 (pre-ReLU)"],
                    [f"{md}.output_hypernetworks_mlps.0.layers.1", "layers.1 (pre-ReLU)"],
                    [f"{md}.output_hypernetworks_mlps.0.layers.2", "layers.2: 32 classifier weights"]],
         note="output_hypernetworks_mlps.1-3 also run, but their masks are dropped by multimask_output=False.")
    node("low_res", "point-wise product", f"{md}[masks]", X, ro + 2,
         "hyper_in @ upscaled_embedding -> masks[:, 0]  (256 x 256 logits)",
         quote="Finally, we predict a mask with a spatially point-wise product",
         internals=[[f"{md}.masks_all", "all 4 mask logits (only #0 is used)"]])
    node("mask_logits", "upscaled mask logits", "postprocess_masks", X, ro + 3,
         "Sam.postprocess_masks: bilinear to 1024, crop padding, bilinear to original size")
    node("mask", "mask", "mask", X, ro + 4, "logits > mask_threshold (0.0)",
         note="rel on a binary mask = sqrt(|XOR|) / sqrt(|mask_best|).")
    edge("final_norm", "iou_tok", "flow", "token 0")
    edge("final_norm", "mask_tok", "flow", "token 1")
    edge(k_prev, "upscaled", "flow", "image embedding")
    edge("iou_tok", "iou_scores", "flow", "")
    edge("mask_tok", "hyper", "flow", "")
    edge("hyper", "low_res", "flow", "classifier weights")
    edge("upscaled", "low_res", "flow", "upscaled embedding")
    edge("low_res", "mask_logits", "flow", "")
    edge("mask_logits", "mask", "flow", "> 0")
    groups.append({"label": "output heads", "code": "mask_decoder + Sam.postprocess_masks",
                   "row0": ro, "row1": ro + 4, "lane0": 0, "lane1": 3})

    ids = [x["id"] for x in nodes]
    assert len(ids) == len(set(ids)), "duplicate node ids"
    eids = [x["id"] for x in edges]
    assert len(eids) == len(set(eids)), "duplicate edge ids"
    return {"nodes": nodes, "edges": edges, "groups": groups, "lanes": LANES,
            "token_groups": [[gid, lab] for gid, (lab, _) in TOKEN_GROUPS.items()]}


def spec_keys(spec) -> list[str]:
    keys = []
    for n in spec["nodes"]:
        keys.append(n["key"])
        keys.extend(k for k, _ in n["internals"])
    return keys


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _finite(x):
    return x is not None and math.isfinite(x)


def _r4(x):
    return float(f"{x:.4g}")


def pair_scalars(rb: dict, rc: dict, spec, detail_keys) -> dict:
    """Flat {(section, id, metric): value} for one pair."""
    s = {}

    def triple(key):
        b, c = rb.get(key), rc.get(key)
        r = None
        if _finite(b) and _finite(c) and c > ZERO:
            r = b / c
        return b, c, r

    for n in spec["nodes"]:
        b, c, r = triple(n["key"])
        s[("nodes", n["id"], "rb")], s[("nodes", n["id"], "rc")], s[("nodes", n["id"], "r")] = b, c, r
    for k in detail_keys:
        b, c, r = triple(k)
        s[("keys", k, "rb")], s[("keys", k, "rc")], s[("keys", k, "r")] = b, c, r
    for e in spec["edges"]:
        ab, ac = s[("nodes", e["src"], "rb")], s[("nodes", e["src"], "rc")]
        bb, bc = s[("nodes", e["dst"], "rb")], s[("nodes", e["dst"], "rc")]
        gb = bb / ab if _finite(ab) and _finite(bb) and ab > ZERO else None
        gc = bc / ac if _finite(ac) and _finite(bc) and ac > ZERO else None
        gs = gb / gc if gb is not None and gc is not None and gc > 0 else None
        s[("edges", e["id"], "gb")], s[("edges", e["id"], "gc")], s[("edges", e["id"], "gs")] = gb, gc, gs
    return s


def median_scalars(dicts: list[dict]) -> dict:
    out = {}
    for key in dicts[0]:
        v = [d[key] for d in dicts if _finite(d[key])]
        out[key] = float(np.median(v)) if v else None
    return out


def aggregate(dicts: list[dict], spread: str) -> dict:
    """[median, lo, hi, n(>1), n] per scalar; spread 'iqr' or 'minmax'."""
    view = {"nodes": {}, "keys": {}, "edges": {}}
    for key in dicts[0]:
        v = np.array([d[key] for d in dicts if _finite(d[key])], dtype=np.float64)
        if v.size == 0:
            stat = None
        else:
            lo, hi = (np.percentile(v, [25, 75]) if spread == "iqr" else (v.min(), v.max()))
            stat = [_r4(np.median(v)), _r4(lo), _r4(hi), int((v > 1).sum()), int(v.size)]
        sect, ident, metric = key
        view[sect].setdefault(ident, {})[metric] = stat
    return view


# ---------------------------------------------------------------------------
# Control boxes and thumbnails
# ---------------------------------------------------------------------------

def control_candidates(best, bad, frame_wh, rng, max_cos, tries):
    """Boxes at the same L-inf distance from best as bad, random direction."""
    d = (bad - best).astype(np.float64)
    linf, dn = float(np.abs(d).max()), float(np.linalg.norm(d))
    W, H = frame_wh
    for _ in range(tries):
        u = rng.standard_normal(4)
        u = u / np.abs(u).max() * linf
        cos = float(u @ d / (np.linalg.norm(u) * dn + EPS))
        if cos > max_cos:
            continue
        c = (best.astype(np.float64) + u).astype(np.float32)
        if not (c[0] < c[2] and c[1] < c[3]):
            continue
        if c[0] < 0 or c[1] < 0 or c[2] > W or c[3] > H:
            continue
        yield c, cos


BEST_BGR, BAD_BGR, CTRL_BGR = (214, 120, 42), (52, 104, 235), (122, 175, 27)


def make_thumb(img_rgb, gt, panels, height):
    """panels: [(box in original pixels, bool mask HxW, bgr)] -> base64 jpeg."""
    H, W = gt.shape
    bx = np.array([p[0] for p in panels], dtype=np.float64)
    x0, y0, x1, y1 = bx[:, 0].min(), bx[:, 1].min(), bx[:, 2].max(), bx[:, 3].max()
    mx, my = 0.15 * (x1 - x0) + 10, 0.15 * (y1 - y0) + 10
    X0, Y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
    X1, Y1 = int(min(W, x1 + mx + 1)), int(min(H, y1 + my + 1))
    scale = height / max(1, Y1 - Y0)
    thick = max(1, int(round(2 / scale)))
    base = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    tiles = []
    for box, mask, color in panels:
        t = base.astype(np.float32)
        t[mask] = t[mask] * 0.45 + np.array(color, np.float32) * 0.55
        t = t.astype(np.uint8)
        cv2.drawContours(t, contours, -1, (255, 255, 255), thick)
        b = np.round(np.asarray(box)).astype(int)
        cv2.rectangle(t, (b[0], b[1]), (b[2], b[3]), color, thick + 1)
        t = t[Y0:Y1, X0:X1]
        t = cv2.resize(t, (max(1, int(round(t.shape[1] * scale))), height), interpolation=cv2.INTER_AREA)
        tiles.append(t)
        tiles.append(np.full((height, 4, 3), 255, np.uint8))
    ok, buf = cv2.imencode(".jpg", np.concatenate(tiles[:-1], axis=1), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def _f(x, nd=4):
    return None if x is None or not math.isfinite(x) else round(float(x), nd)


def run(args) -> int:
    device = torch.device(args.device)
    t0 = time.time()
    predictor = load_predictor(args.checkpoint_path, args.model_type, device)
    sam = predictor.model
    depth = len(sam.mask_decoder.transformer.layers)
    spec = build_spec(depth)
    const_keys = box_independent_keys(depth)
    log = CheckLog()

    cases = json.loads(Path(args.critical_shifts).read_text())
    by_img: OrderedDict[str, list] = OrderedDict()
    for pid, c in enumerate(cases):
        by_img.setdefault(c["image_name"], []).append((pid, c))
    names = list(by_img)[: args.limit_images] if args.limit_images > 0 else list(by_img)
    print(f"{len(cases)} pairs over {len(by_img)} images in {args.critical_shifts}; using {len(names)} images")

    images_dir, masks_dir = Path(args.images_dir), Path(args.masks_dir)
    repro, pairs_info, pair_rel = [], {}, {}
    det_max, det_where = 0.0, ""
    zero_max, zero_where = 0.0, ""
    grid_max = 0.0
    detail_keys = None

    for ii, name in enumerate(names):
        img_path = find_by_stem(images_dir, name, IMAGE_EXTS)
        gt_path = masks_dir / (name + args.mask_ext)
        if img_path is None or not gt_path.is_file():
            why = f"image not found in {images_dir}" if img_path is None else f"mask not found: {gt_path}"
            for pid, c in by_img[name]:
                repro.append({"pid": pid, "image": name, "status": "dropped", "reason": why,
                              "json_best_iou": c["best_iou"], "json_bad_iou": c["bad_iou"]})
            print(f"[{ii + 1}/{len(names)}] {name}: {why}")
            continue
        img = load_image_rgb(img_path)
        gt_np = load_gt(gt_path)
        with torch.inference_mode():
            predictor.set_image(img)
        if gt_np.shape != tuple(predictor.original_size):
            raise RuntimeError(f"{name}: mask {gt_np.shape} != image {predictor.original_size}")
        gt = torch.from_numpy(gt_np).to(device)
        in_h, in_w = predictor.input_size
        or_h, or_w = predictor.original_size
        scale = np.array([in_w / or_w, in_h / or_h, in_w / or_w, in_h / or_h])

        kept = 0
        for pid, c in by_img[name]:
            best = np.asarray(c["best_box"], dtype=np.float32)
            bad = np.asarray(c["bad_box"], dtype=np.float32)
            r = {"pid": pid, "image": name, "json_best_iou": c["best_iou"], "json_bad_iou": c["bad_iou"]}
            # boxes are in SAM's 1024-frame: back in original pixels they sit on the integer grid
            orig = np.stack([best, bad]) / scale
            grid_dev = float(np.abs(orig - np.round(orig)).max())
            grid_max = max(grid_max, grid_dev)
            r["grid_dev_px"] = round(grid_dev, 5)

            T_best, m_best, p_best = trace_and_verify(predictor, best, log, args.trace_tol)
            T_bad, m_bad, p_bad = trace_and_verify(predictor, bad, log, args.trace_tol)
            iou_best, iou_bad = mask_iou(m_best, gt), mask_iou(m_bad, gt)
            r.update(iou_best=iou_best, iou_bad=iou_bad, pred_iou_best=p_best, pred_iou_bad=p_bad,
                     iou_best_vs_bad_mask=mask_iou(m_best, m_bad))
            d_best, d_bad = abs(iou_best - c["best_iou"]), abs(iou_bad - c["bad_iou"])
            if d_best >= args.repro_tol or d_bad >= args.repro_tol:
                r.update(status="dropped",
                         reason=f"not reproduced: |dIoU| best {d_best:.4f}, bad {d_bad:.4f} (tol {args.repro_tol})")
                repro.append(r)
                continue

            # noise floor: the same box twice must give the same activations
            rel_again = rel_divergence(T_best, trace_forward(predictor, best))
            k_again = max(rel_again, key=lambda k: rel_again[k])
            if rel_again[k_again] > det_max:
                det_max, det_where = rel_again[k_again], k_again

            rng = np.random.default_rng([args.seed, pid])
            controls, rel_ctrl, n_tried = [], [], 0
            for cbox, cos in control_candidates(best, bad, (in_w, in_h), rng, args.ctrl_max_cos, args.ctrl_tries):
                n_tried += 1
                m_c, _ = predict_official(predictor, cbox)
                iou_c = mask_iou(m_c, gt)
                if iou_c < args.ctrl_min_iou:
                    continue
                T_c, m_c, p_c = trace_and_verify(predictor, cbox, log, args.trace_tol)
                rel_ctrl.append(rel_divergence(T_best, T_c))
                controls.append({"box": [round(float(v), 2) for v in cbox], "iou": iou_c, "pred_iou": p_c,
                                 "cos_to_attack": round(cos, 3), "mask": m_c})
                del T_c
                if len(controls) >= args.n_controls:
                    break
            r["n_ctrl"], r["ctrl_tried"] = len(controls), n_tried
            r["ctrl_ious"] = [round(x["iou"], 4) for x in controls]
            if not controls:
                r.update(status="dropped", reason=f"no benign control among {n_tried} candidates "
                                                  f"(IoU >= {args.ctrl_min_iou})")
                repro.append(r)
                continue

            rel_bad = rel_divergence(T_best, T_bad)
            for rel_set in [rel_bad] + rel_ctrl:
                for k in const_keys:
                    if rel_set[k] > zero_max:
                        zero_max, zero_where = rel_set[k], k
            rel_c_med = {k: float(np.median([rc[k] for rc in rel_ctrl])) for k in rel_bad}
            if detail_keys is None:
                missing = [k for k in spec_keys(spec) if k not in rel_bad]
                if missing:
                    raise RuntimeError(f"graph keys missing from the trace: {missing}")
                detail_keys = sorted({k for n in spec["nodes"] for k, _ in n["internals"]}
                                     | {k for k in rel_bad if "#" in k})
            pair_rel[pid] = {"rb": rel_bad, "rc": rel_c_med,
                             "rc_min": {k: min(rc[k] for rc in rel_ctrl) for k in rel_bad},
                             "rc_max": {k: max(rc[k] for rc in rel_ctrl) for k in rel_bad}}

            thumb = None
            if args.thumb_h > 0:
                ctrl0 = controls[0]
                thumb = make_thumb(img, gt_np, [
                    (best / scale, m_best.cpu().numpy(), BEST_BGR),
                    (bad / scale, m_bad.cpu().numpy(), BAD_BGR),
                    (np.asarray(ctrl0["box"]) / scale, ctrl0["mask"].cpu().numpy(), CTRL_BGR),
                ], args.thumb_h)
            r.update(status="kept", reason="")
            repro.append(r)
            pairs_info[pid] = {
                "image": name, "best_box": [round(float(v), 2) for v in best],
                "bad_box": [round(float(v), 2) for v in bad],
                "json_best_iou": _f(c["best_iou"]), "json_bad_iou": _f(c["bad_iou"]),
                "iou_best": _f(iou_best), "iou_bad": _f(iou_bad),
                "pred_iou_best": _f(p_best), "pred_iou_bad": _f(p_bad),
                "iou_best_vs_bad_mask": _f(r["iou_best_vs_bad_mask"]),
                "linf_shift": _f(float(np.abs(bad - best).max()), 2),
                "controls": [{k: v for k, v in x.items() if k != "mask"} for x in controls],
                "thumb": thumb,
            }
            for x in pairs_info[pid]["controls"]:
                x["iou"], x["pred_iou"] = _f(x["iou"]), _f(x["pred_iou"])
            kept += 1
            del T_best, T_bad
        print(f"[{ii + 1}/{len(names)}] {name}: kept {kept}/{len(by_img[name])} pairs "
              f"({time.time() - t0:.0f}s)", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_repro_csv(out_dir / "repro.csv", repro)
    n_kept = len(pair_rel)
    print(f"kept {n_kept}/{len(cases)} pairs; trace checks: {log.runs} runs, "
          f"{log.exact_runs} bit-exact, worst rel diff {log.worst:.3g} ({log.worst_where or '-'})")
    if n_kept == 0:
        print("no pair reproduced -- see repro.csv", file=sys.stderr)
        return 1

    # --- views -----------------------------------------------------------
    scal = {pid: pair_scalars(m["rb"], m["rc"], spec, detail_keys) for pid, m in pair_rel.items()}
    images = []
    views = {}
    img_medians = []
    for name in names:
        pids = [pid for pid, _ in by_img[name] if pid in scal]
        if not pids:
            continue
        images.append({"name": name, "pids": pids})
        views[f"img:{name}"] = aggregate([scal[p] for p in pids], "minmax")
        img_medians.append(median_scalars([scal[p] for p in pids]))
        for p in pids:
            views[f"pair:{p}"] = aggregate([scal[p]], "minmax")
    views["all"] = aggregate(img_medians, "iqr")

    checks = {
        "trace_vs_official": log.as_dict(),
        "trace_tol": args.trace_tol,
        "determinism": {"max_rel": det_max, "where": det_where},
        "box_independent_zero": {"keys": const_keys, "max_rel": zero_max, "where": zero_where},
        "grid": {"max_dev_px": grid_max},
    }
    meta = {
        "model": f"SAM {args.model_type}", "checkpoint": Path(args.checkpoint_path).name,
        "json": Path(args.critical_shifts).name, "n_pairs_json": len(cases),
        "n_pairs_kept": n_kept, "n_images": len(images), "n_controls": args.n_controls,
        "ctrl_min_iou": args.ctrl_min_iou, "ctrl_max_cos": args.ctrl_max_cos,
        "repro_tol": args.repro_tol, "device": _device_name(device),
        "torch": torch.__version__, "segment_anything": _sa_path(),
        "created": time.strftime("%Y-%m-%d %H:%M"),
    }
    payload = {"meta": meta, "spec": spec, "views": views, "images": images,
               "pairs": {str(k): v for k, v in pairs_info.items()},
               "repro": [{k: (_f(v) if isinstance(v, float) else v) for k, v in r.items()} for r in repro],
               "checks": checks}
    html_path = out_dir / "activation_graph.html"
    html_path.write_text(render_html(payload), encoding="utf-8")
    _write_pair_csvs(out_dir, pair_rel, scal, spec, pairs_info)
    (out_dir / "checks.json").write_text(json.dumps(checks, indent=2))
    print(f"wrote {html_path} ({html_path.stat().st_size / 1e6:.1f} MB) and CSVs to {out_dir}")
    return 0


def _device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return str(device)


def _sa_path():
    import segment_anything
    return str(Path(segment_anything.__file__).parent)


def _write_repro_csv(path, repro):
    cols = ["pid", "image", "status", "reason", "json_best_iou", "iou_best", "json_bad_iou", "iou_bad",
            "pred_iou_best", "pred_iou_bad", "iou_best_vs_bad_mask", "grid_dev_px", "n_ctrl",
            "ctrl_tried", "ctrl_ious"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in repro:
            w.writerow(r)


def _write_pair_csvs(out_dir, pair_rel, scal, spec, pairs_info):
    with open(out_dir / "per_pair_keys.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid", "image", "key", "rel_bad", "rel_ctrl_median", "rel_ctrl_min", "rel_ctrl_max", "ratio"])
        for pid, m in pair_rel.items():
            for k, b in m["rb"].items():
                c = m["rc"][k]
                ratio = b / c if c > ZERO else ""
                w.writerow([pid, pairs_info[pid]["image"], k, b, c, m["rc_min"][k], m["rc_max"][k], ratio])
    with open(out_dir / "per_pair_edges.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid", "image", "edge", "src", "dst", "role", "gain_bad", "gain_ctrl", "gain_specific"])
        for pid, s in scal.items():
            for e in spec["edges"]:
                g = [s[("edges", e["id"], m)] for m in ("gb", "gc", "gs")]
                w.writerow([pid, pairs_info[pid]["image"], e["id"], e["src"], e["dst"], e["role"],
                            *["" if x is None else x for x in g]])


# ---------------------------------------------------------------------------
# Self-test (no data needed)
# ---------------------------------------------------------------------------

def self_test(args) -> int:
    device = torch.device(args.device)
    torch.manual_seed(0)
    ckpt = args.checkpoint_path or None
    print(f"self-test on {device}, weights: {ckpt or 'random init'}")
    predictor = load_predictor(ckpt, args.model_type, device)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(427, 640, 3), dtype=np.uint8)
    with torch.inference_mode():
        predictor.set_image(img)
    depth = len(predictor.model.mask_decoder.transformer.layers)
    spec = build_spec(depth)
    log = CheckLog()
    boxes = [np.array(b, np.float32) for b in
             ([100.0, 80.0, 700.0, 600.0], [109.6, 70.4, 700.0, 600.0], [0.0, 0.0, 1023.0, 682.0],
              [512.5, 300.25, 530.75, 320.0], [33.6, 25.6, 766.4, 300.8])]
    Ts = []
    for b in boxes:
        T, _, _ = trace_and_verify(predictor, b, log, args.trace_tol)
        Ts.append(T)
    ok = True
    missing = [k for k in spec_keys(spec) if k not in Ts[0]]
    if missing:
        ok = False
        print("FAIL graph keys missing from trace:", missing)
    rel_again = rel_divergence(Ts[0], trace_forward(predictor, boxes[0]))
    det = max(rel_again.values())
    rel01 = rel_divergence(Ts[0], Ts[1])
    zero = {k: rel01[k] for k in box_independent_keys(depth)}
    nonzero_var = [n["key"] for n in spec["nodes"] if n["kind"] == "var" and rel01[n["key"]] <= ZERO]
    const_nonzero = [n["key"] for n in spec["nodes"] if n["kind"] == "const" and rel01[n["key"]] > 0]
    print(f"trace vs official: {log.runs} runs, {log.exact_runs} bit-exact, {log.tensors} tensors, "
          f"worst rel diff {log.worst:.3g} ({log.worst_where or '-'})")
    print(f"modules that fired during predict_torch: {len(log.fired)}")
    print(f"determinism (same box twice): max rel {det:.3g}")
    print(f"box-independent tensors, max rel: {max(zero.values()):.3g}")
    if det != 0.0:
        print("WARN not deterministic")
    if max(zero.values()) != 0.0 or const_nonzero:
        ok = False
        print("FAIL box-independent tensors changed:", {k: v for k, v in zero.items() if v}, const_nonzero)
    if nonzero_var:
        ok = False
        print("FAIL graph nodes marked box-dependent did not change:", nonzero_var)
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def render_html(payload) -> str:
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__PAYLOAD__", data)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SAM activation flow</title>
<style>
:root{
  --scheme:light;
  --bg:#f9f9f7; --surface:#fcfcfb; --surface-2:#f0efec; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10); --sel:#0b0b0b;
  --best:#2a78d6; --bad:#eb6834; --ctrl:#1baf7a; --ok:#006300; --warn:#b5520f; --fail:#d03b3b;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --scheme:dark;
    --bg:#0d0d0d; --surface:#1a1a19; --surface-2:#262624; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#4a4a46; --border:rgba(255,255,255,0.10); --sel:#ffffff;
    --best:#3987e5; --bad:#d95926; --ctrl:#199e70; --ok:#0ca30c; --warn:#ec835a; --fail:#e66767;
  }
}
:root[data-theme="dark"]{
  --scheme:dark;
  --bg:#0d0d0d; --surface:#1a1a19; --surface-2:#262624; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#4a4a46; --border:rgba(255,255,255,0.10); --sel:#ffffff;
  --best:#3987e5; --bad:#d95926; --ctrl:#199e70; --ok:#0ca30c; --warn:#ec835a; --fail:#e66767;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1560px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:16px;margin:24px 0 8px}
h3{font-size:15px;margin:0 0 6px}
.meta{color:var(--ink-2);margin:0 0 12px;font-size:13px}
code,.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px}
.controls{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:flex-end;margin:8px 0 12px}
.controls label{display:flex;flex-direction:column;gap:3px;font-size:12px;color:var(--ink-2)}
.controls label.chk{flex-direction:row;align-items:center;gap:6px;font-size:13px;color:var(--ink)}
select{font:inherit;font-size:13px;color:var(--ink);background:var(--surface);border:1px solid var(--axis);border-radius:6px;padding:4px 6px;max-width:340px}
.summary{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 12px}
.chip{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:6px 10px;font-size:13px}
.chip b{font-weight:650}
.chip .sub{color:var(--ink-2)}
.main{display:grid;grid-template-columns:minmax(0,1fr) 380px;gap:16px;align-items:start}
@media (max-width:1180px){.main{grid-template-columns:minmax(0,1fr)}}
.graph{overflow:auto;width:fit-content;max-width:100%;background:var(--surface);border:1px solid var(--border);border-radius:10px}
.graph svg{display:block}
.legend{display:flex;flex-wrap:wrap;gap:8px 28px;margin:0 0 8px;font-size:12px;color:var(--ink-2)}
.legend .bar{width:220px;height:10px;border-radius:3px;margin:3px 0}
.legend .ticks{display:flex;justify-content:space-between;width:220px;font-variant-numeric:tabular-nums}
.legend .item{display:flex;flex-direction:column}
.legend .keys{display:flex;flex-direction:column;gap:3px}
.legend .keys span{display:flex;align-items:center;gap:6px}
.panel{position:sticky;top:12px;max-height:calc(100vh - 24px);overflow:auto;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
@media (max-width:1180px){.panel{position:static;max-height:none}}
.panel .quote{font-style:italic;color:var(--ink-2);margin:6px 0}
.panel .note{color:var(--ink-2);margin:6px 0}
.panel .code{color:var(--ink-2);word-break:break-all}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{text-align:left;padding:4px 6px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--ink-2);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.tbl{overflow:auto;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:4px 8px}
tr.click{cursor:pointer}
tr.click:hover td{background:var(--surface-2)}
.node{cursor:pointer}
.node:focus{outline:none}
.node:focus rect.frame,.node.sel rect.frame{stroke:var(--sel);stroke-width:2.5}
.edgehit{cursor:pointer}
.tip{position:fixed;z-index:10;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);border-radius:8px;padding:8px 10px;font-size:12.5px;box-shadow:0 4px 14px rgba(0,0,0,.18);max-width:360px}
.tip .v{font-weight:650;font-variant-numeric:tabular-nums}
.tip .l{color:var(--ink-2)}
.thumbs{display:flex;flex-direction:column;gap:12px}
.thumb{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:8px 10px}
.thumb img{max-width:100%;height:auto;border-radius:4px;display:block;margin:6px 0}
.key{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:4px}
details{margin:14px 0;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:8px 12px}
summary{cursor:pointer;font-weight:600}
.ok{color:var(--ok)} .warn{color:var(--warn)} .fail{color:var(--fail)}
.muted{color:var(--muted)}
</style>
</head>
<body>
<div class="wrap">
  <h1>Где ломается SAM: расхождение активаций best → bad против best → control</h1>
  <p class="meta" id="meta"></p>
  <div class="controls">
    <label>Картинка <select id="sel-img"></select></label>
    <label>Пара <select id="sel-pair"></select></label>
    <label>Цвет узлов <select id="sel-nm">
      <option value="r">ratio = rel(best→bad) / rel(best→control)</option>
      <option value="rb">rel L2 best→bad</option>
      <option value="rc">rel L2 best→control</option>
    </select></label>
    <label>Цвет рёбер <select id="sel-em">
      <option value="gs">специфическое усиление gain(bad) / gain(control)</option>
      <option value="gb">усиление best→bad, rel(dst) / rel(src)</option>
      <option value="gc">усиление best→control, rel(dst) / rel(src)</option>
    </select></label>
    <label class="chk"><input type="checkbox" id="chk-pe"> рёбра «re-added prompt tokens» и «image PE»</label>
  </div>
  <div class="summary" id="summary"></div>
  <div class="main">
    <div>
      <div class="legend" id="legend"></div>
      <div class="graph" id="graph"></div>
    </div>
    <aside class="panel" id="panel"></aside>
  </div>
  <div id="thumbs-wrap"></div>
  <h2>Узлы в порядке инференса</h2>
  <div class="tbl" id="tbl-nodes"></div>
  <h2>Рёбра с наибольшим усилением (метрика рёбер)</h2>
  <div class="tbl" id="tbl-edges"></div>
  <details open>
    <summary>Как читать</summary>
    <ul>
      <li><b>Узел</b> — тензор forward-прохода SAM от box prompt до маски; сверху вниз — порядок исполнения. Подписи — термины статьи (Kirillov et al., <i>Segment Anything</i>, 2023, Appendix A); точный путь в коде <code>segment_anything</code> — в панели справа.</li>
      <li><b>rel(a→b)</b> = ‖A<sub>b</sub> − A<sub>a</sub>‖₂ / ‖A<sub>a</sub>‖₂ на тензоре узла; A<sub>a</sub> — активация на best box.</li>
      <li><b>Цвет узла</b> по умолчанию — <b>ratio</b> = rel(best→bad) / rel(best→control): во сколько раз критический сдвиг меняет этот тензор сильнее, чем безобидный сдвиг той же L∞-величины. Серый (×1) — узел реагирует на критический сдвиг как на любой; красный — сильнее; синий — слабее.</li>
      <li><b>Цвет ребра</b> по умолчанию — <b>специфическое усиление</b> gain(bad) / gain(control), где gain = rel(dst) / rel(src). Ровно ratio(dst) = ratio(src) × это число, поэтому слой, который «ломает», — тот, на чьих входящих рёбрах оно ≫ 1. Подписаны только рёбра с усилением ≥ ×2 или ≤ ×0.5.</li>
      <li><b>Штрихованные узлы</b> не зависят от бокса (image embedding, ‘no mask’, image PE, learned tokens). Их расхождение обязано быть ровно 0 — это проверяется (раздел «Проверки»).</li>
      <li><b>Control</b>: K боксов, сдвинутых от best на ту же L∞-величину, что и best→bad, в случайном направлении (косинус с направлением атаки ≤ порога); оставляются только те, у которых маска осталась хорошей (IoU с GT ≥ порога). rel(best→control) — медиана по K.</li>
      <li><b>Агрегация</b>: пара → медиана по парам картинки → медиана по картинкам. Диапазон в «Все картинки» — IQR по картинкам, в виде одной картинки — min–max по её парам. «n&gt;1» — у скольких картинок (пар) значение больше 1.</li>
      <li>Пара попадает в граф, только если пересчёт SAM даёт IoU best и bad в пределах допуска от JSON (раздел «Сверка с JSON»).</li>
    </ul>
  </details>
  <details>
    <summary>Проверки: граф построен на тех активациях, которые реально использовал SAM</summary>
    <div id="checks"></div>
  </details>
  <details>
    <summary>Сверка с JSON (воспроизводимость пар)</summary>
    <div class="tbl" id="repro"></div>
  </details>
</div>
<div id="tip" class="tip" hidden></div>
<script type="application/json" id="payload">__PAYLOAD__</script>
<script>
"use strict";
(() => {
const D = JSON.parse(document.getElementById("payload").textContent);
const S = D.spec;
const NODE = Object.fromEntries(S.nodes.map(n => [n.id, n]));
const SVGNS = "http://www.w3.org/2000/svg";
const LANE_W = 204, NODE_W = 184, NODE_H = 46, ROW_H = 60, GROUP_GAP = 34, PAD = 12, HEAD = 26;
const st = {img: "all", pair: "", nm: "r", em: "gs", pe: false, sel: null};

const $ = id => document.getElementById(id);
function el(tag, attrs, text) {
  const e = document.createElement(tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) { if (k === "class") e.className = v; else e.setAttribute(k, v); }
  if (text != null) e.textContent = text;
  return e;
}
function sv(tag, attrs) {
  const e = document.createElementNS(SVGNS, tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}

// ---------- numbers ----------
function fmtX(x) {
  if (x == null || !isFinite(x)) return "—";
  if (x >= 100) return "×" + x.toFixed(0);
  if (x >= 10) return "×" + x.toFixed(1);
  if (x >= 0.1) return "×" + x.toFixed(2);
  return "×" + x.toPrecision(2);
}
function fmtRel(x) {
  if (x == null || !isFinite(x)) return "—";
  if (x === 0) return "0";
  if (x >= 0.01) return x.toFixed(3);
  return x.toExponential(1);
}
const METRIC = {
  r:  {name: "ratio", long: "rel(best→bad) / rel(best→control)", kind: "div", fmt: fmtX},
  rb: {name: "rel best→bad", long: "‖A_bad − A_best‖ / ‖A_best‖", kind: "seq", fmt: fmtRel},
  rc: {name: "rel best→control", long: "‖A_ctrl − A_best‖ / ‖A_best‖ (медиана по K)", kind: "seq", fmt: fmtRel},
  gs: {name: "специфическое усиление", long: "gain(bad) / gain(control)", kind: "div", fmt: fmtX},
  gb: {name: "усиление best→bad", long: "rel(dst) / rel(src)", kind: "div", fmt: fmtX},
  gc: {name: "усиление best→control", long: "rel(dst) / rel(src)", kind: "div", fmt: fmtX},
};

// ---------- colour ----------
const PAL = {
  light: {div: ["#1c5cab", "#5598e7", "#b7d3f6", "#f0efec", "#f6c3bf", "#e7726f", "#b52e2d"],
          seq: ["#e9f1fc", "#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"],
          edge: ["#1c5cab", "#3987e5", "#a9a8a1", "#e34948", "#a8292a"],
          con: "#e6e5e0"},
  dark:  {div: ["#6da7ec", "#2f6fbf", "#23406a", "#383835", "#6b2b2a", "#b8403e", "#ee7b79"],
          seq: ["#1f2a3a", "#184f95", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"],
          edge: ["#86b6ef", "#3987e5", "#6a6a64", "#e05a59", "#f19a98"],
          con: "#2c2c2a"},
};
function scheme() { return getComputedStyle(document.documentElement).getPropertyValue("--scheme").trim() === "dark" ? "dark" : "light"; }
function hex2rgb(h) { h = h.replace("#", ""); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); }
function rgb2hex(c) { return "#" + c.map(v => Math.round(v).toString(16).padStart(2, "0")).join(""); }
function ramp(stops, t) {
  t = Math.max(0, Math.min(1, t));
  const n = stops.length - 1, i = Math.min(n - 1, Math.floor(t * n)), f = t * n - i;
  const a = hex2rgb(stops[i]), b = hex2rgb(stops[i + 1]);
  return rgb2hex(a.map((v, k) => v + (b[k] - v) * f));
}
function divColor(x, pal) { return ramp(PAL[scheme()][pal || "div"], (Math.log2(x) + 3) / 6); }
function seqColor(x, dom) {
  const l = x > 0 ? Math.log10(x) : dom[0];
  return ramp(PAL[scheme()].seq, (l - dom[0]) / (dom[1] - dom[0]));
}
function textOn(hex) {
  const [r, g, b] = hex2rgb(hex).map(v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 ? "#0b0b0b" : "#ffffff";
}
// nodes are fills (light neutral midpoint); edges are thin lines, so their
// neutral midpoint is a mid gray that stays visible on the surface
function colorFor(metric, x, dom, pal) {
  if (x == null || !isFinite(x)) return null;
  return METRIC[metric].kind === "div" ? divColor(Math.max(x, 1e-9), pal) : seqColor(x, dom);
}

// ---------- data access ----------
function vid() { return st.img === "all" ? "all" : (st.pair ? "pair:" + st.pair : "img:" + st.img); }
function V() { return D.views[vid()]; }
function ns(id, m) { const x = V().nodes[id]; return x ? x[m] : null; }
function es(id, m) { const x = V().edges[id]; return x ? x[m] : null; }
function ks(k, m) { const x = V().keys[k]; return x ? x[m] : null; }
function seqDomain(m) {
  const v = S.nodes.filter(n => n.kind !== "const").map(n => ns(n.id, m)).filter(s => s && s[0] > 0).map(s => Math.log10(s[0]));
  if (!v.length) return [-4, 0];
  let lo = Math.floor(Math.min(...v)), hi = Math.ceil(Math.max(...v));
  if (hi - lo < 1) hi = lo + 1;
  return [lo, hi];
}
function spreadLabel() { return st.img === "all" ? "IQR по картинкам" : (st.pair ? "" : "min–max по парам"); }
function statText(s, fmt) {
  if (!s || s[0] == null) return "—";
  let t = fmt(s[0]);
  if (s[4] > 1) t += "  [" + fmt(s[1]) + " – " + fmt(s[2]) + "]";
  return t;
}
function npos(s) { return s && s[4] > 1 ? s[3] + "/" + s[4] : (s && s[4] === 1 ? (s[3] ? "да" : "нет") : ""); }
function visibleEdge(e) { return st.pe || e.role !== "pe"; }

// ---------- layout ----------
function layout() {
  const starts = new Set(S.groups.map(g => g.row0));
  const maxRow = Math.max(...S.nodes.map(n => n.row));
  const rowY = []; let acc = HEAD;
  for (let r = 0; r <= maxRow; r++) { if (starts.has(r)) acc += GROUP_GAP; rowY.push(PAD + r * ROW_H + acc); }
  const W = 2 * PAD + S.lanes.length * LANE_W, H = rowY[maxRow] + NODE_H + PAD + 8;
  const pos = {};
  for (const n of S.nodes) pos[n.id] = {x: PAD + n.lane * LANE_W + (LANE_W - NODE_W) / 2, y: rowY[n.row], w: NODE_W, h: NODE_H};
  return {rowY, W, H, pos};
}
function spreadPorts(edges, pos) {
  const inc = {}, out = {}, pin = {}, pout = {};
  for (const e of edges) {
    if (pos[e.src].y === pos[e.dst].y) continue;
    (inc[e.dst] = inc[e.dst] || []).push(e); (out[e.src] = out[e.src] || []).push(e);
  }
  const cx = id => pos[id].x;
  for (const es_ of Object.values(inc)) { es_.sort((a, b) => cx(a.src) - cx(b.src) || pos[a.src].y - pos[b.src].y); es_.forEach((e, i) => pin[e.id] = (i - (es_.length - 1) / 2) * 16); }
  for (const es_ of Object.values(out)) { es_.sort((a, b) => cx(a.dst) - cx(b.dst) || pos[a.dst].y - pos[b.dst].y); es_.forEach((e, i) => pout[e.id] = (i - (es_.length - 1) / 2) * 16); }
  return {pin, pout};
}
function edgeGeom(e, pos, ports) {
  const a = pos[e.src], b = pos[e.dst];
  if (a.y === b.y) {
    const right = a.x < b.x;
    const sx = right ? a.x + a.w : a.x, ex = right ? b.x - 3 : b.x + b.w + 3, y = a.y + a.h / 2;
    return {d: `M${sx},${y} L${ex},${y}`, mx: (sx + ex) / 2, my: y - 6};
  }
  const sx = a.x + a.w / 2 + (ports.pout[e.id] || 0), sy = a.y + a.h;
  const ex = b.x + b.w / 2 + (ports.pin[e.id] || 0), ey = b.y - 3;
  const k = Math.min(34, (ey - sy) / 2);
  const p1 = [sx, sy + k], p2 = [ex, ey - k];
  const mx = (sx + 3 * p1[0] + 3 * p2[0] + ex) / 8, my = (sy + 3 * p1[1] + 3 * p2[1] + ey) / 8;
  return {d: `M${sx},${sy} C${p1[0]},${p1[1]} ${p2[0]},${p2[1]} ${ex},${ey}`, mx, my};
}

// ---------- tooltip ----------
const tip = $("tip");
function showTip(ev, rows) {
  tip.replaceChildren();
  for (const [cls, text] of rows) tip.appendChild(el("div", cls ? {class: cls} : null, text));
  tip.hidden = false; moveTip(ev);
}
function moveTip(ev) {
  const r = tip.getBoundingClientRect();
  let x = (ev.clientX ?? 0) + 14, y = (ev.clientY ?? 0) + 14;
  if (x + r.width > innerWidth - 8) x = (ev.clientX ?? 0) - r.width - 14;
  if (y + r.height > innerHeight - 8) y = innerHeight - r.height - 8;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
function hideTip() { tip.hidden = true; }
function nodeTipRows(n) {
  const rows = [["", n.label + "  ·  " + n.code]];
  if (n.kind === "const") { rows.push(["v", "не зависит от бокса: расхождение ≡ 0"]); return rows; }
  for (const m of ["r", "rb", "rc"]) {
    const s = ns(n.id, m), p = npos(s);
    rows.push(["v", METRIC[m].name + ": " + statText(s, METRIC[m].fmt) + (m === "r" && p ? "   (>1: " + p + ")" : "")]);
  }
  return rows;
}
function edgeTipRows(e) {
  const rows = [["", NODE[e.src].label + " → " + NODE[e.dst].label + (e.label ? "  ·  " + e.label : "")]];
  if (NODE[e.src].kind === "const") { rows.push(["l", "источник не зависит от бокса: усиление не определено"]); return rows; }
  for (const m of ["gs", "gb", "gc"]) {
    const s = es(e.id, m), p = npos(s);
    rows.push(["v", METRIC[m].name + ": " + statText(s, fmtX) + (p ? "   (>1: " + p + ")" : "")]);
  }
  return rows;
}

// ---------- graph ----------
function drawGraph() {
  const L = layout();
  const svg = sv("svg", {width: L.W, height: L.H, viewBox: `0 0 ${L.W} ${L.H}`, role: "img",
    "aria-label": "Граф потока данных SAM: prompt encoder и mask decoder, узлы окрашены по выбранной метрике"});
  const defs = sv("defs");
  const mk = sv("marker", {id: "arr", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse"});
  const mp = sv("path", {d: "M0,1 L9,5 L0,9 z"}); mp.style.fill = "var(--muted)"; mk.appendChild(mp); defs.appendChild(mk);
  const pat = sv("pattern", {id: "hatch", width: 7, height: 7, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)"});
  const pr = sv("rect", {width: 7, height: 7}); pr.style.fill = PAL[scheme()].con; pat.appendChild(pr);
  const pl = sv("line", {x1: 0, y1: 0, x2: 0, y2: 7, "stroke-width": 2}); pl.style.stroke = "var(--axis)"; pat.appendChild(pl);
  defs.appendChild(pat); svg.appendChild(defs);

  S.lanes.forEach((l, i) => {
    const t = sv("text", {x: PAD + i * LANE_W + LANE_W / 2, y: PAD + 12, "text-anchor": "middle", "font-size": 11.5});
    t.style.fill = "var(--muted)"; t.textContent = l.label; svg.appendChild(t);
  });
  for (const g of S.groups) {
    const x = PAD + g.lane0 * LANE_W + 3, y = L.rowY[g.row0] - GROUP_GAP + 6;
    const w = (g.lane1 - g.lane0 + 1) * LANE_W - 6, h = L.rowY[g.row1] + NODE_H + 9 - y;
    const r = sv("rect", {x, y, width: w, height: h, rx: 10}); r.style.fill = "none"; r.style.stroke = "var(--grid)"; r.style.strokeWidth = 1.5;
    svg.appendChild(r);
    const t = sv("text", {x: x + 10, y: y + 17, "font-size": 12, "font-weight": 600}); t.style.fill = "var(--ink-2)";
    t.style.stroke = "var(--surface)"; t.style.strokeWidth = 5; t.style.paintOrder = "stroke"; t.style.strokeLinejoin = "round";
    t.textContent = g.label; const c = sv("tspan", {"font-weight": 400, "font-size": 11}); c.style.fill = "var(--muted)";
    c.textContent = "   " + g.code; t.appendChild(c); svg.appendChild(t);
  }

  const edges = S.edges.filter(visibleEdge);
  const ports = spreadPorts(edges, L.pos);
  const eLayer = sv("g"), lLayer = sv("g");
  for (const e of edges) {
    const geo = edgeGeom(e, L.pos, ports);
    const s = es(e.id, st.em);
    const constSrc = NODE[e.src].kind === "const";
    const col = constSrc ? null : colorFor(st.em, s && s[0], null, "edge");
    const p = sv("path", {d: geo.d, fill: "none", "marker-end": "url(#arr)"});
    if (col) { p.style.stroke = col; p.style.strokeWidth = 1.6 + Math.min(3.2, Math.abs(Math.log2(s[0]))); }
    else { p.style.stroke = "var(--axis)"; p.style.strokeWidth = 1.2; p.setAttribute("stroke-dasharray", "4 3"); }
    if (e.role === "pe") { p.setAttribute("stroke-dasharray", "2 3"); p.style.opacity = 0.75; }
    if (st.sel && st.sel.type === "edge" && st.sel.id === e.id) { p.style.strokeWidth = 5; }
    eLayer.appendChild(p);
    const hit = sv("path", {d: geo.d, fill: "none", class: "edgehit", "stroke-width": 12}); hit.style.stroke = "transparent";
    hit.addEventListener("pointerenter", ev => showTip(ev, edgeTipRows(e)));
    hit.addEventListener("pointermove", moveTip);
    hit.addEventListener("pointerleave", hideTip);
    hit.addEventListener("click", () => select({type: "edge", id: e.id}));
    eLayer.appendChild(hit);
    if (col && (s[0] >= 2 || s[0] <= 0.5)) {
      const txt = fmtX(s[0]);
      const bg = sv("rect", {x: geo.mx - 4 * txt.length - 4, y: geo.my - 8, width: 8 * txt.length + 8, height: 16, rx: 4});
      bg.style.fill = "var(--surface)"; bg.style.stroke = col; lLayer.appendChild(bg);
      const t = sv("text", {x: geo.mx, y: geo.my + 4, "text-anchor": "middle", "font-size": 11, "font-weight": 650});
      t.style.fill = "var(--ink)"; t.textContent = txt; lLayer.appendChild(t);
    }
  }
  svg.appendChild(eLayer);

  const dom = seqDomain(st.nm);
  for (const n of S.nodes) {
    const p = L.pos[n.id];
    const g = sv("g", {class: "node" + (st.sel && st.sel.type === "node" && st.sel.id === n.id ? " sel" : ""),
      transform: `translate(${p.x},${p.y})`, tabindex: 0, role: "button", "aria-label": n.label});
    const s = ns(n.id, st.nm);
    const fill = n.kind === "const" ? null : colorFor(st.nm, s && s[0], dom);
    const r = sv("rect", {class: "frame", width: p.w, height: p.h, rx: 7});
    if (fill) r.style.fill = fill; else r.style.fill = n.kind === "const" ? "url(#hatch)" : "var(--surface-2)";
    r.style.stroke = "var(--border)"; r.style.strokeWidth = 1;
    g.appendChild(r);
    const ink = fill ? textOn(fill) : "var(--ink)";
    const t1 = sv("text", {x: 9, y: 18, "font-size": 12, "font-weight": 600}); t1.style.fill = ink; t1.textContent = n.label; g.appendChild(t1);
    const t2 = sv("text", {x: 9, y: 36, "font-size": 12}); t2.style.fill = ink;
    if (n.kind === "const") t2.textContent = "const · ≡ 0";
    else {
      t2.textContent = s && s[0] != null ? METRIC[st.nm].fmt(s[0]) : "—";
      const pp = npos(s);
      if (st.nm === "r" && pp && s[4] > 1) { const ts = sv("tspan", {"font-size": 11}); ts.textContent = "   >1: " + pp; t2.appendChild(ts); }
    }
    g.appendChild(t2);
    g.addEventListener("pointerenter", ev => showTip(ev, nodeTipRows(n)));
    g.addEventListener("pointermove", moveTip);
    g.addEventListener("pointerleave", hideTip);
    g.addEventListener("focus", () => { const b = g.getBoundingClientRect(); showTip({clientX: b.right, clientY: b.top}, nodeTipRows(n)); });
    g.addEventListener("blur", hideTip);
    g.addEventListener("click", () => select({type: "node", id: n.id}));
    g.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); select({type: "node", id: n.id}); } });
    svg.appendChild(g);
  }
  svg.appendChild(lLayer);
  $("graph").replaceChildren(svg);
}

// ---------- legend ----------
function legendBar(metric, dom, pal) {
  const it = el("div", {class: "item"});
  it.appendChild(el("div", null, (metric === st.nm ? "узлы: " : "рёбра: ") + METRIC[metric].name + "  (" + METRIC[metric].long + ")"));
  const bar = el("div", {class: "bar"});
  const stops = [];
  for (let i = 0; i <= 12; i++) {
    const t = i / 12;
    stops.push((METRIC[metric].kind === "div" ? ramp(PAL[scheme()][pal || "div"], t) : ramp(PAL[scheme()].seq, t)) + " " + (t * 100).toFixed(0) + "%");
  }
  bar.style.background = "linear-gradient(90deg," + stops.join(",") + ")";
  it.appendChild(bar);
  const ticks = el("div", {class: "ticks"});
  const labels = METRIC[metric].kind === "div" ? ["≤×1/8", "×1/2", "×1", "×2", "≥×8"]
    : [dom[0], (dom[0] + dom[1]) / 2, dom[1]].map(v => "1e" + (Math.round(v * 10) / 10));
  for (const l of labels) ticks.appendChild(el("span", null, l));
  it.appendChild(ticks);
  return it;
}
function drawLegend() {
  const lg = $("legend"); lg.replaceChildren();
  lg.appendChild(legendBar(st.nm, seqDomain(st.nm)));
  lg.appendChild(legendBar(st.em, null, "edge"));
  const keys = el("div", {class: "keys"});
  const k1 = el("span"); const s1 = el("span", {class: "key"}); s1.style.background = "var(--axis)"; s1.style.height = "2px";
  k1.append(s1, document.createTextNode("штрих — от узла, не зависящего от бокса")); keys.appendChild(k1);
  keys.appendChild(el("span", null, "толщина ребра ∝ |log₂ усиления|"));
  lg.appendChild(keys);
}

// ---------- summary ----------
function chip(title, value, sub) {
  const c = el("div", {class: "chip"});
  c.appendChild(el("span", {class: "sub"}, title + " "));
  c.appendChild(el("b", null, value));
  if (sub) c.appendChild(el("span", {class: "sub"}, "  " + sub));
  return c;
}
function drawSummary() {
  const box = $("summary"); box.replaceChildren();
  const v = V();
  if (st.img === "all") box.appendChild(chip("выборка:", D.meta.n_pairs_kept + " пар", "по " + D.meta.n_images + " картинкам (из " + D.meta.n_pairs_json + " пар в JSON)"));
  const sb = ns("box", "r");
  if (sb) box.appendChild(chip("на входе (box) ratio:", fmtX(sb[0]), "‖Δbox‖ bad / control"));
  const first = S.nodes.find(n => n.kind !== "const" && n.id !== "box" && (s => s && s[0] != null && s[0] >= 2)(ns(n.id, "r")));
  if (first) box.appendChild(chip("первый узел с ratio ≥ ×2:", first.label, "(" + first.code + ")  " + fmtX(ns(first.id, "r")[0])));
  let best = null;
  for (const e of S.edges) { const s = es(e.id, "gs"); if (s && s[0] != null && (!best || s[0] > best.s[0])) best = {e, s}; }
  if (best) box.appendChild(chip("макс. специфическое усиление:", NODE[best.e.src].label + " → " + NODE[best.e.dst].label,
    statText(best.s, fmtX) + (npos(best.s) ? "  (>1: " + npos(best.s) + ")" : "")));
  const sm = ns("mask", "r");
  if (sm) box.appendChild(chip("на выходе (mask) ratio:", fmtX(sm[0]), ""));
}

// ---------- side panel ----------
function metricTable(rows, withN) {
  const t = el("table"); const h = el("tr");
  for (const c of ["", "медиана", spreadLabel() || "", withN ? ">1" : ""]) h.appendChild(el("th", {class: "num"}, c));
  t.appendChild(h);
  for (const [label, s, fmt] of rows) {
    const tr = el("tr");
    tr.appendChild(el("td", null, label));
    tr.appendChild(el("td", {class: "num"}, s && s[0] != null ? fmt(s[0]) : "—"));
    tr.appendChild(el("td", {class: "num"}, s && s[4] > 1 ? fmt(s[1]) + " – " + fmt(s[2]) : ""));
    tr.appendChild(el("td", {class: "num"}, withN ? npos(s) : ""));
    t.appendChild(tr);
  }
  return t;
}
function keyTable(items) {
  const t = el("table"); const h = el("tr");
  for (const c of ["тензор", "rel bad", "rel ctrl", "ratio"]) h.appendChild(el("th", {class: c === "тензор" ? "" : "num"}, c));
  t.appendChild(h);
  for (const [key, label] of items) {
    const tr = el("tr");
    const td = el("td", null, label); td.title = key; tr.appendChild(td);
    tr.appendChild(el("td", {class: "num"}, fmtRel((ks(key, "rb") || [])[0])));
    tr.appendChild(el("td", {class: "num"}, fmtRel((ks(key, "rc") || [])[0])));
    tr.appendChild(el("td", {class: "num"}, fmtX((ks(key, "r") || [])[0])));
    t.appendChild(tr);
  }
  return t;
}
function drawPanel() {
  const P = $("panel"); P.replaceChildren();
  if (!st.sel) {
    P.appendChild(el("h3", null, "Детали"));
    P.appendChild(el("p", {class: "note"}, "Нажмите на узел или ребро графа. Наведение показывает значения, клик — разбивку: внутренности слоя (q/k/v-проекции, веса внимания, подслои MLP), расхождение по группам токенов и входящие/исходящие рёбра."));
    return;
  }
  if (st.sel.type === "node") {
    const n = NODE[st.sel.id];
    P.appendChild(el("h3", null, n.label));
    P.appendChild(el("div", {class: "code mono"}, n.code));
    if (n.quote) P.appendChild(el("p", {class: "quote"}, "«" + n.quote + "»"));
    if (n.note) P.appendChild(el("p", {class: "note"}, n.note));
    if (n.kind === "const") { P.appendChild(el("p", null, "Не зависит от бокса: внутри пары тензор одинаков, расхождение ≡ 0 (проверено).")); }
    else {
      P.appendChild(metricTable([["ratio", ns(n.id, "r"), fmtX], ["rel best→bad", ns(n.id, "rb"), fmtRel], ["rel best→control", ns(n.id, "rc"), fmtRel]], true));
      if (n.internals.length) { P.appendChild(el("h3", {style: "margin-top:12px"}, "Внутри")); P.appendChild(keyTable(n.internals)); }
      if (n.tok) {
        P.appendChild(el("h3", {style: "margin-top:12px"}, "По группам токенов"));
        P.appendChild(keyTable(S.token_groups.map(([gid, lab]) => [n.key + "#" + gid, lab])));
      }
    }
    const inc = S.edges.filter(e => e.dst === n.id), out = S.edges.filter(e => e.src === n.id);
    for (const [title, list, other] of [["Входящие рёбра", inc, "src"], ["Исходящие рёбра", out, "dst"]]) {
      if (!list.length) continue;
      P.appendChild(el("h3", {style: "margin-top:12px"}, title));
      const t = el("table"); const h = el("tr");
      for (const c of ["ребро", "spec", "bad", "ctrl"]) h.appendChild(el("th", {class: c === "ребро" ? "" : "num"}, c));
      t.appendChild(h);
      for (const e of list) {
        const tr = el("tr", {class: "click"});
        tr.appendChild(el("td", null, NODE[e[other]].label + (e.label ? " · " + e.label : "")));
        for (const m of ["gs", "gb", "gc"]) tr.appendChild(el("td", {class: "num"}, fmtX((es(e.id, m) || [])[0])));
        tr.addEventListener("click", () => select({type: "edge", id: e.id}));
        t.appendChild(tr);
      }
      P.appendChild(t);
    }
  } else {
    const e = S.edges.find(x => x.id === st.sel.id);
    P.appendChild(el("h3", null, NODE[e.src].label + " → " + NODE[e.dst].label));
    P.appendChild(el("div", {class: "code mono"}, NODE[e.src].code + "  →  " + NODE[e.dst].code));
    if (e.label) P.appendChild(el("p", {class: "note"}, "роль: " + e.label));
    if (e.role === "pe") P.appendChild(el("p", {class: "quote"}, "«" + (e.label.indexOf("image") === 0
      ? "positional encodings are added to the image embedding whenever they participate in an attention layer"
      : "the entire original prompt tokens (including their positional encodings) are re-added to the updated tokens whenever they participate in an attention layer") + "»"));
    if (NODE[e.src].kind === "const") P.appendChild(el("p", null, "Источник не зависит от бокса — усиление не определено."));
    else P.appendChild(metricTable([["специфическое", es(e.id, "gs"), fmtX], ["best→bad", es(e.id, "gb"), fmtX], ["best→control", es(e.id, "gc"), fmtX]], true));
    const b1 = el("button", {type: "button"}, "← " + NODE[e.src].label); b1.addEventListener("click", () => select({type: "node", id: e.src}));
    const b2 = el("button", {type: "button"}, NODE[e.dst].label + " →"); b2.addEventListener("click", () => select({type: "node", id: e.dst}));
    const row = el("p"); row.append(b1, document.createTextNode(" "), b2); P.appendChild(row);
  }
}
function select(s) { st.sel = s; drawGraph(); drawPanel(); writeHash(); }

// ---------- tables ----------
function drawTables() {
  const t = el("table"); const h = el("tr");
  for (const c of ["узел", "код", "rel best→bad", "rel best→control", "ratio", ">1"]) h.appendChild(el("th", {class: ["узел", "код"].includes(c) ? "" : "num"}, c));
  t.appendChild(h);
  for (const n of S.nodes) {
    if (n.kind === "const") continue;
    const tr = el("tr", {class: "click"});
    tr.appendChild(el("td", null, n.label)); tr.appendChild(el("td", {class: "mono"}, n.key));
    tr.appendChild(el("td", {class: "num"}, statText(ns(n.id, "rb"), fmtRel)));
    tr.appendChild(el("td", {class: "num"}, statText(ns(n.id, "rc"), fmtRel)));
    tr.appendChild(el("td", {class: "num"}, statText(ns(n.id, "r"), fmtX)));
    tr.appendChild(el("td", {class: "num"}, npos(ns(n.id, "r"))));
    tr.addEventListener("click", () => { select({type: "node", id: n.id}); $("graph").scrollIntoView({behavior: "smooth", block: "start"}); });
    t.appendChild(tr);
  }
  $("tbl-nodes").replaceChildren(t);

  const list = S.edges.map(e => ({e, s: es(e.id, st.em)})).filter(x => x.s && x.s[0] != null).sort((a, b) => b.s[0] - a.s[0]).slice(0, 15);
  const t2 = el("table"); const h2 = el("tr");
  for (const c of ["ребро", "роль", METRIC[st.em].name, ">1"]) h2.appendChild(el("th", {class: ["ребро", "роль"].includes(c) ? "" : "num"}, c));
  t2.appendChild(h2);
  for (const {e, s} of list) {
    const tr = el("tr", {class: "click"});
    tr.appendChild(el("td", null, NODE[e.src].label + " → " + NODE[e.dst].label + "  (" + NODE[e.dst].code.split("  ")[0] + ")"));
    tr.appendChild(el("td", null, e.label || ""));
    tr.appendChild(el("td", {class: "num"}, statText(s, fmtX)));
    tr.appendChild(el("td", {class: "num"}, npos(s)));
    tr.addEventListener("click", () => select({type: "edge", id: e.id}));
    t2.appendChild(tr);
  }
  $("tbl-edges").replaceChildren(t2);
}

// ---------- thumbnails ----------
function pairCard(pid) {
  const p = D.pairs[pid];
  const c = el("div", {class: "thumb"});
  c.appendChild(el("h3", null, "пара #" + pid + " · " + p.image));
  const k = el("div");
  for (const [cls, lab] of [["--best", "best"], ["--bad", "bad"], ["--ctrl", "control #1"]]) {
    const s = el("span", {class: "key"}); s.style.background = "var(" + cls + ")"; k.append(s, document.createTextNode(lab + "   "));
  }
  k.appendChild(el("span", {class: "muted"}, "белый контур — GT"));
  c.appendChild(k);
  if (p.thumb) c.appendChild(el("img", {src: "data:image/jpeg;base64," + p.thumb, alt: "best, bad и control маски пары " + pid}));
  const t = el("table"); const h = el("tr");
  for (const x of ["", "box (1024-frame)", "IoU JSON", "IoU пересчёт", "pred IoU"]) h.appendChild(el("th", x.startsWith("IoU") || x === "pred IoU" ? {class: "num"} : null, x));
  t.appendChild(h);
  const rows = [["best", p.best_box, p.json_best_iou, p.iou_best, p.pred_iou_best], ["bad", p.bad_box, p.json_bad_iou, p.iou_bad, p.pred_iou_bad]];
  p.controls.forEach((x, i) => rows.push(["control #" + (i + 1), x.box, null, x.iou, x.pred_iou]));
  for (const r of rows) {
    const tr = el("tr");
    tr.appendChild(el("td", null, r[0])); tr.appendChild(el("td", {class: "mono"}, "[" + r[1].join(", ") + "]"));
    for (const v of r.slice(2)) tr.appendChild(el("td", {class: "num"}, v == null ? "" : v.toFixed(4)));
    t.appendChild(tr);
  }
  c.appendChild(t);
  c.appendChild(el("p", {class: "muted"}, "сдвиг best→bad L∞ = " + p.linf_shift + " px (1024-frame); IoU(маска best, маска bad) = " + (p.iou_best_vs_bad_mask ?? "—")));
  return c;
}
function drawThumbs() {
  const w = $("thumbs-wrap"); w.replaceChildren();
  if (st.img === "all") return;
  const img = D.images.find(x => x.name === st.img);
  const pids = st.pair ? [st.pair] : img.pids.map(String);
  w.appendChild(el("h2", null, st.pair ? "Пара" : "Пары картинки " + st.img));
  const box = el("div", {class: "thumbs"});
  for (const p of pids) box.appendChild(pairCard(p));
  w.appendChild(box);
}

// ---------- checks & repro ----------
function drawChecks() {
  const C = D.checks, box = $("checks"); box.replaceChildren();
  const tv = C.trace_vs_official;
  const items = [
    [tv.worst_rel_diff <= C.trace_tol, "Трасса = SAM: " + tv.runs + " прогонов, " + tv.tensors_compared + " тензоров сверено с выходами модулей (forward-хуки на официальном SamPredictor.predict_torch); побитово совпали " + tv.bit_exact_runs + "/" + tv.runs + " прогонов; худшее отн. расхождение " + tv.worst_rel_diff.toExponential(2) + (tv.worst_where ? " (" + tv.worst_where + ")" : "") + ". При расхождении выше " + C.trace_tol + " скрипт падает."],
    [true, "Покрытие: каждый модуль prompt_encoder / mask_decoder, сработавший при предсказании, сработал ровно один раз и есть в трассе; каждый модуль трассы сработал. Сработало модулей: " + tv.modules_fired.length + "."],
    [C.determinism.max_rel === 0, "Детерминизм: повторный прогон best box, макс. rel = " + C.determinism.max_rel.toExponential(2) + (C.determinism.where ? " (" + C.determinism.where + ")" : "") + "."],
    [C.box_independent_zero.max_rel === 0, "Тензоры, не зависящие от бокса (" + C.box_independent_zero.keys.length + "), дают rel = " + C.box_independent_zero.max_rel.toExponential(2) + (C.box_independent_zero.where ? " (" + C.box_independent_zero.where + ")" : "") + "."],
    [C.grid.max_dev_px < 0.01, "Кадр боксов: JSON-боксы / масштаб ResizeLongestSide отстоят от целой пиксельной сетки исходной картинки максимум на " + C.grid.max_dev_px.toFixed(4) + " px (ожидается ≈0: боксы действительно в 1024-кадре SAM)."],
  ];
  const ul = el("ul");
  for (const [ok, text] of items) { const li = el("li"); li.appendChild(el("b", {class: ok ? "ok" : "warn"}, ok ? "OK  " : "ВНИМАНИЕ  ")); li.appendChild(document.createTextNode(text)); ul.appendChild(li); }
  box.appendChild(ul);
  const d = el("details"); d.appendChild(el("summary", null, "модули, сработавшие при предсказании (" + tv.modules_fired.length + ")"));
  d.appendChild(el("pre", {class: "mono"}, tv.modules_fired.join("\n"))); box.appendChild(d);
}
function drawRepro() {
  const t = el("table"); const h = el("tr");
  for (const c of ["#", "картинка", "статус", "IoU best JSON / пересчёт", "IoU bad JSON / пересчёт", "controls", "причина"]) h.appendChild(el("th", null, c));
  t.appendChild(h);
  const f = v => v == null ? "—" : Number(v).toFixed(4);
  for (const r of D.repro) {
    const tr = el("tr");
    tr.appendChild(el("td", {class: "num"}, String(r.pid)));
    tr.appendChild(el("td", null, r.image));
    tr.appendChild(el("td", {class: r.status === "kept" ? "ok" : "fail"}, r.status === "kept" ? "в графе" : "выброшена"));
    tr.appendChild(el("td", {class: "num"}, f(r.json_best_iou) + " / " + f(r.iou_best)));
    tr.appendChild(el("td", {class: "num"}, f(r.json_bad_iou) + " / " + f(r.iou_bad)));
    tr.appendChild(el("td", {class: "num"}, r.n_ctrl != null ? r.n_ctrl + " из " + r.ctrl_tried : ""));
    tr.appendChild(el("td", null, r.reason || ""));
    t.appendChild(tr);
  }
  $("repro").replaceChildren(t);
}

// ---------- controls ----------
function fillSelects() {
  const si = $("sel-img"); si.replaceChildren();
  si.appendChild(el("option", {value: "all"}, "Все картинки — медиана (" + D.images.length + ")"));
  for (const im of D.images) si.appendChild(el("option", {value: im.name}, im.name + "  (" + im.pids.length + " пар)"));
  si.value = st.img;
  const sp = $("sel-pair"); sp.replaceChildren();
  if (st.img === "all") { sp.appendChild(el("option", {value: ""}, "—")); sp.disabled = true; }
  else {
    sp.disabled = false;
    const im = D.images.find(x => x.name === st.img);
    sp.appendChild(el("option", {value: ""}, "медиана по " + im.pids.length + " парам"));
    for (const p of im.pids) { const q = D.pairs[String(p)]; sp.appendChild(el("option", {value: String(p)}, "#" + p + ": IoU " + q.iou_best.toFixed(3) + " → " + q.iou_bad.toFixed(3))); }
    sp.value = st.pair;
  }
}
// view state lives in the URL hash, so a link reopens the same view
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  const th = h.get("theme");
  if (th === "light" || th === "dark") document.documentElement.setAttribute("data-theme", th);
  const img = h.get("img");
  if (img && D.images.some(x => x.name === img)) st.img = img;
  const pair = h.get("pair");
  if (pair && st.img !== "all" && D.images.find(x => x.name === st.img).pids.map(String).includes(pair)) st.pair = pair;
  if (["r", "rb", "rc"].includes(h.get("nm"))) st.nm = h.get("nm");
  if (["gs", "gb", "gc"].includes(h.get("em"))) st.em = h.get("em");
  st.pe = h.get("pe") === "1";
  const sel = h.get("sel") || "", i = sel.indexOf(":");
  if (i > 0) {
    const t = sel.slice(0, i), id = sel.slice(i + 1);
    if ((t === "node" && NODE[id]) || (t === "edge" && S.edges.some(e => e.id === id))) st.sel = {type: t, id};
  }
  $("sel-nm").value = st.nm; $("sel-em").value = st.em; $("chk-pe").checked = st.pe;
}
function writeHash() {
  const h = new URLSearchParams();
  if (st.img !== "all") h.set("img", st.img);
  if (st.pair) h.set("pair", st.pair);
  if (st.nm !== "r") h.set("nm", st.nm);
  if (st.em !== "gs") h.set("em", st.em);
  if (st.pe) h.set("pe", "1");
  if (st.sel) h.set("sel", st.sel.type + ":" + st.sel.id);
  const th = document.documentElement.getAttribute("data-theme");
  if (th) h.set("theme", th);
  try { history.replaceState(null, "", "#" + h.toString()); } catch (e) { /* file:// in some browsers */ }
}
function render() { fillSelects(); drawLegend(); drawSummary(); drawGraph(); drawPanel(); drawTables(); drawThumbs(); writeHash(); }
$("sel-img").addEventListener("change", e => { st.img = e.target.value; st.pair = ""; render(); });
$("sel-pair").addEventListener("change", e => { st.pair = e.target.value; render(); });
$("sel-nm").addEventListener("change", e => { st.nm = e.target.value; render(); });
$("sel-em").addEventListener("change", e => { st.em = e.target.value; render(); });
$("chk-pe").addEventListener("change", e => { st.pe = e.target.checked; drawGraph(); writeHash(); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);

const m = D.meta;
$("meta").textContent = m.model + " · " + m.checkpoint + " · " + m.json + ": " + m.n_pairs_kept + "/" + m.n_pairs_json +
  " пар воспроизведено (допуск IoU ±" + m.repro_tol + "), " + m.n_images + " картинок · control: K=" + m.n_controls +
  ", IoU ≥ " + m.ctrl_min_iou + ", cos ≤ " + m.ctrl_max_cos + " · " + m.device + ", torch " + m.torch + " · " + m.created;
readHash(); drawChecks(); drawRepro(); render();
})();
</script>
</body>
</html>
"""


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--critical_shifts", default="critical_shifts_coco.json")
    p.add_argument("--images_dir", help="COCO_MVal/img")
    p.add_argument("--masks_dir", help="COCO_MVal/gt")
    p.add_argument("--mask_ext", default=".png")
    p.add_argument("--checkpoint_path", default="", help="sam_vit_b_01ec64.pth")
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--repro_tol", type=float, default=0.02,
                   help="max |IoU recomputed - IoU in JSON| for best and bad")
    p.add_argument("--n_controls", type=int, default=4)
    p.add_argument("--ctrl_min_iou", type=float, default=0.85,
                   help="a control box must keep IoU with GT >= this (the 'good' threshold of find_critical_shifts)")
    p.add_argument("--ctrl_max_cos", type=float, default=0.9,
                   help="max cosine between the control and the attack displacement")
    p.add_argument("--ctrl_tries", type=int, default=300)
    p.add_argument("--trace_tol", type=float, default=1e-5,
                   help="max relative difference trace vs SAM's own module outputs (expected: 0)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit_images", type=int, default=0, help="0 = all")
    p.add_argument("--thumb_h", type=int, default=150, help="thumbnail height in px, 0 = none")
    p.add_argument("--out_dir", default="exp_res/activation_graph")
    p.add_argument("--self_test", action="store_true",
                   help="verify the trace on a random image (random weights unless --checkpoint_path)")
    args = p.parse_args(argv)
    if not args.self_test and not (args.images_dir and args.masks_dir and args.checkpoint_path):
        p.error("--images_dir, --masks_dir and --checkpoint_path are required (or use --self_test)")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
