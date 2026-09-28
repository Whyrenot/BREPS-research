"""
selector_data.py
================
Shared data preparation for the selector experiments: loading the long-format
candidate pool written by `dump_selector_data.py`, carving it into the three
decision problems, and building the derived features that the dump
deliberately leaves out.

The dump's own docstring explains why rank/z-score/margin of `f_pred` are not
written to disk: they depend on WHICH candidates you are choosing among, and
that is decided per question. This module is where that decision is made, once,
so all three questions and both model families see the same matrices.

Three pools, three feature policies
-----------------------------------
  T1  early stop      cand_kind == "grad", 11 checkpoints per case, walked in
                      step order. Features must be CAUSAL IN TIME: a checkpoint
                      at step 5 may use steps 0..5 and nothing later. So the
                      pool-relative block is forbidden (it peeks at step 50) and
                      is replaced by an expanding-window block. x_iou_grad is a
                      hard leak and is dropped; x_iou_undef / x_iou_hsel are
                      drift-from-the-start and ARE causal -- the undefended mask
                      and the step-0 mask both exist before any step is taken --
                      but they are gated behind `causal_x` so the strict variant
                      can be run too.
  T2  best-of-N       the 16 bon candidates plus the undefended mask. The whole
                      pool exists at once, so pool-relative features are fair.
                      x_iou_undef and x_iou_bon are available (they cost nothing
                      beyond the bon run); x_iou_hsel / x_iou_grad are not,
                      since no ascent was run.
  T3  all methods     T2's pool plus grad_final. Everything is available.
                      CAVEAT, inherited from the dump: grad rows read their
                      f_mask_* off the 256px logits while bon/undef rows read
                      them off the full-resolution mask. The two are not on the
                      same scale, so a `c_is_grad` indicator is added and the
                      model is left to condition on it. Nothing can be done
                      about it here; it is a property of the CSV.

Groups
------
weak / mid / strong are cut WITHIN each image, 20/60/20 over the image's 25
annotations ranked by `gt_undef_iou` -- the same stratification used by
regroup_by_image_rank.py and train_step_selector.py. Ranking on the undefended
IoU (not on the first checkpoint) makes the partition identical across all
three tasks, which is the only way their per-group numbers are comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CASE_KEYS = ["image_name", "kind", "user", "attempt"]

# never features: identity, cost, label, or ground-truth derived
NON_FEATURES = {
    "image_name", "kind", "user", "attempt", "case", "dataset", "grp",
    "cand_kind", "cand_id", "method_slot", "start", "step", "y",
    "n_forwards", "n_batched_fwd", "target_iou",
    "gt_undef_iou", "gt_clean_iou", "gt_bad_iou_json", "gt_best_iou_json",
}


def load_pool(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for k in CASE_KEYS:
        if k not in df.columns:
            df[k] = ""
    df[CASE_KEYS] = df[CASE_KEYS].fillna("")
    df["case"] = df[CASE_KEYS].astype(str).agg("|".join, axis=1)
    df["dataset"] = df.image_name.str.split("_").str[0]
    return df.sort_values(["case", "cand_kind", "step", "cand_id"],
                          kind="mergesort").reset_index(drop=True)


def image_groups(df: pd.DataFrame, lo: float = 0.2, hi: float = 0.8) -> pd.Series:
    """weak/mid/strong per case, cut inside each image by undefended IoU."""
    base = df.groupby("case").agg(image_name=("image_name", "first"),
                                  v=("gt_undef_iou", "first"))
    out = pd.Series(index=base.index, dtype=object)
    for _, sub in base.groupby("image_name"):
        o = sub.v.sort_values(kind="mergesort").index
        n = len(o)
        a, b = int(round(lo * n)), int(round(hi * n))
        out[o[:a]] = "weak"
        out[o[a:b]] = "mid"
        out[o[b:]] = "strong"
    return out


def numeric_feature_cols(df: pd.DataFrame, extra_drop=()) -> list[str]:
    drop = NON_FEATURES | set(extra_drop)
    cols = []
    for c in df.columns:
        if c in drop or not c.startswith(("f_", "x_", "c_")):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


def prune(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Drop all-missing and single-valued columns; keep the given order."""
    keep, seen = [], set()
    for c in cols:
        if c in seen:
            continue
        seen.add(c)
        v = df[c]
        if v.notna().sum() == 0 or v.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    return keep


# --------------------------------------------------------------------------
# pool-relative block (T2 / T3): where does this candidate sit inside its pool
# --------------------------------------------------------------------------
def pool_relative(df: pd.DataFrame, cols: list[str], key: str = "case") -> pd.DataFrame:
    """z, margin-to-mean, margin-to-best, span position and percentile rank.

    A selector never has to say how good a mask is in absolute terms -- only
    which of 17 is best. Everything constant within a pool is noise for that
    question, and these transforms are what remove it. For the linear models
    they are not an improvement but a precondition: a fixed weight vector on raw
    features cannot express "highest f_pred of this particular pool".
    """
    g = df.groupby(key, sort=False)[cols]
    agg = g.agg(["mean", "std", "max", "min"])
    idx = df[key].values
    rank = g.rank(pct=True, method="average")
    out = {}
    for c in cols:
        mu = agg[(c, "mean")].reindex(idx).values
        sd = agg[(c, "std")].reindex(idx).values
        mx = agg[(c, "max")].reindex(idx).values
        mn = agg[(c, "min")].reindex(idx).values
        v = df[c].values
        with np.errstate(invalid="ignore", divide="ignore"):
            out["c_z_" + c] = np.where(sd > 0, (v - mu) / sd, np.nan)
            out["c_dmax_" + c] = v - mx
            out["c_dmean_" + c] = v - mu
            span = mx - mn
            out["c_span_" + c] = np.where(span > 0, (v - mn) / span, np.nan)
        out["c_rank_" + c] = rank[c].values
    # pool-level descriptors: they cancel in any within-pool comparison, but
    # they tell a tree HOW contested this pool is, which is what gates whether
    # the margin features mean anything
    for c in ("f_pred", "f_agree_mean", "x_iou_undef"):
        if c in cols:
            out["c_pool_std_" + c] = agg[(c, "std")].reindex(idx).values
            out["c_pool_mean_" + c] = agg[(c, "mean")].reindex(idx).values
    # Borda count over the features that point the same way as quality: one
    # consensus direction, handed to the model already formed. A tree would
    # eventually build something like it out of the individual ranks; a linear
    # model would not, since it cannot rank at all without help.
    borda = [(c, s) for c, s in BORDA_TERMS if c in cols]
    if borda:
        out["c_borda"] = np.mean([s * rank[c].values for c, s in borda], axis=0)
    return pd.DataFrame(out, index=df.index)


# sign = the direction that should mean "better mask"
BORDA_TERMS = (
    ("f_pred", +1), ("f_agree_mean", +1), ("f_agree_min", +1),
    ("f_cluster_frac", +1), ("f_mask_largest_cc_frac", +1),
    ("x_iou_undef", +1), ("f_box_box_iou", +1),
    ("f_mask_n_components", -1), ("f_mask_frac_outside_box", -1),
)


# --------------------------------------------------------------------------
# expanding block (T1): what the trajectory has done SO FAR
# --------------------------------------------------------------------------
def _since_last_max(cases: np.ndarray, at_max: np.ndarray) -> np.ndarray:
    """Checkpoints elapsed since the column last set a running record."""
    out = np.empty(len(cases), dtype=float)
    last = 0.0
    prev_case = None
    for i in range(len(cases)):
        if cases[i] != prev_case:
            prev_case, last = cases[i], 0.0
        out[i] = 0.0 if at_max[i] else last + 1.0
        last = out[i]
    return out


def expanding_causal(df: pd.DataFrame, cols: list[str], key: str = "case") -> pd.DataFrame:
    """Strictly backward-looking trajectory features.

    Rows must already be sorted by (case, step). Every column here is a function
    of checkpoints 0..s only, so a policy that stopped at step s could genuinely
    have computed it. `c_since_max_f_pred` is the one the stop rule is really
    about: the ascent overshoots, and the head's own score plateaus before the
    true IoU turns over.
    """
    g = df.groupby(key, sort=False)[cols]
    prev = g.shift(1)
    prev2 = g.shift(2)
    first = g.transform("first")
    cmax = g.cummax()
    cmin = g.cummin()
    n = (df.groupby(key, sort=False).cumcount() + 1).values.astype(float)
    csum = g.cumsum()
    out = {}
    for c in cols:
        v = df[c].values
        out["c_d1_" + c] = v - prev[c].values
        out["c_d2_" + c] = v - prev2[c].values
        out["c_dfirst_" + c] = v - first[c].values
        out["c_dcmax_" + c] = v - cmax[c].values
        out["c_dcmin_" + c] = v - cmin[c].values
        out["c_dcmean_" + c] = v - csum[c].values / n
    if "f_pred" in cols:
        out["c_acc_f_pred"] = (out["c_d1_f_pred"]
                               - (prev["f_pred"].values - prev2["f_pred"].values))
        out["c_since_max_f_pred"] = _since_last_max(
            df[key].values, np.isclose(df["f_pred"].values, cmax["f_pred"].values))
    out["c_n_seen"] = n
    return pd.DataFrame(out, index=df.index)


# --------------------------------------------------------------------------
# the three task matrices
# --------------------------------------------------------------------------
def build_t1(pool: pd.DataFrame, causal_x: bool = True):
    """Gradient checkpoints, in step order, with the expanding block."""
    d = pool[pool.cand_kind == "grad"].copy()
    d = d.sort_values(["case", "step"], kind="mergesort").reset_index(drop=True)
    drop = ["x_iou_grad", "x_iou_bon"]          # future mask / other pipeline
    if not causal_x:
        drop += ["x_iou_undef", "x_iou_hsel"]
    base = prune(d, numeric_feature_cols(d, extra_drop=drop))
    ext = expanding_causal(d, base)
    d = pd.concat([d, ext], axis=1)
    return d, base + prune(d, list(ext.columns))


def _build_choice(rows: pd.DataFrame, drop_x: list[str]):
    d = rows.sort_values(["case", "cand_kind", "cand_id"],
                         kind="mergesort").reset_index(drop=True)
    d["c_is_undef"] = (d.cand_kind == "undef").astype(float)
    d["c_is_grad"] = (d.cand_kind == "grad").astype(float)
    d["c_is_bon_pick"] = (d.method_slot == "bon").astype(float)
    base = prune(d, numeric_feature_cols(d, extra_drop=drop_x + ["c_is_undef",
                                                                "c_is_grad",
                                                                "c_is_bon_pick"]))
    rel = pool_relative(d, base)
    d = pd.concat([d, rel], axis=1)
    cols = base + ["c_is_undef", "c_is_grad", "c_is_bon_pick"] + list(rel.columns)
    return d, prune(d, cols)


def build_t2(pool: pd.DataFrame):
    """16 bon candidates + the undefended mask."""
    rows = pool[pool.cand_kind.isin(("bon", "undef"))].copy()
    return _build_choice(rows, ["x_iou_hsel", "x_iou_grad"])


def build_t3(pool: pd.DataFrame):
    """T2's pool + the final gradient mask."""
    rows = pool[pool.cand_kind.isin(("bon", "undef"))
                | ((pool.cand_kind == "grad") & (pool.method_slot == "grad"))].copy()
    return _build_choice(rows, [])
