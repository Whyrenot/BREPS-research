"""
steer_gate.py
=============
Which boxes should be steered, and how hard? steer_user_study.py adds one
frozen vector at one alpha to every box: large alphas repair the bad half of
the boxes and start to break the good half, and the alpha that suits natural
images is far too strong on medical ones. Here the dose is decided per box,
from what SAM shows without the GT: a gate (steer at alpha g or not) and an
alpha selector (which alpha of the grid, or none).

DATA  (one GPU pass, dumped to boxes.csv.gz; --from_dump redoes the rest offline)
  tables    COCO_A, COCO_B (the split of steer_user_study), Berkeley, GrabCut,
            ACDC, BUID, MedScribble ... (--datasets), user study --use m
  vector    the frozen v of a finished steer_user_study run (--vector). On
            COCO_A, whose images v was estimated on, every image is steered with
            the v of the other folds instead (cross-fitting: d_img of the four
            other folds of A, --folds), so the labels a gate is trained on there
            are out of sample too; every other table gets the frozen v
  per box   the unsteered forward and one forward per alpha of --alphas
  features  nothing that needs the GT:
              box     pred0 (SAM's predicted IoU), stab0 (stability score of
                      the low-res logits at +-1), fill0 (mask area / box area),
                      mask_frac0, box_frac, log_aspect, and proj_<token>: the
                      L2.norm2 state projected on v, per token ("full" only)
              alpha   alpha, d_pred / d_fill / d_stab (steered - unsteered),
                      chg (|m XOR m0| / |m OR m0|): what the steering did to
                      the mask and to SAM's own estimate
  label     d_alpha = IoU(alpha) - IoU(unsteered)

POLICIES (fitted on the training tables of a regime, applied to the others)
  fixed a            every box at alpha a (--fixed_alphas)
  gate pred-IoU g    alpha g where pred0 < tau, tau = the best mean gain on train
  gate GBM[fs] g     alpha g where the regressor predicts d_g > 0
  select GBM[fs]     alpha = argmax of the predicted d over the grid, none if
                     no predicted d > 0: a per-box alpha regressor, discretised
  oracle             the best alpha of every box by its GT IoU (a ceiling)
  fs = feature set: out (outputs only: pred, stability, fill, mask change) or
  full (+ the state projections). One HistGradientBoostingRegressor per feature
  set on the long table (box x alpha), fixed hyperparameters, nothing tuned on
  a test table. The gate needs one steered forward per box, the selector one
  per alpha -- decoder only, the image is encoded once.

REGIMES  (--regimes, tables joined by +)
  COCO_A                              the gate learns from half A only and is
                                      scored on every other table
  COCO_A+COCO_B+Berkeley+GrabCut      all natural images; scored on the rest
                                      (the medical subsets)
  Training tables are also scored out of fold (image folds), as a sanity check.

CHECKS (checks.json; a failure stops the run): the vector is the one of its
  run (split, sha256) and d_img averages back to it; zero dose bit for bit on
  every image; undefended IoU vs --reference_csv; 12 / 13 groups; with
  --prev_per_case, the steered IoUs of the shared tables and alphas equal the
  earlier run's.

Example
-------
    CUDA_VISIBLE_DEVICES=0 python scripts/steer_gate.py \\
        --root /.../FOR_TEST --checkpoint_path /.../sam_vit_b_01ec64.pth \\
        --vector exp_res/steer_user_study/vector.npz \\
        --prev_per_case exp_res/steer_user_study_med/per_case.csv \\
        --reference_csv results/samfull_m.csv --out_dir exp_res/steer_gate

    # the policies again from the dump, no GPU
    python scripts/steer_gate.py --from_dump exp_res/steer_gate/boxes.csv.gz --out_dir exp_res/steer_gate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import steer_user_study as SUS  # noqa: E402
from steer_sam_decoder import Cfg, Runner, assign_folds, batch_iou, build_sites, feature_of, make_edit, mask_change  # noqa: E402

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
except ImportError:                                   # scikit-learn < 1.0
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KEY, GROUPS, TOKEN_NAMES, _a, _name = SUS.KEY, SUS.GROUPS, SUS.TOKEN_NAMES, SUS._a, SUS._name
PROJ = [f"proj_{t}" for t in TOKEN_NAMES]
BOX_FEATS = OrderedDict([("out", ["pred0", "stab0", "fill0", "mask_frac0", "box_frac", "log_aspect"])])
BOX_FEATS["full"] = BOX_FEATS["out"] + PROJ
ALPHA_FEATS = ["alpha", "d_pred", "chg", "d_fill", "d_stab"]
REF_COLS = OrderedDict([("best_of_n_iou", "best-of-N (csv)"), ("grad_final_iou", "grad (csv)"),
                        ("headsel_iou", "head-select (csv)")])


# ---------------------------------------------------------------------------
# GPU pass
# ---------------------------------------------------------------------------

class PredHook:
    """SAM's predicted IoU of every box of a Runner.forward (mask_decoder
    returns (masks, iou_pred), already cut to token 0 with multimask off)."""

    def __init__(self, sam):
        self.sam, self.buf = sam, []

    def __enter__(self):
        self.h = self.sam.mask_decoder.register_forward_hook(lambda m, i, o: self.buf.append(o[1][:, 0].float()))
        return self

    def __exit__(self, *exc):
        self.h.remove()

    def result(self) -> torch.Tensor:
        return torch.cat(self.buf)


def stability(low: torch.Tensor) -> torch.Tensor:
    """SAM's stability score on the low-res logits: |l > 1| / |l > -1|."""
    hi = (low > 1.0).flatten(1).sum(1).float()
    lo = (low > -1.0).flatten(1).sum(1).float()
    return hi / lo.clamp_min(1.0)


def image_rows(runner: Runner, site, root: Path, table: str, dataset: str, stem: str, recs, v: np.ndarray,
               alphas, tokens: str, checks: dict) -> pd.DataFrame:
    p, dev = runner.p, runner.device
    cases, boxes, gt = SUS.load_image_cases(p, root, stem, recs, dev)
    b = torch.as_tensor(boxes, device=dev)
    Vt = {site.id: {"dm": torch.as_tensor(v, device=dev)}}
    buf = []
    with PredHook(runner.sam) as ph:
        m0, low0 = runner.forward(b, record={site.id: lambda t, ctx: buf.append(feature_of(site, t, ctx).cpu())})
    SUS.check_zero(runner, b, m0, low0, make_edit(Cfg(-1, "steer", site.id, tokens, "dm", 0.0), site, Vt),
                   site.id, checks, f"{table}/{stem}")
    H = torch.cat(buf).numpy()
    in_h, in_w = p.input_size
    bw, bh = (boxes[:, 2] - boxes[:, 0]).clip(min=1e-3), (boxes[:, 3] - boxes[:, 1]).clip(min=1e-3)
    box_frac = bw * bh / float(in_h * in_w)
    frac0 = m0.flatten(1).float().mean(1).cpu().numpy()
    df = pd.DataFrame({"table": table, "dataset": dataset, "image_name": stem,
                       "kind": [c["kind"] for c in cases], "user": [c["user"] for c in cases],
                       "attempt": [c["attempt"] for c in cases],
                       "undef_iou": batch_iou(m0, gt).cpu().numpy(), "pred0": ph.result().cpu().numpy(),
                       "stab0": stability(low0).cpu().numpy(), "mask_frac0": frac0, "box_frac": box_frac,
                       "fill0": frac0 / box_frac, "log_aspect": np.log(bw / bh)})
    vn = np.linalg.norm(v, axis=-1).clip(min=1e-12)
    for i, t in enumerate(TOKEN_NAMES):
        df[f"proj_{t}"] = H[:, i] @ (v[i] / vn[i])
    for a in alphas:
        with PredHook(runner.sam) as ph:
            m, low = runner.forward(b, edits={site.id: make_edit(Cfg(-1, "steer", site.id, tokens, "dm", a), site, Vt)})
        frac = m.flatten(1).float().mean(1).cpu().numpy()
        df[f"iou_{_a(a)}"] = batch_iou(m, gt).cpu().numpy()
        df[f"pred_{_a(a)}"] = ph.result().cpu().numpy()
        df[f"chg_{_a(a)}"] = mask_change(m, m0).cpu().numpy()
        df[f"fill_{_a(a)}"] = frac / box_frac
        df[f"stab_{_a(a)}"] = stability(low).cpu().numpy()
    return df


def gpu_pass(args, out: Path, checks: dict) -> pd.DataFrame | None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    root = Path(args.root)
    datasets = list(dict.fromkeys(["COCO", *args.datasets]))
    index = SUS.index_study(root, args.use, datasets)
    A, B = SUS.split_halves(index["COCO"], args.split_seed)

    device = torch.device(args.device)
    predictor = SUS.load_predictor(args.checkpoint_path, args.model_type, device)
    sites = build_sites(len(predictor.model.mask_decoder.transformer.layers))
    site = next(s for s in sites if s.id == args.site)
    runner = Runner(predictor, sites, args.batch, field_grid=64, field_margin=0.25)

    got = SUS.load_vector(Path(args.vector), site.id, A, checks)
    if got is None:
        return None
    v = got[0]
    z = np.load(args.vector)
    images_A, d_img = [str(s) for s in z["images_A"]], z["d_img"]
    folds = assign_folds(images_A, args.folds, args.seed)
    fold_of = np.array([folds[s] for s in images_A])
    v_fold = {k: d_img[fold_of != k].mean(0) for k in sorted(set(fold_of))}
    checks["cross_fit"] = {"folds": len(v_fold), "images_A_match_split": images_A == A,
                           "d_img_mean_is_v": bool(np.allclose(d_img.mean(0), v, atol=1e-7)),
                           "cos_fold_vs_v": [float(np.dot(vk.ravel(), v.ravel()) / np.linalg.norm(vk) / np.linalg.norm(v))
                                             for vk in v_fold.values()]}
    checks["cross_fit"]["ok"] = checks["cross_fit"]["images_A_match_split"] and checks["cross_fit"]["d_img_mean_is_v"]
    if not checks["cross_fit"]["ok"]:
        return None

    lim = (lambda s: s[:args.limit_images]) if args.limit_images > 0 else (lambda s: s)
    tables = [("COCO_A", "COCO", lim(A)), ("COCO_B", "COCO", lim(B))]
    tables += [(d, d, lim(list(index[d]))) for d in datasets if d != "COCO"]
    parts = []
    t0 = time.time()
    for table, ds, stems in tables:
        print(f"\n[{table}] {len(stems)} images, alphas {', '.join(f'{a:g}' for a in args.alphas)}", flush=True)
        for i, stem in enumerate(stems):
            vv = v_fold[folds[stem]] if table == "COCO_A" else v
            df = image_rows(runner, site, root, table, ds, stem, index[ds][stem], vv, args.alphas, args.tokens, checks)
            if table == "COCO_A":
                df["fold"] = folds[stem]
            parts.append(df)
            print(f"[{table}] [{i + 1}/{len(stems)}] {stem}: {len(df)} boxes, undef {df.undef_iou.mean():.4f}, "
                  f"pred0 {df.pred0.mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
    D = pd.concat(parts, ignore_index=True)
    D["group"] = SUS.assign_groups(D)
    return D


def check_prev(D: pd.DataFrame, path: str, alphas) -> dict:
    """The steered IoUs of the tables and alphas an earlier steer_user_study
    run shares with this one must be identical (same v, same forwards)."""
    p = Path(path) if path else None
    if p is None or not p.is_file():
        return {"status": f"skipped ({path or 'no --prev_per_case'})", "ok": True}
    prev = pd.read_csv(p, dtype={k: str for k in KEY})
    shared = [a for a in alphas if f"dm_iou_{_a(a)}" in prev.columns]
    m = D[D.table != "COCO_A"].merge(prev[KEY + ["undef_iou"] + [f"dm_iou_{_a(a)}" for a in shared]],
                                      on=KEY, how="inner", suffixes=("", "_prev"))
    diffs = {"undef": float((m.undef_iou - m.undef_iou_prev).abs().max()) if len(m) else math.nan}
    for a in shared:
        diffs[f"{a:g}"] = float((m[f"iou_{_a(a)}"] - m[f"dm_iou_{_a(a)}"]).abs().max())
    worst = max(diffs.values()) if diffs else 0.0
    return {"path": str(p), "boxes": int(len(m)), "alphas": shared, "max_abs_diff": diffs, "ok": bool(worst <= 1e-6)}


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def long_xy(D: pd.DataFrame, alphas, fs: str):
    """Long table (alpha-major): box features + alpha features -> d_alpha."""
    bx = D[BOX_FEATS[fs]].to_numpy(float)
    X, y = [], []
    for a in alphas:
        k = _a(a)
        af = np.column_stack([np.full(len(D), a), D[f"pred_{k}"] - D.pred0, D[f"chg_{k}"],
                              D[f"fill_{k}"] - D.fill0, D[f"stab_{k}"] - D.stab0])
        X.append(np.hstack([bx, af]))
        y.append((D[f"iou_{k}"] - D.undef_iou).to_numpy())
    return np.vstack(X), np.concatenate(y)


def fit_tau(pred0: np.ndarray, d: np.ndarray) -> float:
    """Steer where pred0 < tau: the tau with the best mean gain (ties -> the
    one that steers fewer boxes)."""
    cands = np.concatenate([[-np.inf], np.unique(np.quantile(pred0, np.linspace(0, 1, 41))), [np.inf]])
    gains = [(d * (pred0 < t)).mean() for t in cands]
    return float(cands[int(np.argmax(gains))])


def make_gbm(seed: int, classifier: bool = False):
    cls = HistGradientBoostingClassifier if classifier else HistGradientBoostingRegressor
    kw = dict(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=40, l2_regularization=1.0,
              random_state=seed)
    try:
        return cls(early_stopping=False, **kw)
    except TypeError:                                 # scikit-learn < 0.23: no early stopping by default
        return cls(**kw)


class Policies:
    def __init__(self, args):
        self.alphas = list(args.alphas)
        self.gate = [a for a in args.gate_alphas if a in self.alphas]
        self.fixed = [a for a in args.fixed_alphas if a in self.alphas]
        self.fsets, self.seed = list(args.feature_sets), args.seed

    def col(self, a: float) -> int:
        return self.alphas.index(a) + 1

    def fit(self, D: pd.DataFrame) -> "Policies":
        self.tau = {g: fit_tau(D.pred0.to_numpy(), (D[f"iou_{_a(g)}"] - D.undef_iou).to_numpy()) for g in self.gate}
        self.models = {}
        for fs in self.fsets:
            X, y = long_xy(D, self.alphas, fs)
            self.models[fs] = make_gbm(self.seed).fit(X, y)
        return self

    def choose(self, D: pd.DataFrame) -> "OrderedDict[str, np.ndarray]":
        n = len(D)
        cols = OrderedDict([("undef", np.zeros(n, int))])
        for a in self.fixed:
            cols[f"fixed α={a:g}"] = np.full(n, self.col(a))
        for g in self.gate:
            cols[f"gate pred-IoU α={g:g}"] = np.where(D.pred0.to_numpy() < self.tau[g], self.col(g), 0)
        for fs in self.fsets:
            X, _ = long_xy(D, self.alphas, fs)
            P = self.models[fs].predict(X).reshape(len(self.alphas), n).T
            for g in self.gate:
                cols[f"gate GBM[{fs}] α={g:g}"] = np.where(P[:, self.col(g) - 1] > 0, self.col(g), 0)
            cols[f"select GBM[{fs}]"] = np.where(P.max(1) > 0, P.argmax(1) + 1, 0)
        return cols


def iou_matrix(D: pd.DataFrame, alphas) -> np.ndarray:
    return D[["undef_iou"] + [f"iou_{_a(a)}" for a in alphas]].to_numpy(float)


def oof_choose(D: pd.DataFrame, args) -> "OrderedDict[str, np.ndarray]":
    """Every training box chosen by policies fitted on the other image folds."""
    names = sorted(D.image_name.unique())
    folds = assign_folds(names, min(args.folds, len(names)), args.seed)
    f = D.image_name.map(folds).to_numpy()
    out = None
    for k in sorted(set(f)):
        tr, te = f != k, f == k
        cols = Policies(args).fit(D[tr]).choose(D[te])
        if out is None:
            out = OrderedDict((name, np.zeros(len(D), int)) for name in cols)
        for name, c in cols.items():
            out[name][te] = c
    return out


def score(D: pd.DataFrame, cols, alphas, args, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per policy and group: IoU, delta vs undefended with a 95% image
    bootstrap CI, the share of boxes steered and their mean alpha."""
    W = iou_matrix(D, alphas)
    A_ext = np.asarray([0.0, *alphas])
    rows_ = np.arange(len(D))
    named = OrderedDict((name, (W[rows_, c], W[:, 0], c)) for name, c in cols.items())
    named["oracle α"] = (W.max(1), W[:, 0], W.argmax(1))
    if extra is not None:
        for col, label in REF_COLS.items():
            if col in extra.columns:
                named[label] = (extra[col].to_numpy(float), extra.undefended_iou.to_numpy(float), None)
    rows = []
    img, grp = D.image_name.to_numpy(), D.group.to_numpy()
    for name, (x, u, c) in named.items():
        for g in GROUPS:
            m = np.ones(len(D), bool) if g == "all" else grp == g
            d = x[m] - u[m]
            ok = np.isfinite(d)
            lo, hi = SUS.boot_delta(d[ok], img[m][ok], args.n_boot, args.seed)
            rows.append({"policy": name, "group": g, "n": int(m.sum()), "iou": float(np.nanmean(x[m])),
                         "d": float(np.nanmean(d)), "d_lo": lo, "d_hi": hi,
                         "pct": float(np.nanmean(d) / np.nanmean(u[m]) * 100),
                         "steered": float((c[m] > 0).mean()) if c is not None else math.nan,
                         "mean_alpha": float(A_ext[c[m]].mean()) if c is not None else math.nan})
    return pd.DataFrame(rows)


def analyse(D: pd.DataFrame, ref: pd.DataFrame | None, args) -> pd.DataFrame:
    alphas = list(args.alphas)
    tables = list(dict.fromkeys(D.table))
    res = []
    for regime in args.regimes:
        train = [t for t in regime.split("+") if t in tables]
        if not train:
            print(f"[regime {regime}] no training table present -- skipped", flush=True)
            continue
        Dtr = D[D.table.isin(train)].reset_index(drop=True)
        print(f"[regime {regime}] train on {len(Dtr)} boxes; out of fold on the training tables", flush=True)
        oof = oof_choose(Dtr, args)
        for t in train:
            m = (Dtr.table == t).to_numpy()
            r = score(Dtr[m].reset_index(drop=True), OrderedDict((k, c[m]) for k, c in oof.items()), alphas, args)
            res.append(r.assign(regime=regime, eval="oof", table=t))
        pol = Policies(args).fit(Dtr)
        for t in tables:
            if t in train:
                continue
            Dt = D[D.table == t].reset_index(drop=True)
            extra = Dt[KEY].merge(ref, on=KEY, how="left") if ref is not None else None
            r = score(Dt, pol.choose(Dt), alphas, args, extra)
            res.append(r.assign(regime=regime, eval="test", table=t))
            print(f"[regime {regime}] test {t}: tau {', '.join(f'{g:g}->{pol.tau[g]:.3f}' for g in pol.gate)}",
                  flush=True)
    cols = ["regime", "eval", "table", "policy", "group"]
    out = pd.concat(res, ignore_index=True)
    return out[cols + [c for c in out.columns if c not in cols]]


# ---------------------------------------------------------------------------
# Methods table: one row per method, every test table, and their mean
# ---------------------------------------------------------------------------
#   IoU-head selector      alpha (or none) whose steered mask SAM's IoU head
#                          scores highest -- best-of-N over alpha, no training
#   alpha selector         a classifier of the box's best alpha (GT argmax over
#                          none + the grid), logistic regression or GBM, on the
#                          box features + every alpha's features side by side
#   alpha regressor        the gain of every alpha regressed (linear regression,
#                          one per alpha; or the long-table GBM of Policies),
#                          then the alpha with the largest predicted gain, none
#                          if no predicted gain > 0

def alpha_x(D: pd.DataFrame, a: float) -> np.ndarray:
    k = _a(a)
    return np.column_stack([D[f"pred_{k}"] - D.pred0, D[f"chg_{k}"], D[f"fill_{k}"] - D.fill0,
                            D[f"stab_{k}"] - D.stab0])


class Methods:
    def __init__(self, args):
        self.alphas, self.fs, self.seed = list(args.alphas), args.methods_features, args.seed
        self.fixed = [a for a in args.fixed_alphas if a in self.alphas]

    def _bx(self, D):
        return D[BOX_FEATS[self.fs]].to_numpy(float)

    def _wide(self, D):
        return np.hstack([self._bx(D)] + [alpha_x(D, a) for a in self.alphas])

    def fit(self, D: pd.DataFrame) -> "Methods":
        W = iou_matrix(D, self.alphas)
        Xw, best = self._wide(D), W.argmax(1)
        self.sel = OrderedDict([
            ("α-селектор, логистическая регрессия",
             make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(Xw, best)),
            ("α-селектор, GBM", make_gbm(self.seed, classifier=True).fit(Xw, best))])
        self.reg = GainRegressors(self.alphas, self.fs, self.seed).fit(D)
        return self

    def choose(self, D: pd.DataFrame) -> "OrderedDict[str, np.ndarray]":
        n = len(D)
        cols = OrderedDict((f"α = {a:g}", np.full(n, self.alphas.index(a) + 1)) for a in self.fixed)
        cols["IoU-head селектор"] = D[["pred0"] + [f"pred_{_a(a)}" for a in self.alphas]].to_numpy(float).argmax(1)
        Xw = self._wide(D)
        for name, m in self.sel.items():
            cols[name] = m.predict(Xw).astype(int)
        reg = self.reg.choose(D)
        cols["α-регрессор, линейная регрессия"] = reg["lin"]
        cols["α-регрессор, GBM"] = reg["gbm"]
        return cols


class GainRegressors:
    """The gain of every alpha, regressed from the features a box shows
    without its GT: one linear regression per alpha, and one GBM on the long
    table (box x alpha). choose() -> the column of iou_matrix to take: the
    alpha with the largest predicted gain, 0 (no steering) if none is > 0."""

    def __init__(self, alphas, fs: str = "out", seed: int = 0):
        self.alphas, self.fs, self.seed = list(alphas), fs, seed

    def fit(self, D: pd.DataFrame) -> "GainRegressors":
        W = iou_matrix(D, self.alphas)
        bx = D[BOX_FEATS[self.fs]].to_numpy(float)
        self.lin = [make_pipeline(StandardScaler(), LinearRegression()).fit(np.hstack([bx, alpha_x(D, a)]),
                                                                            W[:, j + 1] - W[:, 0])
                    for j, a in enumerate(self.alphas)]
        self.gbm = make_gbm(self.seed).fit(*long_xy(D, self.alphas, self.fs))
        return self

    def choose(self, D: pd.DataFrame) -> dict:
        n = len(D)
        bx = D[BOX_FEATS[self.fs]].to_numpy(float)
        P = {"lin": np.column_stack([m.predict(np.hstack([bx, alpha_x(D, a)]))
                                     for m, a in zip(self.lin, self.alphas)]),
             "gbm": self.gbm.predict(long_xy(D, self.alphas, self.fs)[0]).reshape(len(self.alphas), n).T}
        return {k: np.where(p.max(1) > 0, p.argmax(1) + 1, 0) for k, p in P.items()}


def boot_samples(d: np.ndarray, images: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap means of d over images (the boxes of an image drawn together)."""
    codes, img = np.unique(images, return_inverse=True)
    S = np.bincount(img, weights=d, minlength=len(codes))
    N = np.bincount(img, minlength=len(codes)).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(codes), size=(n_boot, len(codes)))
    return S[idx].sum(1) / N[idx].sum(1)


def methods_table(D: pd.DataFrame, ref: pd.DataFrame | None, args) -> pd.DataFrame:
    """Methods fitted on --methods_train, every other table scored; then the
    mean over the test tables (all / natural / medical), each table weighted
    equally, its CI from a bootstrap that resamples images within each table."""
    alphas = list(args.alphas)
    train = args.methods_train.split("+")
    M = Methods(args).fit(D[D.table.isin(train)].reset_index(drop=True))
    tests = [t for t in dict.fromkeys(D.table) if t not in train]
    per = OrderedDict()
    for t in tests:
        Dt = D[D.table == t].reset_index(drop=True)
        W = iou_matrix(Dt, alphas)
        rows = base_rows(Dt, ref)
        for name, c in M.choose(Dt).items():
            rows[name] = (W[np.arange(len(Dt)), c], W[:, 0])
        per[t] = (Dt, rows)
    return score_blocks(per, args)


def base_rows(Dt: pd.DataFrame, ref: pd.DataFrame | None) -> "OrderedDict[str, tuple]":
    """undef, and grad from the reference csv on the same boxes (against the
    csv's own undefended IoU)."""
    u = Dt.undef_iou.to_numpy(float)
    rows = OrderedDict([("undef", (u, u))])
    if ref is not None:
        e = Dt[KEY].merge(ref, on=KEY, how="left")
        rows["grad"] = (e.grad_final_iou.to_numpy(float), e.undefended_iou.to_numpy(float))
    return rows


def score_blocks(per: "OrderedDict[str, tuple]", args) -> pd.DataFrame:
    """per: block -> (boxes, {method: (iou, undef iou)}), the same methods in
    every block. Per block, method and group: IoU, delta with a 95% image
    bootstrap CI, %; then the mean over the blocks (all / natural / medical),
    each block weighted equally, its CI from bootstraps within each block."""
    tests = list(per)
    out = []
    samples = {}                                       # (table, method, group) -> bootstrap means of d
    for ti, (t, (Dt, rows)) in enumerate(per.items()):
        img, grp = Dt.image_name.to_numpy(), Dt.group.to_numpy()
        for name, (x, u) in rows.items():
            for g in GROUPS:
                m = (np.ones(len(Dt), bool) if g == "all" else grp == g) & np.isfinite(x)
                d = x[m] - u[m]
                s = boot_samples(d, img[m], args.n_boot, args.seed + ti)
                samples[(t, name, g)] = s
                lo, hi = np.percentile(s, [2.5, 97.5])
                out.append({"block": t, "method": name, "group": g, "n": int(m.sum()), "iou": x[m].mean(),
                            "base": u[m].mean(), "d": d.mean(), "d_lo": lo, "d_hi": hi,
                            "pct": (x[m].mean() - u[m].mean()) / u[m].mean() * 100})
    T = pd.DataFrame(out)
    natural = [t for t in tests if t not in SUS.MEDICAL]
    medical = [t for t in tests if t in SUS.MEDICAL]
    means = [(f"среднее по {len(tests)} test", tests), (f"среднее, натуральные ({len(natural)})", natural),
             (f"среднее, медицина ({len(medical)})", medical)]
    macro = []
    for i, (label, ts) in enumerate(means):
        if len(ts) < 2 or (i > 0 and ts == tests):    # a mean of one block, or a repeat of the overall one
            continue
        for name in per[tests[0]][1]:
            for g in GROUPS:
                s = T[(T.block.isin(ts)) & (T.method == name) & (T.group == g)]
                iou, base = s.iou.mean(), s.base.mean()
                boot = np.mean([samples[(t, name, g)] for t in ts], axis=0)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                macro.append({"block": label, "method": name, "group": g, "n": int(s.n.sum()), "iou": iou,
                              "base": base, "d": s.d.mean(), "d_lo": lo, "d_hi": hi,
                              "pct": (iou - base) / base * 100})
    return pd.concat([T, pd.DataFrame(macro)], ignore_index=True)


def md_methods(T: pd.DataFrame, D: pd.DataFrame, args) -> str:
    train = args.methods_train.split("+")
    n_tr = D[D.table.isin(train)]
    L = ["# Steering SAM-1 vit_b: фиксированные α и выбор α по боксу, user study (`m`)", "",
         "**Где и что прибавляется.** Узел `L2.norm2` — выход LayerNorm `norm2` второго (последнего) two-way блока "
         "декодера масок, т.е. поток токенов сразу после cross-attention «токены → картинка». Вектор прибавляется "
         "ко **всем 7 токенам** декодера: IoU-токен, 4 mask-токена (маску даёт mask-токен 0, multimask выключен) "
         "и 2 prompt-токена углов бокса; у каждого токена свой вектор (7 × 256), h' = h + α·v. "
         "Энкодер картинки, prompt encoder и остальной декодер не меняются.", "",
         f"**Вектор.** DM = среднее top50 − среднее bottom50, разность внутри каждой картинки, усреднена по "
         f"25 картинкам COCO (половина A, 625 боксов), заморожен; ‖v‖ ≈ 3.2% ‖h‖ при α = 1. "
         f"**Селекторы и регрессоры** обучены только на {', '.join(train)} ({n_tr.image_name.nunique()} картинок, "
         f"{len(n_tr)} боксов; метки — на векторе других фолдов A), признаки без GT (`{args.methods_features}`: "
         "предсказанный IoU, stability, заполнение бокса маской и то, как каждая α сдвигает маску и прогноз), "
         f"сетка α = {', '.join(f'{a:g}' for a in args.alphas)} или без steering. На тестовых наборах ничего "
         "не подбиралось.", "",
         "- *IoU-head селектор* — α (или без steering), при которой IoU-голова SAM даёт наибольший предсказанный "
         "IoU; без обучения;",
         "- *α-селектор* — классификатор лучшей α бокса (логистическая регрессия / GBM);",
         "- *α-регрессор* — регрессия выигрыша IoU при каждой α (линейная регрессия / GBM), берётся α с наибольшим "
         "предсказанным выигрышем, если он > 0;",
         "- *grad* — градиентная доводка бокса по IoU-голове (`samfull_m.csv`, те же боксы).", "",
         "Ячейка — IoU и прирост к undef в %; Δ — к undef с 95% CI (бутстреп по картинкам; для средних — внутри "
         "каждого набора). Группы — по undef IoU внутри картинки (12 худших / 13 лучших из 25). "
         "Жирным — лучший метод в колонке внутри блока.", ""]
    n_img = {t: D[D.table == t].image_name.nunique() for t in set(D.table)}
    return "\n".join(L) + "\n" + md_blocks(T, n_img)


def block_title(block: str, n_img: dict) -> str:
    if block not in n_img:
        return f"**{block}**"
    return f"**{'COCO (половина B)' if block == 'COCO_B' else _name(block)}**, {n_img[block]} изобр."


def md_blocks(T: pd.DataFrame, n_img: dict) -> str:
    """score_blocks' rows as one sequential table: IoU (% to undef) per group,
    and the delta over all boxes with its CI; bold = the best method of a
    column within its block."""
    L = ["| набор | метод | all | bottom50 | top50 | Δ all [95% CI] |", "|---|---|---|---|---|---|"]
    for block, B in T.groupby("block", sort=False):
        title = block_title(block, n_img)
        best = {g: B[(B.group == g) & (B.method != "undef")].iou.max() for g in GROUPS}
        for i, (name, s) in enumerate(B.groupby("method", sort=False)):
            s = s.set_index("group")
            cells = []
            for g in GROUPS:
                r = s.loc[g]
                c = f"{r.iou:.4f}" if name == "undef" else f"{r.iou:.4f} ({r.pct:+.2f}%)"
                cells.append(f"**{c}**" if name != "undef" and np.isclose(r.iou, best[g]) else c)
            a = s.loc["all"]
            ci = "—" if name == "undef" else f"{a.d:+.4f} [{a.d_lo:+.4f}, {a.d_hi:+.4f}]"
            L.append(f"| {title if i == 0 else ''} | {name} | " + " | ".join(cells) + f" | {ci} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _ci(r) -> str:
    return f"{r['d']:+.4f} [{r['d_lo']:+.4f}, {r['d_hi']:+.4f}]"


def md_policy_table(R: pd.DataFrame) -> str:
    cols = ["политика", "IoU all", "Δ all [95% CI]", "Δ bottom50 [95% CI]", "Δ top50 [95% CI]",
            "% all / b50 / t50", "steered", "ᾱ"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for pol, s in R.groupby("policy", sort=False):
        s = {r["group"]: r for r in s.to_dict("records")}
        a, b, t = s["all"], s["bottom50"], s["top50"]
        st = "-" if not np.isfinite(a["steered"]) else f"{a['steered']:.0%}"
        ma = "-" if not np.isfinite(a["mean_alpha"]) else f"{a['mean_alpha']:.1f}"
        out.append(f"| {pol} | {a['iou']:.4f} | {_ci(a)} | {_ci(b)} | {_ci(t)} | "
                   f"{a['pct']:+.2f} / {b['pct']:+.2f} / {t['pct']:+.2f} | {st} | {ma} |")
    return "\n".join(out)


def summary_md(res: pd.DataFrame, D: pd.DataFrame, args, checks: dict) -> str:
    L = ["# Гейт и выбор α для steering SAM-1 на user study (`m`)", "",
         f"Вектор `{args.vector}` (DM top50 − bottom50, `L2.norm2`, все токены), заморожен; на COCO_A — "
         f"кросс-фит по {args.folds} фолдам картинок. Сетка α = {', '.join(f'{a:g}' for a in args.alphas)}. "
         "Признаки — без GT: предсказанный SAM IoU, stability, заполнение бокса маской, проекции состояния на v "
         "(только `full`), и для каждого α — как steering сдвинул маску и предсказанный IoU. "
         "GBM предсказывает выигрыш IoU при каждом α; гейт применяет α, если предсказанный выигрыш > 0, "
         "селектор берёт α с наибольшим предсказанным выигрышем (или не трогает бокс). "
         "Δ — к undefended, CI — бутстреп по картинкам; steered — доля боксов со steering, ᾱ — средняя α "
         "(0 у нетронутых). Оракул — лучшая α по GT, потолок.", ""]
    for regime, R in res.groupby("regime", sort=False):
        train = regime.split("+")
        L += [f"## Обучение на {', '.join(train)}", ""]
        for (ev, table), s in R.groupby(["eval", "table"], sort=False):
            n_img = D[D.table == table].image_name.nunique()
            tag = "out of fold (обучающая таблица)" if ev == "oof" else "тест"
            L += [f"### {_name(table)} — {tag}, {n_img} картинок", "", md_policy_table(s), ""]
    z, rp, gr = checks.get("zero_dose", {}), checks.get("repro", {}), checks.get("groups", {})
    L += ["## Проверки", "",
          f"- нулевая доза: {z.get('bit_exact', '-')}/{z.get('forwards', '-')} бит в бит;",
          f"- undef vs reference csv: max |Δ| {rp.get('max_abs_diff', math.nan):.2e}, "
          f"доля > {args.repro_tol:g}: {rp.get('frac_over', math.nan):.2%};",
          f"- группы 12/13: отклонений {gr.get('n_images_not_12_13', '-')};",
          f"- вектор: {checks.get('vector_source', {}).get('ok', '-')}, кросс-фит: "
          f"{checks.get('cross_fit', {}).get('ok', '-')}; совпадение с прошлым прогоном: "
          f"{checks.get('prev_run', {}).get('max_abs_diff', checks.get('prev_run', {}).get('status', '-'))}."]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args) -> int:
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checks = {"zero_dose": {"forwards": 0, "bit_exact": 0, "failures": [], "ok": True}}

    def fail(name):
        (out / "checks.json").write_text(json.dumps(checks, indent=2, default=SUS._json_default), encoding="utf-8")
        print(f"\nCHECK FAILED: {name} -- see {out / 'checks.json'}", file=sys.stderr)
        return 1

    ref = SUS.read_reference(args.reference_csv, args.use)
    if args.from_dump:
        D = pd.read_csv(args.from_dump, dtype={k: str for k in KEY})
        prev = Path(args.from_dump).parent / "checks.json"
        if prev.is_file():
            checks = json.loads(prev.read_text(encoding="utf-8"))
    else:
        try:
            D = gpu_pass(args, out, checks)
        except SUS.CheckFailed as e:
            return fail(str(e))
        if D is None:
            return fail("vector / cross-fit")
        D.to_csv(out / "boxes.csv.gz", index=False)
        checks["repro"] = SUS.check_repro(D, ref, args)
        checks["groups"] = SUS.check_groups(D, ref, args, args.limit_images <= 0)
        checks["prev_run"] = check_prev(D, args.prev_per_case, args.alphas)
        failed = [k for k, r in checks.items() if isinstance(r, dict) and r.get("ok") is False]
        if failed:
            return fail(", ".join(failed))
        (out / "checks.json").write_text(json.dumps(checks, indent=2, default=SUS._json_default), encoding="utf-8")

    enc = sys.stdout.encoding or "utf-8"
    if not args.methods_only:
        res = analyse(D, ref, args)
        res.to_csv(out / "gate_results.csv", index=False)
        md = summary_md(res, D, args, checks)
        (out / "summary_gate.md").write_text(md, encoding="utf-8")
        print("\n" + md.encode(enc, "replace").decode(enc))
    T = methods_table(D, ref, args)
    T.to_csv(out / "table_methods.csv", index=False)
    md = md_methods(T, D, args)
    (out / "table_methods.md").write_text(md, encoding="utf-8")
    print("\n" + md.encode(enc, "replace").decode(enc))
    meta = {"args": vars(args), "boxes": int(len(D)), "tables": D.table.value_counts().to_dict(),
            "minutes": round((time.time() - t0) / 60, 1), "created": time.strftime("%Y-%m-%d %H:%M")}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=SUS._json_default), encoding="utf-8")
    print(f"wrote {out} ({(time.time() - t0) / 60:.1f} min)")
    return 0


def parse_args(argv=None):
    fl = SUS._floats
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", help="user study FOR_TEST (images/, masks/, user_masks/)")
    p.add_argument("--use", default="m")
    p.add_argument("--datasets", nargs="*", default=["Berkeley", "GrabCut", *SUS.MEDICAL],
                   choices=[d for d in SUS.DATASETS if d != "COCO"], help="tables besides COCO_A / COCO_B")
    p.add_argument("--checkpoint_path", default="")
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--site", default="L2.norm2")
    p.add_argument("--tokens", default="all")
    p.add_argument("--vector", default="exp_res/steer_user_study/vector.npz", help="steer_user_study's vector.npz")
    p.add_argument("--alphas", type=fl, default=fl("1,2,3,4,6,8,12,16"))
    p.add_argument("--fixed_alphas", type=fl, default=fl("1,4,12"), help="fixed-alpha rows of the report")
    p.add_argument("--gate_alphas", type=fl, default=fl("4,12"), help="alphas a gate switches on")
    p.add_argument("--feature_sets", nargs="*", default=list(BOX_FEATS), choices=list(BOX_FEATS))
    p.add_argument("--regimes", nargs="*", default=["COCO_A", "COCO_A+COCO_B+Berkeley+GrabCut"],
                   help="training tables per regime, joined by +; scored on every other table")
    p.add_argument("--methods_train", default="COCO_A", help="training tables of the methods table, joined by +")
    p.add_argument("--methods_features", default="out", choices=list(BOX_FEATS), help="features of the methods table")
    p.add_argument("--methods_only", action="store_true", help="only the methods table (with --from_dump)")
    p.add_argument("--folds", type=int, default=5, help="image folds: cross-fitting on A, out-of-fold scores")
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reference_csv", default="")
    p.add_argument("--prev_per_case", default="exp_res/steer_user_study_med/per_case.csv",
                   help="an earlier steer_user_study per_case.csv whose steered IoUs must be reproduced")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--repro_tol", type=float, default=1e-3)
    p.add_argument("--repro_max_frac", type=float, default=0.01)
    p.add_argument("--group_tol", type=float, default=1e-3)
    p.add_argument("--limit_images", type=int, default=0, help="per table, for a smoke run; 0 = all")
    p.add_argument("--from_dump", default="", help="boxes.csv.gz of an earlier run: policies only, no GPU")
    p.add_argument("--out_dir", default="exp_res/steer_gate")
    args = p.parse_args(argv)
    if not args.from_dump and not args.root:
        p.error("--root is required (or --from_dump)")
    return args


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
