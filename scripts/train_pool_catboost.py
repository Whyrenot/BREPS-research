"""
train_pool_catboost.py
======================
The CatBoost selector over the paper's candidate pool of one model
(build_candidate_pool.py -> exp_res/pool_<model>/): for every user-study box it
picks one mask among undef, head-select, grad and BoN in both output modes and
steering in both modes at every alpha. CatBoost on the GPU (--task_type GPU) or
the CPU; no torch, no segmentation model.

DATA (--pool_dir)
  pool.csv.gz     one row per (box, candidate): f_* / x_* features, target_iou
  rows.csv.gz     the alpha regressors' features (steer_gate's "out" set), joined
                  onto the pool -- per box: c_reg_box_frac, c_reg_log_aspect and,
                  per output mode, the unsteered predicted IoU, stability, box fill
                  and mask fraction; per candidate (f_reg_*): its stability, box
                  fill, mask change against alpha 0 and d_pred / d_fill / d_stab --
                  for steering at its alpha, for undef / head-select at alpha 0
                  (d_* = 0), NaN for grad / BoN (not measured). Its IoU columns are
                  the GT and are never read as features.
EXCLUDED  the COCO A images (half A of the scan's split.json): the steering
  vector was estimated on them, so they are neither trained on nor scored --
  475 images, 11 875 boxes remain. The steering baseline's alpha* is the one
  thing chosen on them (as the scan chose it), before they are dropped.
FEATURES  every numeric f_* / x_* / c_reg_* column (selector_data.numeric_feature_
  cols), the pool-relative block of selector_data.pool_relative over the f_* /
  x_* ones (z, margin to the pool's best / mean, span position, rank inside the
  box's pool, pool spread, Borda), cand_kind as a categorical feature; columns
  with one value dropped.
MODEL  CatBoostRanker, YetiRank, one group per box, label target_iou; depth 6,
  lr 0.06, 800 iterations as selector_steer.py -- nothing tuned on test data.
PROTOCOL  --folds image folds stratified by subset (train_selectors.make_folds):
  every box is scored by the model of the folds that do not hold its image (out
  of fold). Then one model on all the images. Saved: model_fold<k>.cbm,
  model_full.cbm, folds.json, features.json -- what --apply_model needs.

TRANSFER (--apply_model <out_dir of another model's run>)
  The source run's selectors score this pool. --apply_with fold (default): an
  image is scored by the source model of the fold that held it out, so neither
  the image (on any model) nor this model's candidates were seen in training;
  full: model_full.cbm for every image (the image was seen on the source model).
  Features of the source run missing here are NaN (listed in checks.json).

METRICS (per subset, the natural / medical / all means of the subsets, and
  all / bottom50 / top50 inside each image)
  mean IoU of undef, head-select, grad / BoN in both modes, steering at alpha*
  in both modes, max predicted IoU over the pool, CatBoost, the pool's oracle;
  % to undef; CatBoost's gain over every method in % of it with a paired 95%
  bootstrap CI over images (block means: each subset weighted equally,
  resampled within each); P@1 (the pick is the pool's best), near-best (within
  0.01), regret, and which families CatBoost picks.

OUTPUTS (--out_dir): per_box.csv.gz (one row per box: every method's IoU, the
  pick), oof_scores.csv.gz, table.csv, picks.csv, importance.csv, summary.md,
  checks.json, meta.json, and the models above.

    # train (GPU), then transfer the SAM-3 selectors to SAM-1
    python scripts/train_pool_catboost.py --pool_dir exp_res/pool_SAM3 --out_dir exp_res/cb_SAM3 --devices 0
    python scripts/train_pool_catboost.py --apply_model exp_res/cb_SAM3 --pool_dir exp_res/pool_SAM \\
        --out_dir exp_res/cb_SAM3_on_SAM
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import selector_data as SD  # noqa: E402

KEY = SD.CASE_KEYS
MODES = (0, 1)
PRE = {0: "tok0", 1: "multi"}
STR_COLS = KEY + ["dataset", "half", "group", "cand_kind", "method_slot"]
CAT = ["c_kind"]
MEDICAL = ("ACDC", "BUID", "MedScribble")
ORDER = ["COCO_B", "Berkeley", "GrabCut", "DAVIS", "PASCAL", "ADE20K", "TETRIS", "ACDC", "BUID", "MedScribble"]
GROUPS = ("all", "bottom50", "top50")
SLOTS = ("undef", "hsel", "grad_mm0", "grad_mm1", "bon_mm0", "bon_mm1")
SINGLE = SLOTS + ("steer_mm0", "steer_mm1")
METHODS = SINGLE + ("argmax_pred", "catboost", "oracle")
LABEL = OrderedDict([("undef", "один выход (undef)"), ("hsel", "head-select"),
                     ("grad_mm0", "grad 30, один выход"), ("grad_mm1", "grad 30, multimask"),
                     ("bon_mm0", "BoN, один выход"), ("bon_mm1", "BoN, multimask"),
                     ("steer_mm0", "steering α*, один выход"), ("steer_mm1", "steering α*, multimask"),
                     ("argmax_pred", "max предсказанного IoU по пулу"), ("catboost", "CatBoost"),
                     ("oracle", "oracle пула")])
TIE, NEAR = 1e-6, 0.01


def _a(alpha: float) -> str:
    return f"a{alpha:g}"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(pool_dir: Path):
    """pool.csv.gz sorted by box (CatBoost needs a group's rows together),
    rows.csv.gz, meta.json."""
    t0 = time.time()
    meta = json.loads((pool_dir / "meta.json").read_text(encoding="utf-8"))
    P = pd.read_csv(pool_dir / "pool.csv.gz", dtype={c: str for c in STR_COLS})
    for c in STR_COLS:
        if c in P.columns:
            P[c] = P[c].fillna("")
    P["case"] = P.image_name + "|" + P.kind + "|" + P.user + "|" + P.attempt
    P["key"] = np.where(P.dataset == "COCO", "COCO_" + P.half, P.dataset)
    P = P.sort_values(["case", "cand_kind", "cand_id"], kind="mergesort").reset_index(drop=True)
    for c in P.columns:
        if c.startswith(("f_", "x_")) and pd.api.types.is_float_dtype(P[c]):
            P[c] = P[c].astype(np.float32)
    R = pd.read_csv(pool_dir / "rows.csv.gz", dtype={c: str for c in KEY + ["half", "table", "dataset", "role"]})
    R[KEY] = R[KEY].fillna("")
    R["case"] = R.image_name + "|" + R.kind + "|" + R.user + "|" + R.attempt
    print(f"loaded {pool_dir}: {len(P)} candidates, {P.case.nunique()} boxes, {P.image_name.nunique()} images "
          f"({time.time() - t0:.0f}s)", flush=True)
    return P, R, meta


def find_split(meta: dict, pool_dir: Path):
    scan = meta.get("args", {}).get("scan_dir")
    if not scan:
        return None
    for base in (Path.cwd(), _HERE.parent, pool_dir.parent.parent):
        p = (base / scan / "split.json") if not Path(scan).is_absolute() else Path(scan) / "split.json"
        if p.is_file():
            return p
    return None


def exclude_A(P: pd.DataFrame, meta: dict, pool_dir: Path, checks: dict) -> pd.DataFrame:
    """Drop the COCO A images (the steering vector's own) and prove none is left."""
    A_pool = sorted(set(P.image_name[P.key == "COCO_A"]))
    sp = find_split(meta, pool_dir)
    A_split = json.loads(sp.read_text(encoding="utf-8"))["A"] if sp else None
    K = P[P.key != "COCO_A"].reset_index(drop=True)
    kept = set(K.image_name)
    c = {"A_images_in_pool": len(A_pool), "split_json": str(sp) if sp else None, "kept_images": len(kept),
         "kept_boxes": int(K.case.nunique())}
    if A_split is not None:
        c["kept_in_split_A"] = sorted(kept & set(A_split))
        c["pool_A_in_split_A"] = set(A_pool) <= set(A_split)
        c["ok"] = not c["kept_in_split_A"] and c["pool_A_in_split_A"]
    else:
        c["note"] = "split.json of the scan not found: COCO A is taken from the pool's half column"
        c["ok"] = True
    checks["exclude_A"] = c
    if not c["ok"]:
        raise SystemExit(f"COCO A exclusion failed: {c}")
    return K


def alpha_star(P: pd.DataFrame, alphas) -> dict:
    """The steering baseline: per mode the alpha with the best mean IoU on the
    COCO A boxes (the vector's images, cross-fitted there), ties -> the smaller."""
    A = P[P.key == "COCO_A"]
    out = {}
    for mm in MODES:
        s = A[A.cand_kind == f"steer_mm{mm}"].groupby("cand_id").target_iou.mean().sort_index()
        if len(s):
            j = int(s.idxmax())
            out[mm] = {"cand_id": j, "alpha": float(alphas[j]), "mean_iou_A": float(s.max()), "boxes_A": int(A.case.nunique())}
    return out


def add_regressor_features(P: pd.DataFrame, R: pd.DataFrame, alphas, checks: dict) -> None:
    """The alpha regressors' features (steer_gate BOX_FEATS / ALPHA_FEATS "out"),
    in place: c_reg_* per box, f_reg_* per candidate. Checks the join on the
    columns both files carry (steering's predicted IoU and IoU, alpha)."""
    R = R.drop_duplicates("case")
    pos = pd.Index(R.case).get_indexer(P.case)
    c = {"boxes_missing_in_rows": int(len(set(P.case[pos < 0])))}
    if (pos < 0).any():
        checks["regressor_join"] = c
        raise SystemExit(f"{c['boxes_missing_in_rows']} boxes of the pool are not in rows.csv.gz")

    def g(col):
        return R[col].to_numpy(np.float64)[pos]

    P["c_reg_box_frac"] = g("box_frac").astype(np.float32)
    P["c_reg_log_aspect"] = g("log_aspect").astype(np.float32)
    for mm in MODES:
        for n in ("pred0", "stab0", "fill0", "mask_frac0"):
            P[f"c_reg_{n}_mm{mm}"] = g(f"{PRE[mm]}_{n}").astype(np.float32)
    kind, cid = P.cand_kind.to_numpy(), P.cand_id.to_numpy()
    F = {n: np.full(len(P), np.nan) for n in ("stab", "fill", "chg", "d_pred", "d_fill", "d_stab")}
    for mm in MODES:
        pre = PRE[mm]
        b = kind == ("undef" if mm == 0 else "hsel")
        F["stab"][b], F["fill"][b] = g(f"{pre}_stab0")[b], g(f"{pre}_fill0")[b]
        for n in ("chg", "d_pred", "d_fill", "d_stab"):
            F[n][b] = 0.0
        s = kind == f"steer_mm{mm}"
        if not s.any():
            continue

        def at(f):
            M = np.stack([R[f"{pre}_{f}_{_a(a)}"].to_numpy(np.float64) for a in alphas], axis=1)
            return M[pos[s], cid[s]]
        F["stab"][s], F["fill"][s], F["chg"][s] = at("stab"), at("fill"), at("chg")
        F["d_pred"][s] = at("pred") - g(f"{pre}_pred0")[s]
        F["d_fill"][s] = F["fill"][s] - g(f"{pre}_fill0")[s]
        F["d_stab"][s] = F["stab"][s] - g(f"{pre}_stab0")[s]
        c[f"mm{mm}"] = {"steer_rows": int(s.sum()),
                        "max_abs_pred_vs_pool": float(np.abs(at("pred") - P.f_pred.to_numpy(np.float64)[s]).max()),
                        "max_abs_iou_vs_pool": float(np.abs(at("iou") - P.target_iou.to_numpy(np.float64)[s]).max()),
                        "max_abs_alpha_vs_pool": float(np.abs(np.asarray(alphas)[cid[s]]
                                                              - P.f_steer_alpha.to_numpy(np.float64)[s]).max())}
    for n, v in F.items():
        P[f"f_reg_{n}"] = v.astype(np.float32)
    c["ok"] = all(v["max_abs_iou_vs_pool"] < 1e-4 and v["max_abs_alpha_vs_pool"] < 1e-6
                  for k, v in c.items() if k.startswith("mm"))
    checks["regressor_join"] = c
    if not c["ok"]:
        raise SystemExit(f"rows.csv.gz does not describe the pool's steering candidates: {c}")


def _prune(X: pd.DataFrame, cols) -> list:
    """Drop all-missing and single-valued columns (selector_data.prune, without nunique)."""
    keep = []
    for c in cols:
        if c in CAT:
            keep.append(c)
            continue
        v = X[c].to_numpy(np.float64)
        f = v[np.isfinite(v)]
        if f.size and f.min() != f.max():
            keep.append(c)
    return keep


def make_features(P: pd.DataFrame, spec: dict | None = None):
    """-> (X, spec). spec None: build the feature list on this pool; else
    reproduce a saved one (missing columns NaN)."""
    t0 = time.time()
    if spec is None:
        base = _prune(P, SD.numeric_feature_cols(P))
        rel_base = [c for c in base if c.startswith(("f_", "x_"))]
    else:
        base, rel_base = spec["base"], spec["rel_base"]
    missing = [c for c in dict.fromkeys(base + rel_base) if c not in P.columns]
    Q = P.assign(**{c: np.float32(np.nan) for c in missing}) if missing else P
    rel = SD.pool_relative(Q, rel_base).astype(np.float32)
    X = pd.concat([Q[base].astype(np.float32).reset_index(drop=True), rel.reset_index(drop=True)], axis=1)
    X["c_kind"] = P.cand_kind.astype(str).to_numpy()
    if spec is None:
        cols = _prune(X, list(X.columns))
        spec = {"base": base, "rel_base": rel_base, "features": cols, "cat": CAT}
    else:
        cols = spec["features"]
        for c in cols:
            if c not in X.columns:
                X[c] = np.float32(np.nan)
                missing.append(c)
    spec = dict(spec, missing_here=sorted(set(missing)))
    print(f"features: {len(spec['features'])} ({len(spec['base'])} base, {len(spec['rel_base'])} pool-relative "
          f"sources, categorical {CAT}); missing here: {len(spec['missing_here'])} ({time.time() - t0:.0f}s)", flush=True)
    return X[spec["features"]], spec


def make_folds(P: pd.DataFrame, k: int, seed: int) -> list:
    """train_selectors.make_folds: test-image sets stratified on the subset."""
    from sklearn.model_selection import StratifiedKFold
    im = P[["image_name", "dataset"]].drop_duplicates().sort_values("image_name").reset_index(drop=True)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [sorted(im.image_name.values[te]) for _, te in skf.split(im, im.dataset)]


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

def cb_params(a) -> dict:
    # The learn-set metric (PFound for YetiRank) has no GPU kernel: CatBoost
    # computes it on the CPU, by default every 5 iterations. It is only logged,
    # so compute it when a log line is due.
    kw = dict(loss_function=a.loss, iterations=a.iterations, depth=a.depth, learning_rate=a.lr,
              random_seed=a.seed, verbose=a.verbose, allow_writing_files=False, task_type=a.task_type,
              metric_period=a.verbose if a.verbose > 0 else a.iterations)
    if a.task_type == "GPU":
        kw["devices"] = a.devices
    else:
        kw["thread_count"] = a.threads
    return kw


def fit(X: pd.DataFrame, y: np.ndarray, grp: np.ndarray, a):
    from catboost import CatBoostRanker, Pool
    return CatBoostRanker(**cb_params(a)).fit(Pool(X, y, group_id=grp, cat_features=CAT))


def predict(m, X: pd.DataFrame, grp: np.ndarray) -> np.ndarray:
    from catboost import Pool
    return m.predict(Pool(X, group_id=grp, cat_features=CAT), thread_count=-1)


def check_gpu(a, checks: dict):
    import catboost
    from catboost.utils import get_gpu_device_count
    n = get_gpu_device_count() if a.task_type == "GPU" else 0
    checks["catboost"] = {"version": catboost.__version__, "task_type": a.task_type, "devices": a.devices,
                          "gpus_visible": n}
    print(f"catboost {catboost.__version__}, {a.task_type}" + (f", {n} GPU(s) visible, devices {a.devices}"
                                                              if a.task_type == "GPU" else f", {a.threads} threads"),
          flush=True)
    if a.task_type == "GPU" and n == 0:
        raise SystemExit("--task_type GPU but CatBoost sees no GPU (CUDA_VISIBLE_DEVICES? a CPU-only catboost?): "
                         "--task_type CPU to run on the CPU")


def cfg_hash(spec: dict, a, folds) -> str:
    h = hashlib.sha256(json.dumps({"f": spec["features"], "p": {k: v for k, v in cb_params(a).items()
                                                                if k not in ("verbose", "metric_period", "devices", "thread_count")},
                                   "folds": folds}, sort_keys=True).encode())
    return h.hexdigest()[:16]


def train(P: pd.DataFrame, X: pd.DataFrame, spec: dict, a, out: Path, checks: dict) -> np.ndarray:
    """Out-of-fold scores, the fold models and the full model."""
    folds = make_folds(P, a.folds, a.seed)
    grp = pd.factorize(P.case)[0]
    y = P.target_iou.to_numpy(np.float64)
    h = cfg_hash(spec, a, folds)
    (out / "folds.json").write_text(json.dumps({"seed": a.seed, "folds": folds}, indent=1), encoding="utf-8")
    keys = P.case + "|" + P.cand_kind + "|" + P.cand_id.astype(str)
    cache = out / "oof_scores.csv.gz"
    if cache.is_file() and not a.refit and (out / "model_full.cbm").is_file():
        c = pd.read_csv(cache)
        if c.get("cfg", pd.Series([""])).iloc[0] == h and len(c) == len(P) and (c.key.to_numpy() == keys.to_numpy()).all():
            print(f"cached OOF scores and models in {out} (cfg {h}); --refit to train again", flush=True)
            checks["training"] = json.loads((out / "meta.json").read_text(encoding="utf-8")).get("training", {})
            return c.score.to_numpy()
    s = np.full(len(P), np.nan)
    fold_of = np.full(len(P), -1)
    times = []
    img = P.image_name.to_numpy()
    for k, te_imgs in enumerate(folds):
        te = np.isin(img, te_imgs)
        t0 = time.time()
        m = fit(X[~te], y[~te], grp[~te], a)
        s[te] = predict(m, X[te], grp[te])
        fold_of[te] = k
        m.save_model(str(out / f"model_fold{k}.cbm"))
        times.append(round(time.time() - t0))
        print(f"fold {k}: train {(~te).sum()} / test {te.sum()} candidates, {len(te_imgs)} test images "
              f"({times[-1]}s)", flush=True)
    t0 = time.time()
    m = fit(X, y, grp, a)
    m.save_model(str(out / "model_full.cbm"))
    times.append(round(time.time() - t0))
    print(f"full model on {P.image_name.nunique()} images ({times[-1]}s)", flush=True)
    try:
        # YetiRank's default type (LossFunctionChange) needs the 800k-row pool again
        imp = pd.Series(m.get_feature_importance(type="PredictionValuesChange"),
                        index=spec["features"]).sort_values(ascending=False)
        imp.rename("importance").to_csv(out / "importance.csv", index_label="feature")
    except Exception as e:                       # importance is a report, not a result
        print(f"[warn] feature importance: {e}", flush=True)
    pd.DataFrame({"key": keys, "score": s, "fold": fold_of, "cfg": h}).to_csv(cache, index=False)
    checks["training"] = {"cfg": h, "fold_images": [len(f) for f in folds], "seconds": times,
                          "all_scored": bool(np.isfinite(s).all())}
    return s


def apply(P: pd.DataFrame, X: pd.DataFrame, src: Path, a, checks: dict) -> np.ndarray:
    """Score this pool with another run's selectors (--apply_with fold | full)."""
    from catboost import CatBoostRanker
    grp = pd.factorize(P.case)[0]
    img = P.image_name.to_numpy()
    s = np.full(len(P), np.nan)
    used = {}
    if a.apply_with == "fold":
        folds = json.loads((src / "folds.json").read_text(encoding="utf-8"))["folds"]
        for k, imgs in enumerate(folds):
            sel = np.isin(img, imgs)
            if sel.any():
                m = CatBoostRanker().load_model(str(src / f"model_fold{k}.cbm"))
                s[sel] = predict(m, X[sel], grp[sel])
                used[f"fold{k}"] = int(len(set(img[sel])))
    rest = ~np.isfinite(s)
    if rest.any():
        m = CatBoostRanker().load_model(str(src / "model_full.cbm"))
        s[rest] = predict(m, X[rest], grp[rest])
        used["full"] = int(len(set(img[rest])))
    checks["apply"] = {"source": str(src), "with": a.apply_with, "images_per_model": used,
                       "images_not_in_source_folds": used.get("full", 0) if a.apply_with == "fold" else None}
    return s


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def per_box(P: pd.DataFrame, score: np.ndarray, astar: dict) -> pd.DataFrame:
    """One row per box: every method's IoU, CatBoost's pick, the oracle."""
    T = P[KEY + ["case", "key", "dataset", "group", "cand_kind", "cand_id", "method_slot", "step", "target_iou",
                 "f_pred", "f_steer_alpha"]].assign(score=score)
    gb = T.groupby("case", sort=False)
    B = gb[KEY + ["key", "dataset", "group"]].first()
    B["oracle"] = gb.target_iou.max()
    B["n_candidates"] = gb.size()
    for slot in SLOTS:
        B[slot] = T[T.method_slot == slot].set_index("case").target_iou.reindex(B.index)
    for mm, st in astar.items():
        x = T[(T.cand_kind == f"steer_mm{mm}") & (T.cand_id == st["cand_id"])].set_index("case").target_iou
        B[f"steer_mm{mm}"] = x.reindex(B.index)
    for mm in MODES:
        if f"steer_mm{mm}" not in B:
            B[f"steer_mm{mm}"] = np.nan
    i = gb.f_pred.idxmax()
    B["argmax_pred"] = T.target_iou.loc[i].to_numpy()
    i = gb.score.idxmax()
    B["catboost"] = T.target_iou.loc[i].to_numpy()
    B["cb_kind"] = T.cand_kind.loc[i].to_numpy()
    B["cb_cand_id"] = T.cand_id.loc[i].to_numpy()
    B["cb_alpha"] = T.f_steer_alpha.loc[i].to_numpy()
    B["cb_step"] = T.step.loc[i].to_numpy()
    B["cb_score"] = T.score.loc[i].to_numpy()
    # how sure the ranking is: the top score minus the runner-up's
    S = T[["case", "score"]].sort_values(["case", "score"], ascending=[True, False], kind="mergesort")
    second = S[S.groupby("case", sort=False).cumcount().to_numpy() == 1].set_index("case").score
    B["cb_margin"] = B.cb_score - second.reindex(B.index)
    i = gb.target_iou.idxmax()
    B["oracle_kind"] = T.cand_kind.loc[i].to_numpy()
    B["oracle_cand_id"] = T.cand_id.loc[i].to_numpy()
    return B.reset_index()


def blocks(keys) -> list:
    ks = [k for k in ORDER if k in keys] + sorted(k for k in keys if k not in ORDER)
    nat = [k for k in ks if k not in MEDICAL]
    med = [k for k in ks if k in MEDICAL]
    out = [(k, [k]) for k in ks]
    for name, b in (("натуральные", nat), ("медицина", med), ("все", ks)):
        if len(b) > 1 and (name != "все" or (nat and med)):
            out.append((name, b))
    return out


def boot_cols(X: np.ndarray, img: np.ndarray, n_boot: int, rng) -> np.ndarray:
    """(n_boot, n_cols) means of the columns, the images resampled with their boxes."""
    codes, inv = np.unique(img, return_inverse=True)
    S = np.zeros((len(codes), X.shape[1]))
    np.add.at(S, inv, X)
    N = np.bincount(inv, minlength=len(codes)).astype(float)
    idx = rng.integers(0, len(codes), size=(n_boot, len(codes)))
    return S[idx].sum(1) / N[idx].sum(1)[:, None]


def table(B: pd.DataFrame, a) -> pd.DataFrame:
    """Per block x group x method: IoU, % to undef, CatBoost's gain over it [CI]."""
    meth = [m for m in METHODS if m in B.columns]
    ci, iu, icb = {m: j for j, m in enumerate(meth)}, meth.index("undef"), meth.index("catboost")
    rows = []
    for gi, g in enumerate(GROUPS):
        stat = {}
        for ki, k in enumerate(ORDER + sorted(set(B.key) - set(ORDER))):
            s = B[(B.key == k) & ((B.group == g) | (g == "all"))]
            if not len(s):
                continue
            X = s[meth].to_numpy(np.float64)
            rng = np.random.default_rng([a.seed, ki, gi])
            stat[k] = (X.mean(0), boot_cols(X, s.image_name.to_numpy(), a.n_boot, rng), len(s), s.image_name.nunique())
        for name, ks in blocks(stat):
            mu = np.mean([stat[k][0] for k in ks], axis=0)
            pct = np.mean([(stat[k][0] / stat[k][0][iu] - 1) * 100 for k in ks], axis=0)
            gain = np.mean([(stat[k][0][icb] / stat[k][0] - 1) * 100 for k in ks], axis=0)
            samp = np.mean([(stat[k][1][:, [icb]] / stat[k][1] - 1) * 100 for k in ks], axis=0)
            lo, hi = np.nanpercentile(samp, [2.5, 97.5], axis=0) if a.n_boot > 0 else (np.full(len(meth), np.nan),) * 2
            for m in meth:
                j = ci[m]
                rows.append({"block": name, "group": g, "n_boxes": sum(stat[k][2] for k in ks),
                             "n_images": sum(stat[k][3] for k in ks), "method": m, "iou": mu[j], "pct_undef": pct[j],
                             "cb_gain_pct": gain[j] if m != "catboost" else np.nan,
                             "cb_gain_lo": lo[j] if m != "catboost" else np.nan,
                             "cb_gain_hi": hi[j] if m != "catboost" else np.nan})
    return pd.DataFrame(rows)


def picks(B: pd.DataFrame) -> pd.DataFrame:
    """Per block: P@1, near-best, regret, and the share of each picked family."""
    rows = []
    kinds = sorted(set(B.cb_kind))
    for name, ks in blocks(set(B.key)):
        s = B[B.key.isin(ks)]
        r = {"block": name, "n_boxes": len(s), "p_at_1": float((s.catboost >= s.oracle - TIE).mean()),
             "near_best": float((s.catboost >= s.oracle - NEAR).mean()), "regret": float((s.oracle - s.catboost).mean()),
             "p_at_1_argmax_pred": float((s.argmax_pred >= s.oracle - TIE).mean())}
        r.update({f"pick_{k}": float((s.cb_kind == k).mean()) for k in kinds})
        rows.append(r)
    return pd.DataFrame(rows)


def summary_md(T: pd.DataFrame, K: pd.DataFrame, meta: dict, astar: dict, checks: dict, a, src_meta=None,
               T_kind: pd.DataFrame | None = None) -> str:
    def fmt(x, nd=2, sign=True):
        return "—" if x is None or not np.isfinite(x) else (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}")

    model = meta.get("model_name", "?")
    head = (f"# CatBoost-селектор: {model}" if src_meta is None else
            f"# Перенос CatBoost-селектора: {src_meta.get('model_name', '?')} → {model}")
    L = [head, "",
         f"Пул кандидатов `{a.pool_dir}` ({meta.get('created', '?')}): undef, head-select, grad 30 шагов "
         f"(точки каждые {meta.get('args', {}).get('checkpoint_every', 5)}) и BoN (сдвиг размера и центра, "
         f"{meta.get('args', {}).get('Y', 16)} боксов) в обоих режимах выхода, steering `{meta.get('site', '?')}` "
         f"при α ∈ {{{', '.join(f'{x:g}' for x in meta.get('alphas', []))}}} в обоих режимах. Картинки COCO A "
         f"(на них считался вектор steering) не входят ни в обучение, ни в оценку: "
         f"{checks['exclude_A']['kept_images']} картинок, {checks['exclude_A']['kept_boxes']} боксов.", ""]
    if src_meta is None:
        L.append(f"CatBoostRanker ({a.loss}), {a.iterations} итераций, depth {a.depth}, lr {a.lr}, {a.task_type}; "
                 f"{a.folds} фолдов по картинкам (стратификация по набору), оценка вне фолда. Признаки: f_* / x_* пула, "
                 "признаки регрессора α (стабильность, заполнение бокса, изменение маски, d_pred / d_fill / d_stab), "
                 "относительные признаки внутри пула бокса, cand_kind (категориальный).")
    else:
        L.append(f"Селекторы обучены на пуле {src_meta.get('model_name', '?')} (`{a.apply_model}`), применены к пулу "
                 f"{model} ({'модель фолда, в котором картинка была тестовой: ни картинка, ни кандидаты этой модели в обучении не участвовали' if a.apply_with == 'fold' else 'одна модель на всех картинках источника'}).")
    st = ", ".join(f"{'один выход' if mm == 0 else 'multimask'} α* = {v['alpha']:g}" for mm, v in astar.items())
    kinds = T_kind["ann"].unique().tolist() if T_kind is not None else []
    L += ["", f"Steering α* выбрана на боксах COCO A по среднему IoU: {st or '—'}. Группы — внутри картинки по IoU "
          "одного выхода: худшая половина разметок картинки — bottom50, остальные — top50 (12 / 13 при 25 разметках"
          + ("; разметки m и p ранжируются вместе" if len(kinds) > 1 else "") + "). % — прирост к undef; «CatBoost к "
          "методу» — прирост CatBoost в % от метода, 95% CI парным бутстрепом по картинкам; для блоков — среднее по "
          "наборам.", ""]
    for g in ("bottom50", "top50", "all"):
        L += [f"## {g}: IoU (% к undef)", "",
              "| блок | " + " | ".join(LABEL[m] for m in METHODS if m in set(T.method)) + " |",
              "|---|" + "---|" * len([m for m in METHODS if m in set(T.method)])]
        for name in dict.fromkeys(T.block):
            s = T[(T.block == name) & (T.group == g)].set_index("method")
            cells = [f"{s.loc[m, 'iou']:.4f}" + ("" if m == "undef" else f" ({fmt(s.loc[m, 'pct_undef'])})")
                     for m in METHODS if m in s.index]
            b = f"**{name}**" if name in ("натуральные", "медицина", "все") else name
            L.append(f"| {b} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Прирост CatBoost к каждому методу, % [95% CI]", "",
          "| блок | группа | " + " | ".join(LABEL[m] for m in SINGLE + ("argmax_pred", "oracle") if m in set(T.method)) + " |",
          "|---|---|" + "---|" * len([m for m in SINGLE + ("argmax_pred", "oracle") if m in set(T.method)])]
    for name in [n for n in dict.fromkeys(T.block) if n in ("натуральные", "медицина", "все")] or list(dict.fromkeys(T.block)):
        for g in GROUPS:
            s = T[(T.block == name) & (T.group == g)].set_index("method")
            cells = []
            for m in SINGLE + ("argmax_pred", "oracle"):
                if m not in s.index:
                    continue
                r = s.loc[m]
                star = "*" if np.isfinite(r.cb_gain_lo) and (r.cb_gain_lo > 0 or r.cb_gain_hi < 0) else ""
                cells.append(f"{fmt(r.cb_gain_pct)}{star} [{fmt(r.cb_gain_lo)}, {fmt(r.cb_gain_hi)}]")
            L.append(f"| {name} | {g} | " + " | ".join(cells) + " |")
    L += ["", "\\* — CI не захватывает 0.", ""]
    if len(kinds) > 1:
        cols = [m for m in METHODS if m in set(T_kind.method) and m != "argmax_pred"]
        L += ["## По типу разметки (m — мобильная, p — с ПК): IoU undef и % к нему", "",
              "Одна и та же модель CatBoost (обучена на всех разметках), строки — подвыборки боксов.", "",
              "| разметка | блок | группа | n | " + " | ".join(LABEL[m] for m in cols) + " |",
              "|---|---|---|---|" + "---|" * len(cols)]
        for ann in kinds:
            for name in [n for n in ("натуральные", "медицина", "все") if n in set(T_kind.block)]:
                for g in GROUPS:
                    s = T_kind[(T_kind.ann == ann) & (T_kind.block == name) & (T_kind.group == g)].set_index("method")
                    if s.empty:
                        continue
                    cells = [f"{s.loc[m, 'iou']:.4f}" if m == "undef" else fmt(s.loc[m, "pct_undef"]) for m in cols]
                    L.append(f"| {ann} | {name} | {g} | {int(s.n_boxes.iloc[0])} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Что выбирает CatBoost", "",
          "| блок | P@1 | в пределах 0.01 от лучшего | regret | P@1 max pred IoU | " +
          " | ".join(c[5:] for c in K.columns if c.startswith("pick_")) + " |",
          "|---|---|---|---|---|" + "---|" * len([c for c in K.columns if c.startswith("pick_")])]
    for r in K.itertuples(index=False):
        d = r._asdict()
        L.append(f"| {r.block} | {r.p_at_1:.1%} | {r.near_best:.1%} | {r.regret:.4f} | {r.p_at_1_argmax_pred:.1%} | "
                 + " | ".join(f"{d[c]:.1%}" for c in K.columns if c.startswith("pick_")) + " |")
    rj = checks.get("regressor_join", {})
    L += ["", "## Проверки", "",
          f"- COCO A исключены: {checks['exclude_A'].get('A_images_in_pool')} картинок; split.json: "
          f"{checks['exclude_A'].get('split_json')}; в выборке из A: {len(checks['exclude_A'].get('kept_in_split_A', []))};",
          "- признаки регрессора α совпадают с пулом: " + ", ".join(
              f"{k}: IoU max |Δ| {v['max_abs_iou_vs_pool']:.1e}, pred {v['max_abs_pred_vs_pool']:.1e}"
              for k, v in rj.items() if k.startswith("mm")) + ";"]
    if "training" in checks:
        L.append(f"- обучение: картинок в тестовых фолдах {checks['training'].get('fold_images')}, секунды "
                 f"{checks['training'].get('seconds')}, все кандидаты оценены вне фолда: {checks['training'].get('all_scored')};")
    if "apply" in checks:
        L.append(f"- перенос: модели {checks['apply']['images_per_model']}; признаков источника нет в этом пуле: "
                 f"{len(checks.get('features', {}).get('missing_here', []))};")
    imp = Path(a.out_dir) / "importance.csv"
    if src_meta is None and imp.is_file():
        I = pd.read_csv(imp).head(20)
        L += ["", "## Важность признаков (модель на всех картинках, топ-20)", "", "| признак | важность |", "|---|---|"]
        L += [f"| `{r.feature}` | {r.importance:.2f} |" for r in I.itertuples(index=False)]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(a) -> int:
    t0 = time.time()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checks: dict = {}
    check_gpu(a, checks)
    pool_dir = Path(a.pool_dir)
    P, R, meta = load(pool_dir)
    alphas = meta["alphas"]
    astar = alpha_star(P, alphas)
    add_regressor_features(P, R, alphas, checks)
    del R
    P = exclude_A(P, meta, pool_dir, checks)
    src_meta = None
    if a.apply_model:
        src = Path(a.apply_model)
        spec = json.loads((src / "features.json").read_text(encoding="utf-8"))
        src_meta = json.loads((src / "meta.json").read_text(encoding="utf-8")).get("pool_meta", {})
        X, spec = make_features(P, spec)
        checks["features"] = {"n": len(spec["features"]), "missing_here": spec["missing_here"]}
        s = apply(P, X, src, a, checks)
    else:
        X, spec = make_features(P)
        (out / "features.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
        checks["features"] = {"n": len(spec["features"])}
        s = train(P, X, spec, a, out, checks)
    del X
    B = per_box(P, s, astar)
    B.to_csv(out / "per_box.csv.gz", index=False, float_format="%.6f")
    kinds = sorted(set(B.kind))
    T = table(B, a).assign(ann="+".join(kinds))
    T_kind = pd.concat([T] + ([table(B[B.kind == k].reset_index(drop=True), a).assign(ann=k) for k in kinds]
                              if len(kinds) > 1 else []), ignore_index=True)
    T_kind.to_csv(out / "table.csv", index=False)
    K = picks(B)
    K.to_csv(out / "picks.csv", index=False)
    meta_out = {"script": "train_pool_catboost.py", "created": time.strftime("%Y-%m-%d %H:%M"), "args": vars(a),
                "pool_meta": {k: meta.get(k) for k in ("model_name", "checkpoint", "site", "alphas", "created",
                                                        "layout")},
                "alpha_star": astar, "boxes": int(len(B)), "candidates": int(len(P)),
                "candidates_per_box": float(B.n_candidates.mean()), "training": checks.get("training"),
                "seconds": round(time.time() - t0)}
    (out / "meta.json").write_text(json.dumps(meta_out, indent=1, default=_json_default), encoding="utf-8")
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=_json_default), encoding="utf-8")
    text = summary_md(T, K, meta, astar, checks, a, src_meta, T_kind)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)
    en = {"натуральные": "natural", "медицина": "medical", "все": "all subsets"}
    for name in [n for n in en if n in set(T.block)] or list(dict.fromkeys(T.block)):
        r = T[(T.block == name) & (T.group == "bottom50")].set_index("method")
        print(f"  {en.get(name, name)}, bottom50, % to undef: " + ", ".join(f"{m} {r.loc[m, 'pct_undef']:+.2f}"
                                                            for m in METHODS if m in r.index and m != "undef"), flush=True)
    if len(kinds) > 1:
        for k in kinds:
            r = T_kind[(T_kind.ann == k) & (T_kind.block == "все") & (T_kind.group == "all")].set_index("method")
            if len(r):
                print(f"  kind {k}, all subsets, all boxes ({int(r.n_boxes.iloc[0])}), % to undef: catboost "
                      f"{r.loc['catboost', 'pct_undef']:+.2f}, oracle {r.loc['oracle', 'pct_undef']:+.2f}", flush=True)
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool_dir", required=True, help="out_dir of build_candidate_pool.py (merged)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--apply_model", default="", help="out_dir of another run: score this pool with its selectors")
    p.add_argument("--apply_with", default="fold", choices=["fold", "full"])
    p.add_argument("--task_type", default="GPU", choices=["GPU", "CPU"])
    p.add_argument("--devices", default="0", help="CatBoost GPU devices, e.g. 0 or 0:1")
    p.add_argument("--threads", type=int, default=-1, help="CPU threads (--task_type CPU)")
    p.add_argument("--loss", default="YetiRank")
    p.add_argument("--iterations", type=int, default=800)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.06)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--verbose", type=int, default=100, help="CatBoost log every N iterations (0 = silent)")
    p.add_argument("--refit", action="store_true", help="train again even if this configuration's models exist")
    return p.parse_args(argv)


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
