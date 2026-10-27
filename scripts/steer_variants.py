"""
steer_variants.py
=================
How should the steering vector be estimated, and what does a better estimate
buy? Three questions on the user study (--use m), one GPU pass: every image is
encoded once and every vector variant x alpha runs through the decoder.

The vector is always a difference in means at L2.norm2 over all 7 tokens,
taken inside each training image and averaged over the images (steer_user_study);
what changes is which boxes of an image are "good" and "bad", and which images
it is estimated on.

EXPERIMENTS
  1  annotations per image  the k worst and the k best boxes of every COCO A
                            image by undefended IoU (--extreme_k; 3 ~ a tenth
                            of 25), against the full halves (12 worst / 13 best)
  2  good positives only    positives = top50 boxes with IoU > --pos_thr; an
                            image with none is left out of the mean (it stays a
                            test image like any other); negatives = bottom50, or
                            in a second variant bottom50 boxes with IoU <
                            --pos_thr as well
  3  a vector per dataset   --own_train images of every dataset with all their
                            annotations (COCO: drawn from half A) give the
                            dataset its own vector, scored on the rest of the
                            dataset (COCO: half B) against the COCO A vector,
                            and against COCO vectors of as many images
                            (--coco_sub_draws draws from half A, scored on every
                            test table; draw 0 is the own COCO draw)
  5  one vector for all     the own draws of every dataset pooled into one vector
                            (each image weighted equally, so each dataset too),
                            scored on every dataset's own test images; and per
                            dataset the pool of the OTHER datasets' draws
                            (leave one dataset out: an unseen domain) (--pool)
  4  tight GT box positive  the state of SAM's forward on the tight bbox of the
                            GT mask (the ideal box, best_box of build_user_cases)
                            as a positive: tight / bottom50, top50 + tight /
                            bottom50, tight / all 25 user boxes (--tight)
  For every vector: fixed alphas (--fixed_alphas), alpha* (one alpha for all
  boxes: the best mean gain over the vector's training boxes, none included),
  and steer_gate's gain regressors (linear, GBM; features without the GT)
  fitted on the training boxes.

CROSS-FITTING. The training images of a vector are steered with the vector of
the other folds of its training images (--folds over images), so alpha* and
the regressors learn from out-of-sample effects; test images get the vector of
all training images.

TESTS. Experiments 1 and 2: COCO half B and every --datasets subset, in full.
Experiment 3: per dataset, the images outside its own training draw. Both
report the mean over the test subsets (each weighted equally), natural and
medical apart, with image-bootstrap CIs (steer_gate.score_blocks).

CHECKS (checks.json; a failure stops the run): zero dose bit for bit on every
image; the full-halves COCO A vector equals the one of --vector (same states,
same rule); undefended IoU vs --reference_csv; 12 / 13 groups.

OUTPUTS (--out_dir): rows.csv.gz (box x variant: unsteered features, and per
alpha IoU / predicted IoU / mask change / fill / stability; --from_dump redoes
the analysis from it), vectors.npz, vector_stats.csv, table_exp12.{csv,md},
table_exp3.{csv,md}, summary.md, checks.json, meta.json.

    CUDA_VISIBLE_DEVICES=0 python scripts/steer_variants.py \\
        --root /.../FOR_TEST --checkpoint_path /.../sam_vit_b_01ec64.pth \\
        --vector exp_res/steer_user_study/vector.npz \\
        --reference_csv results/samfull_m.csv --out_dir exp_res/steer_variants
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

import steer_gate as G  # noqa: E402
import steer_user_study as SUS  # noqa: E402
from steer_sam_decoder import (  # noqa: E402
    Cfg, Runner, assign_folds, batch_iou, build_sites, feature_of, make_edit, mask_change, zero_edit,
)

KEY, GROUPS, _a = SUS.KEY, SUS.GROUPS, SUS._a


# ---------------------------------------------------------------------------
# Vector variants
# ---------------------------------------------------------------------------

class Variant:
    """A rule for the good / bad boxes of an image, and the images it is
    estimated on (with their cross-fitting folds)."""

    def __init__(self, name: str, label: str, rule: str, train, folds: dict, k: int = 0, thr: float = math.nan):
        self.name, self.label, self.rule, self.k, self.thr = name, label, rule, k, thr
        self.train, self.folds = list(train), folds

    def split(self, iou: np.ndarray, grp: np.ndarray, case: np.ndarray):
        """-> (pos, neg) boolean masks over the user boxes of one image, and
        whether the tight GT box is a positive too."""
        top, bot = grp == "top50", grp == "bottom50"
        none = np.zeros(len(iou), bool)
        if self.rule == "tight":
            return none, bot, True
        if self.rule == "tight_top":
            return top, bot, True
        if self.rule == "tight_all":
            return none, ~none, True
        pos, neg = self._split_user(iou, top, bot, case)
        return pos, neg, False

    def _split_user(self, iou, top, bot, case):
        if self.rule == "halves":
            return top, bot
        if self.rule == "extreme":
            order = np.lexsort((case, iou))           # by IoU, ties by the case key (as image_groups)
            pos, neg = np.zeros(len(iou), bool), np.zeros(len(iou), bool)
            neg[order[:self.k]] = True
            pos[order[-self.k:]] = True
            return pos, neg
        if self.rule == "pos":
            return top & (iou > self.thr), bot
        if self.rule == "posneg":
            return top & (iou > self.thr), bot & (iou < self.thr)
        raise ValueError(self.rule)

    def fit(self, states: dict):
        """states: stem -> (H, iou, grp, case, h_tight). Sets d_img per used
        image, the vector and its fold vectors."""
        self.d, self.n_pos, self.n_neg, self.skipped = OrderedDict(), 0, 0, []
        for stem in self.train:
            H, iou, grp, case, h_tight = states[stem]
            pos, neg, tight = self.split(iou, grp, case)
            P = np.concatenate([H[pos], h_tight[None]]) if tight else H[pos]
            if len(P) == 0 or not neg.any():
                self.skipped.append(stem)
                continue
            self.d[stem] = P.mean(0) - H[neg].mean(0)
            self.n_pos += len(P)
            self.n_neg += int(neg.sum())
        if not self.d:                                # nothing to estimate from: the variant is dropped
            self.v = None
            return self
        self.v = np.mean(list(self.d.values()), axis=0).astype(np.float32)
        self.v_fold = {}
        for k in sorted(set(self.folds.values())):
            rest = [d for s, d in self.d.items() if self.folds[s] != k]
            self.v_fold[k] = (np.mean(rest, axis=0) if rest else self.v).astype(np.float32)
        return self

    def vector_for(self, stem: str):
        """(vector, role) for an image: the fold vector on a training image."""
        if stem in self.folds:
            return self.v_fold[self.folds[stem]], "train"
        return self.v, "eval"


def draw_images(pool, n: int, key) -> list:
    rng = np.random.default_rng(key)
    return sorted(pool[i] for i in rng.choice(len(pool), size=n, replace=False))


def build_variants(args, A, index) -> tuple:
    """-> (variants on COCO A, {dataset: own variant}, own test stems per dataset)."""
    fA = assign_folds(A, min(args.folds, len(A)), args.seed)
    va = [Variant("all", "все разметки (13 лучших / 12 худших)", "halves", A, fA)]
    for k in args.extreme_k:
        va.append(Variant(f"ext{k}", f"{k} лучших / {k} худших", "extreme", A, fA, k=k))
    t = args.pos_thr
    va.append(Variant("pos", f"лучшие с IoU > {t:g} / 12 худших", "pos", A, fA, thr=t))
    va.append(Variant("posneg", f"лучшие с IoU > {t:g} / худшие с IoU < {t:g}", "posneg", A, fA, thr=t))
    if args.tight:
        va.append(Variant("tight", "tight bbox GT / 12 худших", "tight", A, fA))
        va.append(Variant("tight_top", "13 лучших + tight bbox GT / 12 худших", "tight_top", A, fA))
        va.append(Variant("tight_all", "tight bbox GT / все 25 разметок", "tight_all", A, fA))
    # COCO-n: n images of half A with all their annotations, the COCO counterpart of a dataset's
    # own vector; draw 0 is the own COCO draw itself, the others show what the draw does
    n = args.own_train
    for j in range(args.coco_sub_draws):
        key = [args.seed, 17, list(SUS.DATASETS).index("COCO")] if j == 0 else [args.seed, 23, j]
        tr = draw_images(sorted(A), n, key)
        va.append(Variant(f"coco{n}_d{j}", f"COCO A, {n} изобр. (выборка {j})", "halves", tr,
                          assign_folds(tr, min(args.folds, len(tr)), args.seed)))
    own, own_test = OrderedDict(), OrderedDict()
    for ds in ["COCO", *args.datasets]:
        pool = sorted(A) if ds == "COCO" else sorted(index[ds])
        if len(pool) <= args.own_train:
            print(f"[warn] {ds}: {len(pool)} images, need > {args.own_train} for its own vector -- skipped",
                  flush=True)
            continue
        train = draw_images(pool, args.own_train, [args.seed, 17, list(SUS.DATASETS).index(ds)])
        f = assign_folds(train, min(args.folds, len(train)), args.seed)
        own[ds] = Variant(f"own_{ds}", f"свой вектор ({args.own_train} изобр.)", "halves", train, f)
        table = "COCO_B" if ds == "COCO" else ds
        own_test[table] = (ds, train)
    pooled = []
    if args.pool and len(own) >= 2:
        table_of = {ds: ("COCO_B" if ds == "COCO" else ds) for ds in own}
        tr = [s for v in own.values() for s in v.train]
        pv = Variant("pool", f"общий вектор ({len(own)} наборов × {args.own_train} изобр.)", "halves", tr,
                     assign_folds(tr, min(args.folds, len(tr)), args.seed))
        pv.scope = set(table_of.values())
        pooled.append(pv)
        for ds in own:
            tr = [s for d, v in own.items() if d != ds for s in v.train]
            pv = Variant(f"lodo_{ds}", f"общий вектор без {ds}", "halves", tr,
                         assign_folds(tr, min(args.folds, len(tr)), args.seed))
            pv.scope = {table_of[ds]}
            pooled.append(pv)
    return va, own, own_test, pooled


# ---------------------------------------------------------------------------
# GPU pass
# ---------------------------------------------------------------------------

def base_forward(runner: Runner, site, cases, boxes, gt, checks: dict, where: str, record: bool):
    """The unsteered forward: steer_gate's box features (and the states)."""
    p, dev = runner.p, runner.device
    b = torch.as_tensor(boxes, device=dev)
    buf = []
    rec = {site.id: lambda t, ctx: buf.append(feature_of(site, t, ctx).cpu())} if record else None
    with G.PredHook(runner.sam) as ph:
        m0, low0 = runner.forward(b, record=rec)
    SUS.check_zero(runner, b, m0, low0, zero_edit, site.id, checks, where)
    in_h, in_w = p.input_size
    bw, bh = (boxes[:, 2] - boxes[:, 0]).clip(min=1e-3), (boxes[:, 3] - boxes[:, 1]).clip(min=1e-3)
    box_frac = bw * bh / float(in_h * in_w)
    frac0 = m0.flatten(1).float().mean(1).cpu().numpy()
    base = pd.DataFrame({"kind": [c["kind"] for c in cases], "user": [c["user"] for c in cases],
                         "attempt": [c["attempt"] for c in cases],
                         "undef_iou": batch_iou(m0, gt).cpu().numpy(), "pred0": ph.result().cpu().numpy(),
                         "stab0": G.stability(low0).cpu().numpy(), "mask_frac0": frac0, "box_frac": box_frac,
                         "fill0": frac0 / box_frac, "log_aspect": np.log(bw / bh)})
    H = torch.cat(buf).numpy() if record else None
    return b, m0, base, H


def steer_columns(runner: Runner, site, b, m0, gt, box_frac, v, alphas, tokens) -> dict:
    Vt = {site.id: {"dm": torch.as_tensor(v, device=runner.device)}}
    cols = {}
    for a in alphas:
        with G.PredHook(runner.sam) as ph:
            m, low = runner.forward(b, edits={site.id: make_edit(Cfg(-1, "steer", site.id, tokens, "dm", a), site, Vt)})
        k = _a(a)
        cols[f"iou_{k}"] = batch_iou(m, gt).cpu().numpy()
        cols[f"pred_{k}"] = ph.result().cpu().numpy()
        cols[f"chg_{k}"] = mask_change(m, m0).cpu().numpy()
        cols[f"fill_{k}"] = m.flatten(1).float().mean(1).cpu().numpy() / box_frac
        cols[f"stab_{k}"] = G.stability(low).cpu().numpy()
    return cols


def gpu_pass(args, checks: dict):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    root = Path(args.root)
    datasets = list(dict.fromkeys(["COCO", *args.datasets]))
    index = SUS.index_study(root, args.use, datasets)
    A, B = SUS.split_halves(index["COCO"], args.split_seed)
    va, own, own_test, pooled = build_variants(args, A, index)
    variants = va + list(own.values()) + pooled

    device = torch.device(args.device)
    predictor = SUS.load_predictor(args.checkpoint_path, args.model_type, device)
    sites = build_sites(len(predictor.model.mask_decoder.transformer.layers))
    site = next(s for s in sites if s.id == args.site)
    runner = Runner(predictor, sites, args.batch, field_grid=64, field_margin=0.25)
    ds_of = {s: d for d in datasets for s in index[d]}

    # pass 1: the states of every training image
    train_imgs = list(dict.fromkeys(s for v in variants for s in v.train))
    print(f"\npass 1: states of {len(train_imgs)} training images", flush=True)
    states, t0 = {}, time.time()
    for i, stem in enumerate(train_imgs):
        cases, boxes, gt = SUS.load_image_cases(predictor, root, stem, index[ds_of[stem]][stem], device)
        _, _, base, H = base_forward(runner, site, cases, boxes, gt, checks, f"train/{stem}", record=True)
        base["image_name"] = stem
        grp = SUS.assign_groups(base).to_numpy()
        case = base[KEY].astype(str).agg("|".join, axis=1).to_numpy()
        # the tight GT box: the same best_box for every case of the image (build_user_cases)
        tb = torch.as_tensor(np.asarray(cases[0]["best_box"], np.float32)[None], device=device)
        buf = []
        runner.forward(tb, record={site.id: lambda t, ctx: buf.append(feature_of(site, t, ctx).cpu())})
        states[stem] = (H, base.undef_iou.to_numpy(), grp, case, torch.cat(buf).numpy()[0])
        if (i + 1) % 10 == 0 or i + 1 == len(train_imgs):
            print(f"  [{i + 1}/{len(train_imgs)}] ({time.time() - t0:.0f}s)", flush=True)
    for v in variants:
        v.fit(states)
    dropped = [v.name for v in variants if v.v is None]
    checks["variants_without_data"] = dropped
    if dropped:
        print(f"[warn] no training image with both positives and negatives: {dropped} -- dropped", flush=True)
    if va[0].v is None:
        raise SUS.CheckFailed("the full-halves COCO A vector has no data")
    va = [v for v in va if v.v is not None]
    own = OrderedDict((d, v) for d, v in own.items() if v.v is not None)
    own_test = OrderedDict((t, x) for t, x in own_test.items() if x[0] in own)
    pooled = [v for v in pooled if v.v is not None]
    variants = va + list(own.values()) + pooled
    own_train_all = {s for v in own.values() for s in v.train}
    if args.vector:
        ref_v = np.load(args.vector)["v"]
        all_v = va[0].v
        checks["vector_all_matches"] = {"path": args.vector, "max_abs": float(np.abs(all_v - ref_v).max()),
                                        "ok": bool(np.allclose(all_v, ref_v, atol=1e-6))}
        if not checks["vector_all_matches"]["ok"]:
            raise SUS.CheckFailed(f"the full-halves COCO A vector differs from {args.vector}")

    # pass 2: every test and training image, every variant that concerns it
    lim = (lambda s: s[:args.limit_images]) if args.limit_images > 0 else (lambda s: s)
    tables = [("COCO_A", "COCO", A), ("COCO_B", "COCO", lim(B))]
    tables += [(d, d, lim(list(index[d]))) for d in datasets if d != "COCO"]
    parts, t0 = [], time.time()
    for table, ds, stems in tables:
        print(f"\npass 2: [{table}] {len(stems)} images", flush=True)
        for i, stem in enumerate(stems):
            use = [(v, *v.vector_for(stem)) for v in va]
            ov = own.get(ds)
            # a dataset's own vector: on its training draw (cross-fitted) and its test images;
            # the COCO A images outside the draw are neither
            if ov is not None and (stem in ov.folds or table != "COCO_A"):
                use.append((ov, *ov.vector_for(stem)))
            # a pooled vector: on its training images (cross-fitted) and on the test images of its datasets
            for pv in pooled:
                if stem in pv.folds or (table in pv.scope and stem not in own_train_all):
                    use.append((pv, *pv.vector_for(stem)))
            cases, boxes, gt = SUS.load_image_cases(predictor, root, stem, index[ds][stem], device)
            b, m0, base, _ = base_forward(runner, site, cases, boxes, gt, checks, f"{table}/{stem}", record=False)
            base.insert(0, "image_name", stem)
            base.insert(0, "dataset", ds)
            base.insert(0, "table", table)
            for v, vec, role in use:
                cols = steer_columns(runner, site, b, m0, gt, base.box_frac.to_numpy(), vec, args.alphas, args.tokens)
                parts.append(base.assign(variant=v.name, role=role, **cols))
            print(f"[{table}] [{i + 1}/{len(stems)}] {stem}: {len(cases)} boxes x {len(use)} vectors "
                  f"({time.time() - t0:.0f}s)", flush=True)
    D = pd.concat(parts, ignore_index=True)
    base = D[D.variant == "all"][KEY + ["undef_iou"]].reset_index(drop=True)
    D = D.merge(base.assign(group=SUS.assign_groups(base))[KEY + ["group"]], on=KEY, how="left")
    return D, variants, own_test


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def tune_alpha(D: pd.DataFrame, alphas) -> tuple:
    """alpha* = the alpha (none included) with the best mean gain over all
    boxes of D; ties -> the smaller one. -> (column of iou_matrix, alpha)."""
    W = G.iou_matrix(D, alphas)
    gain = (W - W[:, :1]).mean(0)
    j = int(np.argmax(np.round(gain, 12)))
    return j, ([0.0] + list(alphas))[j]


def variant_methods(Dtr: pd.DataFrame, alphas, fixed, seed: int):
    """Fit alpha* and the gain regressors on a vector's training boxes; return
    a function boxes -> {method: column of iou_matrix} and alpha*."""
    j_star, a_star = tune_alpha(Dtr, alphas)
    reg = G.GainRegressors(alphas, "out", seed).fit(Dtr)

    def choose(Dt):
        n = len(Dt)
        cols = OrderedDict((f"α = {a:g}", np.full(n, list(alphas).index(a) + 1)) for a in fixed)
        cols["α* (подобрана на обучении)"] = np.full(n, j_star)
        r = reg.choose(Dt)
        cols["регрессор α, линейная"] = r["lin"]
        cols["регрессор α, GBM"] = r["gbm"]
        return cols
    return choose, a_star


def analyse(D: pd.DataFrame, ref, args, own_test: dict) -> tuple:
    alphas, fixed = list(args.alphas), [a for a in args.fixed_alphas if a in args.alphas]
    va_names = [v for v in dict.fromkeys(D.variant) if not v.startswith(("own_", "pool", "lodo_"))]
    label = dict(D[["variant", "label"]].drop_duplicates().to_numpy()) if "label" in D else {}
    fitted, a_star = {}, {}
    for v in dict.fromkeys(D.variant):
        tr = D[(D.variant == v) & (D.role == "train")].reset_index(drop=True)
        if len(tr) == 0:
            continue
        fitted[v], a_star[v] = variant_methods(tr, alphas, fixed, args.seed)

    def rows_of(Dt_v, v, prefix, rows):
        W = G.iou_matrix(Dt_v, alphas)
        for name, c in fitted[v](Dt_v).items():
            rows[f"{prefix}: {name}"] = (W[np.arange(len(Dt_v)), c], W[:, 0])

    # experiments 1 + 2: the COCO A variants on every test table
    per12 = OrderedDict()
    for t in [t for t in dict.fromkeys(D.table) if t != "COCO_A"]:
        base = D[(D.table == t) & (D.variant == "all")].reset_index(drop=True)
        rows = G.base_rows(base, ref)
        for v in va_names:
            Dt = D[(D.table == t) & (D.variant == v)].reset_index(drop=True)
            assert (Dt[KEY].to_numpy() == base[KEY].to_numpy()).all()
            rows_of(Dt, v, label.get(v, v), rows)
        per12[t] = (base, rows)
    T12 = G.score_blocks(per12, args)

    # experiment 3: per dataset, its own vector against the COCO A one, on its own test images
    per3 = OrderedDict()
    for t, (ds, train) in own_test.items():
        v = f"own_{ds}"
        if v not in fitted:
            continue
        own = D[(D.table == t) & (D.variant == v) & (D.role == "eval")].reset_index(drop=True)
        if own.empty:                                 # only with --limit_images
            continue
        glob = D[(D.table == t) & (D.variant == "all")].set_index(KEY).loc[
            pd.MultiIndex.from_frame(own[KEY])].reset_index()
        rows = G.base_rows(own, ref)
        rows_of(glob, "all", "вектор COCO A", rows)
        rows_of(own, v, f"свой вектор ({args.own_train} изобр.)", rows)
        per3[t] = (own, rows)
    T3 = G.score_blocks(per3, args) if per3 else pd.DataFrame()
    return T12, T3, a_star


def vector_stats(variants, a_star: dict) -> pd.DataFrame:
    ref = variants[0].v
    rows = []
    for v in variants:
        rows.append({"variant": v.name, "label": v.label, "train_images": len(v.train), "used_images": len(v.d),
                     "skipped": ";".join(v.skipped), "pos_boxes": v.n_pos, "neg_boxes": v.n_neg,
                     "norm": float(np.linalg.norm(v.v)), "norm_rel_all": float(np.linalg.norm(v.v) / np.linalg.norm(ref)),
                     "cos_all": float(np.dot(v.v.ravel(), ref.ravel()) / np.linalg.norm(v.v) / np.linalg.norm(ref)),
                     "alpha_star": a_star.get(v.name, math.nan)})
    return pd.DataFrame(rows)


def summary_md(T12, T3, St, D, args, checks) -> str:
    n12 = {t: D[(D.table == t) & (D.variant == "all")].image_name.nunique() for t in set(D.table)}
    n3 = {t: D[(D.table == t) & (D.role == "eval") & D.variant.str.startswith("own_")].image_name.nunique()
          for t in set(D.table)}
    L = ["# Как считать steering-вектор: разметки на картинку, только хорошие позитивы, свой вектор на набор", "",
         "Узел `L2.norm2` (выход LayerNorm `norm2` второго two-way блока декодера SAM-1 vit_b, после "
         "cross-attention «токены → картинка»), все 7 токенов (IoU, 4 mask, 2 prompt), h' = h + α·v. Вектор — "
         "разность средних «хороших» и «плохих» боксов внутри картинки, усреднённая по обучающим картинкам. "
         "К обучающим картинкам вектора прибавляется вектор других фолдов (кросс-фит); на них подбираются α* "
         "(одна α на все боксы, включая «без steering») и регрессоры выигрыша (линейный, GBM; признаки без GT), "
         "дальше всё заморожено. "
         "Ячейка — IoU (прирост к undef, %); Δ — к undef с 95% CI (бутстреп по картинкам; для средних — внутри "
         "наборов, наборы с равным весом). Группы — по undef IoU внутри картинки (12 / 13).", "",
         "## Векторы", "",
         "| вариант | картинок (использовано / всего) | боксов + / − | ‖v‖ / ‖v_все‖ | cos с v_все | α* |",
         "|---|---|---|---|---|---|"]
    for r in St.to_dict("records"):
        L.append(f"| {r['label'] if not r['variant'].startswith('own_') else r['variant']} | "
                 f"{r['used_images']} / {r['train_images']} | {r['pos_boxes']} / {r['neg_boxes']} | "
                 f"{r['norm_rel_all']:.2f} | {r['cos_all']:+.3f} | {r['alpha_star']:g} |")
    L += ["", "## Эксперименты 1 и 2: векторы на COCO A, тест — COCO B и остальные наборы", "",
          G.md_blocks(T12, n12)]
    if len(T3):
        L += ["## Эксперимент 3: свой вектор на каждый набор против вектора COCO A, на одних и тех же картинках", "",
              f"Свой вектор — {args.own_train} случайных картинок набора со всеми разметками (COCO — из половины A), "
              "тест — остальные картинки набора (COCO — половина B).", "", G.md_blocks(T3, n3)]
    z, rp = checks.get("zero_dose", {}), checks.get("repro", {})
    L += ["## Проверки", "",
          f"- нулевая доза: {z.get('bit_exact', '-')}/{z.get('forwards', '-')} бит в бит;",
          f"- вектор «все разметки» = `--vector`: {checks.get('vector_all_matches', {}).get('max_abs', '-')};",
          f"- undef vs reference csv: max |Δ| {rp.get('max_abs_diff', math.nan):.2e}, "
          f"доля > {args.repro_tol:g}: {rp.get('frac_over', math.nan):.2%};",
          f"- группы 12/13: отклонений {checks.get('groups', {}).get('n_images_not_12_13', '-')}."]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args) -> int:
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checks = {"zero_dose": {"forwards": 0, "bit_exact": 0, "failures": [], "ok": True}}

    def write_checks():
        (out / "checks.json").write_text(json.dumps(checks, indent=2, default=SUS._json_default), encoding="utf-8")

    ref = SUS.read_reference(args.reference_csv, args.use)
    if args.from_dump:
        D = pd.read_csv(args.from_dump, dtype={k: str for k in KEY})
        St = pd.read_csv(Path(args.from_dump).parent / "vector_stats.csv")
        own_test = OrderedDict(json.loads((Path(args.from_dump).parent / "own_test.json").read_text()))
        prev = Path(args.from_dump).parent / "checks.json"
        if prev.is_file():
            checks = json.loads(prev.read_text(encoding="utf-8"))
    else:
        try:
            D, variants, own_test = gpu_pass(args, checks)
        except SUS.CheckFailed as e:
            write_checks()
            print(f"\nCHECK FAILED: {e} -- see {out / 'checks.json'}", file=sys.stderr)
            return 1
        D["label"] = D.variant.map({v.name: v.label for v in variants})
        D.to_csv(out / "rows.csv.gz", index=False)
        np.savez_compressed(out / "vectors.npz", **{f"{v.name}/v": v.v for v in variants},
                            **{f"{v.name}/fold{k}": f for v in variants for k, f in v.v_fold.items()})
        (out / "own_test.json").write_text(json.dumps(own_test, indent=1), encoding="utf-8")
        (out / "meta.json").write_text(json.dumps({"args": vars(args), "rows": int(len(D))}, indent=2,
                                                  default=SUS._json_default), encoding="utf-8")
        St = vector_stats(variants, {})
        St.to_csv(out / "vector_stats.csv", index=False)
        base = D[D.variant == "all"].drop_duplicates(KEY)
        checks["repro"] = SUS.check_repro(base, ref, args)
        checks["groups"] = SUS.check_groups(base, ref, args, args.limit_images <= 0)
        failed = [k for k, r in checks.items() if isinstance(r, dict) and r.get("ok") is False]
        write_checks()
        if failed:
            print(f"\nCHECK FAILED: {', '.join(failed)} -- see {out / 'checks.json'}", file=sys.stderr)
            return 1

    T12, T3, a_star = analyse(D, ref, args, own_test)
    St["alpha_star"] = St.variant.map(a_star)
    St.to_csv(out / "vector_stats.csv", index=False)
    T12.to_csv(out / "table_exp12.csv", index=False)
    if len(T3):
        T3.to_csv(out / "table_exp3.csv", index=False)
    md = summary_md(T12, T3, St, D, args, checks)
    (out / "summary.md").write_text(md, encoding="utf-8")
    enc = sys.stdout.encoding or "utf-8"
    print("\n" + md.encode(enc, "replace").decode(enc))
    meta = {"args": vars(args), "rows": int(len(D)), "minutes": round((time.time() - t0) / 60, 1),
            "created": time.strftime("%Y-%m-%d %H:%M")}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=SUS._json_default), encoding="utf-8")
    print(f"wrote {out} ({(time.time() - t0) / 60:.1f} min)")
    return 0


def parse_args(argv=None):
    fl = SUS._floats
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", help="user study FOR_TEST")
    p.add_argument("--use", default="m")
    p.add_argument("--datasets", nargs="*", default=["Berkeley", "GrabCut", *SUS.MEDICAL],
                   choices=[d for d in SUS.DATASETS if d != "COCO"], help="test subsets besides COCO half B")
    p.add_argument("--checkpoint_path", default="")
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--site", default="L2.norm2")
    p.add_argument("--tokens", default="all")
    p.add_argument("--vector", default="exp_res/steer_user_study/vector.npz",
                   help="the reported COCO A vector: the full-halves variant must reproduce it ('' = no check)")
    p.add_argument("--alphas", type=fl, default=fl("1,2,3,4,6,8,12,16"))
    p.add_argument("--fixed_alphas", type=fl, default=fl("1,4,12"))
    p.add_argument("--extreme_k", type=lambda s: [int(x) for x in s.split(",") if x.strip()], default=[1, 3, 5],
                   help="experiment 1: the k worst / k best boxes per image (3 ~ a tenth of 25)")
    p.add_argument("--pos_thr", type=float, default=0.85, help="experiment 2: IoU a positive must exceed")
    p.add_argument("--own_train", type=int, default=10, help="experiment 3: training images per dataset")
    p.add_argument("--coco_sub_draws", type=int, default=5,
                   help="experiment 3: COCO A vectors of --own_train images (draw 0 = the own COCO draw), "
                        "scored on every test table, the like-for-like comparator of a dataset's own vector")
    p.add_argument("--tight", type=int, default=1, help="1 = the vectors with the tight GT box as a positive")
    p.add_argument("--pool", type=int, default=1,
                   help="1 = one vector from every dataset's own draw, and the leave-one-dataset-out ones")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reference_csv", default="")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--repro_tol", type=float, default=1e-3)
    p.add_argument("--repro_max_frac", type=float, default=0.01)
    p.add_argument("--group_tol", type=float, default=1e-3)
    p.add_argument("--limit_images", type=int, default=0, help="test images per table, for a smoke run")
    p.add_argument("--from_dump", default="", help="rows.csv.gz of an earlier run: analysis only")
    p.add_argument("--out_dir", default="exp_res/steer_variants")
    args = p.parse_args(argv)
    if not args.from_dump and not args.root:
        p.error("--root is required (or --from_dump)")
    return args


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
