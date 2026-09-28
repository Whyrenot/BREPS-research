"""
train_selectors.py
==================
Three selector questions, two model families, one pass over
`results/sam3_selector_pool_m.csv`.

  T1  EARLY STOP    when to stop the gradient ascent. 11 checkpoints per case,
                    features strictly causal in time. Scored against today's
                    behaviour (run all 50 steps) on BOTH axes -- IoU and
                    forwards spent, because stopping early that costs quality is
                    not a win and stopping late that costs compute is not either.
  T2  BEST-OF-N     which of the 16 perturbed-box masks, or the user's own
                    undefended mask, to return. Scored against today's defence,
                    which is argmax of SAM's predicted-IoU head over the 16.
  T3  ALL METHODS   T2's pool plus the final gradient mask: 18 candidates,
                    the full menu the pipeline can produce.

Every question is run with CatBoost and with logistic regression, in four
supervision formulations each (see selector_models.py). Nothing is selected on
the test fold: every head is trained the same way, and the whole table is
printed so the comparison is visible rather than asserted.

HYPERPARAMETERS
  Chosen by tune_selectors.py on an inner 320/80 cut of fold 0's training
  images, scored by selection IoU. Both sweeps came out flat -- CatBoost spans
  0.7804..0.7809 over depth {6,8} x lr {0.06,0.03} x iterations {800,2000}, and
  the logistic 0.7762..0.7772 over C {0.03,0.3,3} x pairs {32,128} -- so the
  cheapest member of each plateau is the default: depth 6, lr 0.06, 800 trees;
  C 0.03, 32 pairs. Which supervision formulation you pick matters far more
  than how you tune it, which is why all four are reported rather than one.

VALIDATION
  Images are the unit. 5 folds, stratified on the source dataset, so each fold
  trains on 400 images and tests on 100 with all ten datasets represented in
  both -- which is the 400/100 split, five times over, and the union of the
  test folds is an out-of-fold prediction for all 500 images. Fold 0 is
  reported on its own as the single held-out 100; the pooled OOF is the same
  procedure with five times the test cases and is what the per-group numbers
  should be read off, since `weak` is only 2500 cases to begin with.

  The 25 annotations of one image share an object, so a random row split would
  leak. Grouping by image is not optional here, and every standard error is
  clustered on the image for the same reason.

GROUPS
  weak / mid / strong are cut inside each image, 20/60/20 by the undefended
  IoU -- identical partition for all three tasks, so their per-group rows are
  comparable with each other.

    python scripts/train_selectors.py --csv results/sam3_selector_pool_m.csv \\
        --out_dir results/selectors
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import selector_data as SD
import selector_models as SM
import selector_policies as SP
from sklearn.model_selection import StratifiedKFold

# head name -> supervision mode. The name is what the tables print.
#   cb_*  CatBoost.
#   lr_*  linear. lr_rank and lr_best are logistic regressions (Bradley-Terry on
#         within-pool feature differences, and "is this the pool's argmax");
#         lr_ridge is a ridge fit on the pool-centred IoU, the linear
#         counterpart of cb_regc, kept so the families line up head for head.
#   *_ens per-case z-scored average of that family's heads.
CB_HEADS = {"cb_reg": "reg", "cb_regc": "regc", "cb_rank": "rank", "cb_best": "best"}
LR_HEADS = {"lr_ridge": "regc", "lr_rank": "rank", "lr_best": "best"}


def make_folds(pool: pd.DataFrame, n_splits: int, seed: int) -> list[np.ndarray]:
    """Test-image sets, stratified on dataset. Splitting at the image level is
    what enforces the grouping -- an image is never split across folds because
    an image is one row of the splitter's input."""
    im = (pool[["image_name", "dataset"]].drop_duplicates()
          .sort_values("image_name").reset_index(drop=True))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [im.image_name.values[te] for _, te in skf.split(im, im.dataset)]


def run_task(name: str, d: pd.DataFrame, cols: list[str], folds, args,
             ref_policy: str, extra_refs: dict, stop_rules: bool,
             method_cols: list[str] | None = None):
    """Fit every head out-of-fold, then score every policy that reads a head."""
    d = d.reset_index(drop=True)
    d["_y_best"] = SM.is_pool_best(d.target_iou, d.case)
    if stop_rules:
        d["_y_cont"] = SM.improves_later(d.target_iou, d.case)

    heads = dict(CB_HEADS)
    lin = dict(LR_HEADS)
    if stop_rules:
        heads["cb_cont"] = "cont"
        lin["lr_cont"] = "cont"
    if args.quick:
        heads = {k: v for k, v in heads.items() if k in ("cb_regc", "cb_cont")}
        lin = {k: v for k, v in lin.items() if k in ("lr_rank", "lr_cont")}
    q_heads = [h for h in list(heads) + list(lin) if not h.endswith("_cont")]

    for h in list(heads) + list(lin):
        d[h] = np.nan
    fold_of = pd.Series(-1, index=d.index)

    for k, te_imgs in enumerate(folds):
        te = d.image_name.isin(te_imgs).values
        fold_of[te] = k
        tr_d, te_d = d[~te], d[te]
        for h, mode in heads.items():
            t0 = time.time()
            d.loc[te, h] = SM.fit_predict_catboost(
                tr_d, te_d, cols, mode, args.iterations, args.threads, args.seed,
                depth=args.depth, learning_rate=args.lr)
            print(f"  [{name} fold {k}] {h:<8} {time.time() - t0:6.1f}s", flush=True)
        for h, mode in lin.items():
            t0 = time.time()
            d.loc[te, h] = SM.fit_predict_linear(
                tr_d, te_d, cols, mode, args.C, args.n_pairs, args.seed)
            print(f"  [{name} fold {k}] {h:<8} {time.time() - t0:6.1f}s", flush=True)

    # Ensembles. Raw scores are on different scales (an IoU, a centred IoU, a
    # logit, a YetiRank score), so they are z-scored WITHIN the pool before
    # averaging -- which is also the only normalisation that leaves the argmax
    # of a single head unchanged.
    def zc(col):
        g = d.groupby("case", sort=False)[col]
        sd = g.transform("std")
        return ((d[col] - g.transform("mean")) / sd.where(sd > 0)).fillna(0.0)

    zs = {h: zc(h) for h in q_heads}
    for fam in ("cb", "lr"):
        members = [h for h in q_heads if h.startswith(fam + "_")]
        if len(members) > 1:
            d[fam + "_ens"] = np.mean([zs[h] for h in members], axis=0)
    if len(q_heads) > 1:
        d["all_ens"] = np.mean([zs[h] for h in q_heads], axis=0)
    ens = [c for c in ("cb_ens", "lr_ens", "all_ens") if c in d.columns]

    d["_fold"] = fold_of.values
    grid = SP.CaseGrid(d)
    grp = SD.image_groups(d).reindex(grid.cases).values
    img = pd.Series(grid.cases).str.split("|").str[0].values

    res = {}
    for nm, col in extra_refs.items():
        res[nm] = grid.argmax(grid.col(d, col))
    res["ORACLE"] = grid.oracle()
    for h in q_heads + ens:
        res[h] = grid.argmax(grid.col(d, h))

    # Choosing the METHOD rather than the candidate. The best fixed method
    # flips with the group -- bon on weak, grad on mid, undef on strong -- so
    # a three-way chooser is a different, cheaper question than an 18-way one,
    # and its oracle says how much of the headroom lives there.
    if method_cols:
        sel = np.zeros((grid.n, grid.t), dtype=bool)
        for c in method_cols:
            sel |= grid.col(d, c) > 0
        res["ORACLE over methods"] = grid.restricted(grid.iou, sel)

    # switch-guard on the best head of each family: stay with today's pick
    # unless the model prefers something else by tau pool-sigmas
    inc = np.argmax(grid.col(d, extra_refs[ref_policy]), axis=1)
    for fam in ("cb", "lr", "all"):
        q = fam + "_ens" if fam + "_ens" in d.columns else None
        if q is None:
            cand = [h for h in q_heads if h.startswith(fam + "_")]
            if not cand:
                continue
            q = cand[0]
        for tau in args.tau:
            res[f"{q} guard {tau:g}s"] = grid.argmax_guarded(grid.col(d, q), inc, tau)
        if method_cols:
            res[f"{q} methods-only"] = grid.restricted(grid.col(d, q), sel)

    if stop_rules:
        for fam in ("cb", "lr"):
            # the stop rule reads the family's strongest quality head; the
            # ensemble if there is one, otherwise the single head that exists
            q = fam + "_ens"
            if q not in d.columns:
                cand = [h for h in q_heads if h.startswith(fam + "_")]
                if not cand:
                    continue
                q = cand[0]
            s = grid.col(d, q)
            for pat in args.patience:
                res[f"{fam}:STOP p={pat:g}"] = grid.stop_patience(s, pat)
            c = f"{fam}_cont"
            if c in d.columns and not d[c].isna().all():
                p = 1.0 / (1.0 + np.exp(-grid.col(d, c)))
                for thr in args.thresholds:
                    res[f"{fam}:STOPc t={thr:g}"] = grid.stop_prob(s, p, thr)

    tabs = []
    for split, m in (("holdout100", (grid.col(d, "_fold")[:, 0] == 0)),
                     ("oof500", np.ones(grid.n, dtype=bool))):
        tabs.append(SP.summarize(res, ref_policy, "ORACLE", img, grp, m, name, split))
    return d, pd.concat(tabs, ignore_index=True), res


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="results/sam3_selector_pool_m.csv")
    p.add_argument("--out_dir", default="results/selectors")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=800)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.06)
    p.add_argument("--threads", type=int, default=0, help="0 = all cores")
    p.add_argument("--C", type=float, default=0.03, help="logistic regularisation")
    p.add_argument("--n_pairs", type=int, default=32,
                   help="within-pool pairs sampled per case for the linear ranker")
    p.add_argument("--patience", type=float, nargs="+", default=[0.002, 0.005, 0.01])
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.35, 0.5])
    p.add_argument("--tau", type=float, nargs="+", default=[0.5, 1.0],
                   help="switch-guard margins, in pool sigmas of the head score")
    p.add_argument("--tasks", nargs="+", default=["t1", "t2", "t3"])
    p.add_argument("--strict_t1", action="store_true",
                   help="drop x_iou_undef / x_iou_hsel from the early-stop features")
    p.add_argument("--quick", action="store_true", help="two heads only, smoke test")
    p.add_argument("--importance", action="store_true",
                   help="also fit one full-data CatBoost per task for feature importance")
    a = p.parse_args()
    if a.threads <= 0:
        import multiprocessing
        a.threads = multiprocessing.cpu_count()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pool = SD.load_pool(a.csv)
    folds = make_folds(pool, a.folds, a.seed)
    slot = (pool[pool.method_slot.notna()]
            .pivot_table(index="case", columns="method_slot", values="target_iou"))
    grp_all = SD.image_groups(pool)
    header = [
        f"{len(pool)} rows | {pool.case.nunique()} cases | "
        f"{pool.image_name.nunique()} images | {pool.dataset.nunique()} datasets",
        "fixed policies over all cases: " + "  ".join(
            f"{k} {slot[k].mean():.4f}" for k in ("undef", "hsel", "bon", "grad")),
        "groups (20/60/20 within image, by undefended IoU): " + "  ".join(
            f"{g} n={int((grp_all == g).sum())} "
            f"undef={pool.groupby('case').gt_undef_iou.first()[grp_all == g].mean():.4f}"
            for g in ("weak", "mid", "strong")),
        f"folds: {a.folds} x {len(folds[0])} test images, stratified by dataset",
        f"catboost: iterations={a.iterations} depth={a.depth} lr={a.lr} | "
        f"logistic: C={a.C} pairs={a.n_pairs}",
    ]
    print("\n".join(header) + "\n")

    specs = {
        # task: (builder, reference, extra fixed baselines, stop rules,
        #        slots that count as a "method" for the restricted oracle)
        "t1": (lambda: SD.build_t1(pool, causal_x=not a.strict_t1),
               "grad_final (today)",
               {"hsel (step 0)": "_first", "grad_final (today)": "_last",
                "argmax f_pred": "f_pred"}, True, None),
        "t2": (lambda: SD.build_t2(pool), "bon argmax f_pred (today)",
               {"undef": "_undef", "bon argmax f_pred (today)": "_bonpick",
                "argmax f_pred (whole pool)": "f_pred"}, False,
               ["_undef", "_bonpick"]),
        "t3": (lambda: SD.build_t3(pool), "grad_final (best fixed)",
               {"undef": "_undef", "bon argmax f_pred": "_bonpick",
                "grad_final (best fixed)": "_grad",
                "argmax f_pred (whole pool)": "f_pred"}, False,
               ["_undef", "_bonpick", "_grad"]),
    }

    all_tabs, report = [], ["\n".join(header)]
    for name in a.tasks:
        build, ref, refs, stop, mcols = specs[name]
        d, cols = build()
        # fixed-policy indicator columns: argmax of these reproduces the policy
        if name == "t1":
            d["_first"] = -d.step.values
            d["_last"] = d.step.values
        else:
            d["_undef"] = (d.cand_kind == "undef").astype(float)
            d["_bonpick"] = (d.method_slot == "bon").astype(float)
            d["_grad"] = (d.cand_kind == "grad").astype(float)
        print(f"=== {name}: {len(d)} rows, {d.groupby('case').size().iloc[0]} "
              f"candidates/case, {len(cols)} features", flush=True)
        t0 = time.time()
        d, tab, res = run_task(name, d, cols, folds, a, ref, refs, stop, mcols)
        print(f"    fitted in {time.time() - t0:.0f}s", flush=True)
        all_tabs.append(tab)
        for split in ("holdout100", "oof500"):
            title = (f"\n{'=' * 78}\n{name.upper()}  --  {split}"
                     f"   (reference: {ref})\n{'=' * 78}")
            body = SP.render(tab[tab.split == split], ref, show_cost=stop)
            print(title)
            print(body, flush=True)
            report += [title, body]
        d[["case", "cand_kind", "cand_id", "step", "target_iou", "_fold"]
          + [c for c in d.columns
             if c.startswith(("cb_", "lr_")) or c == "all_ens"]].to_csv(
            out / f"{name}_scores.csv", index=False)
        # written per task, so a long run is never all-or-nothing
        pd.concat(all_tabs, ignore_index=True).to_csv(out / "metrics.csv", index=False)
        (out / "report.txt").write_text("\n".join(report), encoding="utf-8")
        if a.importance:
            # regc and best only: YetiRank would double the wall clock to say
            # roughly the same thing about which columns carry the signal
            for mode, hd in (("regc", "cb_regc"), ("best", "cb_best")):
                imp = SM.catboost_importance(d, cols, mode, a.iterations, a.threads,
                                             a.seed, depth=a.depth, learning_rate=a.lr)
                imp.rename("importance").to_csv(out / f"{name}_importance_{hd}.csv")
                print(f"\ntop features [{name} {hd}]\n{imp.head(15).to_string()}")

    tab = pd.concat(all_tabs, ignore_index=True)
    tab.to_csv(out / "metrics.csv", index=False)
    (out / "report.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"\nSaved -> {out / 'metrics.csv'}, {out / 'report.txt'}")


if __name__ == "__main__":
    main()
