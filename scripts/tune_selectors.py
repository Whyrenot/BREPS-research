"""
tune_selectors.py
=================
Hyperparameters, chosen without touching a test fold.

The split that matters: fold 0's 400 TRAINING images are cut again, 320/80,
stratified by dataset. Every configuration is fitted on the 320 and scored on
the 80 by the number that actually matters -- the mean IoU of the mask its
argmax selects -- not by RMSE or AUC, which rank configurations differently
when the label is a within-pool ordering.

The 100 test images of fold 0, and the other four folds, are never seen here.
Whatever this prints is then hard-coded into train_selectors.py's defaults and
applied to all five folds unchanged, so the reported numbers are clean.

    python scripts/tune_selectors.py --task t2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

import selector_data as SD
import selector_models as SM


def inner_split(pool, seed):
    im = (pool[["image_name", "dataset"]].drop_duplicates()
          .sort_values("image_name").reset_index(drop=True))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    te0 = set(im.image_name.values[next(iter(skf.split(im, im.dataset)))[1]])
    trim = im[~im.image_name.isin(te0)].reset_index(drop=True)
    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    va = set(trim.image_name.values[next(iter(skf2.split(trim, trim.dataset)))[1]])
    return set(trim.image_name) - va, va


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="results/sam3_selector_pool_m.csv")
    p.add_argument("--task", default="t2", choices=["t1", "t2", "t3"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--family", default="both", choices=["cb", "lr", "both"])
    a = p.parse_args()
    if a.threads <= 0:
        import multiprocessing
        a.threads = multiprocessing.cpu_count()

    pool = SD.load_pool(a.csv)
    d, cols = {"t1": SD.build_t1, "t2": SD.build_t2, "t3": SD.build_t3}[a.task](pool)
    d = d.reset_index(drop=True)
    d["_y_best"] = SM.is_pool_best(d.target_iou, d.case)
    if a.task == "t1":
        d["_y_cont"] = SM.improves_later(d.target_iou, d.case)
    tr_i, va_i = inner_split(pool, a.seed)
    tr = d[d.image_name.isin(tr_i)]
    vam = d.image_name.isin(va_i).values
    va = d[vam]
    print(f"{a.task}: tune on {len(tr_i)} images, validate on {len(va_i)} "
          f"({len(cols)} features)")
    print(f"validation oracle {va.groupby('case').target_iou.max().mean():.4f}\n")

    def score(pred):
        s = va.assign(_p=pred)
        return s.loc[s.groupby("case")._p.idxmax()].target_iou.mean()

    if a.family in ("cb", "both"):
        print(f"{'mode':<6} {'depth':>5} {'lr':>6} {'iters':>6} | {'sel IoU':>8} {'s':>5}")
        print("-" * 44)
        for mode in ("regc", "rank"):
            for depth in (6, 8):
                for lr, iters in ((0.06, 800), (0.03, 2000)):
                    t = time.time()
                    pr = SM.fit_predict_catboost(tr, va, cols, mode, iters,
                                                 a.threads, a.seed,
                                                 depth=depth, learning_rate=lr)
                    print(f"{mode:<6} {depth:>5} {lr:>6.2f} {iters:>6} | "
                          f"{score(pr):>8.4f} {time.time() - t:>5.0f}", flush=True)

    if a.family in ("lr", "both"):
        print(f"\n{'mode':<6} {'C':>7} {'pairs':>6} | {'sel IoU':>8} {'s':>5}")
        print("-" * 36)
        for C in (0.03, 0.3, 3.0):
            for mode, npair in (("rank", 32), ("rank", 128), ("best", 0)):
                t = time.time()
                pr = SM.fit_predict_linear(tr, va, cols, mode, C, npair, a.seed)
                print(f"{mode:<6} {C:>7.2f} {npair:>6} | {score(pr):>8.4f} "
                      f"{time.time() - t:>5.0f}", flush=True)


if __name__ == "__main__":
    main()
