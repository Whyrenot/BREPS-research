"""
steer_user_study.py
===================
Does the steering recipe found on critical shifts help on boxes real people
drew? steer_sam_decoder.py found one site where a fixed vector repairs a broken
box: L2.norm2 (the token stream after token -> image attention of two-way block
2), difference in means over all 7 tokens, alpha = 1. Here the same recipe is
estimated on user-study boxes and applied, frozen, to other user-study images.
A user's box breaks the mask in ordinary ways (too wide, too tight, touching a
neighbour), not adversarially, so the question is whether one direction still
separates good from bad boxes there.

PROTOCOL  (STEERING_USER_STUDY_HANDOFF.md)
--------
  data      user study FOR_TEST, --use m only; three subsets of 50 images x 25
            annotations: COCO (COCO_MVal_*), Berkeley, GrabCut
  box       the tight bbox of the user's mask, in SAM's 1024 frame, read exactly
            as refine_box_iou_grad.build_user_cases does (the producer of
            samfull_m.csv); GT = masks/<stem> > 0 (refine_box_iou_grad._load_gt)
  split     the 50 COCO stems sorted, default_rng(--split_seed).permutation:
            the first 25 are half A, the rest half B -- by image, never by box
  groups    bottom50 / top50 inside each image by the undefended IoU (12 / 13),
            selector_data.image_halves; for reporting only, and always the
            groups of the unsteered forward
  vector    on half A only: per image d_img = mean(top50 states) - mean(bottom50
            states) at --site (7 x 256), v = the mean of d_img over images. The
            difference is taken inside the image, so what the images look like
            does not enter v. Control: a fixed random direction with v's norm
            per token (steer_sam_decoder.random_directions, the direction the
            critical-shift run used at this site)
  apply     on half B, Berkeley and GrabCut: h' = h + alpha * v on --tokens at
            --site, the rest of the forward unchanged, multimask_output=False;
            the same for the random direction. One frozen v for all three test
            sets, every alpha of --alphas reported (none chosen on test data).
            alpha = 1 moves the bottom50 mean onto the top50 mean and is the
            pre-registered configuration; 0.5 and 0.1 are smaller doses of the
            same vector

METRICS (per dataset x alpha x group)
-------
  IoU with GT (inter / (union + 1e-6)), its delta to the undefended forward
  with a paired 95% bootstrap CI over images, the gain in % of undefended,
  the share of boxes whose IoU rose / fell by more than --updown_eps, and the
  mask change |m XOR m0| / |m OR m0| (steer_sam_decoder.mask_change). Under
  each table: head-select, best-of-N and grad from --reference_csv on the same
  boxes (head-select runs multimask, a different decision -- not a like-for-like
  baseline).

CHECKS (checks.json; a failure stops the run before any table is written)
------
  1. zero dose: a zero vector at the site reproduces the unsteered masks and
     logits bit for bit, on every image;
  2. the undefended IoU of every box matches --reference_csv (key image_name,
     user, attempt): max |dIoU|, and the share of boxes with |dIoU| >
     --repro_tol must stay <= --repro_max_frac (that csv ran one box per
     forward, this script a whole image per forward: a few pixels may differ);
  3. groups: 12 bottom50 / 13 top50 on every image; the undefended IoU per
     group over all 50 images of each subset matches the handoff's table (and
     the reference csv) within --group_tol;
  4. split: A and B disjoint, 25 + 25 stems; v averages 25 x 25 boxes;
  5. steer_sam_decoder.py --self_test passes (hooks, trace agreement).

OUTPUTS (--out_dir)
-------
  split.json            the stems of A and B
  vector.npz            v, rand (7 x 256), d_img per A image, the A states
  vector_stats.csv      per token: ||v||, mean ||h|| on A, the dose, cosine with
                        the critical-shift vector (--cs_vectors)
  per_case.csv          one row per box: dataset, image_name, user, attempt,
                        half (A / B / -), group, undef_iou, ref_undef_iou, and
                        per alpha dm_iou_a<alpha>, rand_iou_a<alpha>,
                        chg_dm_a<alpha>, chg_rand_a<alpha>
  table_COCO_B.csv, table_Berkeley.csv, table_GrabCut.csv
  reference_rows.csv    head-select / best-of-N / grad on the same boxes
  summary.md, checks.json, meta.json

Example
-------
    CUDA_VISIBLE_DEVICES=0 python scripts/steer_user_study.py \\
        --root /.../FOR_TEST --use m \\
        --checkpoint_path /.../MODEL_CHECKPOINTS/SAM/sam_vit_b_01ec64.pth \\
        --site L2.norm2 --alphas 1,0.5,0.1 --tokens all --split_seed 0 \\
        --reference_csv results/samfull_m.csv \\
        --out_dir exp_res/steer_user_study
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
import torch

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import selector_data as SD  # noqa: E402
import steer_sam_decoder as SSD  # noqa: E402
from activation_flow_graph import TOKEN_GROUPS, load_predictor  # noqa: E402
from heatmaps.defend_critical_shifts import _find_file, _prepare_image  # noqa: E402
from heatmaps.defend_user_study import index_user_masks  # noqa: E402
from refine_box_iou_grad import _load_gt, build_user_cases  # noqa: E402
from steer_sam_decoder import (  # noqa: E402
    Cfg, Collector, Runner, batch_iou, build_sites, feature_of, make_edit, mask_change, random_directions,
    zero_edit,
)

# dataset -> image-stem prefix in FOR_TEST
DATASETS = OrderedDict([("COCO", "COCO_MVal_"), ("Berkeley", "Berkeley_"), ("GrabCut", "GrabCut_")])
GROUPS = ("all", "bottom50", "top50")
# undefended SAM-1 `m` per group over all 50 images of each subset, from
# samfull_m.csv (handoff section 5): all / bottom50 / top50
REF_UNDEF = {"COCO": (0.8033, 0.7178, 0.8823), "Berkeley": (0.8261, 0.7522, 0.8944),
             "GrabCut": (0.8563, 0.7730, 0.9332)}
REF_METHODS = OrderedDict([("headsel_iou", "head-select (multimask)"), ("best_of_n_iou", "best-of-N"),
                           ("grad_final_iou", "grad")])
TOKEN_NAMES = ["iou", "mask0", "mask1", "mask2", "mask3", "prompt0", "prompt1"]
KEY = SD.CASE_KEYS                     # image_name, kind, user, attempt
N_PER_IMAGE = 25


class CheckFailed(RuntimeError):
    pass


def _a(alpha: float) -> str:
    return f"a{alpha:g}"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def index_study(root: Path, use: str, datasets) -> dict:
    """{dataset: {stem: [UserMaskRecord]}} for the subsets in `datasets`."""
    recs = index_user_masks(root / "user_masks", kinds=tuple(sorted(set(use))))
    out = {d: {} for d in datasets}
    for r in recs:
        for d in datasets:
            if r.stem.startswith(DATASETS[d]):
                out[d].setdefault(r.stem, []).append(r)
    return {d: dict(sorted(v.items())) for d, v in out.items()}


def split_halves(stems, seed: int):
    stems = sorted(stems)
    perm = np.random.default_rng(seed).permutation(len(stems))
    k = len(stems) // 2
    return sorted(stems[i] for i in perm[:k]), sorted(stems[i] for i in perm[k:])


def load_image_cases(predictor, root: Path, stem: str, recs, device):
    """Encode the image and build its user boxes as refine_box_iou_grad does:
    -> (cases, boxes (N, 4) float32 in the 1024 frame, GT bool tensor on device)."""
    img_path = _find_file(str(root / "images"), stem)
    gt_path = _find_file(str(root / "masks"), stem)
    if img_path is None or gt_path is None:
        raise FileNotFoundError(f"{stem}: {'image' if img_path is None else 'mask'} not found under {root}")
    _prepare_image(str(img_path), predictor)
    gt = _load_gt(gt_path, predictor)
    cases = build_user_cases(recs, predictor, gt)
    boxes = np.asarray([c["bad_box"] for c in cases], dtype=np.float32).reshape(-1, 4)
    return cases, boxes, gt.to(device)


def image_halves(df: pd.DataFrame) -> pd.Series:
    """selector_data.image_halves, through the image_groups call it is made of:
    bottom50 / top50 per case inside each image (12 / 13 of 25), ties broken
    by the case key as in the project's tables."""
    return SD.image_groups(df, lo=0.0, hi=0.5).map({"mid": "bottom50", "strong": "top50"})


def assign_groups(df: pd.DataFrame) -> pd.Series:
    d = df.rename(columns={"undef_iou": "gt_undef_iou"})
    case = d[KEY].astype(str).agg("|".join, axis=1)
    if case.duplicated().any():
        raise RuntimeError(f"duplicate case keys {KEY}: {case[case.duplicated()].tolist()[:5]}")
    g = image_halves(d.assign(case=case))
    return case.map(g)


# ---------------------------------------------------------------------------
# Forward passes
# ---------------------------------------------------------------------------

def check_zero(runner: Runner, boxes, m0, low0, edit, site_id: str, checks: dict, where: str):
    """Check 1: a zero vector at the site reproduces the baseline bit for bit."""
    z = checks["zero_dose"]
    m, low = runner.forward(boxes, edits={site_id: edit})
    z["forwards"] += 1
    if torch.equal(m, m0) and torch.equal(low, low0):
        z["bit_exact"] += 1
        return
    z["failures"].append(f"{where}: mask rel {SSD.rel_diff(m, m0):.3g}, logits rel {SSD.rel_diff(low, low0):.3g}")
    z["ok"] = False
    raise CheckFailed(f"zero dose at {where}")


def case_rows(dataset: str, stem: str, half: str, cases, iou0: np.ndarray) -> list:
    return [{"dataset": dataset, "image_name": stem, "kind": c["kind"], "user": c["user"], "attempt": c["attempt"],
             "half": half, "undef_iou": float(v)} for c, v in zip(cases, iou0)]


def collect_source(runner: Runner, site, root: Path, stems, recs: dict, checks: dict, seed: int):
    """Half A: the unsteered forward of every box, its state at the site, and
    the random direction of steer_sam_decoder at the site."""
    dev = runner.device
    rows, states, R = [], [], None
    t0 = time.time()
    for i, stem in enumerate(stems):
        cases, boxes, gt = load_image_cases(runner.p, root, stem, recs[stem], dev)
        b = torch.as_tensor(boxes, device=dev)
        buf = []
        m0, low0 = runner.forward(b, record={site.id: lambda t, ctx: buf.append(feature_of(site, t, ctx).cpu())})
        check_zero(runner, b, m0, low0, zero_edit, site.id, checks, f"A/{stem}")
        if R is None:
            # one box through every site gives random_directions the shapes it
            # enumerates, so the direction at this site is the one the
            # critical-shift run drew there (seed, site index)
            coll = Collector(runner)
            runner.forward(b[:1], record=coll.fns())
            R = random_directions(list(runner.sites.values()), coll.result(), seed)[site.id]
        rows += case_rows("COCO", stem, "A", cases, batch_iou(m0, gt).cpu().numpy())
        states.append(torch.cat(buf).numpy())
        print(f"[A] [{i + 1}/{len(stems)}] {stem}: {len(cases)} boxes ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows), np.concatenate(states), R


def build_vector(df: pd.DataFrame, H: np.ndarray):
    """v = mean over images of mean(top50) - mean(bottom50), inside the image."""
    d_img, used = [], 0
    for stem in sorted(df.image_name.unique()):
        idx = np.where(df.image_name.to_numpy() == stem)[0]
        grp = df.group.to_numpy()[idx]
        top, bot = idx[grp == "top50"], idx[grp == "bottom50"]
        if len(top) == 0 or len(bot) == 0:
            raise RuntimeError(f"{stem}: an empty half ({len(bot)} bottom50, {len(top)} top50)")
        d_img.append(H[top].mean(0) - H[bot].mean(0))
        used += len(top) + len(bot)
    d_img = np.stack(d_img).astype(np.float32)
    return d_img.mean(0), d_img, used


def evaluate_target(runner: Runner, site, root: Path, dataset: str, half: str, stems, recs: dict, Vt: dict,
                    group: str, alphas, checks: dict) -> pd.DataFrame:
    """Every box of every image: unsteered, then dm and rand at every alpha, on
    the same box list (Runner chunks it the same way every time)."""
    dev = runner.device
    zero = make_edit(Cfg(-1, "steer", site.id, group, "dm", 0.0), site, Vt)
    edits = {(m, a): make_edit(Cfg(-1, "steer", site.id, group, method, a), site, Vt)
             for a in alphas for m, method in (("dm", "dm"), ("rand", "dm_rand"))}
    out = []
    t0 = time.time()
    for i, stem in enumerate(stems):
        cases, boxes, gt = load_image_cases(runner.p, root, stem, recs[stem], dev)
        b = torch.as_tensor(boxes, device=dev)
        m0, low0 = runner.forward(b)
        check_zero(runner, b, m0, low0, zero, site.id, checks, f"{dataset}/{stem}")
        df = pd.DataFrame(case_rows(dataset, stem, half, cases, batch_iou(m0, gt).cpu().numpy()))
        for (m, a), edit in edits.items():
            ms, _ = runner.forward(b, edits={site.id: edit})
            df[f"{m}_iou_{_a(a)}"] = batch_iou(ms, gt).cpu().numpy()
            df[f"chg_{m}_{_a(a)}"] = mask_change(ms, m0).cpu().numpy()
        out.append(df)
        a0 = _a(alphas[0])
        print(f"[{dataset}{'/' + half if half != '-' else ''}] [{i + 1}/{len(stems)}] {stem}: {len(cases)} boxes, "
              f"undef {df.undef_iou.mean():.4f}, dm@{alphas[0]:g} {df[f'dm_iou_{a0}'].mean():.4f}, "
              f"rand@{alphas[0]:g} {df[f'rand_iou_{a0}'].mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def read_reference(path, use: str) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"--reference_csv not found: {p}")
    ref = pd.read_csv(p, dtype={k: str for k in KEY})
    return ref[ref.kind.isin(list(use))].reset_index(drop=True)


def check_repro(df: pd.DataFrame, ref: pd.DataFrame | None, args) -> dict:
    """Check 2: undefended IoU per box against the reference csv."""
    if ref is None:
        return {"status": "skipped (no --reference_csv)", "ok": True}
    m = df[KEY + ["undef_iou"]].merge(ref[KEY + ["undefended_iou"]], on=KEY, how="left")
    missing = int(m.undefended_iou.isna().sum())
    d = (m.undef_iou - m.undefended_iou).abs()
    n_over = int((d > args.repro_tol).sum())
    frac = n_over / max(len(m) - missing, 1)
    worst = m.assign(abs_diff=d).nlargest(5, "abs_diff")
    r = {"boxes": len(m), "missing_in_reference": missing, "max_abs_diff": float(d.max()) if len(d) else 0.0,
         "mean_abs_diff": float(d.mean()) if len(d) else 0.0, f"n_over_{args.repro_tol:g}": n_over,
         "frac_over": frac, "max_frac_allowed": args.repro_max_frac,
         "worst": worst[KEY + ["undef_iou", "undefended_iou", "abs_diff"]].to_dict(orient="records")}
    r["ok"] = missing == 0 and frac <= args.repro_max_frac
    return r


def check_groups(df: pd.DataFrame, ref: pd.DataFrame | None, args, full: bool) -> dict:
    """Check 3: 12 / 13 per image; undefended per group over all 50 images."""
    cnt = df.groupby(["image_name", "group"]).size().unstack(fill_value=0)
    for g in ("bottom50", "top50"):
        if g not in cnt:
            cnt[g] = 0
    bad = cnt[(cnt.bottom50 != 12) | (cnt.top50 != 13)]
    r = {"images": int(len(cnt)), "images_not_12_13": bad.index.tolist()[:20], "n_images_not_12_13": int(len(bad))}
    ok = len(bad) == 0
    if not full:
        r["group_means"] = "skipped (--limit_images)"
        r["ok"] = ok
        return r
    def means(s):
        return {"all": s.undef_iou.mean(), **s.groupby("group").undef_iou.mean().to_dict()}

    rows, rr = {}, None
    if ref is not None:
        # the reference's own undefended IoU, grouped the same way: how far
        # this run's groups are from the ones the project's tables used
        rr = ref[ref.image_name.isin(df.image_name.unique())].rename(columns={"undefended_iou": "undef_iou"})
        rr = rr.assign(group=assign_groups(rr))
        mm = df[KEY + ["group"]].merge(rr[KEY + ["group"]], on=KEY, how="left", suffixes=("", "_ref"))
        r["group_label_mismatch_vs_reference"] = int((mm.group != mm.group_ref).sum())
    for ds, vals in REF_UNDEF.items():
        s = df[df.dataset == ds]
        if s.empty:
            continue
        got = means(s)
        row = {g: {"undef": float(got.get(g, math.nan)), "handoff": w} for g, w in zip(GROUPS, vals)}
        if rr is not None:
            exact = means(rr[rr.image_name.str.startswith(DATASETS[ds])])
            for g in GROUPS:
                row[g]["reference_csv"] = float(exact.get(g, math.nan))
        for g in GROUPS:
            diffs = [abs(row[g]["undef"] - row[g][k]) for k in ("handoff", "reference_csv") if k in row[g]]
            row[g]["max_abs_diff"] = max(diffs) if diffs else math.nan
            if not diffs or not max(diffs) <= args.group_tol:
                ok = False
        rows[ds] = row
    r["group_means"] = rows
    r["tol"] = args.group_tol
    r["ok"] = ok
    return r


def run_self_test(args) -> dict:
    """Check 5: steer_sam_decoder.py --self_test with the real weights."""
    argv = ["--self_test", "--device", args.device, "--model_type", args.model_type]
    if args.checkpoint_path:
        argv += ["--checkpoint_path", args.checkpoint_path]
    print("steer_sam_decoder --self_test " + " ".join(argv[1:]), flush=True)
    rc = SSD.main(argv)
    return {"argv": argv, "returncode": int(rc), "ok": rc == 0}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def boot_delta(d: np.ndarray, images: np.ndarray, n_boot: int, seed: int):
    """95% CI of the mean of d (one value per box), resampling images: the
    boxes of one image are drawn together, so their dependence is kept."""
    codes, img = np.unique(images, return_inverse=True)
    S = np.bincount(img, weights=d, minlength=len(codes))
    N = np.bincount(img, minlength=len(codes)).astype(float)
    if len(codes) < 2 or n_boot <= 0:
        return math.nan, math.nan
    idx = np.random.default_rng(seed).integers(0, len(codes), size=(n_boot, len(codes)))
    est = S[idx].sum(1) / N[idx].sum(1)
    lo, hi = np.percentile(est, [2.5, 97.5])
    return float(lo), float(hi)


def dataset_table(df: pd.DataFrame, alphas, args) -> pd.DataFrame:
    rows = []
    for a in alphas:
        for g in GROUPS:
            s = df if g == "all" else df[df.group == g]
            u = s.undef_iou.to_numpy()
            r = {"alpha": a, "group": g, "n": len(s), "n_images": int(s.image_name.nunique()), "undef": u.mean()}
            for m in ("dm", "rand"):
                x = s[f"{m}_iou_{_a(a)}"].to_numpy()
                d = x - u
                lo, hi = boot_delta(d, s.image_name.to_numpy(), args.n_boot, args.seed)
                r.update({m: x.mean(), f"d_{m}": d.mean(), f"d_{m}_lo": lo, f"d_{m}_hi": hi,
                          f"pct_{m}": (x.mean() - u.mean()) / u.mean() * 100,
                          f"up_{m}": float((d > args.updown_eps).mean()), f"down_{m}": float((d < -args.updown_eps).mean()),
                          f"chg_{m}": float(s[f"chg_{m}_{_a(a)}"].mean())})
            rows.append(r)
    return pd.DataFrame(rows)


def reference_rows(df: pd.DataFrame, ref: pd.DataFrame | None, name: str) -> pd.DataFrame:
    """head-select / best-of-N / grad from the reference csv on the same boxes,
    in this run's groups; % against the csv's own undefended IoU."""
    if ref is None:
        return pd.DataFrame()
    cols = ["undefended_iou"] + [c for c in REF_METHODS if c in ref.columns]
    m = df[KEY + ["group"]].merge(ref[KEY + cols], on=KEY, how="left")
    rows = []
    for g in GROUPS:
        s = m if g == "all" else m[m.group == g]
        base = s.undefended_iou.mean()
        for c in cols:
            rows.append({"dataset": name, "group": g, "n": len(s), "method": REF_METHODS.get(c, "undef (csv)"),
                         "column": c, "iou": s[c].mean(), "pct_vs_undef": (s[c].mean() - base) / base * 100})
    return pd.DataFrame(rows)


def _f(x, nd=4):
    return "-" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def md_table(t: pd.DataFrame) -> str:
    cols = ["α", "группа", "n", "undef", "DM", "Δ DM [95% CI]", "% к undef", "случайное", "Δ случ. [95% CI]",
            "% к undef", "↑ боксов DM", "↓ боксов DM", "изм. маски DM"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in t.to_dict("records"):
        out.append(f"| {r['alpha']:g} | {r['group']} | {r['n']} | {_f(r['undef'])} | {_f(r['dm'])} | "
                   f"{r['d_dm']:+.4f} [{r['d_dm_lo']:+.4f}, {r['d_dm_hi']:+.4f}] | {r['pct_dm']:+.2f}% | "
                   f"{_f(r['rand'])} | {r['d_rand']:+.4f} [{r['d_rand_lo']:+.4f}, {r['d_rand_hi']:+.4f}] | "
                   f"{r['pct_rand']:+.2f}% | {r['up_dm']:.1%} | {r['down_dm']:.1%} | {r['chg_dm']:.4f} |")
    return "\n".join(out)


def md_reference(rr: pd.DataFrame) -> str:
    if rr.empty:
        return ""
    out = ["| метод (samfull_m.csv, те же боксы) | all | bottom50 | top50 |", "|---|---|---|---|"]
    for meth, s in rr.groupby("method", sort=False):
        s = s.set_index("group")
        out.append(f"| {meth} | " + " | ".join(f"{s.loc[g, 'iou']:.4f} ({s.loc[g, 'pct_vs_undef']:+.2f}%)"
                                               for g in GROUPS) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Vector diagnostics
# ---------------------------------------------------------------------------

def vector_stats(v: np.ndarray, rand: np.ndarray, H: np.ndarray, alphas, cs_path) -> tuple[pd.DataFrame, dict]:
    hn = np.linalg.norm(H, axis=-1).mean(0)                                  # mean ||h[i]|| on A
    st = pd.DataFrame({"token": TOKEN_NAMES[:len(v)], "v_norm": np.linalg.norm(v, axis=-1),
                       "rand_norm": np.linalg.norm(rand, axis=-1), "h_norm_mean": hn})
    st["v_over_h"] = st.v_norm / st.h_norm_mean
    for a in alphas:
        st[f"dose_{_a(a)}"] = a * st.v_over_h
    whole = {"v_norm": float(np.linalg.norm(v)), "h_norm_mean": float(np.linalg.norm(H.reshape(len(H), -1), axis=1).mean())}
    whole["v_over_h"] = whole["v_norm"] / whole["h_norm_mean"]
    cs = {"path": str(cs_path) if cs_path else None, "found": False}
    p = Path(cs_path) if cs_path else None
    if p is not None and not p.is_file() and (_HERE.parent / p).is_file():
        p = _HERE.parent / p
    if p is not None and p.is_file():
        z = np.load(p)
        if "L2.norm2/dm" in z.files:
            def cos(a, b):
                return float(np.dot(a.ravel(), b.ravel()) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
            c = z["L2.norm2/dm"]
            st["cos_cs_dm"] = [cos(v[i], c[i]) for i in range(len(v))]
            st["cs_dm_norm"] = np.linalg.norm(c, axis=-1)
            cs.update(found=True, cos_all=cos(v, c), cs_norm=float(np.linalg.norm(c)))
            if "L2.norm2/dm_rand" in z.files:
                # the same random direction as the critical-shift run: per token cosine 1
                cr = z["L2.norm2/dm_rand"]
                cs["cos_rand_vs_cs_rand_min"] = min(cos(rand[i], cr[i]) for i in range(len(v)))
    return st, {"whole": whole, "critical_shifts": cs}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def run(args) -> int:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)
    full = args.limit_images <= 0
    checks = {"zero_dose": {"forwards": 0, "bit_exact": 0, "failures": [], "ok": True}}

    def write_checks():
        (out / "checks.json").write_text(json.dumps(checks, indent=2, default=_json_default), encoding="utf-8")

    def fail(name):
        write_checks()
        print(f"\nCHECK FAILED: {name} -- see {out / 'checks.json'}; no tables written", file=sys.stderr)
        return 1

    # 5. the hooks themselves, before anything is computed with them
    if args.skip_self_test:
        checks["self_test"] = {"status": "skipped (--skip_self_test)", "ok": True}
    else:
        checks["self_test"] = run_self_test(args)
        if not checks["self_test"]["ok"]:
            return fail("self_test")

    # data and split
    datasets = list(dict.fromkeys(["COCO", *args.datasets]))
    index = index_study(root, args.use, datasets)
    for d in datasets:
        n_box = sum(len(v) for v in index[d].values())
        print(f"[{d}] {len(index[d])} images, {n_box} user masks ({args.use})", flush=True)
        if not index[d]:
            raise SystemExit(f"no user masks for {d} under {root / 'user_masks'}")
    A, B = split_halves(index["COCO"], args.split_seed)
    checks["split"] = {"n_A": len(A), "n_B": len(B), "overlap": sorted(set(A) & set(B)),
                       "ok": not (set(A) & set(B)) and len(A) == len(B) == 25}
    (out / "split.json").write_text(json.dumps({"seed": args.split_seed, "rule": "sorted stems, "
                                                "np.random.default_rng(seed).permutation, first half -> A",
                                                "A": A, "B": B}, indent=2), encoding="utf-8")
    if not checks["split"]["ok"]:
        return fail("split")
    if not full:
        A, B = A[:args.limit_images], B[:args.limit_images]
    targets = [("COCO", "B", B)] + [(d, "-", list(index[d])[:args.limit_images] if not full else list(index[d]))
                                    for d in datasets if d != "COCO"]

    device = torch.device(args.device)
    predictor = load_predictor(args.checkpoint_path, args.model_type, device)
    sites = build_sites(len(predictor.model.mask_decoder.transformer.layers))
    site = next((s for s in sites if s.id == args.site), None)
    if site is None or site.kind != "tok":
        raise SystemExit(f"--site must be a token-stream site, one of {[s.id for s in sites if s.kind == 'tok']}")
    runner = Runner(predictor, sites, args.batch, field_grid=64, field_margin=0.25)
    ref = read_reference(args.reference_csv, args.use)

    # pass 1: half A -> the vector
    print(f"\npass 1: half A ({len(A)} COCO images), states at {site.id}", flush=True)
    try:
        dfA, H, R = collect_source(runner, site, root, A, index["COCO"], checks, args.seed)
    except CheckFailed as e:
        return fail(str(e))
    dfA["group"] = assign_groups(dfA)
    checks["repro_A"] = check_repro(dfA, ref, args)
    if not checks["repro_A"]["ok"]:
        return fail("undefended IoU vs --reference_csv on half A")
    v, d_img, used = build_vector(dfA, H)
    rand = (R * np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)
    want = len(A) * N_PER_IMAGE
    checks["split"].update(vector_boxes=used, vector_images=len(d_img), vector_boxes_expected=want)
    if full and used != want:
        checks["split"]["ok"] = False
        return fail(f"vector averaged {used} boxes, expected {want}")
    sha = hashlib.sha256(np.ascontiguousarray(v).tobytes() + np.ascontiguousarray(rand).tobytes()).hexdigest()
    checks["vector_sha256"] = sha
    np.savez_compressed(out / "vector.npz", v=v, rand=rand, d_img=d_img, images_A=np.asarray(sorted(dfA.image_name.unique())),
                        states_A=H.astype(np.float32), undef_A=dfA.undef_iou.to_numpy(np.float32),
                        group_A=dfA.group.to_numpy().astype(str), site=site.id)
    st, vmeta = vector_stats(v, rand, H, args.alphas, args.cs_vectors)
    st.to_csv(out / "vector_stats.csv", index=False)
    checks["vector"] = vmeta
    print(f"\nv at {site.id}: ||v|| {vmeta['whole']['v_norm']:.4f} = {vmeta['whole']['v_over_h']:.4f} of mean ||h|| "
          f"(per token below); cosine with the critical-shift dm: "
          f"{vmeta['critical_shifts'].get('cos_all', 'n/a') if vmeta['critical_shifts']['found'] else 'not found'}")
    print(st.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    # pass 2: test images, frozen v
    Vt = {site.id: {"dm": torch.as_tensor(v, device=device), "dm_rand": torch.as_tensor(rand, device=device)}}
    parts = [dfA]
    for ds, half, stems in targets:
        print(f"\npass 2: {ds}{' half ' + half if half != '-' else ''} ({len(stems)} images), "
              f"alpha {', '.join(f'{a:g}' for a in args.alphas)}", flush=True)
        try:
            parts.append(evaluate_target(runner, site, root, ds, half, stems, index[ds], Vt, args.tokens,
                                         args.alphas, checks))
        except CheckFailed as e:
            return fail(str(e))
    df = pd.concat(parts, ignore_index=True)
    df["group"] = assign_groups(df)       # inside each image, so A's labels are unchanged
    if ref is not None:
        df = df.merge(ref[KEY + ["undefended_iou"]].rename(columns={"undefended_iou": "ref_undef_iou"}),
                      on=KEY, how="left")
    lead = ["dataset", "image_name", "kind", "user", "attempt", "half", "group", "undef_iou"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    df.to_csv(out / "per_case.csv", index=False)

    checks["repro"] = check_repro(df, ref, args)
    checks["groups"] = check_groups(df, ref, args, full)
    steered = [c for c in df.columns if c.startswith("chg_")]
    checks["steering_changed_boxes"] = {c: float((df[c] > 0).mean()) for c in steered}
    failed = [k for k, r in checks.items() if isinstance(r, dict) and r.get("ok") is False]
    if failed:
        return fail(", ".join(failed))
    write_checks()

    # tables
    tables, refs = {}, []
    for ds, half, _ in targets:
        name = f"{ds}_{half}" if half != "-" else ds
        s = df[(df.dataset == ds) & (df.half == half)]
        t = dataset_table(s, args.alphas, args)
        t.to_csv(out / f"table_{name}.csv", index=False)
        tables[name] = (t, s)
        refs.append(reference_rows(s, ref, name))
    rr = pd.concat(refs, ignore_index=True) if refs else pd.DataFrame()
    if not rr.empty:
        rr.to_csv(out / "reference_rows.csv", index=False)

    md = summary_md(args, site, A, B, st, vmeta, tables, rr, checks, df)
    (out / "summary.md").write_text(md, encoding="utf-8")
    enc = sys.stdout.encoding or "utf-8"            # a cp1251 console has no α / Δ / −
    print("\n" + md.encode(enc, "replace").decode(enc))
    meta = {"model": f"SAM {args.model_type}", "checkpoint": Path(args.checkpoint_path).name, "site": site.id,
            "tokens": args.tokens, "alphas": args.alphas, "datasets": datasets, "boxes": int(len(df)),
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "torch": torch.__version__, "args": vars(args), "minutes": round((time.time() - t0) / 60, 1),
            "created": time.strftime("%Y-%m-%d %H:%M")}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=_json_default), encoding="utf-8")
    print(f"wrote {out} ({(time.time() - t0) / 60:.1f} min)")
    return 0


def summary_md(args, site, A, B, st, vmeta, tables, rr, checks, df) -> str:
    L = [f"# Steering SAM-1 {args.model_type} на боксах user study (`{args.use}`)", "",
         f"Вектор: DM top50 − bottom50 внутри картинки, узел `{site.id}`, токены `{args.tokens}`, "
         f"посчитан на половине A COCO ({len(A)} картинок, {int((df.half == 'A').sum())} боксов), "
         f"заморожен; α = {', '.join(f'{a:g}' for a in args.alphas)} (α = 1 — заранее выбранная конфигурация, "
         "остальные — меньшие дозы того же вектора, ничего не подбиралось на тесте). "
         "Группы — по undefended IoU внутри картинки (12 / 13). Δ — к undefended, CI — бутстреп по картинкам "
         f"({args.n_boot}). ↑/↓ — доля боксов с изменением IoU больше {args.updown_eps:g}.", "",
         "## Вектор", "",
         f"‖v‖ = {vmeta['whole']['v_norm']:.4f}, средняя ‖h‖ на A = {vmeta['whole']['h_norm_mean']:.2f}, "
         f"доза α=1: {vmeta['whole']['v_over_h']:.2%}."]
    cs = vmeta["critical_shifts"]
    if cs["found"]:
        L.append(f"Косинус с вектором critical shifts (`{cs['path']}`, `L2.norm2/dm`): {cs['cos_all']:+.3f} по всему "
                 f"тензору; ‖v_cs‖ = {cs['cs_norm']:.4f}.")
    else:
        L.append(f"Вектор critical shifts не найден ({cs['path']}): косинус не посчитан.")
    L += ["", "| токен | ‖v‖ | ‖h‖ (A) | ‖v‖/‖h‖ | " + " | ".join(f"доза α={a:g}" for a in args.alphas)
          + (" | cos с CS dm |" if "cos_cs_dm" in st else " |"),
          "|" + "|".join(["---"] * (4 + len(args.alphas) + ("cos_cs_dm" in st))) + "|"]
    for _, r in st.iterrows():
        L.append(f"| {r.token} | {r.v_norm:.4f} | {r.h_norm_mean:.2f} | {r.v_over_h:.4f} | "
                 + " | ".join(f"{r[f'dose_{_a(a)}']:.4f}" for a in args.alphas)
                 + (f" | {r.cos_cs_dm:+.3f} |" if "cos_cs_dm" in st else " |"))
    for name, (t, s) in tables.items():
        L += ["", f"## {name}: {s.image_name.nunique()} картинок, {len(s)} боксов", "", md_table(t)]
        sub = rr[rr.dataset == name] if not rr.empty else rr
        if not sub.empty:
            L += ["", "Сравнение (head-select работает в режиме multimask — другое решение, не прямой бейзлайн):",
                  "", md_reference(sub)]
    z, rp, gr = checks["zero_dose"], checks.get("repro", {}), checks.get("groups", {})
    L += ["", "## Проверки", "",
          f"- нулевая доза: {z['bit_exact']}/{z['forwards']} прогонов бит в бит;",
          f"- undef vs reference csv: max |Δ| {rp.get('max_abs_diff', math.nan):.2e}, "
          f"доля |Δ| > {args.repro_tol:g}: {rp.get('frac_over', math.nan):.2%}, нет в csv: "
          f"{rp.get('missing_in_reference', '-')};",
          f"- группы 12/13: картинок с другим разбиением {gr.get('n_images_not_12_13', '-')}; "
          f"меток, не совпавших с разбиением по csv: {gr.get('group_label_mismatch_vs_reference', '-')};",
          f"- сплит: A ∩ B = ∅, {checks['split']['n_A']} + {checks['split']['n_B']}; в v вошло "
          f"{checks['split'].get('vector_boxes')} боксов;",
          f"- self-test steer_sam_decoder: {'ok' if checks['self_test']['ok'] else 'FAIL'}"
          f"{' (пропущен)' if args.skip_self_test else ''}."]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _floats(s: str) -> list:
    return [float(x) for x in s.split(",") if x.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="user study FOR_TEST (images/, masks/, user_masks/)")
    p.add_argument("--use", default="m", help="user mask kinds: m (mobile), p, or pm")
    p.add_argument("--datasets", nargs="*", default=["Berkeley", "GrabCut"], choices=[d for d in DATASETS if d != "COCO"],
                   help="test subsets besides COCO half B")
    p.add_argument("--checkpoint_path", required=True, help="sam_vit_b_01ec64.pth")
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--site", default="L2.norm2", help="token-stream site of steer_sam_decoder.build_sites")
    p.add_argument("--alphas", type=_floats, default=_floats("1,0.5,0.1"),
                   help="steering strengths, all reported; dm at 1 moves the bottom50 mean onto the top50 mean")
    p.add_argument("--tokens", default="all", choices=["all", *SSD.TOK_ROWS], help="token group steered")
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0, help="random direction (as steer_sam_decoder) and bootstrap")
    p.add_argument("--reference_csv", default="", help="per-case SAM1 results, e.g. results/samfull_m.csv")
    p.add_argument("--cs_vectors", default="exp_res/steer_sam/vectors.npz",
                   help="steer_sam_decoder's vectors, for the cosine with L2.norm2/dm")
    p.add_argument("--batch", type=int, default=64, help="boxes per predict_torch call")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--updown_eps", type=float, default=0.01, help="IoU change counted as up / down")
    p.add_argument("--repro_tol", type=float, default=1e-3, help="|undef - reference| counted as a mismatch")
    p.add_argument("--repro_max_frac", type=float, default=0.01, help="max share of mismatched boxes")
    p.add_argument("--group_tol", type=float, default=1e-3, help="max |group mean undef - reference|")
    p.add_argument("--limit_images", type=int, default=0, help="per half / subset, for a smoke run; 0 = all")
    p.add_argument("--skip_self_test", action="store_true")
    p.add_argument("--out_dir", default="exp_res/steer_user_study")
    return p.parse_args(argv)


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
