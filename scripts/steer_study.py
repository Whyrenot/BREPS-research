"""
steer_study.py
==============
The full steering study on the user study (m + p) for SAM-1, SAM-2.1 and
SAM-3 with the same code: which site, which vector -- own, pooled, leave one
dataset out -- and how far to steer each box, out of fold on every image.

What changes against steer_universal.py (A / B halves, L2.norm3, 13 best - 12 worst)
  positives    D10 - D1: inside every image the best and the worst 10% of its
               annotations by multimask=False IoU (exactly 10% of them, the
               deciles of export_combined_deciles.py); --rules decile,halves adds
               the old 13 best - 12 worst (the runner's default), compared box by
               box in pairs.csv
  sites        L2.norm2, L2.norm3 (token stream, all tokens) and hyper (the
               outputs of all mask tokens' hypernetwork MLPs), --sites
  folds        5 image folds stratified by subset over all images (COCO too) for
               the vectors, the alpha regressor and the CatBoost selector after
               it: every image is steered with vectors that never saw its fold, so
               the whole user study is the test set, and the steering candidates
               can go into the pool without leaking
  regressor    CatBoost regresses the gain of every alpha per box, out of fold
  annotations  m + p (--use mp); every image is decoded in one batch of all its
               boxes as build_candidate_pool.py does, so undef / head-select are
               the pool's bit for bit (checked against --pool_dir) and the
               pool's grad / BoN candidates stay valid next to the new steering

VECTORS (per site, every row of the site's tensor)
  per image   D10 - D1 of the site's states; per subset the mean over its images
  an image of subset d in fold k gets (every subset's vector without fold k):
    own       d's
    pool      the mean of all subsets' (each subset weighted equally)
    lodo      the mean of the other subsets' -- d never seen
    coco      COCO's (the current practice's analogue)
    rand      a fixed random direction with pool's norm per row (the control)
    pool_nat  (--variants) the natural subsets'
  dose        every vector rescaled to the norm of its site's full pool vector
              (--norm pool; the factor of each full vector, its out-of-fold
              versions get the same one): alpha = 1 is one pool DM for every
              variant. --alphas for the token sites, --alphas_hyper for hyper

ALPHA POLICIES (analyse; per site x variant x output mode; out of fold)
  CatBoost    the gain of every alpha per box from features without the GT: the
              box (predicted IoU, stability, box fill, mask fraction, box size /
              aspect, annotation kind), the alpha (its value and what it does to
              the predicted IoU, the mask, the fill, the stability) and the box's
              whole response curve (predicted IoU and mask change at every alpha);
              the alpha with the largest predicted gain, none if <= 0. Trained on
              the other folds; lodo for subset d only on the other subsets' rows,
              each under its own lodo vector, so d is never seen
  alpha*      one alpha (or none), the best mean gain on the same training rows
  oracle      the subset's best single alpha in hindsight (the direction at its
              best dose); per box the best alpha (the ceiling) in per_box

STAGES (scripts/run_steer_study.sh runs them all)
  states   GPU, --shard i --n_shards n: one unsteered forward per image, the sites
           recorded -> the per-image DMs (shard_<i>/states.npz)
  vectors  CPU: every subset's vector per site, full and per fold -> vectors.npz,
           folds.json, vector_stats.csv, study.json
  steer    GPU, shards: every image, unsteered + site x variant x alpha, both modes
  merge    CPU: shards -> steer.npz, boxes.csv.gz, checks.json
  analyse  CPU + CatBoost (--task_type GPU): policies, tables, summary.md
  The GPU stages take the model's arguments; steer reads the study's sites,
  rules, variants and alphas from study.json, whatever its command line says.

CHECKS (checks.json; * stop the run)
  * own decoder pass = MaskDecoder.forward (the first image of every subset in a
    shard); * zero dose bit for bit at every site (states: first image of every
    subset in a shard; steer: every image); * every image of the plan in exactly
    one shard, the steer stage's plan = the states stage's; the states and the
    steer pass give the same undef; undef / head-select vs the pool's per_box
    (--pool_dir, reported); groups round(n / 2); TF32 matmuls off

OUTPUTS (--out_dir)
  study.json, folds.json, vectors.npz, vector_stats.csv
  steer.npz       per config "<rule>|<site>|<variant>": (boxes, 1 + alphas, 2 modes, 6):
                  iou, pred, stab, frac, chg, head; column 0 unsteered
  boxes.csv.gz    the boxes in steer.npz's order: keys, dataset, fold, group, rank
  per_box.csv.gz  every method's IoU and alpha per box (analyse)
  table.csv, pairs.csv, table_kind.csv, alpha_star.csv, geometry_<rule>_<site>.csv,
  summary.md, checks.json

    # one model by hand (the runner does this on every GPU)
    python scripts/steer_study.py --stage states --model_name SAM --root /.../FOR_TEST \\
        --checkpoint_path /.../sam_vit_b_01ec64.pth --shard 0 --n_shards 1 --out_dir exp_res/steer_study/SAM
    python scripts/steer_study.py --stage vectors --out_dir exp_res/steer_study/SAM
    python scripts/steer_study.py --stage steer --model_name SAM --root ... --checkpoint_path ... \\
        --shard 0 --n_shards 1 --out_dir exp_res/steer_study/SAM
    python scripts/steer_study.py --stage merge --out_dir exp_res/steer_study/SAM --pool_dir exp_res/pool_SAM_mp
    python scripts/steer_study.py --stage analyse --out_dir exp_res/steer_study/SAM --task_type GPU
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

from heatmaps.env_dispatch import maybe_dispatch_to_env  # noqa: E402

EPS = 1e-12
MODES = (0, 1)
Q = ("iou", "pred", "stab", "frac", "chg", "head")
KEYS = ["image_name", "kind", "user", "attempt"]
VARIANTS = ("own", "pool", "lodo", "coco", "rand", "pool_nat")
RULES = ("decile", "halves")
GROUPS = ("all", "bottom50", "top50")
POLICIES = ("CatBoost", "α*", "оракул α")


def _floats(s: str) -> list:
    return [float(x) for x in s.split(",") if x.strip()]


def _list(s: str) -> list:
    return [x.strip() for x in s.split(",") if x.strip()]


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_json_default), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def cos(a, b) -> float:
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), EPS))


def alphas_of(study: dict, site: str) -> list:
    return list(study["alphas_hyper"] if site == "hyper" else study["alphas"])


def cfg_name(rule: str, site: str, variant: str) -> str:
    return f"{rule}|{site}|{variant}"


def site_label(study: dict, rule: str, site: str) -> str:
    return site if len(study["rules"]) == 1 else f"{site}/{rule}"


# ---------------------------------------------------------------------------
# Plan, folds, deciles
# ---------------------------------------------------------------------------

def plan_of(index: dict, datasets, limit: int) -> list:
    """Every (subset, stem), subsets in order, stems sorted; --limit_images per subset."""
    plan = []
    for d in datasets:
        stems = sorted(index[d])
        plan += [[d, s] for s in (stems[:limit] if limit > 0 else stems)]
    return plan


def make_folds(plan: list, datasets, k: int, seed: int) -> dict:
    """stem -> fold: inside every subset its sorted stems permuted, position mod k
    (stratified by subset, no sklearn: the sam3 env has none)."""
    fold = {}
    for i, d in enumerate(datasets):
        stems = sorted(s for dd, s in plan if dd == d)
        perm = np.random.default_rng([seed, 17, i]).permutation(len(stems))
        for pos, j in enumerate(perm):
            fold[stems[j]] = int(pos % k)
    return fold


def case_rank(rows: list, u0: np.ndarray) -> np.ndarray:
    """rank of every annotation inside its image by undef IoU, ties by the case key
    (as selector_data.image_halves)."""
    case = np.asarray(["|".join(str(r[k]) for k in KEYS) for r in rows])
    order = np.lexsort((case, u0))
    rank = np.empty(len(u0), int)
    rank[order] = np.arange(len(u0))
    return rank


def decile_weights(rank: np.ndarray, n: np.ndarray, k: int = 10) -> np.ndarray:
    """export_combined_deciles.decile_weights: (boxes, k), every decile's weights
    sum to 1 inside an image (a decile is exactly 10% of its annotations)."""
    lo, hi = rank / n, (rank + 1) / n
    edges = np.arange(k + 1) / k
    w = np.minimum(hi[:, None], edges[None, 1:]) - np.maximum(lo[:, None], edges[None, :-1])
    return np.clip(w, 0, None) * k


def image_dm(X: np.ndarray, rule: str, W: np.ndarray, grp: np.ndarray) -> np.ndarray:
    """One image's DM of the site's states X (boxes, ...)."""
    if rule == "decile":
        w1, w10 = W[:, 0], W[:, 9]
        return np.tensordot(w10, X, 1) / w10.sum() - np.tensordot(w1, X, 1) / w1.sum()
    return X[grp == "top50"].mean(0) - X[grp == "bottom50"].mean(0)


# ---------------------------------------------------------------------------
# The model (GPU stages)
# ---------------------------------------------------------------------------

def new_checks() -> dict:
    return {"zero_dose": {"forwards": 0, "bit_exact": 0, "ok": True, "failures": []},
            "own_forward": {"compared": 0, "bit_exact": 0, "ok": True, "failures": []},
            "public_api": {"compared": 0, "max_abs_iou_diff": 0.0, "n_over_1e-4": 0, "where": []},
            "public_single": {"compared": 0, "max_abs_iou_diff": 0.0, "n_over_1e-3": 0, "where": []}}


def load_decoder(args, site_ids):
    import steer_models_scan as SMS
    from heatmaps.comp_hw_smoothed import load_model

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    predictor = load_model(model_name=args.model_name, model_type=args.model_type,
                           checkpoint=args.checkpoint_path, device=torch.device(args.device))
    for prm in predictor.model.parameters():
        prm.requires_grad_(False)
    # sam3 turns TF32 matmuls on at import; the scan and the pool run in fp32
    tf32 = {"matmul_after_load": bool(torch.backends.cuda.matmul.allow_tf32), "cudnn": bool(torch.backends.cudnn.allow_tf32)}
    if not args.keep_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
    tf32["matmul_run"] = bool(torch.backends.cuda.matmul.allow_tf32)
    dec = SMS.Decoder(predictor)
    if dec.training:
        raise SMS.CheckFailed("mask decoder is in training mode: the stability fallback would be off")
    by = {s.id: s for s in SMS.build_sites(dec)}
    sites = []
    for sid in site_ids:
        s = by.get(sid)
        if s is None or s.kind not in ("tok", "vec"):
            raise SystemExit(f"--sites {sid}: a token-stream site (L1/L2.norm1-3, final_norm) or hyper is needed")
        sites.append(s)
    return predictor, dec, sites, SMS.Runner(dec, sites, args.batch), tf32


def study_index(root: Path, use: str, datasets):
    import steer_user_study as SUS
    index = SUS.index_study(root, use, datasets)
    for d in datasets:
        if not index[d]:
            raise SystemExit(f"no user masks for {d} under {root}")
    return index


# ---------------------------------------------------------------------------
# Stage 1: states
# ---------------------------------------------------------------------------

def stage_states(args) -> int:
    import steer_models_scan as SMS
    import steer_user_study as SUS

    out = Path(args.out_dir)
    sh = out / f"shard_{args.shard}"
    sh.mkdir(parents=True, exist_ok=True)
    checks = new_checks()
    try:
        t0 = time.time()
        _, dec, sites, runner, tf32 = load_decoder(args, args.sites)
        checks["tf32"] = tf32
        root = Path(args.root)
        datasets = list(dict.fromkeys(args.datasets or SUS.DATASETS))
        if "COCO" not in datasets and "coco" in args.variants:
            raise SystemExit("the coco variant needs COCO among --datasets")
        index = study_index(root, args.use, datasets)
        plan = plan_of(index, datasets, args.limit_images)
        mine = list(range(args.shard, len(plan), args.n_shards))
        study = {"script": "steer_study.py", "created": time.strftime("%Y-%m-%d %H:%M"), "model_name": args.model_name,
                 "checkpoint": args.checkpoint_path, "root": str(root), "use": args.use, "datasets": datasets,
                 "medical": [d for d in datasets if d in SUS.MEDICAL],
                 "natural": [d for d in datasets if d not in SUS.MEDICAL], "sites": args.sites, "rules": args.rules,
                 "variants": args.variants, "alphas": args.alphas, "alphas_hyper": args.alphas_hyper,
                 "folds": args.folds, "seed": args.seed, "norm": args.norm, "batch": args.batch,
                 "limit_images": args.limit_images, "n_shards": args.n_shards, "plan": plan,
                 "layout": dec.layout(), "site_kinds": {s.id: s.kind for s in sites}}
        write_json(sh / "study.json", study)
        print(f"{args.model_name} states, shard {args.shard}/{args.n_shards}: {len(mine)} of {len(plan)} images; sites "
              f"{args.sites}; rules {args.rules}; TF32 matmul {tf32['matmul_after_load']} -> {tf32['matmul_run']}", flush=True)
        dms = {(r, s.id): [] for r in args.rules for s in sites}
        hn = {s.id: [0.0, 0] for s in sites}
        stems, undef, counts, seen = [], [], [], set()
        for n_done, pi in enumerate(mine):
            d, stem = plan[pi]
            item = SMS.load_item(dec, root, d, "-", stem, index[d], -1)
            rec = SMS.Recorder(sites)
            low0, iou0 = runner.forward(item["b"], record=rec.fns())
            base = SMS.score_modes(dec, low0, iou0, item["gt"])
            if d not in seen:
                seen.add(d)
                SMS.check_official(dec, item, low0, iou0, base, checks)
                SMS.check_zero(runner, item["b"], low0, iou0, [s.id for s in sites], checks, f"{d}/{stem}")
            u0 = base[0][0]
            n = len(u0)
            W = decile_weights(case_rank(item["rows"], u0), np.full(n, n, float))
            grp = SUS.assign_groups(pd.DataFrame(item["rows"]).assign(undef_iou=u0)).to_numpy()
            for s in sites:
                X = rec.raw(s.id).float().cpu().numpy()
                hn[s.id][0] += float(np.linalg.norm(X.reshape(n, -1), axis=1).sum())
                hn[s.id][1] += n
                for r in args.rules:
                    dms[(r, s.id)].append(image_dm(X, r, W, grp))
            stems.append(stem)
            undef.append(u0.astype(np.float32))
            counts.append(n)
            print(f"[states {d}] [{n_done + 1}/{len(mine)}] {stem}: {n} boxes, undef {u0.mean():.4f}, head-select "
                  f"{base[1][0].mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
            del item, rec
        npz = {"plan_idx": np.asarray(mine, int), "stems": np.asarray(stems), "counts": np.asarray(counts, int),
               "undef": np.concatenate(undef) if undef else np.zeros(0, np.float32)}
        for (r, sid), v in dms.items():
            if v:
                npz[f"dm/{r}/{sid}"] = np.stack(v).astype(np.float32)
        for sid, (hs, hc) in hn.items():
            npz[f"hn/{sid}"] = np.asarray([hs, hc], float)
        np.savez_compressed(sh / "states.npz", **npz)
        print(f"states done in {(time.time() - t0) / 60:.1f} min -> {sh}", flush=True)
        return 0
    finally:
        write_json(sh / "states_checks.json", checks)


# ---------------------------------------------------------------------------
# Stage 2: vectors
# ---------------------------------------------------------------------------

def variant_keys(variants, datasets) -> list:
    out = []
    for v in variants:
        if v in ("own", "lodo"):
            out += [f"{v}:{d}" for d in datasets]
        else:
            out.append(v)
    return out


def raw_vector(key: str, ds: dict, natural, u: np.ndarray) -> np.ndarray:
    import steer_universal as U
    if key.startswith("own:"):
        return ds[key.split(":", 1)[1]]
    return U.combine(key, ds, natural, u)


def sum_checks(parts: list) -> dict:
    """The shards' check dicts added up (counters summed, ok and-ed, lists joined)."""
    out = {}
    for p in parts:
        for name, c in p.items():
            if not isinstance(c, dict):
                continue
            o = out.setdefault(name, {})
            for k, v in c.items():
                if isinstance(v, bool):
                    o[k] = o.get(k, True) and v
                elif isinstance(v, (int, float)) and k.startswith("max"):
                    o[k] = max(o.get(k, 0.0), v)
                elif isinstance(v, (int, float)):
                    o[k] = o.get(k, 0) + v
                elif isinstance(v, list):
                    o[k] = o.get(k, []) + v
                else:
                    o[k] = v
    return out


def stage_vectors(args) -> int:
    out = Path(args.out_dir)
    shards = sorted(out.glob("shard_*"), key=lambda p: int(p.name.split("_")[1]))
    shards = [p for p in shards if (p / "states.npz").is_file()]
    if not shards:
        raise SystemExit(f"no shard_*/states.npz under {out}")
    studies = [read_json(p / "study.json") for p in shards]
    study = dict(studies[0])
    same = ("model_name", "use", "datasets", "sites", "rules", "variants", "alphas", "alphas_hyper", "folds", "seed",
            "norm", "plan", "n_shards")
    for s in studies[1:]:
        bad = [k for k in same if s[k] != study[k]]
        if bad:
            raise SystemExit(f"shards disagree on {bad}")
    if len(shards) != study["n_shards"]:
        raise SystemExit(f"{len(shards)} shards with states, the study has {study['n_shards']}")
    Z = [np.load(p / "states.npz") for p in shards]
    plan = study["plan"]
    pidx = np.concatenate([z["plan_idx"] for z in Z])
    coverage = {"images": len(plan), "in_shards": int(len(pidx)), "unique": int(len(set(pidx.tolist()))),
                "ok": sorted(pidx.tolist()) == list(range(len(plan)))}
    if not coverage["ok"]:
        raise SystemExit(f"the shards do not cover the plan once: {coverage}")
    order = np.argsort(pidx)
    datasets, natural, rules, sites = study["datasets"], study["natural"], study["rules"], study["sites"]
    sets = np.asarray([plan[i][0] for i in pidx[order]])
    stems = np.asarray([plan[i][1] for i in pidx[order]])
    fold_of = make_folds(plan, datasets, study["folds"], study["seed"])
    fold = np.asarray([fold_of[s] for s in stems])
    counts = np.concatenate([z["counts"] for z in Z])[order]
    undef_parts = []
    for z in Z:
        c = z["counts"]
        undef_parts += np.split(z["undef"], np.cumsum(c)[:-1]) if len(c) else []
    undef = np.concatenate([undef_parts[i] for i in order])
    h_norm = {sid: float(sum(z[f"hn/{sid}"][0] for z in Z) / max(sum(z[f"hn/{sid}"][1] for z in Z), 1)) for sid in sites}
    k = study["folds"]
    npz = {"stems": stems, "datasets": sets, "fold": fold, "counts": counts, "undef_states": undef}
    scale, empty, stats = {}, [], []
    for si, sid in enumerate(sites):
        for rule in rules:
            D = np.concatenate([z[f"dm/{rule}/{sid}"] for z in Z if f"dm/{rule}/{sid}" in z.files])[order]
            if f"u/{sid}" not in npz:
                u = np.random.default_rng([study["seed"], 202, si]).standard_normal(D.shape[1:]).astype(np.float32)
                npz[f"u/{sid}"] = u / np.linalg.norm(u, axis=-1, keepdims=True)
            u = npz[f"u/{sid}"]
            full = {}
            for d in datasets:
                sel = sets == d
                full[d] = D[sel].mean(0).astype(np.float32)
                npz[f"own/{rule}/{sid}/{d}"], npz[f"dimg/{rule}/{sid}/{d}"] = full[d], D[sel]
                for j in range(k):
                    fs = sel & (fold != j)
                    if not fs.any():
                        if args.allow_empty_fold or study["limit_images"] > 0:
                            empty.append(f"{rule}/{sid}/{d}/fold{j}")
                            fs = sel
                        else:
                            raise SystemExit(f"{d}: no image outside fold {j}")
                    npz[f"own/{rule}/{sid}/{d}/fold{j}"] = D[fs].mean(0).astype(np.float32)
            pool = raw_vector("pool", full, natural, u)
            target = float(np.linalg.norm(pool))
            for key in variant_keys(study["variants"], datasets):
                v = raw_vector(key, full, natural, u)
                n = float(np.linalg.norm(v))
                scale[f"{rule}|{sid}|{key}"] = target / max(n, EPS) if study["norm"] == "pool" else 1.0
                sub = key.split(":", 1)[1] if ":" in key else ""
                stats.append({"rule": rule, "site": sid, "vector": key, "raw_norm": n,
                              "scale": scale[f"{rule}|{sid}|{key}"], "dose_alpha1": target / max(h_norm[sid], EPS)
                              if study["norm"] == "pool" else n / max(h_norm[sid], EPS),
                              "cos_pool": cos(v, pool), "cos_coco": cos(v, full["COCO"]) if "COCO" in full else math.nan,
                              "cos_own_subset": cos(v, full[sub]) if sub else math.nan})
    np.savez_compressed(out / "vectors.npz", **npz)
    pd.DataFrame(stats).to_csv(out / "vector_stats.csv", index=False)
    study.update({"folds_of": fold_of, "scale": scale, "h_norm": h_norm, "empty_folds": empty,
                  "n_images": len(plan), "n_boxes": int(counts.sum())})
    study.pop("n_shards", None)
    write_json(out / "study.json", study)
    write_json(out / "folds.json", {"k": k, "seed": study["seed"], "folds": fold_of})
    sc = sum_checks([read_json(p / "states_checks.json") for p in shards if (p / "states_checks.json").is_file()])
    sc["coverage"] = coverage
    write_json(out / "checks_states.json", sc)
    print(f"vectors: {len(plan)} images, {int(counts.sum())} boxes, {len(sites)} sites x {len(rules)} rules; folds "
          f"{np.bincount(fold).tolist()}; empty folds {len(empty)}; zero dose {sc.get('zero_dose', {}).get('bit_exact')}/"
          f"{sc.get('zero_dose', {}).get('forwards')}, own forward {sc.get('own_forward', {}).get('bit_exact')}/"
          f"{sc.get('own_forward', {}).get('compared')}", flush=True)
    return 0


class StudyVectors:
    """vectors.npz -> the variants of an image (subset d, fold k), rescaled."""

    def __init__(self, out: Path, study: dict):
        self.z = np.load(out / "vectors.npz")
        self.study = study

    def for_image(self, rule: str, site: str, d: str, k: int) -> "OrderedDict[str, np.ndarray]":
        s = self.study
        ds = {e: self.z[f"own/{rule}/{site}/{e}/fold{k}"] for e in s["datasets"]}
        u = self.z[f"u/{site}"]
        out = OrderedDict()
        for v in s["variants"]:
            key = f"{v}:{d}" if v in ("own", "lodo") else v
            out[v] = (raw_vector(key, ds, s["natural"], u) * s["scale"][f"{rule}|{site}|{key}"]).astype(np.float32)
        return out


# ---------------------------------------------------------------------------
# Stage 3: steer
# ---------------------------------------------------------------------------

def stage_steer(args) -> int:
    import steer_models_scan as SMS
    import steer_universal as U
    import steer_user_study as SUS
    from heatmaps.comp_hw_smoothed import get_original_size
    from segment_anything.utils.transforms import ResizeLongestSide

    out = Path(args.out_dir)
    study = read_json(out / "study.json")
    sh = out / f"shard_{args.shard}"
    sh.mkdir(parents=True, exist_ok=True)
    checks = new_checks()
    try:
        t0 = time.time()
        _, dec, sites, runner, tf32 = load_decoder(args, study["sites"])
        checks["tf32"] = tf32
        if dec.layout()["token_names"] != study["layout"]["token_names"]:
            raise SMS.CheckFailed("the model is not the one of the states stage (token layout)")
        root = Path(args.root)
        index = study_index(root, study["use"], study["datasets"])
        plan = plan_of(index, study["datasets"], study["limit_images"])
        checks["plan"] = {"images": len(plan), "same_as_states": plan == study["plan"]}
        if plan != study["plan"]:
            raise SMS.CheckFailed("the images of --root are not the states stage's")
        vec = StudyVectors(out, study)
        z = vec.z
        st_undef = np.split(z["undef_states"], np.cumsum(z["counts"])[:-1])
        cfgs = [(r, s, v) for r in study["rules"] for s in sites for v in study["variants"]]
        mine = list(range(args.shard, len(plan), args.n_shards))
        log_a = {s.id: alphas_of(study, s.id).index(min(alphas_of(study, s.id), key=lambda x: abs(x - args.log_alpha)))
                 + 1 for s in sites}
        print(f"{args.model_name} steer, shard {args.shard}/{args.n_shards}: {len(mine)} images; {len(cfgs)} configs, "
              f"{sum(len(alphas_of(study, s.id)) for _, s, _ in cfgs)} steered forwards per image", flush=True)
        arrs = {cfg_name(r, s.id, v): [] for r, s, v in cfgs}
        boxes, cache = [], {}
        checks["baseline_repeat"] = {"images": 0, "max_abs_iou_diff": 0.0}
        for n_done, pi in enumerate(mine):
            d, stem = plan[pi]
            k = study["folds_of"][stem]
            item = SMS.load_item(dec, root, d, "-", stem, index[d], k)
            b, gt = item["b"], item["gt"]
            low0, iou0 = runner.forward(b)
            SMS.check_zero(runner, b, low0, iou0, [s.id for s in sites], checks, f"{d}/{stem}")
            base, m0 = {}, {}
            for mm in MODES:
                base[mm], m0[mm] = U.describe(dec, low0, iou0, mm, gt, None)
            dd = float(np.abs(st_undef[pi] - base[0]["iou"]).max())
            r_ = checks["baseline_repeat"]
            r_["images"] += 1
            r_["max_abs_iou_diff"] = max(r_["max_abs_iou_diff"], dd)
            N = len(base[0]["iou"])
            log = {}
            for r, s, v in cfgs:
                key = (r, s.id, d, k)
                if key not in cache:
                    cache[key] = {n: torch.as_tensor(x, device=b.device) for n, x in vec.for_image(r, s.id, d, k).items()}
                w = cache[key][v]
                al = alphas_of(study, s.id)
                A = np.zeros((N, len(al) + 1, len(MODES), len(Q)), np.float32)
                for mm in MODES:
                    A[:, 0, mm] = np.stack([base[mm][q] for q in Q], 1)
                for j, a in enumerate(al, start=1):
                    edit = SMS.edit_vec(a * w) if s.kind == "vec" else SMS.edit_rows(a * w)
                    low_all, iou_all = runner.forward(b, edits={s.id: edit})
                    for mm in MODES:
                        rr, _ = U.describe(dec, low_all, iou_all, mm, gt, m0[mm])
                        A[:, j, mm] = np.stack([rr[q] for q in Q], 1)
                arrs[cfg_name(r, s.id, v)].append(A)
                if v in ("pool", "lodo", "own"):
                    log[f"{s.id} {v}"] = A[:, log_a[s.id], 0, 0].mean()
            u0 = base[0]["iou"]
            hw = get_original_size(dec.p)
            ih, iw = ResizeLongestSide.get_preprocess_shape(hw[0], hw[1], 1024)
            bx = item["boxes"]
            bw, bh = (bx[:, 2] - bx[:, 0]).clip(min=1e-3), (bx[:, 3] - bx[:, 1]).clip(min=1e-3)
            grp = SUS.assign_groups(pd.DataFrame(item["rows"]).assign(undef_iou=u0)).to_numpy()
            rank = case_rank(item["rows"], u0)
            for i, row in enumerate(item["rows"]):
                boxes.append({"plan_idx": pi, "box_idx": i, "dataset": d, **{kk: row[kk] for kk in KEYS}, "fold": k,
                              "group": grp[i], "rank": int(rank[i]), "n_ann": N,
                              "box_frac": float(bw[i] * bh[i] / float(ih * iw)), "log_aspect": float(np.log(bw[i] / bh[i]))})
            print(f"[steer {d}] [{n_done + 1}/{len(mine)}] {stem}: {N} boxes, undef {u0.mean():.4f}, hsel "
                  f"{base[1]['iou'].mean():.4f}; mm0 " + ", ".join(f"{n} {x:.4f}" for n, x in log.items())
                  + f" ({time.time() - t0:.0f}s)", flush=True)
            del item
        np.savez_compressed(sh / "steer.npz", **{c: (np.concatenate(v) if v else np.zeros((0,), np.float32))
                                                 for c, v in arrs.items()})
        pd.DataFrame(boxes).to_csv(sh / "boxes.csv.gz", index=False)
        print(f"steer done in {(time.time() - t0) / 60:.1f} min -> {sh}", flush=True)
        return 0
    finally:
        write_json(sh / "steer_checks.json", checks)


# ---------------------------------------------------------------------------
# Stage 4: merge
# ---------------------------------------------------------------------------

def group_check(B: pd.DataFrame) -> dict:
    c = B.groupby(["image_name", "group"]).size().unstack(fill_value=0)
    n = c.get("bottom50", 0) + c.get("top50", 0)
    off = int((c.get("bottom50", 0) != (n * 0.5).round().astype(int)).sum())
    return {"images": int(len(c)), "n_images_off": off,
            "annotations_per_image": {int(a): int(b) for a, b in n.value_counts().sort_index().items()}, "ok": off == 0}


def stage_merge(args) -> int:
    out = Path(args.out_dir)
    study = read_json(out / "study.json")
    shards = sorted((p for p in out.glob("shard_*") if (p / "steer.npz").is_file()), key=lambda p: int(p.name.split("_")[1]))
    if not shards:
        raise SystemExit(f"no shard_*/steer.npz under {out}")
    Bs = [pd.read_csv(p / "boxes.csv.gz", dtype={k: str for k in KEYS}) for p in shards]
    Zs = [np.load(p / "steer.npz") for p in shards]
    B = pd.concat(Bs, ignore_index=True)
    B[KEYS] = B[KEYS].fillna("")
    order = np.lexsort((B.box_idx.to_numpy(), B.plan_idx.to_numpy()))
    B = B.iloc[order].reset_index(drop=True)
    names = list(Zs[0].files)
    Z = {c: np.concatenate([z[c] for z in Zs if z[c].size])[order] for c in names}
    imgs = sorted(set(B.plan_idx.tolist()))
    checks = read_json(out / "checks_states.json") if (out / "checks_states.json").is_file() else {}
    st = sum_checks([read_json(p / "steer_checks.json") for p in shards if (p / "steer_checks.json").is_file()])
    checks["steer"] = st
    dup = int(B.duplicated(["plan_idx", "box_idx"]).sum())       # a stale shard of an earlier run
    want = [cfg_name(r, s, v) for r in study["rules"] for s in study["sites"] for v in study["variants"]]
    checks["steer_coverage"] = {"images": len(study["plan"]), "merged": len(imgs), "duplicate_boxes": dup,
                                "configs_missing": sorted(set(want) - set(names)),
                                "ok": imgs == list(range(len(study["plan"]))) and dup == 0 and set(want) <= set(names)}
    checks["groups"] = group_check(B)
    first = Z[names[0]]
    B["undef"], B["hsel"] = first[:, 0, 0, 0], first[:, 0, 1, 0]
    same = {c: bool(np.array_equal(Z[c][:, 0], first[:, 0])) for c in names}
    checks["base_same_in_every_config"] = all(same.values())
    if args.pool_dir and (Path(args.pool_dir) / "per_box.csv.gz").is_file():
        P = pd.read_csv(Path(args.pool_dir) / "per_box.csv.gz", dtype={k: str for k in KEYS})
        P[KEYS] = P[KEYS].fillna("")
        J = B[KEYS + ["undef", "hsel"]].merge(P[KEYS + ["undef_iou", "hsel_iou"]], on=KEYS, how="left")
        r = {"pool_dir": args.pool_dir, "boxes": int(len(J)), "matched": int(J.undef_iou.notna().sum())}
        for a, b_ in (("undef", "undef_iou"), ("hsel", "hsel_iou")):
            dlt = (J[a] - J[b_]).abs()
            r[f"{a}_max_abs_diff"], r[f"{a}_n_over_1e-6"] = float(dlt.max()), int((dlt > 1e-6).sum())
        checks["vs_pool"] = r
    else:
        checks["vs_pool"] = {"pool_dir": args.pool_dir, "available": False}
    np.savez_compressed(out / "steer.npz", **Z)
    B.to_csv(out / "boxes.csv.gz", index=False, float_format="%.6f")
    write_json(out / "checks.json", checks)
    vp = checks["vs_pool"]
    print(f"merge: {len(imgs)} images, {len(B)} boxes, {len(names)} configs; zero dose {st.get('zero_dose', {}).get('bit_exact')}/"
          f"{st.get('zero_dose', {}).get('forwards')}; baseline repeat {st.get('baseline_repeat', {}).get('max_abs_iou_diff')}; "
          f"groups off {checks['groups']['n_images_off']}; vs pool undef "
          f"{vp.get('undef_max_abs_diff', 'n/a')} ({vp.get('undef_n_over_1e-6', '-')} boxes > 1e-6)", flush=True)
    bad = [k for k in ("steer_coverage", "groups") if not checks[k]["ok"]]
    if bad or not st.get("zero_dose", {}).get("ok", True):
        print(f"[check failed] {bad or 'zero dose'}", file=sys.stderr)
        return 2
    return 0


# ---------------------------------------------------------------------------
# Stage 5: analyse
# ---------------------------------------------------------------------------

def box_parts(A: np.ndarray, mm: int, B: pd.DataFrame) -> dict:
    """One config and mode -> the arrays the policies use."""
    X = A[:, :, mm, :]
    iou, pred, stab, frac, chg = (X[..., Q.index(q)] for q in ("iou", "pred", "stab", "frac", "chg"))
    bf = B.box_frac.to_numpy(float)
    fill = frac / np.maximum(bf, 1e-6)[:, None]
    box = np.column_stack([pred[:, 0], stab[:, 0], fill[:, 0], frac[:, 0], bf, B.log_aspect.to_numpy(float),
                           (B.kind == "m").to_numpy(float)])
    curve = np.column_stack([pred[:, 1:] - pred[:, :1], chg[:, 1:]])
    return {"iou": iou, "pred": pred, "stab": stab, "fill": fill, "chg": chg, "box": box, "curve": curve}


def long_table(p: dict, idx: np.ndarray, alphas) -> tuple:
    """(boxes x alphas) rows, alpha-major: box + alpha + curve features -> gain."""
    Xs, ys = [], []
    for j, a in enumerate(alphas, start=1):
        Xs.append(np.column_stack([p["box"][idx], np.full(len(idx), a), p["pred"][idx, j] - p["pred"][idx, 0],
                                   p["chg"][idx, j], p["fill"][idx, j] - p["fill"][idx, 0],
                                   p["stab"][idx, j] - p["stab"][idx, 0], p["curve"][idx]]))
        ys.append(p["iou"][idx, j] - p["iou"][idx, 0])
    return np.vstack(Xs).astype(np.float32), np.concatenate(ys).astype(np.float32)


def best_column(W: np.ndarray) -> int:
    """alpha* = the column (0 = none) with the best mean gain; ties -> the smaller alpha."""
    return int(np.argmax(np.round((W - W[:, :1]).mean(0), 12)))


def policies(p: dict, alphas, B: pd.DataFrame, lodo: bool, args, cache_dir: Path, tag: str) -> dict:
    """-> {policy: column per box}, the chosen alpha* per fit, out of fold."""
    from catboost import CatBoostRegressor

    W = p["iou"]
    N = len(W)
    fold, sets = B.fold.to_numpy(), B.dataset.to_numpy()
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(W).tobytes())
    h.update(np.ascontiguousarray(p["curve"]).tobytes())
    h.update(json.dumps([tag, list(alphas), lodo, args.cb_iterations, args.cb_depth, args.cb_lr, args.seed,
                         fold.tolist()]).encode())
    cpath = cache_dir / f"{h.hexdigest()[:24]}.npz"
    if cpath.is_file():
        c = np.load(cpath)
        return {"CatBoost": c["cb"], "α*": c["astar"], "fits": json.loads(str(c["fits"]))}
    cb, astar, fits = np.zeros(N, int), np.zeros(N, int), []
    units = [(k, None) for k in sorted(set(fold.tolist()))] if not lodo else \
        [(k, d) for k in sorted(set(fold.tolist())) for d in dict.fromkeys(sets.tolist())]
    for k, d in units:
        te = (fold == k) if d is None else (fold == k) & (sets == d)
        tr = (fold != k) if d is None else (fold != k) & (sets != d)
        if not te.any():
            continue
        tr_i, te_i = np.where(tr)[0], np.where(te)[0]
        j = best_column(W[tr_i])
        astar[te_i] = j
        X, y = long_table(p, tr_i, alphas)
        kw = dict(iterations=args.cb_iterations, depth=args.cb_depth, learning_rate=args.cb_lr, loss_function="RMSE",
                  random_seed=args.seed, verbose=False, allow_writing_files=False, task_type=args.task_type)
        if args.task_type == "GPU":
            kw["devices"] = args.devices
        else:
            kw["thread_count"] = args.threads
        m = CatBoostRegressor(**kw).fit(X, y)
        Xt, _ = long_table(p, te_i, alphas)
        P = m.predict(Xt).reshape(len(alphas), len(te_i)).T
        cb[te_i] = np.where(P.max(1) > 0, P.argmax(1) + 1, 0)
        fits.append({"fold": int(k), "dataset": d or "", "train_boxes": int(len(tr_i)), "alpha_star": ([0.0] + list(alphas))[j]})
    np.savez_compressed(cpath, cb=cb, astar=astar, fits=json.dumps(fits))
    return {"CatBoost": cb, "α*": astar, "fits": fits}


def stage_analyse(args) -> int:
    import steer_gate as G
    import steer_universal as U

    t0 = time.time()
    out = Path(args.out_dir)
    study = read_json(out / "study.json")
    checks = read_json(out / "checks.json") if (out / "checks.json").is_file() else {}
    B = pd.read_csv(out / "boxes.csv.gz", dtype={k: str for k in KEYS})
    B[KEYS] = B[KEYS].fillna("")
    Z = np.load(out / "steer.npz")
    datasets, natural, rules, sites = study["datasets"], study["natural"], study["rules"], study["sites"]
    variants = study["variants"]
    cache = out / "policy_cache"
    cache.mkdir(exist_ok=True)
    N = len(B)
    ar = np.arange(N)
    u, hsel = B.undef.to_numpy(float), B.hsel.to_numpy(float)
    res, alpha_cols, fit_rows = OrderedDict(), OrderedDict(), []
    print(f"analyse: {N} boxes, {len(Z.files)} configs, CatBoost {args.cb_iterations} it. on {args.task_type}", flush=True)
    for rule in rules:
        for site in sites:
            al = alphas_of(study, site)
            grid = np.asarray([0.0] + list(al))
            for v in variants:
                A = Z[cfg_name(rule, site, v)]
                for mm in MODES:
                    p = box_parts(A, mm, B)
                    W = p["iou"]
                    pol = policies(p, al, B, v == "lodo", args, cache, f"{rule}|{site}|{v}|{mm}")
                    for f in pol["fits"]:
                        fit_rows.append({"rule": rule, "site": site, "variant": v, "mode": f"mm{mm}", **f})
                    ora = np.zeros(N, int)
                    for d in datasets:
                        s = (B.dataset == d).to_numpy()
                        ora[s] = best_column(W[s])
                    lab = f"mm{mm} · {site_label(study, rule, site)} · {v}"
                    for name, col in (("CatBoost", pol["CatBoost"]), ("α*", pol["α*"]), ("оракул α", ora)):
                        res[f"{lab} · {name}"] = W[ar, col]
                        alpha_cols[f"{lab} · {name} [α]"] = grid[col]
                    res[f"{lab} · оракул бокса"] = W.max(1)
                    print(f"  {lab}: CatBoost {W[ar, pol['CatBoost']].mean():.4f}, α* {W[ar, pol['α*']].mean():.4f}, "
                          f"oracle α {W[ar, ora].mean():.4f}, undef {u.mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
    PB = pd.concat([B[["dataset", *KEYS, "fold", "group", "rank", "n_ann", "undef", "hsel"]].reset_index(drop=True),
                    pd.DataFrame(res), pd.DataFrame(alpha_cols)], axis=1)
    PB.to_csv(out / "per_box.csv.gz", index=False, float_format="%.6f")
    pd.DataFrame(fit_rows).to_csv(out / "alpha_star.csv", index=False)

    methods = [m for m in res if not m.endswith("оракул бокса")]
    per, per2 = OrderedDict(), OrderedDict()
    ref = "L2.norm3" if "L2.norm3" in sites else sites[0]
    for d in datasets:
        s = (B.dataset == d).to_numpy()
        Dt = B[s][["image_name", "group"]].reset_index(drop=True)
        rows = OrderedDict([("undef", (u[s], u[s])), ("head-select", (hsel[s], u[s]))])
        rows.update((m, (res[m][s], u[s])) for m in methods)
        per[d] = (Dt, rows)
        r2 = OrderedDict()
        for rule in rules:
            for mm in MODES:
                for site in sites:
                    L = f"mm{mm} · {site_label(study, rule, site)}"
                    for x, y in (("pool", "own"), ("lodo", "own"), ("lodo", "coco"), ("pool", "coco"), ("pool", "rand"),
                                 ("pool_nat", "pool")):
                        if x in variants and y in variants:
                            r2[f"{L} · {x} − {y} · CatBoost"] = (res[f"{L} · {x} · CatBoost"][s], res[f"{L} · {y} · CatBoost"][s])
                    for v in ("pool", "lodo", "own"):
                        if v in variants:
                            r2[f"{L} · {v} · CatBoost − α*"] = (res[f"{L} · {v} · CatBoost"][s], res[f"{L} · {v} · α*"][s])
                    if mm == 1 and "pool" in variants:
                        r2[f"{L} · pool · CatBoost − head-select"] = (res[f"{L} · pool · CatBoost"][s], hsel[s])
                for site in sites:
                    if site == ref:
                        continue
                    for v in ("own", "pool", "lodo"):
                        if v in variants:
                            a_ = f"mm{mm} · {site_label(study, rule, site)} · {v} · CatBoost"
                            b_ = f"mm{mm} · {site_label(study, rule, ref)} · {v} · CatBoost"
                            r2[f"mm{mm} · {v}: {site_label(study, rule, site)} − {site_label(study, rule, ref)} · CatBoost"] = \
                                (res[a_][s], res[b_][s])
        if "decile" in rules and "halves" in rules:
            for mm in MODES:
                for site in sites:
                    for v in ("own", "pool", "lodo"):
                        if v in variants:
                            a_, b_ = (f"mm{mm} · {site_label(study, r, site)} · {v} · CatBoost" for r in ("decile", "halves"))
                            r2[f"mm{mm} · {site} · {v}: D10−D1 − половины · CatBoost"] = (res[a_][s], res[b_][s])
        per2[d] = (Dt, r2)
    T = G.score_blocks(per, args)
    T.to_csv(out / "table.csv", index=False)
    P = G.score_blocks(per2, args)
    P.to_csv(out / "pairs.csv", index=False)
    kind_rows = []
    for kind in sorted(B.kind.unique()):
        for m in ["head-select", *methods]:
            x = hsel if m == "head-select" else res[m]
            pct = [(x[(B.dataset == d).to_numpy() & (B.kind == kind).to_numpy()].mean()
                    / u[(B.dataset == d).to_numpy() & (B.kind == kind).to_numpy()].mean() - 1) * 100 for d in datasets]
            kind_rows.append({"kind": kind, "method": m, "pct_mean_subsets": float(np.mean(pct)),
                              **{d: p_ for d, p_ in zip(datasets, pct)}})
    TK = pd.DataFrame(kind_rows)
    TK.to_csv(out / "table_kind.csv", index=False)
    vz = np.load(out / "vectors.npz")
    geo = {}
    for rule in rules:
        for site in sites:
            like = {}
            for d in datasets:
                like[f"own/{d}"], like[f"dimg/{d}"] = vz[f"own/{rule}/{site}/{d}"], vz[f"dimg/{rule}/{site}/{d}"]
            C, Gm = U.geometry(like, datasets, natural, args.n_split, args.seed)
            C.to_csv(out / f"cos_own_{rule}_{site}.csv")
            Gm.to_csv(out / f"geometry_{rule}_{site}.csv", index=False)
            geo[(rule, site)] = (C, Gm)
    vs = pd.read_csv(out / "vector_stats.csv")
    text = summary_md(T, P, TK, pd.DataFrame(fit_rows), PB, geo, vs, study, checks)
    (out / "summary.md").write_text(text, encoding="utf-8")
    enc = sys.stdout.encoding or "utf-8"
    print("\n" + text.encode(enc, "replace").decode(enc))
    print(f"wrote {out} ({(time.time() - t0) / 60:.1f} min)")
    return 0


# ---------------------------------------------------------------------------
# summary.md
# ---------------------------------------------------------------------------

VLABEL = {"own": "свой вектор набора", "pool": "общий: среднее векторов всех наборов",
          "lodo": "без своего набора (leave one dataset out): домен не виден",
          "coco": "вектор COCO (аналог текущей практики)", "rand": "случайное направление с нормами pool (контроль)",
          "pool_nat": "общий вектор натуральных наборов"}


def summary_md(T, P, TK, A, PB, geo, vs, study, checks) -> str:
    import steer_universal as U
    datasets, sites, rules, variants = study["datasets"], study["sites"], study["rules"], study["variants"]
    M = U._means(T, datasets)
    n_img = study.get("n_images")
    L = [f"# Steering, полное исследование: {study['model_name']}, разметка `{study['use']}`", "",
         f"{n_img} картинок, {study.get('n_boxes')} боксов, {study['folds']} фолдов по картинкам (стратификация по набору). "
         f"Векторы — D10 − D1 внутри картинки ({', '.join(rules)}), узлы {', '.join(f'`{s}`' for s in sites)}; "
         "картинка стирится векторами, посчитанными без её фолда, регрессор α (CatBoost) обучен на других фолдах: "
         "**весь user study — тест**. Все векторы приведены к норме общего вектора своего узла (α = 1 — один "
         f"общий DM); α ∈ {{{', '.join(f'{a:g}' for a in study['alphas'])}}}"
         + (f", для hyper {{{', '.join(f'{a:g}' for a in study['alphas_hyper'])}}}" if "hyper" in sites else "") + ".",
         "", "Ячейка — прирост IoU к undef (multimask=False, без steering) в %; среднее по наборам, каждый с равным весом; "
         "b50 / t50 — худшая / лучшая половина аннотаций картинки по undef. mm0 — multimask=False, mm1 — multimask "
         "с выбором головы (его α = 0 — head-select).", "", "**Векторы:**", ""]
    L += [f"- `{v}` — {VLABEL[v]}" for v in variants]
    L += ["", "**Политики α:** `CatBoost` — регрессор выигрыша каждой α по боксу (признаки без GT: бокс, сама α и вся "
          "кривая отклика бокса), α с наибольшим предсказанным выигрышем или без steering; `α*` — одна α на обучающих "
          "строках; `оракул α` — лучшая одна α набора задним числом. LODO: регрессор для набора d учится только на "
          "других наборах.", ""]

    def main_table(meths):
        cols = [("all", "all"), ("nat", "all"), ("med", "all"), ("all", "bottom50"), ("all", "top50")]
        out = ["| метод | все | натур. | медицина | все b50 | все t50 | Δ IoU все [95% CI] |", "|---|---|---|---|---|---|---|"]
        for m in meths:
            out.append(f"| {m} | " + " | ".join(U._cell(T, M[b], m, g) for b, g in cols) + " | "
                       + (U._cell(T, M["all"], m, "all", "ci") if m != "undef" else "—") + " |")
        return out

    L += ["## Главное: CatBoost-регрессор α", ""]
    for mm in MODES:
        meths = (["undef", "head-select"] if mm == 0 else [])
        for rule in rules:
            for site in sites:
                meths += [f"mm{mm} · {site_label(study, rule, site)} · {v} · CatBoost" for v in variants]
        L += main_table(meths) + [""]
    L += ["## Все политики α", ""]
    for mm in MODES:
        meths = [f"mm{mm} · {site_label(study, rule, site)} · {v} · {p}" for rule in rules for site in sites
                 for v in variants for p in POLICIES]
        L += main_table(meths) + [""]

    MP = U._means(P, datasets)
    L += ["## Попарно (тот же бокс; 95% CI — бутстреп по картинкам внутри наборов)", "",
          "`x − y`: Δ > 0 — x лучше; % — к y.", "",
          "| сравнение | натур. all | медицина all | все all | все b50 | все t50 |", "|---|---|---|---|---|---|"]
    for m in dict.fromkeys(P.method):
        cells = [U._cell(P, MP[b], m, "all", "ci") + f" ({U._cell(P, MP[b], m, 'all')})" if MP[b] else "—"
                 for b in ("nat", "med", "all")]
        cells += [U._cell(P, MP["all"], m, g, "ci") for g in ("bottom50", "top50")]
        L.append(f"| {m} | " + " | ".join(cells) + " |")

    L += ["", "## По наборам: CatBoost, all, % к undef", "",
          "| метод | " + " | ".join(datasets) + " |", "|---|" + "---|" * len(datasets)]
    for mm in MODES:
        meths = (["undef", "head-select"] if mm == 0 else [])
        meths += [f"mm{mm} · {site_label(study, rule, site)} · {v} · CatBoost" for rule in rules for site in sites
                  for v in variants if v in ("own", "pool", "lodo")]
        for m in meths:
            vals = [T[(T.block == d) & (T.method == m) & (T.group == "all")].pct for d in datasets]
            L.append(f"| {m} | " + " | ".join(f"{x.iloc[0]:+.2f}%" if len(x) else "—" for x in vals) + " |")

    L += ["", "## m и p отдельно: CatBoost, all, среднее по наборам, % к undef", "",
          "| метод | " + " | ".join(sorted(TK.kind.unique())) + " |", "|---|" + "---|" * TK.kind.nunique()]
    for m in ["head-select"] + [x for x in dict.fromkeys(TK.method) if x.endswith("· CatBoost")]:
        s = TK[TK.method == m].set_index("kind").pct_mean_subsets
        L.append(f"| {m} | " + " | ".join(f"{s[k]:+.2f}%" for k in sorted(TK.kind.unique())) + " |")

    L += ["", "## Доза", "", "Средняя α, выбранная CatBoost, и α* на обучении (по фолдам).", "",
          "| узел · вектор · режим | α CatBoost (среднее, доля без steering) | α* по фолдам |", "|---|---|---|"]
    for rule in rules:
        for site in sites:
            for v in variants:
                for mm in MODES:
                    lab = f"mm{mm} · {site_label(study, rule, site)} · {v}"
                    a = PB[f"{lab} · CatBoost [α]"]
                    s = A[(A.rule == rule) & (A.site == site) & (A.variant == v) & (A["mode"] == f"mm{mm}")]
                    L.append(f"| {lab} | {a.mean():.1f} ({(a == 0).mean() * 100:.0f}% без) | "
                             f"{', '.join(f'{x:g}' for x in sorted(set(s.alpha_star)))} |")
    edge = [f"{r.site}/{r.variant}/{r.mode}" for r in A.itertuples()
            if r.alpha_star == max(alphas_of(study, r.site))]
    if edge:
        L += ["", f"α* на краю сетки: {', '.join(sorted(set(edge)))} — доза может быть недобрана."]

    L += ["", "## Геометрия векторов (сырые, все картинки набора)", ""]
    for (rule, site), (C, Gm) in geo.items():
        nat = [d for d in datasets if d in study["natural"]]
        med = [d for d in datasets if d in study["medical"]]
        off = lambda a, b: np.nanmean([C.loc[x, y] for x in a for y in b if x != y]) if a and b else math.nan  # noqa: E731
        g = Gm.set_index("dataset")
        L += [f"**{site_label(study, rule, site)}**: косинус натур.–натур. {off(nat, nat):+.2f} (шум половин "
              f"{g.loc[nat].split_half.mean():+.2f}), мед.–мед. {off(med, med):+.2f}, натур.–мед. {off(nat, med):+.2f}. "
              "cos(свой, LODO) / √надёжность: " + ", ".join(
                  f"{d} {g.loc[d, 'cos_lodo'] / math.sqrt(g.loc[d, 'reliability']):.2f}"
                  if g.loc[d, "reliability"] > 0 else f"{d} —" for d in datasets), ""]
    show = vs[vs.vector.isin(["pool", "coco", "rand", "pool_nat"])]
    L += ["| правило · узел · вектор | ‖v‖ сырой | множитель | доза при α = 1 | cos pool | cos COCO |",
          "|---|---|---|---|---|---|"]
    for r in show.itertuples():
        L.append(f"| {r.rule} · {r.site} · {r.vector} | {r.raw_norm:.3f} | {r.scale:.2f} | {r.dose_alpha1:.4f} | "
                 f"{r.cos_pool:+.2f} | {r.cos_coco:+.2f} |")

    L += ["", "## Проверки", ""]
    st = checks.get("steer", {})
    z0, of = checks.get("zero_dose", {}), checks.get("own_forward", {})
    z1, br = st.get("zero_dose", {}), st.get("baseline_repeat", {})
    vp, gr = checks.get("vs_pool", {}), checks.get("groups", {})
    L += [f"- states: нулевая доза {z0.get('bit_exact')}/{z0.get('forwards')}, свой проход = MaskDecoder.forward "
          f"{of.get('bit_exact')}/{of.get('compared')}, покрытие плана {checks.get('coverage', {}).get('ok')};",
          f"- steer: нулевая доза {z1.get('bit_exact')}/{z1.get('forwards')}, повтор baseline (states vs steer) max |ΔIoU| "
          f"{br.get('max_abs_iou_diff')}, покрытие {checks.get('steer_coverage', {}).get('ok')};",
          f"- undef / head-select против пула `{vp.get('pool_dir')}`: max |Δ| {vp.get('undef_max_abs_diff', '—')} / "
          f"{vp.get('hsel_max_abs_diff', '—')}, боксов > 1e-6: {vp.get('undef_n_over_1e-6', '—')} / "
          f"{vp.get('hsel_n_over_1e-6', '—')} (найдено {vp.get('matched', '—')} из {vp.get('boxes', '—')});",
          f"- группы round(n/2): отклонений {gr.get('n_images_off')} из {gr.get('images')};"
          + (f" пустые фолды (smoke): {len(study.get('empty_folds', []))}." if study.get("empty_folds") else "")]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=["states", "vectors", "steer", "merge", "analyse"])
    p.add_argument("--out_dir", required=True, help="the study's directory (one model)")
    g = p.add_argument_group("model (GPU stages)")
    g.add_argument("--model_name", choices=["SAM", "SAM2.1", "SAM3"])
    g.add_argument("--checkpoint_path", default=None, help="'' = SAM3 from the HF cache")
    g.add_argument("--model_type", default="vit_b")
    g.add_argument("--root", help="user study FOR_TEST")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--batch", type=int, default=64, help="boxes per decoder pass (build_candidate_pool's)")
    g.add_argument("--keep_tf32", action="store_true")
    g.add_argument("--shard", type=int, default=0)
    g.add_argument("--n_shards", type=int, default=1)
    g.add_argument("--log_alpha", type=float, default=4.0, help="the alpha in the steer stage's log line")
    s = p.add_argument_group("study (states stage; later stages read study.json)")
    s.add_argument("--use", default="mp", help="annotation kinds: mp (default), m or p")
    s.add_argument("--datasets", nargs="*", default=None, help="subsets (default: all ten)")
    s.add_argument("--sites", type=_list, default=_list("L2.norm2,L2.norm3,hyper"))
    s.add_argument("--rules", type=_list, default=_list("decile"), help=f"of {', '.join(RULES)}")
    s.add_argument("--variants", type=_list, default=_list("own,pool,lodo,coco,rand"), help=f"of {', '.join(VARIANTS)}")
    s.add_argument("--alphas", type=_floats, default=_floats("1,2,3,4,6,8,12,16,24,32"))
    s.add_argument("--alphas_hyper", type=_floats, default=_floats("1,2,3,4,6,8,12,16,24,32,48,64"))
    s.add_argument("--folds", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--norm", choices=["pool", "raw"], default="pool")
    s.add_argument("--limit_images", type=int, default=0, help="per subset, for a smoke run")
    s.add_argument("--allow_empty_fold", action="store_true", help="a fold-out vector of no image falls back to all")
    m = p.add_argument_group("merge / analyse")
    m.add_argument("--pool_dir", default="", help="build_candidate_pool run of the same model and --use (per_box.csv.gz)")
    m.add_argument("--cb_iterations", type=int, default=400)
    m.add_argument("--cb_depth", type=int, default=6)
    m.add_argument("--cb_lr", type=float, default=0.08)
    m.add_argument("--task_type", choices=["CPU", "GPU"], default="CPU")
    m.add_argument("--devices", default="0")
    m.add_argument("--threads", type=int, default=-1)
    m.add_argument("--n_boot", type=int, default=2000)
    m.add_argument("--n_split", type=int, default=200)
    a = p.parse_args(argv)
    if a.stage in ("states", "steer"):
        miss = [f"--{k}" for k in ("model_name", "root") if not getattr(a, k)]
        if a.checkpoint_path is None:
            miss.append("--checkpoint_path")
        if miss:
            p.error(f"--stage {a.stage} needs {', '.join(miss)}")
    bad = [v for v in a.variants if v not in VARIANTS] + [r for r in a.rules if r not in RULES]
    if bad:
        p.error(f"unknown variants / rules {bad}")
    return a


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.stage in ("states", "steer"):
        # SAM-3 lives in its own conda env; re-execs there and exits (no-op otherwise)
        maybe_dispatch_to_env(args.model_name, __file__)
    import steer_models_scan as SMS
    stage = {"states": stage_states, "vectors": stage_vectors, "steer": stage_steer, "merge": stage_merge,
             "analyse": stage_analyse}[args.stage]
    try:
        return stage(args)
    except SMS.CheckFailed as e:
        print(f"[check failed] {e}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
