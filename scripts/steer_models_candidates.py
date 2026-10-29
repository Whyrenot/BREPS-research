"""
steer_models_candidates.py
==========================
What the per-box alpha regressor and the CatBoost selector need, for the
steering vector of a steer_models_scan.py run -- SAM-1, SAM-2.1 or SAM-3, one
GPU pass; everything after it runs on a laptop from the dump.

VECTOR   the scan's DM at --site (default L2.norm2), all tokens, from its
         vectors.npz: on the COCO A images (the vector's own) the vector of the
         other folds (cross-fitting, as steer_gate / steer_multimask), the all-A
         vector everywhere else. The scan's split, folds and decoder layout are
         checked against this model before anything runs.
PER BOX  every user-study box of --datasets (all ten by default), the user's
         box: the unsteered forward and one forward per alpha of --alphas, both
         output modes from the same decoder pass -- mm0 = multimask=False as the
         model ships (SAM-2.1 / SAM-3: with the stability fallback), mm1 =
         multimask=True with head-select.

OUTPUTS (--out_dir)
  rows.csv.gz     steer_multimask.py's dump schema (tok0_* = mm0, multi_* = mm1):
                  per mode iou0 / pred0 / stab0 / mask_frac0 / fill0 / head0 and per
                  alpha iou / pred / chg / fill / stab / head; box_frac, log_aspect,
                  role (train = COCO A), group. The alpha regressors (steer_gate's
                  "out" features: predicted IoU, stability, box fill, mask change)
                  and the tables against head-select, without a GPU:
                    python scripts/steer_multimask.py --from_dump <out_dir>/rows.csv.gz \\
                        --alphas <the same> --out_dir <...>
  candidates_mm0.csv, candidates_mm1.csv
                  dump_steer_candidates.py's schema (= dump_selector_data's): one row
                  per (case, alpha), cand_kind "steer", f_* on the steered mask (mm1:
                  the head-select mask, f_head and the heads' spread), agreement
                  within the case's steering pool, x_* against undef / head-select /
                  the best_of_n pick (the pool's own sampler and seed), target_iou,
                  chk_*. For the CatBoost selector:
                    python scripts/selector_steer.py --pool <the model's selector pool> \\
                        --steer <out_dir>/candidates_mm0.csv --out_dir <...>
  checks.json, meta.json

CHECKS (* stop the run)
  * the scan is this model's (decoder layout, split A); zero dose in both modes on
    every image; own decoder pass = MaskDecoder.forward (first image of every
    subset); the scan's IoUs at the alphas both grids share (max |d|, reported);
    undef / head-select vs --reference_csv (reported); groups 12 / 13.
  TF32 matmuls are off, as in the scan (sam3 turns them on at import).

    CUDA_VISIBLE_DEVICES=0 python scripts/steer_models_candidates.py --model_name SAM2.1 \\
        --root /.../FOR_TEST --checkpoint_path /.../sam2.1_hiera_base_plus.pt \\
        --scan_dir exp_res/steer_scan_SAM2.1 --out_dir exp_res/steer_cand_SAM2.1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heatmaps.env_dispatch import maybe_dispatch_to_env  # noqa: E402

NAN = float("nan")
MODES = ((0, "tok0"), (1, "multi"))


def _a(alpha: float) -> str:
    return f"a{alpha:g}"


def stability(low: torch.Tensor) -> torch.Tensor:
    """steer_gate.stability: |l > 1| / |l > -1| on the low-res logits."""
    hi = (low > 1.0).flatten(1).sum(1).float()
    lo = (low > -1.0).flatten(1).sum(1).float()
    return hi / lo.clamp_min(1.0)


def mask_change(m: torch.Tensor, m0: torch.Tensor) -> torch.Tensor:
    """steer_sam_decoder.mask_change: |m XOR m0| / |m OR m0|, 0 when both are empty."""
    x = (m ^ m0).flatten(1).sum(1).float()
    u = (m | m0).flatten(1).sum(1).float()
    return x / u.clamp_min(1.0)


def load_scan(scan: Path, site: str, dec, A, args) -> dict:
    """The scan's vectors at the site, and the checks that it is this model's."""
    meta = json.loads((scan / "meta.json").read_text(encoding="utf-8"))
    split = json.loads((scan / "split.json").read_text(encoding="utf-8"))
    z = np.load(scan / "vectors.npz")
    folds = int(meta["folds"])
    V = {"all": z[f"all/{site}/dm"]}
    for k in range(folds):
        V[k] = z[f"{k}/{site}/dm"]
    lay = meta["layout"]
    c = {"scan_dir": str(scan), "model": meta["model_name"], "checkpoint": meta["checkpoint"],
         "same_model": meta["model_name"] == args.model_name,
         "same_checkpoint": meta["checkpoint"] == args.checkpoint_path,
         "same_layout": lay["token_names"] == dec.token_names and list(lay["emb_hw"]) == list(dec.emb_hw),
         "same_split_A": sorted(split["A"]) == sorted(A), "folds": folds, "seed": meta["args"]["seed"],
         "vector_shape": list(V["all"].shape)}
    c["ok"] = bool(c["same_model"] and c["same_layout"] and c["same_split_A"]
                   and tuple(V["all"].shape) == (dec.T, 256))
    return {"V": V, "meta": meta, "check": c}


def bon_pick(case, predictor, device, a) -> torch.Tensor:
    """The selector pool's best_of_n pick: dump_selector_data's sampler, seed and call."""
    import refine_box_iou_grad as R
    boxes = R.sample_bon_boxes(case["bad_box"], predictor, a.Y, a.sigma, a.sigma_center, a.perturb_mode, a.bon_seed)
    masks, scores = R._predict_boxes(boxes.float().to(device), predictor, multimask=False)
    return masks[int(torch.argmax(scores[:, 0])), 0].bool().cpu()


def hsel_mask(box, predictor, device) -> torch.Tensor:
    """Multimask head-select on the user's box, through the pool's own call."""
    import refine_box_iou_grad as R
    masks, scores = R._predict_boxes(torch.as_tensor(np.asarray(box, np.float32)[None], device=device), predictor,
                                     multimask=True)
    return masks[0, int(torch.argmax(scores[0]))].bool().cpu()


def image_pass(runner, dec, site, item: dict, v: torch.Tensor, alphas, checks: dict, first: bool, stride: int) -> dict:
    """Unsteered + every alpha -> per mode: IoU, predicted IoU, stability, mask
    fraction, mask change, head, and the strided masks (for the candidates)."""
    import steer_models_scan as SMS
    b, gt, st = item["b"], item["gt"], max(1, stride)
    low0, iou0 = runner.forward(b)
    lz, iz = runner.forward(b, edits={site.id: SMS.zero_edit})
    z = checks["zero_dose"]
    z["forwards"] += 1
    if not (torch.equal(lz, low0) and torch.equal(iz, iou0)):
        z["ok"] = False
        z["failures"].append(f"{item['dataset']}/{item['stem']}")
        raise SMS.CheckFailed(f"zero dose at {item['dataset']}/{item['stem']}")
    z["bit_exact"] += 1
    checks["zero_dose_multi"] = z            # one decoder pass serves both modes
    if first:
        SMS.check_official(dec, item, low0, iou0, SMS.score_modes(dec, low0, iou0, gt), checks)
    Vt = {site.id: {"dm": v}}
    runs = [(low0, iou0)] + [runner.forward(b, edits={site.id: SMS.make_edit(SMS.Cfg(-1, site.id, "all", "dm", a),
                                                                           site, dec, Vt, {})}) for a in alphas]
    ar = torch.arange(len(b), device=b.device)
    out = {}
    for mm, _ in MODES:
        res = {"iou": [], "pred": [], "stab": [], "frac": [], "chg": [], "head": [], "small": [], "heads_pred": []}
        m_base = None
        for low_all, iou_all in runs:
            low, head = dec.pick(low_all, iou_all, bool(mm))
            m = dec.masks(low)
            if m_base is None:
                m_base = m
            res["iou"].append(SMS.batch_iou(m, gt).cpu().numpy())
            res["pred"].append(iou_all[ar, head].float().cpu().numpy())
            res["stab"].append(stability(low).cpu().numpy())
            res["frac"].append(m.flatten(1).float().mean(1).cpu().numpy())
            res["chg"].append(mask_change(m, m_base).cpu().numpy())
            res["head"].append(head.cpu().numpy())
            res["small"].append(m[:, ::st, ::st].cpu())
            res["heads_pred"].append(iou_all[:, 1:].float().cpu().numpy())
        out[mm] = res
    return out


def multimask_rows(item: dict, res: dict, alphas, box_frac, log_aspect, table: str) -> pd.DataFrame:
    """One image in steer_multimask.py's rows.csv.gz schema."""
    cols = {}
    for mm, pre in MODES:
        r = res[mm]
        cols[f"{pre}_iou0"], cols[f"{pre}_pred0"], cols[f"{pre}_stab0"] = r["iou"][0], r["pred"][0], r["stab"][0]
        cols[f"{pre}_mask_frac0"], cols[f"{pre}_fill0"] = r["frac"][0], r["frac"][0] / box_frac
        cols[f"{pre}_head0"] = r["head"][0]
        for j, a in enumerate(alphas, start=1):
            k = _a(a)
            cols[f"{pre}_iou_{k}"], cols[f"{pre}_pred_{k}"], cols[f"{pre}_chg_{k}"] = r["iou"][j], r["pred"][j], r["chg"][j]
            cols[f"{pre}_fill_{k}"], cols[f"{pre}_stab_{k}"] = r["frac"][j] / box_frac, r["stab"][j]
            cols[f"{pre}_head_{k}"] = r["head"][j]
    rows = item["rows"]
    return pd.DataFrame({"table": table, "dataset": item["dataset"], "image_name": item["stem"],
                         "role": "train" if table == "COCO_A" else "eval", "kind": [r["kind"] for r in rows],
                         "user": [r["user"] for r in rows], "attempt": [r["attempt"] for r in rows],
                         "half": item["half"], "fold": item["fold"], "box_frac": box_frac, "log_aspect": log_aspect,
                         **cols})


def candidate_rows(item: dict, res: dict, mm: int, alphas, cases, predictor, dev, a, refs: dict, clean: float,
                   tag: str) -> list:
    """One image in dump_steer_candidates.py's schema, for one output mode."""
    import dump_selector_data as DS
    import refine_box_iou_grad as R
    from heatmaps.comp_hw_smoothed import get_original_size
    r = res[mm]
    st, orig_hw = max(1, a.stride), tuple(get_original_size(predictor))
    out = []
    for i, case in enumerate(cases):
        box = np.asarray(case["bad_box"], dtype=np.float64)
        small = torch.stack([r["small"][j + 1][i] for j in range(len(alphas))])
        M = R._iou_matrix(small)
        lab = R._cluster(M, 0.90)
        sizes = np.bincount(lab)
        ref = {"undef": res[0]["small"][0][i], "hsel": refs["hsel"][i][::st, ::st], "bon": refs["bon"][i][::st, ::st]}
        for j, al in enumerate(alphas):
            row = {"image_name": item["stem"], "kind": case["kind"], "user": case["user"], "attempt": case["attempt"],
                   "cand_kind": "steer", "cand_id": j, "method_slot": "", "start": -1, "step": -1, "y": -1,
                   "n_forwards": j + 1, "n_batched_fwd": 1, "f_step": NAN, "f_step_frac": NAN, "f_is_start0": NAN}
            row.update(DS.cand_feats(small[j], box, box, orig_hw, st, float(r["pred"][j + 1][i]),
                                     head=int(r["head"][j + 1][i]),
                                     head_preds=r["heads_pred"][j + 1][i] if mm else None))
            row.update(DS.agree_feats(M[j], j, sizes[lab[j]] / len(alphas), sizes.size))
            row["f_steer_alpha"] = float(al)
            row["target_iou"] = float(r["iou"][j + 1][i])
            for nm, mk in ref.items():
                row[f"x_iou_{nm}"] = R._pair_iou(small[j], mk) if tuple(mk.shape) == tuple(small[j].shape) else NAN
            row["x_iou_grad"] = NAN
            row["gt_undef_iou"] = float(res[0]["iou"][0][i])
            row["gt_clean_iou"] = clean
            row["gt_bad_iou_json"] = NAN
            row["gt_best_iou_json"] = NAN
            row["chk_undef_iou"] = float(res[0]["iou"][0][i])
            row["chk_bon_pick_iou"] = refs["bon_iou"][i]
            row["chk_hsel_iou"] = refs["hsel_iou"][i]
            row["chk_vector"] = tag
            row["chk_mode"] = f"mm{mm}"
            out.append(row)
    return out


def scan_agreement(D: pd.DataFrame, scan: Path, site: str, alphas) -> dict:
    """The steered IoUs at the alphas both grids share vs the scan's own."""
    import selector_data as SD
    cfg = pd.read_csv(scan / "configs.csv")
    out = {}
    for mm, pre in MODES:
        sel = cfg[(cfg.site == site) & (cfg.group == "all") & (cfg.method == "dm")]
        shared = [(a, int(sel[np.isclose(sel.alpha, a)].cid.iloc[0])) for a in alphas if np.isclose(sel.alpha, a).any()]
        p = scan / f"per_case_mm{mm}.csv.gz"
        if not shared or not p.is_file():
            out[f"mm{mm}"] = {"available": False}
            continue
        S = pd.read_csv(p, usecols=SD.CASE_KEYS + ["base_iou"] + [f"iou_c{c}" for _, c in shared],
                        dtype={k: str for k in SD.CASE_KEYS})
        S[SD.CASE_KEYS] = S[SD.CASE_KEYS].fillna("")
        m = D[SD.CASE_KEYS + [f"{pre}_iou0"] + [f"{pre}_iou_{_a(a)}" for a, _ in shared]].merge(S, on=SD.CASE_KEYS)
        d = [np.abs(m[f"{pre}_iou0"] - m.base_iou).max()] + [np.abs(m[f"{pre}_iou_{_a(a)}"] - m[f"iou_c{c}"]).max()
                                                            for a, c in shared]
        out[f"mm{mm}"] = {"available": True, "boxes": int(len(m)), "alphas": [a for a, _ in shared],
                          "max_abs_iou_diff": float(np.max(d))}
    return out


def run(args) -> int:
    import selector_data as SD
    import steer_models_scan as SMS
    import steer_user_study as SUS
    from heatmaps.comp_hw_smoothed import get_original_size, load_model
    from segment_anything.utils.transforms import ResizeLongestSide
    from steer_sam_decoder import assign_folds

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checks = {"zero_dose": {"forwards": 0, "bit_exact": 0, "ok": True, "failures": []},
              "own_forward": {"compared": 0, "bit_exact": 0, "ok": True, "failures": []},
              "public_api": {"compared": 0, "max_abs_iou_diff": 0.0, "n_over_1e-4": 0, "where": []},
              "public_single": {"compared": 0, "max_abs_iou_diff": 0.0, "n_over_1e-3": 0, "where": []}}

    def write_checks():
        (out / "checks.json").write_text(json.dumps(checks, indent=1, default=SMS._json_default), encoding="utf-8")

    predictor = load_model(model_name=args.model_name, model_type=args.model_type,
                           checkpoint=args.checkpoint_path, device=device)
    for prm in predictor.model.parameters():
        prm.requires_grad_(False)
    tf32 = {"matmul_after_load": bool(torch.backends.cuda.matmul.allow_tf32), "cudnn": bool(torch.backends.cudnn.allow_tf32)}
    if not args.keep_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
    tf32["matmul_run"] = bool(torch.backends.cuda.matmul.allow_tf32)
    checks["tf32"] = tf32
    dec = SMS.Decoder(predictor)
    sites = SMS.build_sites(dec)
    site = next(s for s in sites if s.id == args.site)
    runner = SMS.Runner(dec, sites, args.batch)

    root = Path(args.root)
    datasets = list(dict.fromkeys(["COCO", *args.datasets]))
    index = SUS.index_study(root, args.use, datasets)
    A, B = SUS.split_halves(index["COCO"].keys(), args.split_seed)
    scan = load_scan(Path(args.scan_dir), site.id, dec, A, args)
    checks["scan"] = scan["check"]
    print(f"{args.model_name}: {dec.T} tokens, fallback {dec.dynamic}, TF32 matmul {tf32['matmul_after_load']} -> "
          f"{tf32['matmul_run']}; scan {args.scan_dir}: {scan['check']}", flush=True)
    if not scan["check"]["ok"]:
        write_checks()
        raise SMS.CheckFailed(f"--scan_dir is not this model's run: {scan['check']}")
    fold_of = assign_folds(A, scan["check"]["folds"], scan["check"]["seed"])
    Vdev = {k: torch.as_tensor(v, device=device) for k, v in scan["V"].items()}
    alphas = list(args.alphas)
    lim = (lambda xs: list(xs)[:args.limit_images]) if args.limit_images > 0 else list
    plan = [("COCO_A", "COCO", "A", s) for s in lim(A)] + [("COCO_B", "COCO", "B", s) for s in lim(B)]
    plan += [(d, d, "-", s) for d in datasets[1:] for s in lim(index[d].keys())]

    parts, writers, files, n_cand, seen = [], {}, {}, 0, set()
    try:
        for i, (table, ds, half, stem) in enumerate(plan):
            fold = fold_of[stem] if half == "A" else -1
            item = SMS.load_item(dec, root, ds, half, stem, index[ds], fold)
            first = table not in seen
            seen.add(table)
            res = image_pass(runner, dec, site, item, Vdev[fold if half == "A" else "all"], alphas, checks, first,
                             args.stride)
            hw = get_original_size(predictor)
            ih, iw = ResizeLongestSide.get_preprocess_shape(hw[0], hw[1], 1024)
            bx = item["boxes"]
            bw, bh = (bx[:, 2] - bx[:, 0]).clip(min=1e-3), (bx[:, 3] - bx[:, 1]).clip(min=1e-3)
            box_frac = bw * bh / float(ih * iw)
            parts.append(multimask_rows(item, res, alphas, box_frac, np.log(bw / bh), table))
            if not args.no_candidates:
                cases = [{"bad_box": list(map(float, bx[k])), "kind": r["kind"], "user": r["user"],
                          "attempt": r["attempt"]} for k, r in enumerate(item["rows"])]
                gt_cpu = item["gt"].cpu()
                bon = [bon_pick(c, predictor, device, args) for c in cases]
                hs = [hsel_mask(bx[k], predictor, device) for k in range(len(cases))]
                refs = {"bon": bon, "hsel": hs,
                        "bon_iou": [float(SMS.batch_iou(m[None], gt_cpu)[0]) for m in bon],
                        "hsel_iou": [float(SMS.batch_iou(m[None], gt_cpu)[0]) for m in hs]}
                # the GT's tight box, as the pool's gt_clean_iou (single output)
                low_c, _ = dec.pick(*runner.forward(dec.model_boxes(np.asarray([item["best_box"]], np.float32))), False)
                clean = float(SMS.batch_iou(dec.masks(low_c), item["gt"])[0])
                tag = f"fold{fold}" if half == "A" else "full"
                for mm, _ in MODES:
                    rows = candidate_rows(item, res, mm, alphas, cases, predictor, device, args, refs, clean, tag)
                    if mm not in writers:
                        files[mm] = open(out / f"candidates_mm{mm}.csv", "w", newline="", encoding="utf-8")
                        writers[mm] = csv.DictWriter(files[mm], fieldnames=list(rows[0]))
                        writers[mm].writeheader()
                    writers[mm].writerows(rows)
                    files[mm].flush()
                    n_cand += len(rows)
            print(f"[{table}] [{i + 1}/{len(plan)}] {stem}: undef {res[0]['iou'][0].mean():.4f}, head-select "
                  f"{res[1]['iou'][0].mean():.4f}, a={alphas[-1]:g}: {res[0]['iou'][-1].mean():.4f} / "
                  f"{res[1]['iou'][-1].mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
            del item, res
    except SMS.CheckFailed as e:
        write_checks()
        print(f"[check failed] {e}", file=sys.stderr, flush=True)
        return 2
    finally:
        for f in files.values():
            f.close()

    D = pd.concat(parts, ignore_index=True)
    D["group"] = SUS.assign_groups(D.rename(columns={"tok0_iou0": "undef_iou"})).to_numpy()
    gcount = D.groupby(["image_name", "group"]).size().unstack(fill_value=0)
    bad = int((~((gcount.get("bottom50", 0) == 12) & (gcount.get("top50", 0) == 13))).sum())
    checks["groups"] = {"images": int(len(gcount)), "n_images_not_12_13": bad, "ok": bad == 0}
    D.to_csv(out / "rows.csv.gz", index=False, float_format="%.6f")
    ref = SMS.read_reference(args.reference_csv, args.use)
    for key, col, ref_col in (("repro", "tok0_iou0", "undefended_iou"), ("repro_headsel", "multi_iou0", "headsel_iou")):
        r = SMS.repro(D, ref, col, ref_col, args.repro_tol)
        if r.get("available"):
            r["max_abs_diff"], r["frac_over"] = r["max_abs"], r["frac_over_tol"]     # steer_multimask's names
        checks[key] = r
    checks["scan_agreement"] = scan_agreement(D, Path(args.scan_dir), site.id, alphas)
    write_checks()
    meta = {"script": "steer_models_candidates.py", "created": time.strftime("%Y-%m-%d %H:%M"),
            "model_name": args.model_name, "checkpoint": args.checkpoint_path, "site": site.id, "tokens": "all",
            "alphas": alphas, "layout": dec.layout(), "args": vars(args), "rows": int(len(D)),
            "candidates": n_cand, "versions": SMS.package_versions(), "seconds": round(time.time() - t0)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=SMS._json_default), encoding="utf-8")
    sa = checks["scan_agreement"]
    print(f"done in {time.time() - t0:.0f}s -> {out}: {len(D)} boxes, {n_cand} candidates; zero dose "
          f"{checks['zero_dose']['bit_exact']}/{checks['zero_dose']['forwards']}, own forward "
          f"{checks['own_forward']['bit_exact']}/{checks['own_forward']['compared']}, vs scan "
          + ", ".join(f"{k} {v.get('max_abs_iou_diff', 'n/a')}" for k, v in sa.items())
          + f", groups off {bad}", flush=True)
    return 0


def _floats(s: str) -> list:
    return [float(x) for x in s.split(",") if x.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", required=True, choices=["SAM", "SAM2.1", "SAM3"])
    p.add_argument("--checkpoint_path", required=True, help="the scan's checkpoint ('' = SAM3 from the HF cache)")
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--root", required=True, help="user study FOR_TEST")
    p.add_argument("--use", default="m")
    p.add_argument("--datasets", nargs="*", default=["Berkeley", "GrabCut", "DAVIS", "PASCAL", "ADE20K", "TETRIS",
                                                     "ACDC", "BUID", "MedScribble"],
                   help="subsets besides COCO (default: every subset of the user study)")
    p.add_argument("--scan_dir", required=True, help="out_dir of steer_models_scan.py for this model")
    p.add_argument("--site", default="L2.norm2")
    p.add_argument("--alphas", type=_floats, default=_floats("1,2,3,4,6,8,12,16,24,32"))
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--stride", type=int, default=4, help="feature masks, as dump_selector_data")
    p.add_argument("--Y", type=int, default=16, help="best_of_n: dump_selector_data's defaults (the pool's pick)")
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--sigma_center", type=float, default=0.03)
    p.add_argument("--perturb_mode", default="size", choices=["size", "size_center"])
    p.add_argument("--bon_seed", type=int, default=42)
    p.add_argument("--no_candidates", action="store_true", help="rows.csv.gz only (no best_of_n / head-select calls)")
    p.add_argument("--reference_csv", default="")
    p.add_argument("--repro_tol", type=float, default=1e-3)
    p.add_argument("--keep_tf32", action="store_true")
    p.add_argument("--limit_images", type=int, default=0, help="per half / subset, for a smoke run")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    maybe_dispatch_to_env(args.model_name, __file__)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
