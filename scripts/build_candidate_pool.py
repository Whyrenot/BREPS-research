"""
build_candidate_pool.py
=======================
Every candidate mask of the paper for one model, in one GPU pass, with the
features the CatBoost selector is trained on: SAM-1, SAM-2.1 or SAM-3 on the
user study (--use m, all ten subsets, 12 500 boxes). Everything after it -- the
selector, the tables, the per-box file of all methods -- runs on a CPU.

CANDIDATES per box (the tight bbox of the user's mask, 1024 frame)
  undef       multimask=False on the user's box (SAM-2.1 / SAM-3: with the
              stability fallback, as the models ship)
  hsel        multimask=True, the head with the highest predicted IoU
  grad_mm0    gradient ascent of the predicted IoU of the multimask=False output:
              refine_box_iou_grad.refine_boxes_multistart's ascent -- (cx, cy, w,
              h) in the model's frame, w, h >= 2, Adam --lr, --steps steps --
              with one start, the user's box; a candidate every
              --checkpoint_every steps (step 0 is undef, not repeated)
  grad_mm1    the same ascent of the best multimask head (step 0 = hsel)
  bon_mm0     the Y boxes of refine_box_iou_grad.sample_bon_boxes (--perturb_mode
              size_center: width / height and centre noise; the user's box
              first), multimask=False on each
  bon_mm1     the same Y boxes, head-select on each
  steer_mm0   the scan's DM at --site (L2.norm3), all tokens -- on COCO A the
              vector of the other folds (cross-fitting) -- at every alpha of
              --alphas, multimask=False
  steer_mm1   the same, head-select
  Methods (method_slot marks their row): undef, hsel, grad_mm{0,1} = the last
  checkpoint, bon_mm{0,1} = argmax of the predicted IoU over the Y.

EFFICIENCY  the image is encoded once and all its boxes go through the decoder
  together: the BoN boxes of every box of the image as one batch (in chunks of
  --batch), and the ascents of all of them at once -- the boxes share no
  parameters and Adam is element-wise, so one summed backward is as many
  independent ascents (refine_boxes_multistart batches its starts the same way).
  Both output modes of BoN and steering come from the same decoder pass.

OUTPUTS (--out_dir)
  pool.csv.gz     one row per (box, candidate), dump_selector_data's schema:
                  identity, cand_kind, cand_id, method_slot, step, cost
                  (n_forwards); f_* features of the candidate (predicted IoU, head
                  and the heads' spread, mask shape, box move, box prior),
                  f_agree_* / f_cluster_* within its own family (BoN, steering;
                  NaN for one trajectory), f_steer_alpha, f_step*; x_iou_* against
                  the methods' masks; target_iou (full resolution); gt_* (never
                  features)
  per_box.csv.gz  one row per box: IoU and predicted IoU of every method, the
                  ascents' checkpoints, steering at every alpha, the pool's oracle
  rows.csv.gz     steering per alpha in steer_multimask's dump schema (the alpha
                  regressors: python scripts/steer_multimask.py --from_dump ...)
  checks.json, meta.json

CHECKS (* stop the run)
  * the scan is this model's run (decoder layout, split A, vector shape)
  * zero dose on every image, own decoder pass = MaskDecoder.forward (first image
    of every subset), as steer_models_scan
  the batched ascent vs refine_boxes_multistart one box at a time, and BoN vs
  refine_box_iou_grad.best_of_n_batched (--check_boxes boxes of the first image of
  every subset; max |dIoU| reported); undef / hsel vs --reference_csv; groups 12 / 13.
  TF32 matmuls are off (sam3 turns them on at import).

    CUDA_VISIBLE_DEVICES=0 python scripts/build_candidate_pool.py --model_name SAM2.1 \\
        --root /.../FOR_TEST --checkpoint_path /.../sam2.1_hiera_base_plus.pt \\
        --scan_dir exp_res/steer_scan_SAM2.1 --out_dir exp_res/pool_SAM2.1
"""

from __future__ import annotations

import argparse
import csv
import gzip
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
MODES = (0, 1)


def _a(alpha: float) -> str:
    return f"a{alpha:g}"


class Frames:
    """The model's input frame <-> this repo's 1024 frame (refine_boxes_multistart's boxes_to_1024)."""

    def __init__(self, dec):
        self.dec = dec

    def to_1024(self, b: torch.Tensor) -> np.ndarray:
        from heatmaps.comp_hw_smoothed import get_original_size
        from heatmaps.defend_critical_shifts import original_to_1024
        x = b.detach().cpu().numpy().astype(np.float64).reshape(-1, 4)
        if self.dec.sam1:
            return x
        H, W = get_original_size(self.dec.p)
        sc = np.array([W, H, W, H], dtype=np.float64)
        return original_to_1024(x / self.dec.frame * sc, (H, W))


def ascend(dec, b0: torch.Tensor, gt: torch.Tensor, steps: int, lr: float, multimask: bool, every: int,
           stride: int, frames: Frames) -> list:
    """refine_boxes_multistart's ascent for every box of b0 at once -> one
    record per checkpoint: IoU, predicted IoU, head, the heads' predictions,
    the box (1024 frame) and the strided mask."""
    import steer_models_scan as SMS
    N = b0.shape[0]
    cx, cy = (b0[:, 0] + b0[:, 2]) / 2, (b0[:, 1] + b0[:, 3]) / 2
    w, h = (b0[:, 2] - b0[:, 0]).clamp(min=2.0), (b0[:, 3] - b0[:, 1]).clamp(min=2.0)
    params = torch.stack([cx, cy, w, h], dim=1).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([params], lr=lr)

    def to_boxes(p):
        ww, hh = p[:, 2].clamp(min=2.0), p[:, 3].clamp(min=2.0)
        return torch.stack([p[:, 0] - ww / 2, p[:, 1] - hh / 2, p[:, 0] + ww / 2, p[:, 1] + hh / 2], dim=1)

    ar = torch.arange(N, device=b0.device)
    out = []
    with torch.inference_mode(False), torch.enable_grad():
        for step in range(steps + 1):
            bx = to_boxes(params)
            low_all, iou_all = dec.decode(bx)
            low, head = dec.pick(low_all, iou_all, multimask)
            scores = iou_all[ar, head]
            if step % every == 0 or step == steps:
                with torch.no_grad():
                    m = dec.masks(low.detach())
                    out.append({"step": step, "iou": SMS.batch_iou(m, gt).cpu().numpy(),
                                "pred": scores.detach().float().cpu().numpy(), "head": head.cpu().numpy(),
                                "heads_pred": iou_all[:, 1:].detach().float().cpu().numpy(),
                                "box_1024": frames.to_1024(bx), "small": m[:, ::stride, ::stride].cpu()})
            if step == steps:
                break
            opt.zero_grad()
            (-scores.sum()).backward()
            opt.step()
    return out


def bon_pass(runner, dec, boxes_1024: np.ndarray, gt, a, stride: int) -> dict:
    """The Y BoN boxes of every box -> per mode (N, Y) IoU / predicted IoU / head,
    the heads' predictions, the strided masks, and the boxes."""
    import refine_box_iou_grad as R
    import steer_models_scan as SMS
    N = len(boxes_1024)
    pert = np.stack([R.sample_bon_boxes(list(map(float, boxes_1024[i])), dec.p, a.Y, a.sigma, a.sigma_center,
                                        a.perturb_mode, a.bon_seed).cpu().numpy().reshape(-1, 4) for i in range(N)])
    Y = pert.shape[1]
    flat = pert.reshape(-1, 4)
    res = {mm: {"iou": [], "pred": [], "head": [], "heads_pred": [], "small": []} for mm in MODES}
    for s in range(0, len(flat), a.batch):
        b = dec.model_boxes(flat[s:s + a.batch])
        low_all, iou_all = runner.forward(b)
        ar = torch.arange(len(b), device=b.device)
        for mm in MODES:
            low, head = dec.pick(low_all, iou_all, bool(mm))
            m = dec.masks(low)
            r = res[mm]
            r["iou"].append(SMS.batch_iou(m, gt).cpu().numpy())
            r["pred"].append(iou_all[ar, head].float().cpu().numpy())
            r["head"].append(head.cpu().numpy())
            r["heads_pred"].append(iou_all[:, 1:].float().cpu().numpy())
            r["small"].extend(list(m[:, ::stride, ::stride].cpu()))
    out = {"boxes": pert}
    for mm in MODES:
        r = res[mm]
        out[mm] = {k: np.concatenate(r[k]).reshape((N, Y) + np.concatenate(r[k]).shape[1:])
                   for k in ("iou", "pred", "head", "heads_pred")}
        out[mm]["small"] = [r["small"][i * Y:(i + 1) * Y] for i in range(N)]
    return out


def feat_row(DS, small, box, user_box, orig_hw, stride, pred, head, head_preds) -> dict:
    return DS.cand_feats(small, box, user_box, orig_hw, stride, float(pred), head=int(head), head_preds=head_preds)


def agree_block(R, DS, smalls) -> list:
    """f_agree_* / f_cluster_* of every member of one family (dump_selector_data.agree_feats)."""
    S = torch.stack(list(smalls))
    M = R._iou_matrix(S)
    lab = R._cluster(M, 0.90)
    sizes = np.bincount(lab)
    return [DS.agree_feats(M[j], j, sizes[lab[j]] / len(S), sizes.size) for j in range(len(S))]


def image_rows(ctx: dict, item: dict, res: dict, bon: dict, grad: dict, clean: float, groups: np.ndarray,
               alphas, a) -> tuple:
    """The pool rows and the per-box rows of one image."""
    import dump_selector_data as DS
    import refine_box_iou_grad as R
    from heatmaps.comp_hw_smoothed import get_original_size
    st, orig_hw = a.stride, tuple(get_original_size(ctx["predictor"]))
    nan_agree = DS.agree_feats(None, 0, NAN, NAN)
    steps_ck = [c["step"] for c in grad[0]]
    pool, per_box = [], []
    for i, row in enumerate(item["rows"]):
        ub = item["boxes"][i].astype(np.float64)
        ident = {"image_name": item["stem"], "kind": row["kind"], "user": row["user"], "attempt": row["attempt"],
                 "dataset": item["dataset"], "half": item["half"], "fold": item["fold"], "group": groups[i]}
        bpick = {mm: int(np.argmax(bon[mm]["pred"][i])) for mm in MODES}
        ref = {"undef": res[0]["small"][0][i], "hsel": res[1]["small"][0][i],
               "bon_mm0": bon[0]["small"][i][bpick[0]], "bon_mm1": bon[1]["small"][i][bpick[1]],
               "grad_mm0": grad[0][-1]["small"][i], "grad_mm1": grad[1][-1]["small"][i]}
        gt_ref = {"gt_undef_iou": float(res[0]["iou"][0][i]), "gt_hsel_iou": float(res[1]["iou"][0][i]),
                  "gt_clean_iou": clean}
        cands = []

        def add(kind, cid, slot, small, box, pred, head, head_preds, target, n_fwd, step=-1, alpha=NAN, agree=None):
            r = dict(ident)
            r.update({"cand_kind": kind, "cand_id": cid, "method_slot": slot, "start": 0 if step >= 0 else -1,
                      "step": step, "y": -1, "n_forwards": n_fwd, "n_batched_fwd": 1,
                      "f_step": float(step) if step >= 0 else NAN,
                      "f_step_frac": step / max(1, a.steps) if step >= 0 else NAN,
                      "f_is_start0": 1.0 if step >= 0 else NAN, "f_steer_alpha": alpha})
            r.update(feat_row(DS, small, box, ub, orig_hw, st, pred, head, head_preds))
            r.update(agree if agree is not None else nan_agree)
            for k, mk in ref.items():
                r[f"x_iou_{k}"] = R._pair_iou(small, mk)
            r["target_iou"] = float(target)
            r.update(gt_ref)
            cands.append(r)

        add("undef", 0, "undef", ref["undef"], ub, res[0]["pred"][0][i], res[0]["head"][0][i], None,
            res[0]["iou"][0][i], 1)
        add("hsel", 0, "hsel", ref["hsel"], ub, res[1]["pred"][0][i], res[1]["head"][0][i],
            res[1]["heads_pred"][0][i], res[1]["iou"][0][i], 1)
        for mm in MODES:
            ck = [c for c in grad[mm] if c["step"] > 0]
            for j, c in enumerate(ck):
                add(f"grad_mm{mm}", j, f"grad_mm{mm}" if j == len(ck) - 1 else "", c["small"][i], c["box_1024"][i],
                    c["pred"][i], c["head"][i], c["heads_pred"][i] if mm else None, c["iou"][i], c["step"] + 1,
                    step=c["step"])
            ag = agree_block(R, DS, bon[mm]["small"][i])
            for j in range(bon[mm]["iou"].shape[1]):
                add(f"bon_mm{mm}", j, f"bon_mm{mm}" if j == bpick[mm] else "", bon[mm]["small"][i][j],
                    bon["boxes"][i][j], bon[mm]["pred"][i, j], bon[mm]["head"][i, j],
                    bon[mm]["heads_pred"][i, j] if mm else None, bon[mm]["iou"][i, j], j + 1, agree=ag[j])
            r = res[mm]
            ag = agree_block(R, DS, [r["small"][j + 1][i] for j in range(len(alphas))])
            for j, al in enumerate(alphas):
                add(f"steer_mm{mm}", j, "", r["small"][j + 1][i], ub, r["pred"][j + 1][i], r["head"][j + 1][i],
                    r["heads_pred"][j + 1][i] if mm else None, r["iou"][j + 1][i], j + 1, alpha=float(al), agree=ag[j])
        pool += cands
        pb = dict(ident)
        pb.update({"undef_iou": res[0]["iou"][0][i], "undef_pred": res[0]["pred"][0][i],
                   "undef_head": int(res[0]["head"][0][i]), "hsel_iou": res[1]["iou"][0][i],
                   "hsel_pred": res[1]["pred"][0][i], "clean_iou": clean})
        for mm in MODES:
            for c in grad[mm]:
                pb[f"grad_mm{mm}_s{c['step']}_iou"] = c["iou"][i]
            pb[f"grad_mm{mm}_iou"], pb[f"grad_mm{mm}_pred"] = grad[mm][-1]["iou"][i], grad[mm][-1]["pred"][i]
            pb[f"bon_mm{mm}_iou"] = bon[mm]["iou"][i, bpick[mm]]
            pb[f"bon_mm{mm}_pred"] = bon[mm]["pred"][i, bpick[mm]]
            pb[f"bon_mm{mm}_oracle_iou"] = bon[mm]["iou"][i].max()
            for j, al in enumerate(alphas):
                pb[f"steer_mm{mm}_{_a(al)}_iou"] = res[mm]["iou"][j + 1][i]
        pb["oracle_iou"] = max(c["target_iou"] for c in cands)
        pb["n_candidates"] = len(cands)
        per_box.append(pb)
    return pool, per_box, steps_ck


def check_against_repo(ctx, item, bon: dict, grad: dict, a, checks: dict):
    """The batched ascent and BoN vs the repo's own one-box functions."""
    import refine_box_iou_grad as R
    import steer_models_scan as SMS
    c = checks["repo_equivalence"]
    gt_cpu = item["gt"].cpu()
    for i in range(min(a.check_boxes, len(item["rows"]))):
        box = list(map(float, item["boxes"][i]))
        for mm in MODES:
            _, ex = R.refine_boxes_multistart(box, ctx["predictor"], ctx["device"], n_starts=1, steps=a.steps, lr=a.lr,
                                              multimask=bool(mm), gt_tensor=None, checkpoint_every=0)
            d = abs(float(SMS.batch_iou(ex["masks"][:1], gt_cpu)[0]) - float(grad[mm][-1]["iou"][i]))
            c[f"grad_mm{mm}"]["compared"] += 1
            c[f"grad_mm{mm}"]["max_abs_iou_diff"] = max(c[f"grad_mm{mm}"]["max_abs_iou_diff"], d)
        m, _, _ = R.best_of_n_batched(box, ctx["predictor"], ctx["device"], a.Y, a.sigma, a.sigma_center,
                                      a.perturb_mode, a.bon_seed)
        ours = float(bon[0]["iou"][i, int(np.argmax(bon[0]["pred"][i]))])
        d = abs(float(SMS.batch_iou(m.bool()[None], gt_cpu)[0]) - ours)
        c["bon_mm0"]["compared"] += 1
        c["bon_mm0"]["max_abs_iou_diff"] = max(c["bon_mm0"]["max_abs_iou_diff"], d)


def run(args) -> int:
    import selector_data as SD
    import steer_models_candidates as SMC
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
              "public_single": {"compared": 0, "max_abs_iou_diff": 0.0, "n_over_1e-3": 0, "where": []},
              "repo_equivalence": {k: {"compared": 0, "max_abs_iou_diff": 0.0} for k in ("grad_mm0", "grad_mm1", "bon_mm0")}}

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
    frames = Frames(dec)
    ctx = {"predictor": predictor, "device": device}

    root = Path(args.root)
    datasets = list(dict.fromkeys(["COCO", *args.datasets]))
    index = SUS.index_study(root, args.use, datasets)
    A, B = SUS.split_halves(index["COCO"].keys(), args.split_seed)
    scan = SMC.load_scan(Path(args.scan_dir), site, dec, A, args)
    checks["scan"] = scan["check"]
    print(f"{args.model_name}: {dec.T} tokens, fallback {dec.dynamic}, TF32 matmul {tf32['matmul_after_load']} -> "
          f"{tf32['matmul_run']}; steering {site.id} from {args.scan_dir}: ok {scan['check']['ok']}", flush=True)
    if not scan["check"]["ok"]:
        write_checks()
        raise SMS.CheckFailed(f"--scan_dir is not this model's run: {scan['check']}")
    fold_of = assign_folds(A, scan["check"]["folds"], scan["check"]["seed"])
    Vdev = {k: torch.as_tensor(v, device=device) for k, v in scan["V"].items()}
    alphas = list(args.alphas)
    lim = (lambda xs: list(xs)[:args.limit_images]) if args.limit_images > 0 else list
    plan = [("COCO", "A", s) for s in lim(A)] + [("COCO", "B", s) for s in lim(B)]
    plan += [(d, "-", s) for d in datasets[1:] for s in lim(index[d].keys())]

    fh = gzip.open(out / "pool.csv.gz", "wt", newline="", encoding="utf-8")
    writer, per_box, mm_rows, seen, n_pool, steps_ck = None, [], [], set(), 0, None
    try:
        for k, (ds, half, stem) in enumerate(plan):
            table = f"COCO_{half}" if ds == "COCO" else ds
            fold = fold_of[stem] if half == "A" else -1
            item = SMS.load_item(dec, root, ds, half, stem, index[ds], fold)
            first = table not in seen
            seen.add(table)
            res = SMC.image_pass(runner, dec, site, item, Vdev[fold if half == "A" else "all"], alphas, checks,
                                 first, args.stride)
            bon = bon_pass(runner, dec, item["boxes"], item["gt"], args, args.stride)
            grad = {mm: ascend(dec, item["b"], item["gt"], args.steps, args.lr, bool(mm), args.checkpoint_every,
                               args.stride, frames) for mm in MODES}
            if first and args.check_boxes > 0:
                check_against_repo(ctx, item, bon, grad, args, checks)
            low_c, _ = dec.pick(*runner.forward(dec.model_boxes(np.asarray([item["best_box"]], np.float32))), False)
            clean = float(SMS.batch_iou(dec.masks(low_c), item["gt"])[0])
            g = pd.DataFrame(item["rows"]).assign(undef_iou=res[0]["iou"][0])
            groups = SUS.assign_groups(g).to_numpy()
            pool, pb, steps_ck = image_rows(ctx, item, res, bon, grad, clean, groups, alphas, args)
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(pool[0]))
                writer.writeheader()
            writer.writerows(pool)
            n_pool += len(pool)
            per_box += pb
            hw = get_original_size(predictor)
            ih, iw = ResizeLongestSide.get_preprocess_shape(hw[0], hw[1], 1024)
            bx = item["boxes"]
            bw, bh = (bx[:, 2] - bx[:, 0]).clip(min=1e-3), (bx[:, 3] - bx[:, 1]).clip(min=1e-3)
            mm_rows.append(SMC.multimask_rows(item, res, alphas, bw * bh / float(ih * iw), np.log(bw / bh), table))
            P = pd.DataFrame(pb)
            print(f"[{table}] [{k + 1}/{len(plan)}] {stem}: undef {P.undef_iou.mean():.4f} hsel {P.hsel_iou.mean():.4f} "
                  f"grad {P.grad_mm0_iou.mean():.4f}/{P.grad_mm1_iou.mean():.4f} bon {P.bon_mm0_iou.mean():.4f}/"
                  f"{P.bon_mm1_iou.mean():.4f} oracle {P.oracle_iou.mean():.4f}, {len(pool)} candidates "
                  f"({time.time() - t0:.0f}s)", flush=True)
            del item, res, bon, grad
    except SMS.CheckFailed as e:
        write_checks()
        print(f"[check failed] {e}", file=sys.stderr, flush=True)
        return 2
    finally:
        fh.close()

    PB = pd.DataFrame(per_box)
    PB.to_csv(out / "per_box.csv.gz", index=False, float_format="%.6f")
    D = pd.concat(mm_rows, ignore_index=True)
    D["group"] = SUS.assign_groups(D.rename(columns={"tok0_iou0": "undef_iou"})).to_numpy()
    D.to_csv(out / "rows.csv.gz", index=False, float_format="%.6f")
    gcount = PB.groupby(["image_name", "group"]).size().unstack(fill_value=0)
    bad = int((~((gcount.get("bottom50", 0) == 12) & (gcount.get("top50", 0) == 13))).sum())
    checks["groups"] = {"images": int(len(gcount)), "n_images_not_12_13": bad, "ok": bad == 0}
    ref = SMS.read_reference(args.reference_csv, args.use)
    for key, col, ref_col in (("repro", "undef_iou", "undefended_iou"), ("repro_headsel", "hsel_iou", "headsel_iou")):
        r = SMS.repro(PB, ref, col, ref_col, args.repro_tol)
        if r.get("available"):
            r["max_abs_diff"], r["frac_over"] = r["max_abs"], r["frac_over_tol"]
        checks[key] = r
    write_checks()
    meta = {"script": "build_candidate_pool.py", "created": time.strftime("%Y-%m-%d %H:%M"),
            "model_name": args.model_name, "checkpoint": args.checkpoint_path, "site": site.id, "tokens": "all",
            "alphas": alphas, "grad_checkpoints": steps_ck, "layout": dec.layout(), "args": vars(args),
            "boxes": int(len(PB)), "pool_rows": n_pool, "candidates_per_box": float(PB.n_candidates.mean()),
            "versions": SMS.package_versions(), "seconds": round(time.time() - t0)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=SMS._json_default), encoding="utf-8")
    eq = checks["repo_equivalence"]
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}: {len(PB)} boxes, {n_pool} candidates; zero dose "
          f"{checks['zero_dose']['bit_exact']}/{checks['zero_dose']['forwards']}, own forward "
          f"{checks['own_forward']['bit_exact']}/{checks['own_forward']['compared']}; vs repo one box at a time: "
          + ", ".join(f"{k} {v['max_abs_iou_diff']:.3g} ({v['compared']})" for k, v in eq.items())
          + f"; groups off {bad}", flush=True)
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
                                                     "ACDC", "BUID", "MedScribble"])
    p.add_argument("--scan_dir", required=True, help="steer_models_scan.py run of this model (the steering vector)")
    p.add_argument("--site", default="L2.norm3")
    p.add_argument("--alphas", type=_floats, default=_floats("1,2,3,4,6,8,12,16,24,32"))
    p.add_argument("--steps", type=int, default=30, help="gradient ascent steps")
    p.add_argument("--lr", type=float, default=1.0, help="Adam lr (~pixels per step), as grad_results")
    p.add_argument("--checkpoint_every", type=int, default=5)
    p.add_argument("--Y", type=int, default=16, help="BoN boxes (the user's box included)")
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--sigma_center", type=float, default=0.03)
    p.add_argument("--perturb_mode", default="size_center", choices=["size", "size_center"])
    p.add_argument("--bon_seed", type=int, default=42)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=64, help="boxes per decoder pass")
    p.add_argument("--stride", type=int, default=4, help="feature masks, as dump_selector_data")
    p.add_argument("--check_boxes", type=int, default=3, help="boxes per subset checked against the repo's functions")
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
