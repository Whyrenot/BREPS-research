"""
dump_steer_candidates.py
========================
Steering masks as candidates for the CatBoost selector: one row per (case,
alpha), in the schema of dump_selector_data.py, so the rows can be appended to
its pool (results/sam1_selector_pool_m.csv) on a laptop and the selector
retrained there (scripts/selector_steer.py). Nothing of the existing pool is
recomputed.

CANDIDATES  every user-study case (--use m, all subsets): the user's box with
            the frozen COCO A vector of steer_user_study (--vector) added at
            L2.norm2, all 7 tokens, at every alpha of --alphas. On the COCO A
            images, which the vector was estimated on, the vector of the other
            folds of A is used instead (cross-fitting, as steer_gate).

ROW SCHEMA  as dump_selector_data: cand_kind "steer", cand_id = alpha index,
  method_slot "", start / step / y = -1, n_forwards = alpha index + 1,
  n_batched_fwd 1, f_step* NaN;
  f_*       dump_selector_data.cand_feats on the steered mask (strided by
            --stride) with the user's box as the prompt (the box does not move:
            f_box_* = no displacement) and f_pred = the IoU head's score of the
            STEERED forward -- the IoU token is steered too, so the score rises
            with alpha; f_steer_alpha carries alpha so a selector can correct for
            it; f_agree_* / f_cluster_frac / f_n_clusters over the case's own
            steering pool, as the bon rows over theirs;
  x_*       agreement with the undefended mask, the multimask head-select mask
            on the user's box, and the best_of_n pick (same sampler, same seed:
            the pool's own pick); x_iou_grad = NaN -- the ascent is not re-run;
  target_iou  full resolution; gt_undef_iou / gt_clean_iou as the pool.
  chk_*     not features: this run's undefended, best_of_n-pick and
            head-select IoUs, for selector_steer.py to verify against the pool
            that the rows describe the same cases.

    CUDA_VISIBLE_DEVICES=0 python scripts/dump_steer_candidates.py \\
        --root /.../FOR_TEST --checkpoint_path /.../sam_vit_b_01ec64.pth \\
        --vector exp_res/steer_user_study/vector.npz \\
        --out_csv results/sam1_steer_candidates_m.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import dump_selector_data as DS  # noqa: E402
import refine_box_iou_grad as R  # noqa: E402
import steer_gate as G  # noqa: E402
import steer_user_study as SUS  # noqa: E402
from steer_sam_decoder import Cfg, Runner, assign_folds, batch_iou, build_sites, make_edit, zero_edit  # noqa: E402

NAN = float("nan")


def bon_pick_mask(case, predictor, device, a) -> torch.Tensor:
    """The pool's best_of_n pick: dump_selector_data's sampler, seed and call."""
    boxes = R.sample_bon_boxes(case["bad_box"], predictor, a.Y, a.sigma, a.sigma_center, a.perturb_mode, a.bon_seed)
    masks, scores = R._predict_boxes(boxes.float().to(device), predictor, multimask=False)
    return masks[int(torch.argmax(scores[:, 0])), 0].bool().cpu()


def hsel_mask(box, predictor, device) -> torch.Tensor:
    """Multimask head-select on the user's box: the pool's hsel slot (start 0,
    step 0 of the multimask ascent)."""
    masks, scores = R._predict_boxes(torch.as_tensor(box[None], dtype=torch.float32, device=device), predictor,
                                     multimask=True)
    return masks[0, int(torch.argmax(scores[0]))].bool().cpu()


def image_rows(runner: Runner, site, root: Path, ds: str, stem: str, recs, v: np.ndarray, vec_tag: str, a,
               checks: dict) -> list:
    p, dev, st = runner.p, runner.device, max(1, a.stride)
    cases, boxes, gt = SUS.load_image_cases(p, root, stem, recs, dev)
    gt_cpu = gt.cpu()
    orig_hw = tuple(p.original_size)
    b = torch.as_tensor(boxes, device=dev)
    m0, low0 = runner.forward(b)
    SUS.check_zero(runner, b, m0, low0, zero_edit, site.id, checks, f"{ds}/{stem}")
    iou0 = batch_iou(m0, gt).cpu().numpy()
    Vt = {site.id: {"dm": torch.as_tensor(v, device=dev)}}
    steered, preds = [], []
    for al in a.alphas:
        with G.PredHook(runner.sam) as ph:
            m, _ = runner.forward(b, edits={site.id: make_edit(Cfg(-1, "steer", site.id, a.tokens, "dm", al), site, Vt)})
        steered.append(m.cpu())
        preds.append(ph.result().cpu().numpy())
    # the GT's tight box (the same for every case of the image), as the pool's gt_clean_iou
    best = torch.as_tensor(np.asarray(cases[0]["best_box"], np.float32)[None], device=dev) if cases else None
    clean = float(batch_iou(runner.forward(best)[0], gt)[0]) if cases else NAN
    rows = []
    for i, case in enumerate(cases):
        box = np.asarray(case["bad_box"], dtype=np.float64)
        undef_small = m0[i, ::st, ::st].cpu()
        bon = bon_pick_mask(case, p, dev, a)
        hs = hsel_mask(boxes[i], p, dev)
        ref = {"undef": undef_small, "hsel": hs[::st, ::st], "bon": bon[::st, ::st]}
        full = [steered[j][i] for j in range(len(a.alphas))]
        small = torch.stack([f[::st, ::st] for f in full])
        M = R._iou_matrix(small)
        lab = R._cluster(M, 0.90)
        sizes = np.bincount(lab)
        for j, al in enumerate(a.alphas):
            r = {"image_name": stem, "kind": case["kind"], "user": case["user"], "attempt": case["attempt"],
                 "cand_kind": "steer", "cand_id": j, "method_slot": "", "start": -1, "step": -1, "y": -1,
                 "n_forwards": j + 1, "n_batched_fwd": 1, "f_step": NAN, "f_step_frac": NAN, "f_is_start0": NAN}
            r.update(DS.cand_feats(small[j], box, box, orig_hw, st, float(preds[j][i])))
            r.update(DS.agree_feats(M[j], j, sizes[lab[j]] / len(full), sizes.size))
            r["f_steer_alpha"] = float(al)
            r["target_iou"] = R._iou(gt_cpu, full[j])
            for nm, mk in ref.items():
                r[f"x_iou_{nm}"] = R._pair_iou(small[j], mk) if tuple(mk.shape) == tuple(small[j].shape) else NAN
            r["x_iou_grad"] = NAN
            r["gt_undef_iou"] = float(iou0[i])
            r["gt_clean_iou"] = clean
            r["gt_bad_iou_json"] = NAN
            r["gt_best_iou_json"] = NAN
            r["chk_undef_iou"] = float(iou0[i])
            r["chk_bon_pick_iou"] = R._iou(gt_cpu, bon)
            r["chk_hsel_iou"] = R._iou(gt_cpu, hs)
            r["chk_vector"] = vec_tag
            rows.append(r)
    return rows


def run(a) -> int:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    t0 = time.time()
    root = Path(a.root)
    datasets = list(SUS.DATASETS) if not a.datasets else list(dict.fromkeys(["COCO", *a.datasets]))
    index = SUS.index_study(root, a.use, datasets)
    A, _ = SUS.split_halves(index["COCO"], a.split_seed)
    device = torch.device(a.device)
    predictor = SUS.load_predictor(a.checkpoint_path, a.model_type, device)
    sites = build_sites(len(predictor.model.mask_decoder.transformer.layers))
    site = next(s for s in sites if s.id == a.site)
    runner = Runner(predictor, sites, a.batch, field_grid=64, field_margin=0.25)

    checks = {"zero_dose": {"forwards": 0, "bit_exact": 0, "failures": [], "ok": True}}
    out = Path(a.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    chk_path = out.with_suffix(".checks.json")
    got = SUS.load_vector(Path(a.vector), site.id, A, checks)
    if got is None:
        chk_path.write_text(json.dumps(checks, indent=2, default=SUS._json_default), encoding="utf-8")
        print(f"CHECK FAILED: --vector {a.vector} -- see {chk_path}", file=sys.stderr)
        return 1
    v = got[0]
    z = np.load(a.vector)
    images_A, d_img = [str(s) for s in z["images_A"]], z["d_img"]
    folds = assign_folds(images_A, a.folds, a.seed)
    fold_of = np.array([folds[s] for s in images_A])
    v_fold = {k: d_img[fold_of != k].mean(0) for k in sorted(set(fold_of))}

    fh = open(out, "w", newline="", encoding="utf-8")
    writer, n_rows = None, 0
    try:
        for ds in datasets:
            stems = list(index[ds])[:a.limit_images] if a.limit_images > 0 else list(index[ds])
            print(f"\n[{ds}] {len(stems)} images", flush=True)
            for i, stem in enumerate(stems):
                vv, tag = (v_fold[folds[stem]], f"fold{folds[stem]}") if stem in folds else (v, "full")
                rows = image_rows(runner, site, root, ds, stem, index[ds][stem], vv, tag, a, checks)
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    writer.writeheader()
                writer.writerows(rows)
                fh.flush()
                n_rows += len(rows)
                print(f"[{ds}] [{i + 1}/{len(stems)}] {stem}: {len(rows)} rows ({time.time() - t0:.0f}s)", flush=True)
    except SUS.CheckFailed as e:
        print(f"CHECK FAILED: {e}", file=sys.stderr)
        return 1
    finally:
        fh.close()
        chk_path.write_text(json.dumps(checks, indent=2, default=SUS._json_default), encoding="utf-8")
    print(f"\n{n_rows} steering candidates -> {out} ({(time.time() - t0) / 60:.1f} min)")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="user study FOR_TEST")
    p.add_argument("--use", default="m")
    p.add_argument("--datasets", nargs="*", default=[], choices=[d for d in SUS.DATASETS if d != "COCO"],
                   help="subsets besides COCO (default: every subset of the user study)")
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--site", default="L2.norm2")
    p.add_argument("--tokens", default="all")
    p.add_argument("--vector", default="exp_res/steer_user_study/vector.npz")
    p.add_argument("--alphas", type=SUS._floats, default=SUS._floats("1,2,3,4,6,8,12,16"))
    p.add_argument("--folds", type=int, default=5, help="cross-fitting folds on COCO A (as steer_gate)")
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--stride", type=int, default=4, help="as dump_selector_data")
    # best_of_n: dump_selector_data's defaults, so the pick is the pool's own
    p.add_argument("--Y", type=int, default=16)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--sigma_center", type=float, default=0.03)
    p.add_argument("--perturb_mode", default="size", choices=["size", "size_center"])
    p.add_argument("--bon_seed", type=int, default=42)
    p.add_argument("--limit_images", type=int, default=0, help="per subset, for a smoke run")
    p.add_argument("--out_csv", default="results/sam1_steer_candidates_m.csv")
    return p.parse_args(argv)


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
