"""
dump_selector_data.py
=====================
ONE pass over the cases, ONE long-format CSV: every mask any policy in this
project could return, described by the same feature schema and labelled with
its true IoU. The point is that the GPU is used once, on the server, and every
selector question after that is a pandas groupby on a laptop.

Four questions come out of the same file:

  1. early stop / step selection -- rows where cand_kind == "grad", grouped by
     case and walked in `step` order.
  2. best_of_n by a learned selector -- rows where cand_kind == "bon", against
     the argmax of `f_pred`, which is today's defence.
  3. choosing between methods -- rows where method_slot is set (it is written
     empty elsewhere, which pandas reads back as NaN, so filter on .notna()):
     exactly one row per (case, method) for undef / hsel / bon / grad.
  4. CatBoost vs LightGBM vs a linear model -- the same three matrices, a
     different estimator. Nothing needs re-running to answer this.

COLUMN CONVENTIONS -- the whitelist that keeps all four honest:

  f_*        CAUSAL features. Available at the moment that candidate is
             produced, without knowing what any later or competing candidate
             did. Safe everywhere, including early stop.
  x_*        CROSS-METHOD features: agreement with the other methods' final
             masks. These exist only once the whole pipeline has run, so they
             are legitimate for (2) and (3) and are a LEAK for (1) -- a grad
             checkpoint at step 5 cannot know the mask of step 20.
  target_iou the label.
  gt_*       true IoUs of the case's reference masks. For stratifying and
             reporting ONLY; every one of them is derived from the ground
             truth and is never a feature.
  everything else is identity, position or cost.

Deliberately NOT written: within-case rank, z-score and margin of f_pred.
Those depend on which candidates you are choosing among, and that is decided
per question at analysis time -- ranking a grad checkpoint against the bon pool
would be meaningless. They are two lines of pandas over `f_pred`.

COST. `n_forwards` is what it took to have this candidate in hand, so an
early-stop policy can be priced, not just scored. For the bon pool it also
means a "best of the first k" analysis falls out of the same rows.

Resolution. Feature masks are subsampled by --stride; every f_mask_* is a
ratio, and at stride 4 they move in the 4th decimal (verified against stride
1). target_iou is always measured on the full-resolution mask. The one
exception is the grad rows, whose f_* block is read off the 256px logits --
that is how the ascent already computes them, for free -- so grad rows are
never pooled with bon rows in any of the four questions above.

    python scripts/dump_selector_data.py --model_name SAM3 \\
        --checkpoint_path /.../SAM3/sam3.pt \\
        --dataset user_study --root /.../user_study/FOR_TEST --use m \\
        --out_csv results/sam3_selector_pool.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refine_box_iou_grad as R
from heatmaps.comp_hw_smoothed import get_original_size, load_model
from heatmaps.defend_critical_shifts import (
    _find_file,
    _predict_single_box,
    _prepare_image,
    boxes_to_original,
)
from heatmaps.env_dispatch import maybe_dispatch_to_env

_AGREE_KEYS = ("f_agree_mean", "f_agree_min", "f_agree_max",
               "f_cluster_frac", "f_n_clusters")


def cand_feats(mask_small, box_1024, bad_box_1024, orig_hw, stride,
               pred, head=0, head_preds=None) -> dict:
    """The unified f_* block: one candidate mask, described the same way
    whichever policy produced it.

    mask_small is already subsampled by `stride`; box_1024 is this candidate's
    prompt box in the repo's 1024 frame, bad_box_1024 the user's own box that
    every displacement is measured against.
    """
    box_orig = boxes_to_original(np.asarray(box_1024, dtype=np.float64)[None],
                                 orig_hw)[0].astype(np.float64)
    f = {"f_pred": float(pred), "f_head": float(head)}
    if head_preds is not None and np.size(head_preds) > 1:
        hp = np.sort(np.asarray(head_preds, dtype=np.float64).ravel())
        f["f_head_pred_spread"] = float(hp[-1] - hp[0])
        f["f_head_pred_std"] = float(hp.std())
    else:
        # single-output token: there are no sibling heads to disagree with.
        # NaN, not 0 -- "no disagreement measured" is not "the heads agree".
        f["f_head_pred_spread"] = f["f_head_pred_std"] = float("nan")
    f.update(R._mask_shape_feats(mask_small, np.asarray(box_orig) / float(stride),
                                 "f_mask"))
    f.update(R._box_move_feats(box_1024, bad_box_1024, "f_box"))
    f.update(R._box_prior_feats(box_orig, orig_hw, "f_prior"))
    return f


def agree_feats(agree_row, self_idx, cluster_frac, n_clusters) -> dict:
    """How far this candidate agrees with the others in ITS OWN pool -- the
    block SAM's predicted-IoU head has no access to, and the reason to expect a
    learned selector to beat it.

    A candidate with no pool (the undefended mask) gets NaN, not 1.0: a
    reduction over no pairs is missing, not perfect agreement, and CatBoost
    reads NaN as missing.
    """
    if agree_row is None:
        return {k: float("nan") for k in _AGREE_KEYS}
    oth = np.delete(np.asarray(agree_row, dtype=np.float64), self_idx)
    if oth.size == 0:
        return {k: float("nan") for k in _AGREE_KEYS}
    return {"f_agree_mean": float(oth.mean()),
            "f_agree_min": float(oth.min()),
            "f_agree_max": float(oth.max()),
            "f_cluster_frac": float(cluster_frac),
            "f_n_clusters": float(n_clusters)}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="user_study",
                   choices=["user_study", "critical_shifts"])
    p.add_argument("--root", default=None, help="user_study root (FOR_TEST)")
    p.add_argument("--use", default="m", help="user_study annotation kinds: m|p|mp")
    p.add_argument("--critical_shifts", default="critical_shifts.json")
    p.add_argument("--images_dir", default=None)
    p.add_argument("--masks_dir", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sample_images", type=int, default=0,
                   help="random subset of IMAGES, keeping all of each one's "
                        "annotations (0 = all)")
    p.add_argument("--sample_seed", type=int, default=0)

    p.add_argument("--model_name", default="SAM")
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument("--steps", type=int, default=20, help="gradient ascent steps")
    p.add_argument("--lr", type=float, default=3.0, help="Adam lr (~pixels/step)")
    p.add_argument("--multimask", action="store_true", default=False)
    p.add_argument("--multistart", type=int, default=4,
                   help="parallel ascents. >1 is what makes f_agree_* exist on "
                        "the grad rows, and agreement across starts was the "
                        "strongest feature the step selector had. One batched "
                        "forward per step regardless of how many starts, so "
                        "raising this is cheaper than it looks.")
    p.add_argument("--multistart_seed", type=int, default=0)
    p.add_argument("--checkpoint_every", type=int, default=5,
                   help="emit a grad candidate every k steps (the last step "
                        "is always emitted)")
    p.add_argument("--bon_start", action="store_true", default=False,
                   help="seed the ascent with the best_of_n box as an extra "
                        "start, as the main script does. OFF here so the grad "
                        "pool is self-contained and its n_forwards does not "
                        "silently include the Y best_of_n forwards.")

    p.add_argument("--Y", type=int, default=16)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--sigma_center", type=float, default=0.03)
    p.add_argument("--perturb_mode", default="size", choices=["size", "size_center"])
    p.add_argument("--bon_seed", type=int, default=42,
                   help="must match the main script's best_of_n seed (42) for "
                        "the dumped pool to be the pool the defence searched")

    p.add_argument("--stride", type=int, default=4,
                   help="subsample masks by this factor for the f_* features. "
                        "target_iou is always measured at full resolution.")
    p.add_argument("--out_csv", default="results/selector_pool.csv")
    return p.parse_args()


def main():
    a = parse_args()
    # SAM-HQ2 / SAM3 live in separate conda envs; re-execs and exits, no-op
    # for the other backends.
    maybe_dispatch_to_env(a.model_name, __file__)

    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    by_image, images_dir, masks_dir, dataset_kind = R.load_tasks(a)
    predictor = load_model(model_name=a.model_name, model_type=a.model_type,
                           checkpoint=a.checkpoint_path, device=device)
    # freeze: backward then only flows to the box leaf
    for prm in predictor.model.parameters():
        prm.requires_grad_(False)

    st = max(1, a.stride)
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    fh = open(a.out_csv, "w", newline="", encoding="utf-8")
    writer = None
    n_rows = n_cases = 0
    stopped = False

    R._install_stop_handlers()
    pbar = tqdm(total=sum(len(v) for v in by_image.values()),
                desc=f"{a.model_name} cases", unit="case", dynamic_ncols=True)
    for image_name, raw_tasks in by_image.items():
        if R._STOP_REQUESTED:
            stopped = True
            break
        image_path = _find_file(images_dir, image_name)
        mask_path = _find_file(masks_dir, image_name)
        if image_path is None or mask_path is None:
            pbar.write(f"[warn] missing image/mask for {image_name}, skipping")
            pbar.update(len(raw_tasks))
            continue
        try:
            _prepare_image(str(image_path), predictor)
            gt_tensor = R._load_gt(mask_path, predictor)
        except Exception as e:                          # noqa: BLE001
            pbar.write(f"[warn] setup failed for {image_name}: {e}")
            pbar.update(len(raw_tasks))
            continue

        cases = (R.build_user_cases(raw_tasks, predictor, gt_tensor)
                 if dataset_kind == "user_study" else raw_tasks)
        # build_user_cases drops degenerate user masks -- keep the bar honest
        pbar.update(max(0, len(raw_tasks) - len(cases)))
        orig_hw = get_original_size(predictor)

        for case in cases:
            if R._STOP_REQUESTED:
                stopped = True
                break
            bad_box_np = np.asarray(case["bad_box"], dtype=np.float64)
            bad_box = torch.tensor(case["bad_box"], dtype=torch.float32)

            rows: list[dict] = []
            small: list = []        # small[i] is the strided mask of rows[i]

            def emit(row, mask_full, _r=rows, _s=small):
                """One candidate: label it, and keep its mask for the x_* pass."""
                row["target_iou"] = R._iou(gt_tensor, mask_full)
                _r.append(row)
                _s.append(mask_full[::st, ::st].bool())

            # ---- undefended: token-0 on the user's own box, one forward ----
            undef_mask, undef_pred = _predict_single_box(
                bad_box, predictor, device, boxes_already_transformed=True,
                return_score=True)
            r = {"cand_kind": "undef", "cand_id": 0, "method_slot": "undef",
                 "start": -1, "step": -1, "y": -1,
                 "n_forwards": 1, "n_batched_fwd": 1,
                 "f_step": float("nan"), "f_step_frac": float("nan"),
                 "f_is_start0": float("nan")}
            r.update(cand_feats(undef_mask[::st, ::st], bad_box_np, bad_box_np,
                                orig_hw, st, undef_pred))
            r.update(agree_feats(None, 0, 0, 0))
            emit(r, undef_mask)
            undef_iou = rows[-1]["target_iou"]

            # Clean reference: the un-attacked box. NOT a candidate -- no policy
            # can produce it at inference -- but it is the ceiling the defence
            # is measured against, so it rides along as a gt_* column.
            clean_mask, _ = _predict_single_box(
                torch.tensor(case["best_box"], dtype=torch.float32), predictor,
                device, boxes_already_transformed=True, return_score=True)
            clean_iou = R._iou(gt_tensor, clean_mask)

            # ---- best_of_n pool: the Y boxes the defence actually searches --
            perturbed = R.sample_bon_boxes(case["bad_box"], predictor, a.Y,
                                           a.sigma, a.sigma_center,
                                           a.perturb_mode, a.bon_seed)
            bon_masks, bon_scores = R._predict_boxes(
                perturbed.float().to(device), predictor, multimask=False)
            bon_masks = bon_masks[:, 0].cpu().bool()
            bon_s = bon_scores[:, 0].float().cpu().numpy().astype(np.float64)
            bon_1024 = perturbed.cpu().numpy().astype(np.float64).reshape(-1, 4)
            bon_small = bon_masks[:, ::st, ::st]
            bon_M = R._iou_matrix(bon_small)
            bon_lab = R._cluster(bon_M, 0.90)
            bon_sizes = np.bincount(bon_lab)
            n_bon = int(bon_masks.shape[0])
            bon_pick = int(np.argmax(bon_s))
            for i in range(n_bon):
                r = {"cand_kind": "bon", "cand_id": i,
                     "method_slot": "bon" if i == bon_pick else "",
                     "start": -1, "step": -1, "y": i,
                     # candidates are usable in draw order, so this doubles as
                     # the x-axis of a "best of the first k" curve
                     "n_forwards": i + 1, "n_batched_fwd": 1,
                     "f_step": float("nan"), "f_step_frac": float("nan"),
                     "f_is_start0": float("nan")}
                r.update(cand_feats(bon_small[i], bon_1024[i], bad_box_np,
                                    orig_hw, st, bon_s[i]))
                r.update(agree_feats(bon_M[i], i,
                                     bon_sizes[bon_lab[i]] / n_bon,
                                     bon_sizes.size))
                emit(r, bon_masks[i])
            bon_pick_small = bon_small[bon_pick]

            # ---- gradient pool: n_starts x checkpoints ---------------------
            # The ascent stays written once, in refine_boxes_multistart, with
            # its three backend branches; this callback only decides what a
            # checkpoint looks like as a row.
            grad_rows: list = []

            def row_fn(ctx, _gr=grad_rows):
                r = {"cand_kind": "grad", "cand_id": len(_gr),
                     "method_slot": "", "start": ctx["start"],
                     "step": ctx["step"], "y": -1,
                     # every start advances in the same batched forward, so the
                     # batched count is the one an early-stop policy saves on
                     "n_forwards": (ctx["step"] + 1) * a.multistart,
                     "n_batched_fwd": ctx["step"] + 1,
                     "f_step": float(ctx["step"]),
                     "f_step_frac": float(ctx["step_frac"]),
                     "f_is_start0": float(ctx["start"] == 0)}
                r.update(cand_feats(ctx["mask_low"], ctx["box_1024"], bad_box_np,
                                    orig_hw, 1.0 / ctx["lo"],
                                    ctx["pred"][ctx["head"]],
                                    head=ctx["head"], head_preds=ctx["pred"]))
                # agreement across the STARTS at this step, read off the 256px
                # logits: causal (nothing from a later step) and free
                r.update(agree_feats(ctx["agree"], ctx["start"],
                                     ctx["cluster_frac"], ctx["n_clusters"]))
                _gr.append((r, ctx["full_mask"]()))
                return r

            R.refine_boxes_multistart(
                case["bad_box"], predictor, device, n_starts=a.multistart,
                steps=a.steps, lr=a.lr, multimask=a.multimask,
                gt_tensor=None, seed=a.multistart_seed,
                extra_start=(bon_1024[bon_pick] if a.bon_start else None),
                checkpoint_every=max(1, a.checkpoint_every),
                cand_row_fn=row_fn)
            for r, full in grad_rows:
                emit(r, full)

            # head-select and grad_final are POSITIONS in the grad pool, not
            # separate computations: start 0 at step 0 is the multimask
            # head-select on the user's box, and start 0's last checkpoint is
            # grad_final. Marking them costs nothing and makes question (3) a
            # one-line filter.
            s0 = [i for i, r in enumerate(rows)
                  if r["cand_kind"] == "grad" and r["start"] == 0]
            if s0:
                rows[min(s0, key=lambda i: rows[i]["step"])]["method_slot"] = "hsel"
                rows[max(s0, key=lambda i: rows[i]["step"])]["method_slot"] = "grad"
            slot = {r["method_slot"]: i for i, r in enumerate(rows) if r["method_slot"]}

            # ---- x_*: agreement with each method's own output ---------------
            # Post-hoc by construction: a grad checkpoint cannot know these, so
            # they are a leak for question (1) and legitimate for (2) and (3).
            ref = {"undef": small[slot["undef"]] if "undef" in slot else None,
                   "hsel": small[slot["hsel"]] if "hsel" in slot else None,
                   "bon": bon_pick_small,
                   "grad": small[slot["grad"]] if "grad" in slot else None}
            ident = {"image_name": image_name, "kind": case.get("kind", ""),
                     "user": case.get("user", ""),
                     "attempt": case.get("attempt", "")}
            out = []
            for i, r in enumerate(rows):
                for nm, m in ref.items():
                    # every kind's mask enters `small` at full resolution
                    # strided by --stride, grad checkpoints included, so these
                    # are comparable across kinds -- unlike f_mask_*, which the
                    # grad rows read off the 256px logits
                    ok = m is not None and tuple(m.shape) == tuple(small[i].shape)
                    r[f"x_iou_{nm}"] = R._pair_iou(small[i], m) if ok else float("nan")
                r["gt_undef_iou"] = undef_iou
                r["gt_clean_iou"] = clean_iou
                r["gt_bad_iou_json"] = float(case.get("bad_iou", float("nan")))
                r["gt_best_iou_json"] = float(case.get("best_iou", float("nan")))
                out.append({**ident, **r})

            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(out[0]))
                writer.writeheader()
            writer.writerows(out)
            n_rows += len(out)
            n_cases += 1
            pbar.update(1)
            pbar.set_postfix(rows=n_rows, refresh=False)
        if stopped:
            break

    pbar.close()
    fh.close()
    if stopped:
        print(f"[stop] interrupted after {n_cases} cases -- the CSV is valid, "
              f"just partial", file=sys.stderr)
    n_ck = len(range(0, a.steps + 1, max(1, a.checkpoint_every)))
    if a.steps % max(1, a.checkpoint_every):
        n_ck += 1                                   # the last step is forced
    n_starts = a.multistart + (1 if a.bon_start else 0)
    print(f"\n{n_rows} candidate rows over {n_cases} cases -> {a.out_csv}")
    print(f"  per case: 1 undef + {a.Y} bon + {n_starts} starts x {n_ck} "
          f"checkpoints = {1 + a.Y + n_starts * n_ck}")
    print("  features: f_* everywhere; + x_* only for the bon and "
          "method-choice questions; never gt_* or target_iou")


if __name__ == "__main__":
    main()
