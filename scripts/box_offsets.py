"""How far are users' boxes from the tight box of the GT mask: the BREPS user study
against a transfer set of fetch_transfer_sets.py (e.g. Open Images 2k). CPU only, no model.

Both boxes of a case are put in the same convention -- pixel edges, [x1, x2) -- in the
image the model sees (BREPS: the user's rectangle mask -> its bbox, as
refine_box_iou_grad.build_user_cases; the GT mask's bbox = the tight box; transfer set: the
annotated box and the tight box of its mask, recomputed the same way).

Per box (T = tight box, w, h its size; U = the user's box):
  iou            IoU(U, T)
  off_<side>     signed offset of each side, + = U is looser (outside T), - = U cuts the object
  side_rel       mean |offset| of the 4 sides, left/right / w, top/bottom / h  (fraction of size)
  side_px1024    mean |offset| of the 4 sides in pixels of SAM's 1024 frame
  loose_rel      mean signed offset (fraction of size): > 0 looser on average, < 0 tighter
  corner_rel     mean L2 distance of the two corners / diag(T)
  center_rel     centre shift / diag(T)
  log_area       log(area U / area T)
  cuts           any side cuts the object by more than 1% of its size

    python scripts/box_offsets.py --breps_root /.../FOR_TEST --oi_dir /.../OpenImagesV7_test2000
"""
import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PREFIX = {"COCO": "COCO_MVal_", "Berkeley": "Berkeley_", "GrabCut": "GrabCut_", "DAVIS": "DAVIS_",
          "PASCAL": "PASCAL_", "ADE20K": "ADE20K_", "TETRIS": "TETRIS_", "ACDC": "ACDC_", "BUID": "BUID_",
          "MedScribble": "MedScribble_"}
MEDICAL = ("ACDC", "BUID", "MedScribble")
USER_MASK_RE = re.compile(r"^(?P<stem>.+?)_(?P<inst>\d+)_(?P<user>[0-9a-f]+)--(?P<attempt>[0-9a-f]+)_(?P<kind>[pm])\.png$")
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".PNG")
METRICS = ["iou", "side_rel", "loose_rel", "corner_rel", "center_rel", "log_area", "side_px1024", "cuts"]


def edges(mask: np.ndarray):
    """bbox of a binary mask as pixel edges [x1, y1, x2, y2) (inclusive max + 1)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64)


def find(d: Path, stem: str):
    for e in EXTS:
        p = d / (stem + e)
        if p.exists():
            return p
    return None


def geometry(U: np.ndarray, T: np.ndarray, hw) -> dict:
    w, h = T[2] - T[0], T[3] - T[1]
    off = {"off_left": T[0] - U[0], "off_top": T[1] - U[1], "off_right": U[2] - T[2], "off_bottom": U[3] - T[3]}
    rel = np.array([off["off_left"] / w, off["off_right"] / w, off["off_top"] / h, off["off_bottom"] / h])
    ix = max(0.0, min(U[2], T[2]) - max(U[0], T[0]))
    iy = max(0.0, min(U[3], T[3]) - max(U[1], T[1]))
    inter = ix * iy
    aU, aT = (U[2] - U[0]) * (U[3] - U[1]), w * h
    diag = np.hypot(w, h)
    s = 1024.0 / max(hw)
    c1 = np.hypot(U[0] - T[0], U[1] - T[1])
    c2 = np.hypot(U[2] - T[2], U[3] - T[3])
    return {**off, "iou": inter / (aU + aT - inter), "side_rel": float(np.abs(rel).mean()),
            "loose_rel": float(rel.mean()),
            "side_px1024": float(np.mean(np.abs(list(off.values())))) * s,
            "corner_rel": (c1 + c2) / 2 / diag,
            "center_rel": np.hypot((U[0] + U[2] - T[0] - T[2]) / 2, (U[1] + U[3] - T[1] - T[3]) / 2) / diag,
            "log_area": float(np.log(aU / aT)), "cuts": float((rel < -0.01).any()),
            "obj_frac": aT / float(hw[0] * hw[1])}


def breps_image(job):
    root, ds, stem, recs = job
    img = find(root / "images", stem)
    gtp = find(root / "masks", stem)
    if img is None or gtp is None:
        return []
    im = cv2.imread(str(img))
    H, W = im.shape[:2]
    if H > 1024 and W > 1024:          # defend_critical_shifts._prepare_image squashes these to 1024 x 1024
        H, W = 1024, 1024
    gt = cv2.imread(str(gtp), cv2.IMREAD_GRAYSCALE)
    if gt.shape != (H, W):
        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
    T = edges(gt > 0)
    if T is None:
        return []
    rows = []
    for kind, user, attempt, path in recs:
        um = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if um is None:
            continue
        if um.shape != (H, W):
            um = cv2.resize(um, (W, H), interpolation=cv2.INTER_NEAREST)
        U = edges(um > 0)
        if U is None or U[2] - U[0] <= 1 or U[3] - U[1] <= 1:
            continue
        rows.append({"set": "BREPS", "dataset": ds, "kind": kind, "image_name": stem, "user": user,
                     "attempt": attempt, **geometry(U, T, (H, W))})
    return rows


def breps(root: Path, workers: int) -> pd.DataFrame:
    by = {}
    for p in sorted((root / "user_masks").iterdir()):
        m = USER_MASK_RE.match(p.name)
        if m is None:
            continue
        ds = next((d for d, pre in PREFIX.items() if m["stem"].startswith(pre)), None)
        if ds is None:
            continue
        by.setdefault((ds, m["stem"]), []).append((m["kind"], m["user"], m["attempt"], p))
    jobs = [(root, ds, stem, recs) for (ds, stem), recs in sorted(by.items())]
    print(f"BREPS: {sum(len(r) for r in by.values())} user boxes on {len(jobs)} images", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for rs in ex.map(breps_image, jobs, chunksize=4) for r in rs]
    return pd.DataFrame(rows)


def transfer(d: Path, name: str) -> pd.DataFrame:
    ann = pd.read_csv(d / "annotations.csv", dtype={"image_id": str, "rad_id": str})
    rows = []
    for r in ann.itertuples():
        mp = find(d / "masks", r.image_id)
        if mp is None:
            continue
        gt = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        H, W = gt.shape
        T = edges(gt > 0)
        sx, sy = W / float(r.width), H / float(r.height)
        U = np.array([r.x1 * sx, r.y1 * sy, r.x2 * sx, r.y2 * sy], dtype=np.float64)   # already pixel edges
        rows.append({"set": name, "dataset": name, "kind": "-", "image_name": r.image_id,
                     "user": getattr(r, "rad_id", "u00"), "attempt": "own", **geometry(U, T, (H, W))})
    print(f"{name}: {len(rows)} boxes", flush=True)
    return pd.DataFrame(rows)


def summarize(D: pd.DataFrame) -> pd.DataFrame:
    blocks = [(f"BREPS {k}", D[(D.set == "BREPS") & (D.dataset == k)]) for k in PREFIX]
    b = D[D.set == "BREPS"]
    blocks += [("BREPS natural", b[~b.dataset.isin(MEDICAL)]), ("BREPS medical", b[b.dataset.isin(MEDICAL)]),
               ("BREPS all", b)]
    for k in ("m", "p"):
        blocks.append((f"BREPS natural, {k}", b[(~b.dataset.isin(MEDICAL)) & (b.kind == k)]))
    blocks += [(s, D[D.set == s]) for s in D.set.unique() if s != "BREPS"]
    rows = []
    for name, s in blocks:
        if not len(s):
            continue
        r = {"block": name, "boxes": len(s), "images": s.image_name.nunique()}
        for m in METRICS:
            r[f"{m}_mean"] = s[m].mean()
            if m != "cuts":
                r[f"{m}_median"] = s[m].median()
        r["obj_frac_median"] = s.obj_frac.median()
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--breps_root", default="/home/jovyan/shares/SR006.nfs2/pishugin/user_study/FOR_TEST")
    p.add_argument("--oi_dir", action="append", default=[], help="fetch_transfer_sets.py output dir (repeatable)")
    p.add_argument("--oi_name", action="append", default=[], help="its name (default: the dir name)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out_dir", default="exp_res/box_offsets")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parts = [breps(Path(a.breps_root), a.workers)] if a.breps_root else []
    for i, d in enumerate(a.oi_dir):
        parts.append(transfer(Path(d), a.oi_name[i] if i < len(a.oi_name) else Path(d).name))
    D = pd.concat(parts, ignore_index=True)
    D.to_csv(out / "per_box.csv.gz", index=False, float_format="%.6f")
    S = summarize(D)
    S.to_csv(out / "summary.csv", index=False, float_format="%.4f")
    show = ["block", "boxes", "iou_mean", "iou_median", "side_rel_mean", "side_rel_median", "loose_rel_mean",
            "corner_rel_mean", "center_rel_mean", "side_px1024_mean", "side_px1024_median", "cuts_mean",
            "obj_frac_median"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(S[show].round(4).to_string(index=False))
    print(f"-> {out}/summary.csv, per_box.csv.gz")


if __name__ == "__main__":
    main()
