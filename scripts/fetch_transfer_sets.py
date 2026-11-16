"""Fetch two small transfer sets for the "do our methods hurt?" check.

openimages : N random images of the Open Images V7 *test* split, one instance each.
             The box is the human-drawn one the mask was annotated from (BoxXMin.. in
             the segmentation CSV); test masks were hand-painted separately.
vindr      : N random VinDr-CXR *train* images where one finding was boxed by all three
             radiologists (one box each, pairwise IoU >= --min-iou). No masks: stability only.
nucls      : NuCLS Unbiased control (breast cancer H&E, 0.2 um/px): ~25 participants (NP
             non-pathologists, JP junior / SP senior pathologists) boxed the nuclei of the same
             FOVs by hand; SP.3's hand-traced boundaries are the GT. One item per nucleus (the
             whole FOV as the image, that nucleus as the mask), every rater's box matched to it.

Images are downscaled to a long side of --max-side (1024, as in BREPS); boxes are scaled
with them. Output per set: images/, (masks/), annotations.csv.

  python scripts/fetch_transfer_sets.py openimages --root $ROOT --n 500
  python scripts/fetch_transfer_sets.py vindr --root $ROOT --n 50 --user <physionet login>
"""
import argparse
import getpass
import io
import random
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

OI = "https://storage.googleapis.com/openimages"
OI_IMG = "https://open-images-dataset.s3.amazonaws.com/test/{}.jpg"
PHYSIONET = "https://physionet.org/files/vindr-cxr/1.0.0"


def fetch(url, dst, opener=None, tries=4):
    """Download url -> dst (skipped if dst exists). url may be a list of alternatives."""
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
    urls = [url] if isinstance(url, str) else list(url)
    err = None
    for k in range(tries):
        for u in urls:
            try:
                with (opener.open if opener else urllib.request.urlopen)(u, timeout=120) as r, \
                        open(tmp, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                tmp.replace(dst)
                return dst
            except urllib.error.HTTPError as e:
                err = e
                if e.code in (401, 403):
                    raise
            except Exception as e:  # noqa: BLE001 -- timeouts, resets
                err = e
        time.sleep(2 * (k + 1))
    raise RuntimeError(f"{urls[0]}: {err}")


def scale_to(img, max_side):
    s = min(1.0, max_side / max(img.size))
    if s < 1.0:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
    return img, s


# ---------------------------------------------------------------- Open Images
def openimages(a):
    out = Path(a.root) / f"OpenImagesV7_test{a.n}"
    raw = out / "raw"
    for d in (raw, out / "images", out / "masks"):
        d.mkdir(parents=True, exist_ok=True)

    seg = pd.read_csv(fetch(f"{OI}/v5/test-annotations-object-segmentation.csv",
                            raw / "test-annotations-object-segmentation.csv"))
    names = pd.read_csv(fetch(f"{OI}/v7/oidv7-class-descriptions.csv", raw / "class-descriptions.csv"))
    seg["area"] = (seg.BoxXMax - seg.BoxXMin) * (seg.BoxYMax - seg.BoxYMin)
    seg = seg[(seg.area >= a.min_area) & (seg.area <= a.max_area)]

    rng = random.Random(a.seed)
    ids = sorted(seg.ImageID.unique())
    rng.shuffle(ids)
    ids = ids[: int(a.n * 1.1) + 10]  # spare for broken images / empty masks
    by_img = seg[seg.ImageID.isin(ids)].groupby("ImageID")
    pick = pd.DataFrame([g.iloc[rng.randrange(len(g))] for _, g in by_img])
    pick = pick.set_index("ImageID").loc[ids].reset_index()  # keep the shuffled order
    print(f"[oi] {seg.ImageID.nunique()} test images with a mask after the area filter; "
          f"{len(pick)} candidates")

    # masks: 16 zips (~11 MB each) by the first hex digit of the ImageID
    want = {}
    for r in pick.itertuples():
        want.setdefault(r.ImageID[0].lower(), set()).add(r.MaskPath)
    masks = {}
    for p, need in sorted(want.items()):
        z = fetch(f"{OI}/v5/test-masks/test-masks-{p}.zip", raw / f"test-masks-{p}.zip")
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if Path(n).name in need:
                    masks[Path(n).name] = zf.read(n)
    print(f"[oi] masks found {len(masks)}/{len(pick)}")

    def get_img(i):
        try:
            return i, fetch(OI_IMG.format(i), raw / f"{i}.jpg")
        except Exception as e:  # noqa: BLE001
            print(f"[oi] image {i}: {e}", file=sys.stderr)
            return i, None

    with ThreadPoolExecutor(a.workers) as ex:
        imgs = dict(ex.map(get_img, pick.ImageID))

    rows = []
    for r in pick.itertuples():
        if len(rows) == a.n:
            break
        if imgs.get(r.ImageID) is None or r.MaskPath not in masks:
            continue
        img = Image.open(imgs[r.ImageID]).convert("RGB")
        img, _ = scale_to(img, a.max_side)
        m = Image.open(io.BytesIO(masks[r.MaskPath])).convert("L").resize(img.size, Image.NEAREST)
        m = np.asarray(m) > 0
        if m.sum() < 16:
            continue
        W, H = img.size
        ys, xs = np.nonzero(m)
        img.save(out / "images" / f"{r.ImageID}.jpg", quality=95)
        Image.fromarray(m.astype(np.uint8) * 255).save(out / "masks" / f"{r.ImageID}.png")
        rows.append(dict(
            image_id=r.ImageID, label=r.LabelName, width=W, height=H,
            # human box (pixels of the saved image), x1 y1 x2 y2
            x1=r.BoxXMin * W, y1=r.BoxYMin * H, x2=r.BoxXMax * W, y2=r.BoxYMax * H,
            # tight box of the mask, for the tight-vs-human comparison
            tx1=xs.min(), ty1=ys.min(), tx2=xs.max() + 1, ty2=ys.max() + 1,
            mask_path=r.MaskPath, box_id=r.BoxID))
    ann = pd.DataFrame(rows).merge(names, how="left", left_on="label", right_on="LabelName")
    ann = ann.drop(columns="LabelName").rename(columns={"DisplayName": "class_name"})
    ann.to_csv(out / "annotations.csv", index=False)
    print(f"[oi] wrote {len(ann)} images to {out}")
    if len(ann) < a.n:
        print(f"[oi] WARNING: only {len(ann)} of {a.n}", file=sys.stderr)


# ---------------------------------------------------------------- VinDr-CXR
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def dicom_to_uint8(path):
    import pydicom
    try:
        from pydicom.pixels import apply_voi_lut  # pydicom >= 3
    except ImportError:
        from pydicom.pixel_data_handlers.util import apply_voi_lut

    d = pydicom.dcmread(path)
    x = apply_voi_lut(d.pixel_array, d).astype(np.float32)
    if getattr(d, "PhotometricInterpretation", "") == "MONOCHROME1":
        x = x.max() - x
    x -= x.min()
    x /= max(float(x.max()), 1e-6)
    return (x * 255).astype(np.uint8)


def vindr(a):
    out = Path(a.root) / f"VinDrCXR_train{a.n}"
    raw = out / "raw"
    for d in (raw, out / "images"):
        d.mkdir(parents=True, exist_ok=True)

    pw = getpass.getpass(f"PhysioNet password for {a.user}: ")
    mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    mgr.add_password(None, "https://physionet.org/", a.user, pw)
    opener = urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(mgr))

    ann = pd.read_csv(fetch([f"{PHYSIONET}/annotations/annotations_train.csv",
                             f"{PHYSIONET}/Annotations/annotations_train.csv"],
                            raw / "annotations_train.csv", opener))
    ann = ann.dropna(subset=["x_min"])  # "No finding" rows carry no box

    # objects boxed once by each of the 3 radiologists, boxes agreeing pairwise
    cand = []
    for (img_id, cls), g in ann.groupby(["image_id", "class_name"]):
        per_rad = g.groupby("rad_id")
        if per_rad.ngroups < 3 or (per_rad.size() != 1).any():
            continue
        b = g[["x_min", "y_min", "x_max", "y_max"]].to_numpy(float)
        if min(iou(b[i], b[j]) for i in range(len(b)) for j in range(i + 1, len(b))) >= a.min_iou:
            cand.append((img_id, cls))
    imgs = sorted({i for i, _ in cand})
    print(f"[vindr] {len(cand)} objects boxed by all 3 radiologists on {len(imgs)} images")
    rng = random.Random(a.seed)
    rng.shuffle(imgs)
    imgs = imgs[: a.n]
    chosen = {}
    for i, c in cand:
        if i in imgs:
            chosen.setdefault(i, []).append(c)
    chosen = {i: rng.choice(sorted(cs)) for i, cs in chosen.items()}  # one object per image

    def get(i):
        return i, fetch([f"{PHYSIONET}/train/{i}.dicom", f"{PHYSIONET}/Train/{i}.dicom"],
                        raw / f"{i}.dicom", opener)

    with ThreadPoolExecutor(a.workers) as ex:
        dcm = dict(ex.map(get, imgs))

    rows = []
    for i in imgs:
        x = dicom_to_uint8(dcm[i])
        img, s = scale_to(Image.fromarray(x), a.max_side)
        img.save(out / "images" / f"{i}.png")
        g = ann[(ann.image_id == i) & (ann.class_name == chosen[i])]
        for r in g.itertuples():
            rows.append(dict(image_id=i, class_name=r.class_name, rad_id=r.rad_id,
                             width=img.width, height=img.height,
                             x1=r.x_min * s, y1=r.y_min * s, x2=r.x_max * s, y2=r.y_max * s))
    pd.DataFrame(rows).to_csv(out / "annotations.csv", index=False)
    print(f"[vindr] wrote {len(imgs)} images, {len(rows)} boxes to {out}")


# ---------------------------------------------------------------- NuCLS
# Unbiased control, raw data of each annotator (sites.google.com/view/nucls/multi-rater):
# every participant boxed the nuclei of the same FOVs without algorithmic suggestions;
# SP.3 (a pathologist) traced the boundaries by hand -> the GT masks.
NUCLS_FOLDERS = {"csv": "1OF9Herx3aHH2HQ0RBt0ge4Pq7MFPIcvU", "rgb": "15_Mxl-4iBSFFmgBIfqw8Isq8g5xGsBA_"}
GDRIVE_LIST = "https://drive.google.com/embeddedfolderview?id={}"
GDRIVE_FILE = "https://drive.usercontent.google.com/download?id={}&export=download"


def gdrive_list(folder_id):
    import re
    with urllib.request.urlopen(GDRIVE_LIST.format(folder_id), timeout=120) as r:
        h = r.read().decode("utf-8")
    return dict((n, f) for f, n in re.findall(r'file/d/([A-Za-z0-9_-]{20,}).*?flip-entry-title">([^<]+)', h, flags=re.S))


def nucls_name(name):
    """'NP.5_#_U-control_#_TCGA-..._id-<hex>_left-L_top-T_bottom-B_right-R.csv' -> dict."""
    import re
    part, _, rest = Path(name).stem.split("_#_")
    m = re.match(r"(TCGA-[^_]+)_id-[0-9a-f]+_left-(\d+)_top-(\d+)_bottom-(\d+)_right-(\d+)$", rest)
    slide, l, t, b, r = m.group(1), *map(int, m.groups()[1:])
    return {"participant": part, "slide": slide, "left": l, "top": t, "bottom": b, "right": r}


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def nucls(a):
    from PIL import ImageDraw
    out = Path(a.root) / "NuCLS_Ucontrol"
    raw = out / "raw"
    for d in (raw / "csv", raw / "rgb", out / "images", out / "masks"):
        d.mkdir(parents=True, exist_ok=True)
    csvs, rgbs = gdrive_list(NUCLS_FOLDERS["csv"]), gdrive_list(NUCLS_FOLDERS["rgb"])
    meta = pd.DataFrame([{"name": n, "id": f, **nucls_name(n)} for n, f in csvs.items()])
    print(f"[nucls] {len(meta)} annotation files, {meta.participant.nunique()} participants", flush=True)

    # one FOV = the same place of a slide (each participant's window is shifted by a few px)
    meta["cx"], meta["cy"] = (meta.left + meta.right) / 2, (meta.top + meta.bottom) / 2
    fov = np.full(len(meta), -1)
    cents = []
    for i, r in enumerate(meta.itertuples()):
        for k, (s, x, y) in enumerate(cents):
            if s == r.slide and abs(x - r.cx) < 150 and abs(y - r.cy) < 150:
                fov[i] = k
                break
        else:
            cents.append((r.slide, r.cx, r.cy))
            fov[i] = len(cents) - 1
    meta["fov"] = fov
    gt_files = meta[meta.participant == a.gt]
    meta = meta[meta.fov.isin(gt_files.fov)]
    print(f"[nucls] {len(cents)} FOVs, {len(gt_files)} with {a.gt}'s traced boundaries; "
          f"raters per FOV {meta.groupby('fov').participant.nunique().describe()[['min', '50%', 'max']].tolist()}",
          flush=True)

    def get_csv(r):
        return fetch(GDRIVE_FILE.format(r.id), raw / "csv" / f"{r.id}.csv")   # short names (Windows MAX_PATH)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(get_csv, meta.itertuples()))
    for r in gt_files.itertuples():
        fetch(GDRIVE_FILE.format(rgbs[Path(r.name).with_suffix(".png").name]), raw / "rgb" / f"{r.id}.png")
    meta[["name", "id", "participant", "slide", "left", "top", "bottom", "right", "fov"]].to_csv(raw / "files.csv", index=False)

    rows, diag = [], []
    rng = random.Random(a.seed)
    for g in gt_files.itertuples():
        img = Image.open(raw / "rgb" / f"{g.id}.png").convert("RGB")
        W, H = img.size
        f = W / float(g.right - g.left)      # pixels of the FOV image per unit of the file-name coordinates
        G = pd.read_csv(raw / "csv" / f"{g.id}.csv")
        G = G[G.type == "polyline"].reset_index(drop=True)
        polys = [list(zip(map(float, str(r.coords_x).split(",")), map(float, str(r.coords_y).split(","))))
                 for r in G.itertuples()]
        masks = []
        for poly in polys:
            m = Image.new("L", (W, H), 0)
            ImageDraw.Draw(m).polygon(poly, fill=255, outline=255)
            masks.append(np.asarray(m) > 0)
        tight = []
        for m in masks:
            ys, xs = np.nonzero(m)
            tight.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1] if len(xs) else None)
        matched = {k: [] for k in range(len(polys))}
        for p in meta[(meta.fov == g.fov) & (meta.participant != a.gt)].itertuples():
            P = pd.read_csv(raw / "csv" / f"{p.id}.csv")
            P = P[P.type == "rectangle"]
            dx, dy = (p.left - g.left) * f, (p.top - g.top) * f
            boxes = [[r.xmin + dx, r.ymin + dy, r.xmax + dx, r.ymax + dy] for r in P.itertuples()]
            pairs = sorted(((box_iou(b, t), i, k) for i, b in enumerate(boxes) for k, t in enumerate(tight) if t),
                           reverse=True)
            used_b, used_t = set(), set()
            for v, i, k in pairs:                      # greedy one-to-one by IoU
                if v < a.min_iou:
                    break
                if i in used_b or k in used_t:
                    continue
                used_b.add(i)
                used_t.add(k)
                matched[k].append((p.participant, boxes[i]))
                t = tight[k]
                diag.append({"participant": p.participant, "iou": v,
                             "dcx": (boxes[i][0] + boxes[i][2] - t[0] - t[2]) / 2,
                             "dcy": (boxes[i][1] + boxes[i][3] - t[1] - t[3]) / 2})
        fov_stem = f"{g.slide}_l{g.left}_t{g.top}"
        for k, lst in matched.items():
            if len(lst) < a.min_raters or tight[k] is None:
                continue
            t = tight[k]
            if t[0] <= 0 or t[1] <= 0 or t[2] >= W or t[3] >= H:     # nucleus cut by the FOV border
                continue
            stem = f"{fov_stem}_n{k:03d}"
            img.save(out / "images" / f"{stem}.png")
            Image.fromarray(masks[k].astype(np.uint8) * 255).save(out / "masks" / f"{stem}.png")
            for part, b in lst:
                rows.append(dict(image_id=stem, rad_id=part, kind=part.split(".")[0], width=W, height=H,
                                 x1=b[0], y1=b[1], x2=b[2], y2=b[3], tx1=t[0], ty1=t[1], tx2=t[2], ty2=t[3],
                                 class_name=G.main_classification[k], fov=fov_stem))
    ann = pd.DataFrame(rows)
    if a.n and ann.image_id.nunique() > a.n:
        keep = set(rng.sample(sorted(ann.image_id.unique()), a.n))
        for s in set(ann.image_id) - keep:
            (out / "images" / f"{s}.png").unlink(missing_ok=True)
            (out / "masks" / f"{s}.png").unlink(missing_ok=True)
        ann = ann[ann.image_id.isin(keep)]
    ann.to_csv(out / "annotations.csv", index=False)
    D = pd.DataFrame(diag)
    D.groupby("participant").agg(n=("iou", "size"), iou=("iou", "median"), dcx=("dcx", "median"),
                                 dcy=("dcy", "median")).to_csv(out / "participants.csv")
    print(f"[nucls] alignment: median box-centre offset to the traced nucleus dx {D.dcx.median():+.2f} "
          f"dy {D.dcy.median():+.2f} px (should be ~0); per participant -> participants.csv")
    per = ann.groupby("image_id").size()
    print(f"[nucls] wrote {ann.image_id.nunique()} nuclei on {ann.fov.nunique()} FOVs, {len(ann)} boxes "
          f"(raters per nucleus min {per.min()} median {per.median():.0f} max {per.max()}; "
          f"kinds {ann.kind.value_counts().to_dict()}) to {out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("nucls")
    s.add_argument("--root", required=True)
    s.add_argument("--n", type=int, default=0, help="keep N random nuclei (0 = all)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--gt", default="SP.3", help="the participant whose traced boundaries are the GT")
    s.add_argument("--min-iou", type=float, default=0.3, help="min IoU of a rater's box with the traced nucleus' box")
    s.add_argument("--min-raters", type=int, default=10, help="nuclei boxed by fewer raters are dropped")
    for name, n in (("openimages", 500), ("vindr", 50)):
        s = sub.add_parser(name)
        s.add_argument("--root", required=True)
        s.add_argument("--n", type=int, default=n)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--max-side", type=int, default=1024)
        s.add_argument("--workers", type=int, default=8)
    o = sub.choices["openimages"]
    o.add_argument("--min-area", type=float, default=0.01, help="min human-box area, fraction of image")
    o.add_argument("--max-area", type=float, default=0.95)
    v = sub.choices["vindr"]
    v.add_argument("--user", required=True, help="PhysioNet login (credentialed, DUA signed)")
    v.add_argument("--min-iou", type=float, default=0.1, help="min pairwise IoU of the 3 boxes")
    a = p.parse_args()
    {"openimages": openimages, "vindr": vindr, "nucls": nucls}[a.cmd](a)


if __name__ == "__main__":
    main()
