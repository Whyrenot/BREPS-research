"""Fetch two small transfer sets for the "do our methods hurt?" check.

openimages : N random images of the Open Images V7 *test* split, one instance each.
             The box is the human-drawn one the mask was annotated from (BoxXMin.. in
             the segmentation CSV); test masks were hand-painted separately.
vindr      : N random VinDr-CXR *train* images where one finding was boxed by all three
             radiologists (one box each, pairwise IoU >= --min-iou). No masks: stability only.

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


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
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
    {"openimages": openimages, "vindr": vindr}[a.cmd](a)


if __name__ == "__main__":
    main()
