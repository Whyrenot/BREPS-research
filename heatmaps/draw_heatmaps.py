import ast
from functools import partial
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from loguru import logger
from matplotlib import cm, colors
from tqdm import tqdm as tqdmbar

from segment_anything.utils.transforms import ResizeLongestSide


def boxes_to_original(
    boxes: Union[np.ndarray, torch.Tensor],
    original_size: Tuple[int, int],
    target_length: int,
) -> Union[np.ndarray, torch.Tensor]:
    old_h, old_w = original_size
    new_h, new_w = ResizeLongestSide.get_preprocess_shape(old_h, old_w, target_length)

    if isinstance(boxes, np.ndarray):
        if boxes.size == 0:
            return boxes.copy()
        b = boxes.astype(np.float32, copy=True).reshape(-1, 2, 2)
        scale_x = float(old_w) / float(new_w)
        scale_y = float(old_h) / float(new_h)
        b[..., 0] *= scale_x
        b[..., 1] *= scale_y
        return b.reshape(-1, 4)

    if isinstance(boxes, torch.Tensor):
        if boxes.numel() == 0:
            return boxes.clone()
        b = boxes.to(torch.float32).reshape(-1, 2, 2)
        scale_x = float(old_w) / float(new_w)
        scale_y = float(old_h) / float(new_h)
        b[..., 0] *= scale_x
        b[..., 1] *= scale_y
        return b.reshape(-1, 4)

    raise TypeError(f"Unsupported boxes type: {type(boxes)}")


def get_bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def imshow_to_numpy(img, *, cmap: str = "viridis", dpi: int = 100) -> np.ndarray:
    arr = np.asarray(img)

    if arr.ndim == 2:
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        cmap_obj = cm.get_cmap(cmap)
        rgba = cmap_obj(norm(arr))
        rgb = rgba[..., :3]
        return rgb.astype(np.float32, copy=False)

    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        out = arr.astype(np.float32, copy=False)
        if np.issubdtype(arr.dtype, np.integer):
            out /= np.iinfo(arr.dtype).max
        elif out.max() > 1.1:
            out /= 255.0
        out = np.clip(out, 0.0, 1.0)
        return out[..., :3]

    raise ValueError(f"Unsupported image shape {arr.shape}")


def build_heatmaps(
    df: pd.DataFrame,
    img_height: int = 1024,
    img_width: int = 1024,
    im_name: str = "",
) -> np.ndarray:
    H, W = img_height, img_width

    xyxy = np.vstack(df["bbox"].to_numpy()).astype(np.int32)
    widths = xyxy[:, 2] - xyxy[:, 0]
    heights = xyxy[:, 3] - xyxy[:, 1]
    valid = (widths > 0) & (heights > 0)

    if not np.any(valid):
        logger.warning(f"No valid boxes for image {im_name}")
        return np.zeros((H, W), dtype=np.float32)

    xyxy = xyxy[valid]
    df_valid = df[valid].reset_index(drop=True)

    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, W - 1)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, H - 1)

    xmin, ymin, xmax, ymax = xyxy.T
    ious = df_valid["iou"].to_numpy(np.float32)

    sum_exact = np.zeros((H, W), np.float32)
    count_exact = np.zeros_like(sum_exact)

    for x1, y1, x2, y2, iou in tqdmbar(
        zip(xmin, ymin, xmax, ymax, ious),
        total=len(ious),
        desc=f"building heatmap {im_name}",
    ):
        sum_exact[y1, x1] += iou
        sum_exact[y2, x2] += iou
        sum_exact[y2, x1] += iou
        sum_exact[y1, x2] += iou

        count_exact[y1, x1] += 1
        count_exact[y2, x2] += 1
        count_exact[y2, x1] += 1
        count_exact[y1, x2] += 1

    exact_canvas = np.zeros_like(sum_exact)
    mask_nonzero = count_exact > 0
    exact_canvas[mask_nonzero] = sum_exact[mask_nonzero] / count_exact[mask_nonzero]

    return exact_canvas


def process_image(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    df_path: Union[str, Path],
    dataset_name: str,
    model_name: str,
) -> None:
    image_path = Path(image_path)
    mask_path = Path(mask_path)
    df_path = Path(df_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image at {image_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask at {mask_path}")

    mask = ((mask > 0) * 255).astype(np.uint8)

    df = pd.read_csv(df_path)
    df["bbox"] = df["bbox"].apply(
        lambda v: v if isinstance(v, (list, tuple, np.ndarray)) else ast.literal_eval(v)
    )

    box_resize = partial(
        boxes_to_original, original_size=image.shape[:2], target_length=1024
    )
    df["bbox"] = df["bbox"].apply(
        lambda x: np.round(box_resize(np.array(x))[0]).astype(np.int32)
    )

    output_dir = Path("heatmaps") / model_name / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        heatmap_exact = build_heatmaps(
            df,
            img_height=image.shape[0],
            img_width=image.shape[1],
            im_name=str(image_path),
        )
    except Exception:
        logger.error(f"Failed to build heatmap for image={image_path}, mask={mask_path}")
        logger.debug(f"image.shape={image.shape}, mask.shape={mask.shape}")
        raise

    np.save(
        output_dir
        / f"heatmap_{dataset_name}_{image_path.stem}_exact.npy",
        heatmap_exact,
    )

    heatmap_rgb = imshow_to_numpy(heatmap_exact, cmap="jet")
    heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)
    heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)

    mask_rgb = np.repeat(mask[..., None], 3, axis=-1)

    assert (
        heatmap_bgr.shape == image.shape == mask_rgb.shape
    ), f"{heatmap_bgr.shape} {image.shape} {mask_rgb.shape}"

    blended = cv2.addWeighted(image.copy(), 1.0, heatmap_bgr, 0.6, 0)

    try:
        blended_masked = cv2.addWeighted(blended, 0.6, mask_rgb, 0.7, 10)
    except Exception as e:
        logger.debug(
            f"image.shape={image.shape}, mask.shape={mask_rgb.shape}, heatmap.shape={heatmap_bgr.shape}"
        )
        logger.debug(
            f"image.dtype={image.dtype}, mask.dtype={mask_rgb.dtype}, heatmap.dtype={heatmap_bgr.dtype}"
        )
        logger.error(e)
        raise

    cv2.imwrite(
        str(
            output_dir
            / f"heatmap_{dataset_name}_{image_path.stem}_exact_blended.png"
        ),
        blended,
    )
    cv2.imwrite(
        str(
            output_dir
            / f"heatmap_{dataset_name}_{image_path.stem}_exact_masked.png"
        ),
        blended_masked,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_names",
        type=lambda s: [item.strip() for item in s.split(",")],
        required=True,
        help="Comma-separated list of dataset names (e.g. ds1,ds2,ds3)",
    )
    parser.add_argument(
        "--input_data",
        type=str,
        default="computed_bboxes/",
        help="Folder containing precomputed results for each dataset",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers for image processing",
    )
    args = parser.parse_args()

    model_name = Path(args.input_data).stem

    for dataset_name in args.dataset_names:
        dataset_path = Path("../tetris/datasets") / dataset_name
        image_paths = list(dataset_path.glob("images/*.*"))
        mask_paths = list(dataset_path.glob("masks/*.png"))

        image_mask_df_paths = []
        for image_path in image_paths:
            image_name = image_path.stem
            mask_candidates = [p for p in mask_paths if image_name in str(p)]
            assert (
                len(mask_candidates) == 1
            ), f"Expected exactly one mask for {image_path}, got {len(mask_candidates)}"
            mask_path = mask_candidates[0]

            df_path = Path(args.input_data) / dataset_name / f"res_final_{image_name}_.csv"
            if not df_path.exists():
                continue

            image_mask_df_paths.append(
                (image_path, mask_path, df_path, dataset_name, model_name)
            )

        if not image_mask_df_paths:
            logger.warning(f"No CSV files found for dataset {dataset_name}")
            continue

        if args.num_workers > 1:
            from multiprocessing import Pool

            with Pool(processes=args.num_workers) as pool:
                try:
                    for _ in tqdmbar(
                        pool.starmap(process_image, image_mask_df_paths),
                        total=len(image_mask_df_paths),
                        desc=f"{dataset_name}",
                    ):
                        pass
                except Exception:
                    pool.terminate()
                    pool.join()
                    raise
        else:
            for paths in tqdmbar(
                image_mask_df_paths, desc=f"{dataset_name}"
            ):
                process_image(*paths)


if __name__ == "__main__":
    main()
