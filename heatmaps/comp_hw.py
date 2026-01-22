import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from loguru import logger
from tqdm import tqdm


def generate_random_bboxes(
    n: int,
    height: int,
    width: int,
    min_box_w: int = 1,
    min_box_h: int = 1,
    max_box_w: int | None = None,
    max_box_h: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if n < 0:
        raise ValueError("n must be non-negative")

    max_box_w = max_box_w or width
    max_box_h = max_box_h or height

    if not (1 <= min_box_w <= max_box_w <= width):
        raise ValueError("min_box_w ≤ max_box_w ≤ width")
    if not (1 <= min_box_h <= max_box_h <= height):
        raise ValueError("min_box_h ≤ max_box_h ≤ height")

    rng = np.random.default_rng(seed)

    bw = rng.integers(min_box_w, max_box_w + 1, size=n, dtype=np.int32)
    bh = rng.integers(min_box_h, max_box_h + 1, size=n, dtype=np.int32)

    x_min = rng.integers(0, width - bw + 1, size=n, dtype=np.int32)
    y_min = rng.integers(0, height - bh + 1, size=n, dtype=np.int32)

    boxes = np.stack((x_min, y_min, x_min + bw, y_min + bh), axis=1)
    return boxes


def generate_bounding_boxes(
    x: int,
    y: int,
    height: int,
    width: int,
    max_h: int,
    max_w: int,
) -> np.ndarray:
    bboxes: list[tuple[int, int, int, int]] = []

    for h in range(max_h + 1):
        for w in range(max_w + 1):
            xmin = x - w // 2
            xmax = x + (w + 1) // 2
            ymin = y - h // 2
            ymax = y + (h + 1) // 2

            xmin_clipped = int(np.clip(xmin, 0, width - 1))
            xmax_clipped = int(np.clip(xmax, 0, width - 1))
            ymin_clipped = int(np.clip(ymin, 0, height - 1))
            ymax_clipped = int(np.clip(ymax, 0, height - 1))

            bboxes.append(
                (xmin_clipped, ymin_clipped, xmax_clipped, ymax_clipped)
            )

    bboxes_arr = np.stack(bboxes).astype(np.int32)
    bboxes_arr = np.unique(bboxes_arr, axis=0)
    return bboxes_arr


def get_bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(rmin), int(rmax), int(cmin), int(cmax)


def mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    h, w = mask.shape
    img_diag = np.sqrt(h**2 + w**2)
    dilation = int(round(dilation_ratio * img_diag))
    dilation = max(dilation, 1)

    new_mask = cv2.copyMakeBorder(
        mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]

    return mask - mask_erode


def mask_to_polygon(mask: np.ndarray, dilation_coeff: float = 0.0) -> tuple[int, int, int, int]:
    mask = mask.astype(np.uint8)

    h, w = mask.shape
    kernel_size = max(1, int(min(h, w) * dilation_coeff))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    bbox = get_bbox_from_mask(mask)
    return bbox


def batch_iou_torch(
    gt_masks: torch.Tensor,
    pred_masks: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if gt_masks.ndim == 4 and gt_masks.size(1) == 1:
        gt = gt_masks.squeeze(1).bool()
    else:
        gt = gt_masks.bool()

    if pred_masks.ndim == 4 and pred_masks.size(1) == 1:
        pred = pred_masks.squeeze(1).bool()
    else:
        pred = pred_masks.bool()

    gt_flat = gt.flatten(1)
    pred_flat = pred.flatten(1)

    inter = (gt_flat & pred_flat).sum(dim=1).float()
    union = (gt_flat | pred_flat).sum(dim=1).float()

    return inter / (union + eps)


def get_iou(gt_mask, pred_mask, ignore_label: int = -1):
    if isinstance(gt_mask, torch.Tensor):
        ignore_gt_mask_inv = gt_mask != ignore_label
        obj_gt_mask = gt_mask == 1

        intersection = torch.logical_and(
            torch.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv
        ).sum()
        union = torch.logical_and(
            torch.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv
        ).sum()

        return intersection / union

    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(
        np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv
    ).sum()
    union = np.logical_and(
        np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv
    ).sum()

    return intersection / union


def get_boundary_iou(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    ignore_label: int = -1,
) -> float:
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1
    obj_gt_mask = mask_to_boundary(obj_gt_mask.astype(np.uint8))
    pred_mask = mask_to_boundary(pred_mask.astype(np.uint8))

    intersection = np.logical_and(
        np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv
    ).sum()
    union = np.logical_and(
        np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv
    ).sum()

    return float(intersection / union) if union > 0 else 0.0


def sliding_window_gen(list1, list2, size: int):
    output: list[list[int]] = []
    for tl, br in tqdm(zip(list1, list2), total=len(list1)):
        if len(output) == size:
            yield output
            output = []
        output.append([*tl, *br])
    if output:
        yield output


class DDPWithAttrs(torch.nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except Exception:
            return getattr(self.module, name)


def load_sam_model(
    checkpoint: str,
    device: torch.device,
    model_type: str = "vit_b",
):
    from segment_anything import sam_model_registry, SamPredictor

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    for _, p in sam.named_parameters():
        p.requires_grad = False

    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor


def load_samhq_model(
    checkpoint: str,
    device: torch.device,
    model_type: str = "vit_b",
):
    from segment_anything_hq import SamPredictor, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor


def get_model_type(checkpoint_path: str) -> str:
    if "base_plus" in checkpoint_path:
        return "configs/sam2.1/sam2.1_hiera_b+.yaml"
    if "large" in checkpoint_path:
        return "configs/sam2.1/sam2.1_hiera_l.yaml"
    if "small" in checkpoint_path:
        return "configs/sam2.1/sam2.1_hiera_s.yaml"
    if "tiny" in checkpoint_path:
        return "configs/sam2.1/sam2.1_hiera_t.yaml"

    raise ValueError(
        f"Could not infer SAM2.1 config from checkpoint path: {checkpoint_path}"
    )


def load_sam2_model(
    checkpoint: str,
    device: torch.device,
    model_type: str = "vit_b",
):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model_config = get_model_type(checkpoint)
    sam2 = build_sam2(model_config, checkpoint, device=device)

    for _, p in sam2.named_parameters():
        p.requires_grad = False

    sam2.to(device)
    predictor = SAM2ImagePredictor(sam2)
    return predictor


def load_model(model_name: str, **kwargs):
    match model_name:
        case "SAM":
            return load_sam_model(**kwargs)
        case "SAM-HQ":
            return load_samhq_model(**kwargs)
        case "SAM2.1":
            return load_sam2_model(**kwargs)
        case _:
            raise ValueError(f"Unknown model name: {model_name}")


def prepare_input(
    image_path: str,
    predictor,
    all_bboxes: np.ndarray,
) -> tuple[np.ndarray, torch.Tensor]:
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if image.shape[0] > 1024 and image.shape[1] > 1024:
        image = cv2.resize(image, (1024, 1024))

    predictor.set_image(image)

    boxes_tensor = torch.as_tensor(
        all_bboxes,
        device="cpu",
        dtype=torch.float32,
    )

    if hasattr(predictor, "transform"):
        transformed_boxes = predictor.transform.apply_boxes_torch(
            boxes_tensor,
            predictor.original_size,
        )
    else:
        transformed_boxes = boxes_tensor

    return image, transformed_boxes


@torch.inference_mode()
def multi_bbox_inference(
    image_path: str,
    predictor,
    all_bboxes_list: np.ndarray,
    rank: int,
    batch_size: int = 8,
):
    _, transformed_boxes = prepare_input(
        image_path=image_path,
        predictor=predictor,
        all_bboxes=all_bboxes_list,
    )

    predictor.model.eval()
    indice = 0

    for bboxes in transformed_boxes.split(batch_size, dim=0):
        if hasattr(predictor, "predict_torch"):
            masks, scores, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=bboxes.to(rank),
                multimask_output=False,
            )
        else:
            masks, scores, _ = predictor.predict_batch_single_image(
                point_coords_batch=None,
                point_labels_batch=None,
                box_batch=bboxes.to(rank),
                multimask_output=False,
            )
            masks = torch.stack(masks).squeeze(0)

        yield {
            "indice": indice,
            "masks": masks,
            "scores": scores,
            "boxes": bboxes,
        }

        indice += bboxes.size(0)


def evaluate_mask(args, device: torch.device, rank: int):
    mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.warning(f"Failed to read mask: {args.mask_path}")
        return None

    if mask.shape[0] > 1024 and mask.shape[1] > 1024:
        mask = cv2.resize(mask, (1024, 1024), cv2.INTER_LINEAR)

    mask = mask > 0

    logger.debug(f"Mask shape: {mask.shape}")

    if mask.sum() == 0:
        return None

    rmin, rmax, cmin, cmax = get_bbox_from_mask(mask)
    bbox = np.array([cmin, rmin, cmax, rmax], dtype=np.int32)

    bbox_center = [
        (bbox[2] - bbox[0]) // 2 + bbox[0],
        (bbox[3] - bbox[1]) // 2 + bbox[1],
    ]

    height, width = mask.shape

    if args.random_boxes:
        all_bboxes = generate_random_bboxes(
            n=1024**2 * 10,
            height=1024,
            width=1024,
            max_box_h=1023,
            max_box_w=1023,
            seed=424242,
        )
    else:
        all_bboxes = generate_bounding_boxes(
            width=width,
            height=height,
            x=bbox_center[0],
            y=bbox_center[1],
            max_h=height * 2,
            max_w=width * 2,
        )

    local_boxes = all_bboxes
    results: dict[str, list[dict]] = {
        args.image_path: [],
        "original_bbox": bbox.tolist(),
    }

    batch_size = args.batch_size

    predictor = load_model(
        model_name=args.model_name,
        model_type=args.model_type,
        checkpoint=args.checkpoint_path,
        device=device,
    )

    gt_masks = (
        torch.from_numpy(mask)
        .to(rank)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    total_batches = (len(local_boxes) + batch_size - 1) // batch_size

    for res in tqdm(
        multi_bbox_inference(
            args.image_path,
            predictor=predictor,
            all_bboxes_list=local_boxes,
            rank=rank,
            batch_size=batch_size,
        ),
        total=total_batches,
        desc=f"Processing {args.image_path}",
    ):
        gt_batch = gt_masks.repeat_interleave(
            repeats=res["masks"].size(0),
            dim=0,
        )

        iou_scores = batch_iou_torch(res["masks"], gt_batch).cpu()

        for bbox_item, iou_score in zip(
            res["boxes"].tolist(),
            iou_scores.tolist(),
        ):
            results[args.image_path].append(
                {
                    "bbox": bbox_item,
                    "iou": iou_score,
                }
            )

    return results


def get_output_filename(image_name: str, random_boxes: bool) -> str:
    stem = os.path.splitext(image_name)[0]
    suffix = "_random" if random_boxes else "_"
    return f"res_final_{stem}{suffix}.csv"


def get_output_path(output_dir: str, image_name: str, random_boxes: bool) -> str:
    return os.path.join(output_dir, get_output_filename(image_name, random_boxes))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate SAM with bounding boxes"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="../datasets/mvp_attacks/dataset/clean",
        help="directory containing input images",
    )
    parser.add_argument(
        "--masks_dir",
        type=str,
        default="../datasets/mvp_attacks/dataset/masks",
        help="directory containing ground truth masks",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="directory for output files",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="MODEL_CHECKPOINTS/SAM/sam_vit_b_01ec64.pth",
        help="model checkpoint path",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["SAM", "SAM2.1", "SAM-HQ", "MobileSAM", "MedSAM"],
        default="SAM",
        help="model name to use",
    )
    parser.add_argument(
        "--random_boxes",
        action="store_true",
        default=False,
        help="sample random boxes instead of centered grid",
    )
    parser.add_argument(
        "--continue_from",
        action="store_true",
        default=False,
        help="skip images that already have output CSV",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="vit_b",
        help="model type / backbone name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="batch size",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    all_images = sorted(
        [
            f
            for f in os.listdir(args.images_dir)
            if os.path.isfile(os.path.join(args.images_dir, f))
        ]
    )

    my_images: list[str] = []

    for image_name in all_images:
        output_path = get_output_path(
            output_dir=args.output_dir,
            image_name=image_name,
            random_boxes=bool(args.random_boxes),
        )

        if args.continue_from and os.path.exists(output_path):
            logger.info(f"Skipping existing result: {output_path}")
            continue

        my_images.append(image_name)

    my_images = sorted(my_images)
    my_images = my_images[rank::world_size]

    for image_name in my_images:
        image_path = os.path.join(args.images_dir, image_name)
        mask_name = os.path.splitext(image_name)[0] + ".png"
        mask_path = os.path.join(args.masks_dir, mask_name)

        if not os.path.exists(mask_path):
            logger.warning(
                f"No corresponding mask found for {image_name}, skipping..."
            )
            continue

        args.image_path = image_path
        args.mask_path = mask_path

        output_path = get_output_path(
            output_dir=args.output_dir,
            image_name=image_name,
            random_boxes=bool(args.random_boxes),
        )

        results = evaluate_mask(args, device=device, rank=rank)

        if results is None:
            logger.warning(f"Failed to process item: {args.image_path}")
            continue

        pd.DataFrame(results[args.image_path]).to_csv(
            output_path,
            index=False,
        )

    dist.destroy_process_group()
