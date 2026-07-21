import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from loguru import logger
from tqdm import tqdm
from torchvision.ops import box_iou


def sample_shifted_boxes_normal(base_box: torch.Tensor, image_shape, num_samples=10, X=0.05, seed=0):
    H, W = image_shape
    np.random.seed(seed)
    boxes = []
    gx0, gy0, gx1, gy1 = base_box.tolist()
    w, h = gx1 - gx0, gy1 - gy0
    
    for i in range(num_samples * 10):
        dx = np.random.normal(loc=0.0, scale=X) * w
        dy = np.random.normal(loc=0.0, scale=X) * h
        cand = torch.tensor([gx0 + dx, gy0 + dy, gx1 + dx, gy1 + dy])
        cand[0::2].clamp_(0, W - 1)
        cand[1::2].clamp_(0, H - 1)
        
        iou = box_iou(cand.unsqueeze(0), base_box.unsqueeze(0))[0, 0].item()
        if iou > 0.5:
            boxes.append(cand)
        if len(boxes) >= num_samples:
            break
            
    if not boxes:
        boxes = [base_box]
        
    while len(boxes) < num_samples:
        boxes.append(boxes[-1])
        
    return torch.stack(boxes[:num_samples]).to(torch.float32)


def sample_size_perturbed_boxes(
    base_box: torch.Tensor,
    image_shape: tuple[int, int],
    num_samples: int = 10,
    sigma_w: float = 0.05,
    sigma_h: float = 0.05,
    seed: int = 0,
) -> torch.Tensor:
    """Randomized Smoothing defence: generate box copies with perturbed width/height.

    The center of the bounding box is kept fixed; only width and height are
    varied independently using zero-mean normal noise scaled by ``sigma_w`` and
    ``sigma_h`` (fractions of the original side length).  Always includes the
    original box as the first sample.

    Args:
        base_box: [x0, y0, x1, y1] tensor (float).
        image_shape: (H, W) of the image.
        num_samples: total number of boxes to return (including the original).
        sigma_w: std-dev of relative width perturbation (e.g. 0.05 = ±5% of w).
        sigma_h: std-dev of relative height perturbation (e.g. 0.05 = ±5% of h).
        seed: random seed for reproducibility.

    Returns:
        Tensor of shape ``(num_samples, 4)`` with [x0, y0, x1, y1] boxes.
    """
    H, W = image_shape
    rng = np.random.default_rng(seed)

    gx0, gy0, gx1, gy1 = base_box.tolist()
    w = gx1 - gx0
    h = gy1 - gy0
    cx = (gx0 + gx1) / 2.0
    cy = (gy0 + gy1) / 2.0

    boxes: list[torch.Tensor] = [base_box.float()]

    attempts = 0
    while len(boxes) < num_samples and attempts < num_samples * 20:
        attempts += 1
        dw = rng.normal(loc=0.0, scale=sigma_w) * w
        dh = rng.normal(loc=0.0, scale=sigma_h) * h
        new_w = max(1.0, w + dw)
        new_h = max(1.0, h + dh)

        x0 = cx - new_w / 2.0
        x1 = cx + new_w / 2.0
        y0 = cy - new_h / 2.0
        y1 = cy + new_h / 2.0

        cand = torch.tensor(
            [
                float(np.clip(x0, 0, W - 1)),
                float(np.clip(y0, 0, H - 1)),
                float(np.clip(x1, 0, W - 1)),
                float(np.clip(y1, 0, H - 1)),
            ],
            dtype=torch.float32,
        )
        boxes.append(cand)

    # Pad with the original box if we could not generate enough samples
    while len(boxes) < num_samples:
        boxes.append(base_box.float())

    return torch.stack(boxes[:num_samples])


def sample_size_and_center_perturbed_boxes(
    base_box: torch.Tensor,
    image_shape: tuple[int, int],
    num_samples: int = 10,
    sigma_w: float = 0.05,
    sigma_h: float = 0.05,
    sigma_cx: float = 0.03,
    sigma_cy: float = 0.03,
    seed: int = 0,
) -> torch.Tensor:
    """Randomized Smoothing defence: perturb width, height AND center position.

    Each sample independently draws:
      - width  noise ~ N(0, sigma_w) * w
      - height noise ~ N(0, sigma_h) * h
      - cx     noise ~ N(0, sigma_cx) * w   (scaled by box width)
      - cy     noise ~ N(0, sigma_cy) * h   (scaled by box height)

    The first sample is always the original (unperturbed) box.
    All coordinates are clipped to the valid image range.

    Args:
        base_box:   [x0, y0, x1, y1] tensor (float).
        image_shape: (H, W) of the image.
        num_samples: total number of boxes to return (incl. original).
        sigma_w:    std-dev of relative width  perturbation.
        sigma_h:    std-dev of relative height perturbation.
        sigma_cx:   std-dev of relative center-x shift (fraction of w).
        sigma_cy:   std-dev of relative center-y shift (fraction of h).
        seed:       random seed.

    Returns:
        Tensor of shape ``(num_samples, 4)`` with [x0, y0, x1, y1] boxes.
    """
    H, W = image_shape
    rng = np.random.default_rng(seed)

    gx0, gy0, gx1, gy1 = base_box.tolist()
    w  = gx1 - gx0
    h  = gy1 - gy0
    cx = (gx0 + gx1) / 2.0
    cy = (gy0 + gy1) / 2.0

    boxes: list[torch.Tensor] = [base_box.float()]

    attempts = 0
    while len(boxes) < num_samples and attempts < num_samples * 20:
        attempts += 1

        new_w  = max(1.0, w  + rng.normal(0.0, sigma_w)  * w)
        new_h  = max(1.0, h  + rng.normal(0.0, sigma_h)  * h)
        new_cx = cx + rng.normal(0.0, sigma_cx) * w
        new_cy = cy + rng.normal(0.0, sigma_cy) * h

        x0 = new_cx - new_w / 2.0
        x1 = new_cx + new_w / 2.0
        y0 = new_cy - new_h / 2.0
        y1 = new_cy + new_h / 2.0

        cand = torch.tensor(
            [
                float(np.clip(x0, 0, W - 1)),
                float(np.clip(y0, 0, H - 1)),
                float(np.clip(x1, 0, W - 1)),
                float(np.clip(y1, 0, H - 1)),
            ],
            dtype=torch.float32,
        )
        boxes.append(cand)

    while len(boxes) < num_samples:
        boxes.append(base_box.float())

    return torch.stack(boxes[:num_samples])


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


def get_samhq2_model_type(checkpoint_path: str) -> str:
    if "base_plus" in checkpoint_path:
        return "configs/sam2.1/sam2.1_hq_hiera_b+.yaml"
    if "large" in checkpoint_path:
        return "configs/sam2.1/sam2.1_hq_hiera_l.yaml"
    if "small" in checkpoint_path:
        return "configs/sam2.1/sam2.1_hq_hiera_s.yaml"
    if "tiny" in checkpoint_path:
        return "configs/sam2.1/sam2.1_hq_hiera_t.yaml"

    raise ValueError(
        f"Could not infer SAM-HQ2 config from checkpoint path: {checkpoint_path}"
    )


def load_samhq2_model(
    checkpoint: str,
    device: torch.device,
    model_type: str = "vit_l",
):
    """SysCV/sam-hq -- the sam-hq2 subproject (github.com/SysCV/sam-hq).

    Its Python package is ALSO named ``sam2`` -- the same import path as the
    official facebookresearch/sam2 package already used by load_sam2_model()
    above. The two cannot coexist on one interpreter's sys.path, so this
    only works when actually running inside the ``sam_hq2`` conda env
    created by scripts/setup_repo.sh. Callers should route through
    heatmaps.env_dispatch.maybe_dispatch_to_env("SAM-HQ2", ...) first (all
    scripts in this repo that accept --model_name already do).
    """
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model_config = get_samhq2_model_type(checkpoint)
    sam2_hq = build_sam2(model_config, checkpoint, device=device)

    for _, p in sam2_hq.named_parameters():
        p.requires_grad = False

    sam2_hq.to(device)
    predictor = SAM2ImagePredictor(sam2_hq)
    return predictor


def load_sam3_model(
    checkpoint: str,
    device: torch.device,
    model_type: str = "vit_b",
):
    """NOT YET WIRED IN.

    SAM3 (facebookresearch/sam3) uses a different predictor
    (``Sam3Processor`` / ``build_sam3_image_model()``), not the
    SamPredictor/SAM2ImagePredictor box-prompt API the rest of this repo
    assumes. It is UNCONFIRMED whether SAM3 exposes ``model.prompt_encoder``
    / ``model.mask_decoder`` the way scripts/refine_box_iou_grad.py's
    gradient-ascent path (refine_box_by_iou_grad) needs for autograd through
    the box prompt.

    Before wiring this up: in the ``sam3`` conda env (scripts/setup_repo.sh),
    inspect build_sam3_image_model()'s returned model for a
    prompt_encoder/mask_decoder pair with the same forward signature as
    SAM1/SAM2, or adapt the gradient path to Sam3Processor's actual API.
    """
    raise NotImplementedError(
        "SAM3 is not wired into load_model() yet -- its predictor API "
        "(Sam3Processor / build_sam3_image_model) differs from SAM1/SAM2's "
        "SamPredictor/SAM2ImagePredictor, and it's unconfirmed whether the "
        "differentiable box->mask_decoder path refine_box_iou_grad.py needs "
        "exists for SAM3. Verify on the `sam3` conda env first, then "
        "implement load_sam3_model() here."
    )


def load_model(model_name: str, **kwargs):
    match model_name:
        case "SAM":
            return load_sam_model(**kwargs)
        case "SAM-HQ":
            return load_samhq_model(**kwargs)
        case "SAM-HQ2":
            return load_samhq2_model(**kwargs)
        case "SAM2.1":
            return load_sam2_model(**kwargs)
        case "SAM3":
            return load_sam3_model(**kwargs)
        case _:
            raise ValueError(f"Unknown model name: {model_name}")


def get_original_size(predictor) -> tuple:
    """(H, W) of the last image passed to predictor.set_image(), portable
    across predictor backends.

    segment_anything / segment_anything_hq's SamPredictor exposes a single
    ``.original_size`` (H, W) tuple. SAM2ImagePredictor (used for both
    SAM2.1 and SAM-HQ2, which share the sam2 package) has no such attribute
    -- it keeps ``._orig_hw``, a LIST of (H, W) tuples (one per image, since
    its API also supports batched set_image_batch()); index 0 is the single
    image set by set_image().
    """
    if hasattr(predictor, "original_size"):
        return predictor.original_size
    orig_hw = getattr(predictor, "_orig_hw", None)
    if orig_hw:
        return tuple(orig_hw[0])
    raise AttributeError(
        f"{type(predictor).__name__} exposes neither .original_size nor "
        "._orig_hw -- unknown predictor backend, cannot determine the "
        "original image size"
    )


def prepare_input(
    image_path: str,
    predictor,
    all_bboxes: np.ndarray,
) -> tuple[np.ndarray, torch.Tensor, tuple[int, int]]:
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_shape = image.shape[:2]

    if image.shape[0] > 1024 and image.shape[1] > 1024:
        image = cv2.resize(image, (1024, 1024))

    predictor.set_image(image)

    boxes_tensor = torch.as_tensor(
        all_bboxes,
        device="cpu",
        dtype=torch.float32,
    )

    return image, boxes_tensor, image_shape


@torch.inference_mode()
def multi_bbox_inference_smoothed(
    image_path: str,
    predictor,
    all_bboxes_list: np.ndarray,
    rank: int,
    Y: int,
    X: float,
    averaging_mode: str,
    batch_size: int = 1024,
):
    _, base_boxes_tensor, image_shape = prepare_input(
        image_path=image_path,
        predictor=predictor,
        all_bboxes=all_bboxes_list,
    )

    predictor.model.eval()
    thresh = 0.0

    base_batch_size = max(1, batch_size // Y)

    for i in range(0, len(base_boxes_tensor), base_batch_size):
        base_batch = base_boxes_tensor[i : i + base_batch_size]

        all_noisy_boxes = []
        for j, base_box in enumerate(base_batch):
            noisy_boxes = sample_shifted_boxes_normal(
                base_box,
                image_shape=image_shape,
                num_samples=Y,
                X=X,
                seed=42
            )
            all_noisy_boxes.append(noisy_boxes)
            
        all_noisy_boxes = torch.cat(all_noisy_boxes, dim=0)

        if hasattr(predictor, "transform"):
            transformed_boxes = predictor.transform.apply_boxes_torch(
                all_noisy_boxes,
                predictor.original_size,
            )
        else:
            transformed_boxes = all_noisy_boxes
            
        transformed_boxes = transformed_boxes.to(rank)

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            if hasattr(predictor, "predict_torch"):
                try:
                    masks_logits, scores, _ = predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False,
                        return_logits=True,
                    )
                except TypeError:
                    masks_logits, scores, _ = predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False,
                    )
            else:
                try:
                    masks_logits, scores, _ = predictor.predict_batch_single_image(
                        point_coords_batch=None,
                        point_labels_batch=None,
                        box_batch=transformed_boxes,
                        multimask_output=False,
                        return_logits=True,
                    )
                except (TypeError, RuntimeError):
                    masks_logits, scores, _ = predictor.predict_batch_single_image(
                        point_coords_batch=None,
                        point_labels_batch=None,
                        box_batch=transformed_boxes,
                        multimask_output=False,
                    )
                masks_logits = torch.stack(masks_logits).squeeze(0)

        masks_logits = masks_logits.float()
        
        B = base_batch.size(0)
        _, C, H, W = masks_logits.shape
        masks_logits = masks_logits.view(B, Y, C, H, W)
        scores = scores.view(B, Y, -1)

        if "logit" in averaging_mode:
            avg_res = masks_logits.mean(dim=1)
            smoothed_prob = torch.sigmoid(avg_res)
        elif "sigmoid" in averaging_mode:
            p_perturb = torch.sigmoid(masks_logits)
            smoothed_prob = p_perturb.mean(dim=1)
        elif "binary" in averaging_mode:
            m_perturb = (masks_logits > thresh).float()
            smoothed_prob = m_perturb.mean(dim=1)
        else:
            avg_res = masks_logits.mean(dim=1)
            smoothed_prob = torch.sigmoid(avg_res)

        smoothed_mask = (smoothed_prob > 0.5)

        yield {
            "masks": smoothed_mask,
            "scores": scores.mean(dim=1),
            "boxes": base_batch,
        }


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

    batch_size = args.batch_size if hasattr(args, "batch_size") else 1024
    base_batch_size = max(1, batch_size // args.Y)
    total_batches = (len(local_boxes) + base_batch_size - 1) // base_batch_size

    for res in tqdm(
        multi_bbox_inference_smoothed(
            args.image_path,
            predictor=predictor,
            all_bboxes_list=local_boxes,
            rank=rank,
            Y=args.Y,
            X=args.X,
            averaging_mode=args.averaging_mode,
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


def get_output_filename(image_name: str, random_boxes: bool, Y: int, X: float, averaging_mode: str) -> str:
    stem = os.path.splitext(image_name)[0]
    suffix = "_random" if random_boxes else "_"
    return f"res_final_{stem}{suffix}smoothed_{averaging_mode}_Y{Y}_X{X}.csv"


def get_output_path(output_dir: str, image_name: str, random_boxes: bool, Y: int, X: float, averaging_mode: str) -> str:
    return os.path.join(output_dir, get_output_filename(image_name, random_boxes, Y, X, averaging_mode))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate SAM with smoothed bounding boxes (Randomized Smoothing)"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/home/jovyan/shares/SR006.nfs2/pishugin/rclicks/datasets/Berkeley/images",
        help="directory containing input images",
    )
    parser.add_argument(
        "--masks_dir",
        type=str,
        default="/home/jovyan/shares/SR006.nfs2/pishugin/rclicks/datasets/Berkeley/masks",
        help="directory containing ground truth masks",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_smoothed",
        help="directory for output files",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/jovyan/shares/SR006.nfs2/pishugin/rclicks/MODEL_CHECKPOINTS/SAM/sam_vit_b_01ec64.pth",
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
        "--mask_ext",
        type=str,
        default=".png",
        help="extension of mask files (e.g. .png, .bmp)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="batch size for model inference",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="limit number of images to process (0 for all)",
    )
    parser.add_argument(
        "--Y",
        type=int,
        default=10,
        help="Number of samples (batch size) for smoothing",
    )
    parser.add_argument(
        "--X",
        type=float,
        default=0.05,
        help="Standard deviation fraction for normal noise",
    )
    parser.add_argument(
        "--averaging_mode",
        type=str,
        choices=["logit", "sigmoid", "binary"],
        default="logit",
        help="Averaging mode strategy.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    all_images = sorted(
        [
            f
            for f in os.listdir(args.images_dir)
            if os.path.isfile(os.path.join(args.images_dir, f))
        ]
    )
    if args.limit > 0:
        all_images = all_images[:args.limit]

    my_images: list[str] = []

    for image_name in all_images:
        output_path = get_output_path(
            output_dir=args.output_dir,
            image_name=image_name,
            random_boxes=bool(args.random_boxes),
            Y=args.Y, X=args.X, averaging_mode=args.averaging_mode
        )

        if args.continue_from and os.path.exists(output_path):
            logger.info(f"Skipping existing result: {output_path}")
            continue

        my_images.append(image_name)

    my_images = sorted(my_images)
    my_images = my_images[rank::world_size]

    for image_name in my_images:
        image_path = os.path.join(args.images_dir, image_name)
        mask_name = os.path.splitext(image_name)[0] + args.mask_ext
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
            Y=args.Y, X=args.X, averaging_mode=args.averaging_mode
        )

        results = evaluate_mask(args, device=device, rank=rank)

        if results is None:
            logger.warning(f"Failed to process item: {args.image_path}")
            continue

        pd.DataFrame(results[args.image_path]).to_csv(
            output_path,
            index=False,
        )

    if world_size > 1:
        dist.destroy_process_group()
