import argparse
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision.ops import box_iou, complete_box_iou_loss
from tqdm import tqdm


# =================== utilities ===================
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_gt_box_from_mask(mask: np.ndarray) -> np.ndarray:
    """Return bbox [x0,y0,x1,y1] from binary mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.array([0, 0, 0, 0], dtype=np.float32)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def sample_shifted_boxes(gt_box: torch.Tensor, image_shape, num_samples=10,
                         max_shift_frac=0.3, min_iou=0.5, seed=0):
    """Generate multiple shifted boxes (IoU>min_iou)."""
    H, W = image_shape
    set_all_seeds(seed)
    boxes = []
    gx0, gy0, gx1, gy1 = gt_box.tolist()
    w, h = gx1 - gx0, gy1 - gy0
    for i in range(num_samples * 5):  # oversample a bit
        dx = random.uniform(-max_shift_frac, max_shift_frac) * w
        dy = random.uniform(-max_shift_frac, max_shift_frac) * h
        cand = torch.tensor([gx0 + dx, gy0 + dy, gx1 + dx, gy1 + dy])
        cand[0::2].clamp_(0, W - 1)
        cand[1::2].clamp_(0, H - 1)
        iou = box_iou(cand.unsqueeze(0), gt_box.unsqueeze(0))[0, 0].item()
        if iou > min_iou:
            boxes.append(cand)
        if len(boxes) >= num_samples:
            break
    if not boxes:
        boxes = [gt_box]
    return torch.stack(boxes[:num_samples])


def batch_iou_torch(gt_masks: torch.Tensor, pred_masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
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


# class Task:
#     image
#     mask
#     gt_bbox
#     shifted_bbox

# def run_inference(task:Task):
#     input_box = task.shifted_bbox.numpy()
#     masks, _, _ = predictor.predict(point_coords=None,
#                                     point_labels=None,
#                                     box=input_box[None, :],
#                                     multimask_output=False)
#     pred_mask = torch.from_numpy(masks[0]).bool()
#     gt_mask_t = torch.from_numpy(gt_mask).bool()

#     mask_iou = compute_mask_iou(pred_mask, gt_mask_t)
#     ciou = box_iou(sbox.unsqueeze(0), gt_box.unsqueeze(0))[0, 0].item()

#     shifted_bbox_results.append({
#         "image": os.path.basename(img_path),
#         "shift_id": idx,
#         "shifted_box": input_box.tolist(),
#         "mask_iou": mask_iou,
#         "cIoU": ciou,
#     })
def main(args):
    set_all_seeds(42)

    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint).to("cuda")
    predictor = SamPredictor(sam)
    resize = ResizeLongestSide(1024)
    image_files = sorted([
        os.path.join(args.images_dir, f)
        for f in os.listdir(args.images_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    ])
    mask_files = sorted([
        os.path.join(args.masks_dir, f)
        for f in os.listdir(args.masks_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ])
    assert len(image_files) == len(mask_files), "Mismatched image/mask count."

    shifted_bbox_results = []
    N_jitters = 15
    for img_path, mask_path in tqdm(zip(image_files, mask_files), total=len(image_files)*N_jitters):
        image = np.array(Image.open(img_path).convert("RGB"))
        image = resize.apply_image(image)
        gt_mask = np.array(Image.open(mask_path).convert("L"))
        gt_mask = resize.apply_image_torch(torch.Tensor(gt_mask).unsqueeze(0).unsqueeze(0))
        H, W = gt_mask.shape[2:]
        gt_box_np = get_gt_box_from_mask(gt_mask[0, 0, ...])
        gt_box = torch.tensor(gt_box_np, dtype=torch.float32)


        # generate 10 jitters
        shifted_boxes = sample_shifted_boxes(gt_box, (H, W),
                                             num_samples=N_jitters,
                                             max_shift_frac=args.jitter_pow,
                                             min_iou=0.5,
                                             seed=42)
        shifted_boxes = shifted_boxes[:10, :]
        # shifted_boxes = gt_box
        predictor.set_image(image)
        # for idx, sbox in enumrate(shifted_boxes):
            # input_box = sbox.numpy()
        shifted_boxes = resize.apply_boxes_torch(shifted_boxes, image.shape[:2])
        masks, _, _ = predictor.predict_torch(point_coords=None,
                                        point_labels=None,
                                        boxes=shifted_boxes.to("cuda").to(torch.int32),
                                        multimask_output=False)
        # pred_masks = torch.from_numpy(masks[0]).bool()
        # gt_mask_t = torch.from_numpy(gt_mask).bool()
        # logger.debug(gt_mask.unsqueeze(0).repeat_interleave(N_jitters, 0).shape)
        mask_iou = batch_iou_torch(masks[:, 0, ...].bool(), gt_mask.unsqueeze(0).repeat_interleave(masks.size(0), 0).to("cuda"))
        # Save concatenated masks for debugging
        # Convert mask predictions to RGB for drawing boxes
        # For each mask and shifted bbox pair
        # for i in range(N_jitters):
        #     # Create RGB versions of prediction and GT masks
        #     pred_mask_rgb = torch.repeat_interleave(masks[i, 0, ...].bool().cpu().unsqueeze(-1), 3, dim=-1)
        #     gt_mask_rgb = torch.repeat_interleave(gt_mask[0, 0, ...].unsqueeze(-1), 3, dim=-1)

        #     # Draw GT bbox in red on both images
        #     x0,y0,x1,y1 = gt_box.int()
        #     pred_mask_rgb[y0:y0+2, x0:x1, 0] = 1
        #     pred_mask_rgb[y1-2:y1, x0:x1, 0] = 1
        #     pred_mask_rgb[y0:y1, x0:x0+2, 0] = 1
        #     pred_mask_rgb[y0:y1, x1-2:x1, 0] = 1

        #     gt_mask_rgb[y0:y0+2, x0:x1, 0] = 1
        #     gt_mask_rgb[y1-2:y1, x0:x1, 0] = 1
        #     gt_mask_rgb[y0:y1, x0:x0+2, 0] = 1
        #     gt_mask_rgb[y0:y1, x1-2:x1, 0] = 1

        #     # Draw shifted bbox in blue on both images
        #     x0,y0,x1,y1 = shifted_boxes[i].int()
        #     pred_mask_rgb[y0:y0+2, x0:x1, 2] = 1
        #     pred_mask_rgb[y1-2:y1, x0:x1, 2] = 1
        #     pred_mask_rgb[y0:y1, x0:x0+2, 2] = 1
        #     pred_mask_rgb[y0:y1, x1-2:x1, 2] = 1

        #     gt_mask_rgb[y0:y0+2, x0:x1, 2] = 1
        #     gt_mask_rgb[y1-2:y1, x0:x1, 2] = 1
        #     gt_mask_rgb[y0:y1, x0:x0+2, 2] = 1
        #     gt_mask_rgb[y0:y1, x1-2:x1, 2] = 1

        #     # Concatenate GT and prediction horizontally
        #     debug_img = torch.cat([gt_mask_rgb, pred_mask_rgb], dim=1)

        #     # Save image for this pair
        #     output_path = f"debug_masks_{os.path.basename(img_path).split('.')[0]}_{i}.png"
        #     Image.fromarray((debug_img.numpy() * 255).astype(np.uint8)).save(output_path)
        ciou = complete_box_iou_loss(shifted_boxes, gt_box.unsqueeze(0).repeat_interleave(repeats=shifted_boxes.shape[0],dim= 0)).detach().cpu().numpy()
        for idx in range(shifted_boxes.shape[0]):
            shifted_bbox_results.append({
                "image": os.path.basename(img_path),
                "mask_iou": mask_iou[idx].item(),
                "ciou": ciou[idx].item(),
                "gt_bbox": gt_box.tolist(),
                "shifted_bbox": shifted_boxes[idx].tolist()
            })
        # shifted_bbox_results.append({
        #     "image": os.path.basename(img_path),
        #     # "shift_id": idx,
        #     "shifted_box": input_box.tolist(),
        #     "mask_iou": mask_iou,
        #     "cIoU": ciou,
        # })


    # with tqdm_joblib(tqdm(desc="Processing", total=len(tasks))) as progress_bar:
    #     results = Parallel(n_jobs=n_jobs)(
    #         delayed(fit_single_segment)(x) for x in tasks
    #     )
    # print("\n=== Example Results (first 5) ===")
    df = pd.DataFrame(shifted_bbox_results)
    df.to_csv(f"jitter_results/{Path(args.images_dir).parent.name}_{len(shifted_bbox_results)}_{args.jitter_pow}.csv", index=False)

    print(f"\nTotal evaluated instances: {len(shifted_bbox_results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Path to dataset_name/images/")
    parser.add_argument("--masks_dir", required=True, help="Path to dataset_name/masks/")
    parser.add_argument("--sam_checkpoint", required=True,
                        help="path to sam vit-h checkpoint (e.g. sam_vit_h_4b8939.pth)")
    parser.add_argument("--jitter_pow", type=float, default=0.3,
                        help="path to sam vit-h checkpoint (e.g. sam_vit_h_4b8939.pth)")

    args = parser.parse_args()
    main(args)
