import os
import argparse
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from segment_anything import sam_model_registry, SamPredictor

def unnormalize_box(box_1024, orig_shape):
    # Convert box from 1024x1024 space back to original image space
    H, W = orig_shape
    scale_x = W / 1024.0
    scale_y = H / 1024.0
    x1, y1, x2, y2 = box_1024
    
    # Actually, SAM resize longest side to 1024
    scale = 1024.0 / max(H, W)
    new_h, new_w = int(H * scale + 0.5), int(W * scale + 0.5)
    
    x1 = x1 * (W / new_w)
    x2 = x2 * (W / new_w)
    y1 = y1 * (H / new_h)
    y2 = y2 * (H / new_h)
    
    return [int(x1), int(y1), int(x2), int(y2)]

def draw_image_with_masks(image, gt_mask, pred_mask, bbox, title, text_info):
    # image: BGR image
    # gt_mask: binary mask (H,W)
    # pred_mask: binary mask (H,W)
    # bbox: [x1, y1, x2, y2]
    
    out_img = image.copy().astype(np.float32)
    
    # Add GT mask as white overlay (alpha blend)
    gt_rgb = np.zeros_like(out_img)
    gt_rgb[gt_mask > 0] = [255, 255, 255]
    alpha_gt = 0.4 * (gt_mask > 0)[:, :, None]
    out_img = out_img * (1 - alpha_gt) + gt_rgb * alpha_gt
    
    # Add predicted mask as blue overlay
    pred_rgb = np.zeros_like(out_img)
    pred_rgb[pred_mask > 0] = [255, 0, 0] # BGR -> Blue
    alpha_pred = 0.5 * (pred_mask > 0)[:, :, None]
    out_img = out_img * (1 - alpha_pred) + pred_rgb * alpha_pred
    
    out_img = out_img.astype(np.uint8)
    
    # Draw bounding box
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw Text Background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    y_offset = out_img.shape[0] - 20 * len(text_info) - 10
    
    for line in text_info:
        (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.rectangle(out_img, (10, y_offset - text_h - 5), (10 + text_w + 10, y_offset + 5), (0, 0, 0), -1)
        cv2.putText(out_img, line, (15, y_offset), font, font_scale, (255, 255, 255), thickness)
        y_offset += 25
        
    return out_img

def predict_and_draw(sam_predictor, image_path, mask_path, cases, out_dir):
    image_name = Path(image_path).name
    stem = Path(image_path).stem
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image {image_path}")
        return
        
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        print(f"Cannot read mask {mask_path}")
        return
    gt_mask = (gt_mask > 0).astype(np.uint8)
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image_rgb)
    
    rows = []
    for case in cases:
        best_box_1024 = case["best_box"]
        bad_box_1024 = case["bad_box"]
        best_iou = case["best_iou"]
        bad_iou = case["bad_iou"]
        
        best_box_tensor = torch.tensor([best_box_1024], device=sam_predictor.device)
        bad_box_tensor = torch.tensor([bad_box_1024], device=sam_predictor.device)
        
        masks_best, _, _ = sam_predictor.predict_torch(
            point_coords=None, point_labels=None,
            boxes=best_box_tensor, multimask_output=False,
        )
        mask_best = masks_best[0, 0].cpu().numpy().astype(np.uint8)
        
        masks_bad, _, _ = sam_predictor.predict_torch(
            point_coords=None, point_labels=None,
            boxes=bad_box_tensor, multimask_output=False,
        )
        mask_bad = masks_bad[0, 0].cpu().numpy().astype(np.uint8)
        
        best_box_orig = unnormalize_box(best_box_1024, image.shape[:2])
        bad_box_orig = unnormalize_box(bad_box_1024, image.shape[:2])
        
        text_best = [
            f"Base BBox IoU: {best_iou:.3f}",
            f"BBox: {best_box_orig}"
        ]
        img_best_drawn = draw_image_with_masks(image, gt_mask, mask_best, best_box_orig, "Good BBox", text_best)
        
        text_bad = [
            f"Bad BBox IoU: {bad_iou:.3f}", 
            f"BBox: {bad_box_orig}"
        ]
        img_bad_drawn = draw_image_with_masks(image, gt_mask, mask_bad, bad_box_orig, "Bad BBox", text_bad)
        
        row_img = np.hstack([img_best_drawn, np.zeros((img_best_drawn.shape[0], 10, 3), dtype=np.uint8), img_bad_drawn])
        rows.append(row_img)
        
    if rows:
        pad_h = 10
        final_img = rows[0]
        for r in rows[1:]:
            sep = np.zeros((pad_h, final_img.shape[1], 3), dtype=np.uint8)
            final_img = np.vstack([final_img, sep, r])
            
        out_path = Path(out_dir) / f"{stem}_comparison.png"
        cv2.imwrite(str(out_path), final_img)
        print(f"Saved visualization for {stem} to {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shifts_json', type=str, default='critical_shifts.json', help='Output from find_critical_shifts.py')
    parser.add_argument('--images_dir', type=str, default='../datasets/mvp_attacks/dataset/clean', help='Directory with source images')
    parser.add_argument('--masks_dir', type=str, default='../datasets/mvp_attacks/dataset/masks', help='Directory with GT masks')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to SAM model checkpoint')
    parser.add_argument('--model_type', type=str, default='vit_b', help='SAM model type')
    parser.add_argument('--out_dir', type=str, default='visualizations', help='Output directory for drawn images')
    parser.add_argument('--limit', type=int, default=5, help='How many pairs to visualize from JSON (0 for all)')
    parser.add_argument('--continue_from', action='store_true', help='Skip images that are already generated')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.shifts_json, 'r') as f:
        shifts = json.load(f)

    if not shifts:
        print("No shifts to process.")
        return

    print("Loading SAM model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint_path)
    sam.to(device)
    sam_predictor = SamPredictor(sam)

    from collections import defaultdict
    shifts_by_img = defaultdict(list)
    for shift in shifts:
        shifts_by_img[shift["image_name"]].append(shift)

    img_names = list(shifts_by_img.keys())
    if args.limit > 0:
        img_names = img_names[:args.limit]

    for img_name in tqdm(img_names, desc="Drawing images"):
        cases = shifts_by_img[img_name]
        
        # Searching for image file (could be png/jpg)
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            p = Path(args.images_dir) / f"{img_name}{ext}"
            if p.exists():
                img_path = str(p)
                break
                
        # Format masks
        mask_path = None
        for ext in ['.png', '.bmp', '.jpg']:
            p = Path(args.masks_dir) / f"{img_name}{ext}"
            if p.exists():
                mask_path = str(p)
                break

        if not img_path or not mask_path:
            print(f"Could not find image/mask for {img_name}")
            continue

        out_path = Path(args.out_dir) / f"{img_name}_comparison.png"
        if args.continue_from and out_path.exists():
            continue

        predict_and_draw(sam_predictor, img_path, mask_path, cases, args.out_dir)

if __name__ == '__main__':
    main()
