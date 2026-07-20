from datetime import timedelta
from pathlib import Path
from loguru import logger
from numba import njit
import torch
import numpy as np
from isegm.data.datasets import (
    GrabCutDataset,
    BerkeleyDataset,
    DavisDataset,
    SBDEvaluationDataset,
    TETRISDataset,
    # InteractionDataset,
    Split1kDataset,
    # HeLaNucDataset,
    # WBCDataset,
    ACDCDataset,
    BUIDDataset,
)
from isegm.utils.serialization import load_model
from isegm.utils.misc import get_bbox_from_mask
from isegm.inference.clicker import Click
import cv2
import os
import random








def setup_deterministic(seed, is_full=True):
    """
    set every seed
    """
    torch.set_printoptions(precision=16)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(is_full)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if is_full:
        torch.set_deterministic_debug_mode(2)


def get_time_metrics(all_ious, elapsed_time):
    n_images = len(all_ious)
    n_clicks = sum(map(len, all_ious))

    mean_spc = elapsed_time / n_clicks
    mean_spi = elapsed_time / n_images

    return mean_spc, mean_spi


def load_is_model(checkpoint, device, **kwargs):
    if isinstance(checkpoint, (str, Path)):
        state_dict = torch.load(checkpoint, map_location="cpu")
    else:
        state_dict = checkpoint

    if isinstance(state_dict, list):
        model = load_single_is_model(state_dict[0], device, **kwargs)
        models = [load_single_is_model(x, device, **kwargs) for x in state_dict]

        return model, models
    else:
        return load_single_is_model(state_dict, device, **kwargs)


def load_single_is_model(state_dict, device, **kwargs):
    model = load_model(state_dict["config"], **kwargs)
    model.load_state_dict(state_dict["state_dict"], strict=False)

    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    return model


def get_dataset(dataset_name, cfg, args):

    if dataset_name == "GrabCut":
        dataset = GrabCutDataset(cfg.GRABCUT_PATH, args)
    elif dataset_name == "Berkeley":
        dataset = BerkeleyDataset(cfg.BERKELEY_PATH, args)
    elif dataset_name == "DAVIS":
        dataset = DavisDataset(cfg.DAVIS_PATH, args)
    elif dataset_name == "SBD":
        dataset = SBDEvaluationDataset(cfg.SBD_PATH, args)
    elif dataset_name == "SBD_Train":
        dataset = SBDEvaluationDataset(cfg.SBD_PATH, args, split="train")
    elif dataset_name == "PascalVOC":
        dataset = Split1kDataset(cfg.PASCALVOC_PATH, args)
    elif dataset_name == "COCO_MVal":
        dataset = DavisDataset(cfg.COCO_MVAL_PATH, args)
    elif dataset_name == "TETRIS":
        dataset = TETRISDataset(cfg.TETRIS_PATH, args)
    elif dataset_name == "ADE20K":
        dataset = Split1kDataset(cfg.ADE20K_PATH, args)
    elif dataset_name == "INTERACTIONS":
        dataset = InteractionDataset(cfg.INTERACTIONSET_PATH, args)
    # elif dataset_name == "HELA":
    #     dataset = HeLaNucDataset(cfg.HELA_PATH, args)
    elif dataset_name == "WBC":
        dataset = WBCDataset(cfg.WBC_PATH, args)
    elif dataset_name == "ACDC":
        dataset = ACDCDataset(cfg.ACDC_PATH, args)
    elif dataset_name == "BUID":
        dataset = BUIDDataset(cfg.BUID_PATH, args)
    elif dataset_name == "USNEW":
        dataset = UserStudyNewDataset(cfg.USER_STUDY_NEW_PATH, args)
    else:
        dataset = None

    return dataset

# ---------------------- numba iou
@njit
def get_iou_fast(gt_mask, pred_mask, ignore_label=-1):
    intersection = 0
    union = 0
    for i in range(gt_mask.shape[0]):
        for j in range(gt_mask.shape[1]):
            if gt_mask[i, j] == ignore_label:
                continue
            gt_obj = gt_mask[i, j] == 1
            pred_obj = pred_mask[i, j]
            if gt_obj and pred_obj:
                intersection += 1
            if gt_obj or pred_obj:
                union += 1
    return intersection / union if union != 0 else 0

def mask_to_boundary_fast(mask, dilation=3):  # Fixed dilation instead of ratio-based
    h, w = mask.shape
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    return mask - new_mask_erode[1:h+1, 1:w+1]

def get_boundary_iou_fast(gt_mask, pred_mask, ignore_label=-1):
    obj_gt_mask = (gt_mask == 1).astype(np.uint8)
    gt_boundary = mask_to_boundary(obj_gt_mask)
    pred_boundary = mask_to_boundary(pred_mask.astype(np.uint8))
    return compute_boundary_iou_numba(gt_mask, gt_boundary, pred_boundary, ignore_label)

@njit
def compute_boundary_iou_numba(gt_mask, gt_boundary, pred_boundary, ignore_label=-1):
    intersection = 0
    union = 0
    for i in range(gt_mask.shape[0]):
        for j in range(gt_mask.shape[1]):
            if gt_mask[i, j] == ignore_label:
                continue
            gt_b = gt_boundary[i, j] > 0
            pred_b = pred_boundary[i, j] > 0
            if gt_b and pred_b:
                intersection += 1
            if gt_b or pred_b:
                union += 1
    return intersection / union if union != 0 else 0
# ---------------------------------------------

def get_iou(gt_mask, pred_mask, ignore_label=-1):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(
        np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv
    ).sum()
    union = np.logical_and(
        np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv
    ).sum()

    return intersection / union


def mask_to_polygon(mask, dilation_coeff: float = 0.1) -> np.ndarray:
    """
    Convert binary mask to a rectangle [x_min, y_min, height, width]
    around the object (always a single object in the mask and masks are dilated with a set coefficient)

    :param mask: binary mask of shape [W, H]
    :param dilation_coeff: coefficient for mask dilation, defaults to 0.1
    :return: rectangle coordinates as [x_min, y_min, height, width]
    """
    mask = mask.astype(np.uint8)

    # Dilate mask by 10%
    h, w = mask.shape
    kernel_size = max(1, int(min(h, w) * dilation_coeff))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cv2.dilate(mask, kernel, iterations=1)
    bbox = get_bbox_from_mask(mask)
    return bbox


def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h**2 + w**2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def get_boundary_iou(gt_mask, pred_mask, ignore_label=-1):
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
    return intersection / union


def compute_noc_metric(all_ious, iou_thrs, max_clicks=20):
    def _get_noc(iou_arr, iou_thr):
        if isinstance(iou_arr, dict) and 'iou' in iou_arr:
            ious = np.array(iou_arr['iou'])
            if ious.ndim > 1:
                ious = ious[0] if len(ious) > 0 and isinstance(ious[0], np.ndarray) else ious
        else:
            ious = np.array([x[0] if isinstance(x, (list, np.ndarray, tuple)) else x for x in iou_arr])
        vals = ious >= iou_thr
        return np.argmax(vals) + 1 if np.any(vals) else max_clicks

    noc_list = []
    over_max_list = []
    for iou_thr in iou_thrs:
        scores_arr = np.array(
            [_get_noc(iou_arr, iou_thr) for iou_arr in all_ious], dtype=int
        )

        score = scores_arr.mean()
        over_max = (scores_arr == max_clicks).sum()

        noc_list.append(score)
        over_max_list.append(over_max)

    return noc_list, over_max_list


def find_checkpoint(weights_folder, checkpoint_name):
    weights_folder = Path(weights_folder)
    if ":" in checkpoint_name:
        model_name, checkpoint_name = checkpoint_name.split(":")
        models_candidates = [
            x for x in weights_folder.glob(f"{model_name}*") if x.is_dir()
        ]
        assert len(models_candidates) == 1
        model_folder = models_candidates[0]
    else:
        model_folder = weights_folder

    if checkpoint_name.endswith(".pth") or checkpoint_name.endswith(".pt"):
        if Path(checkpoint_name).exists():
            checkpoint_path = checkpoint_name
        else:
            checkpoint_path = weights_folder / checkpoint_name
    else:
        model_checkpoints = list(model_folder.rglob(f"{checkpoint_name}*.pt*"))
        assert len(model_checkpoints) == 1
        checkpoint_path = model_checkpoints[0]

    return str(checkpoint_path)


def get_results_table(
    noc_list,
    over_max_list,
    brs_type,
    dataset_name,
    mean_spc,
    elapsed_time,
    n_clicks=20,
    model_name=None,
):
    table_header = (
        f'|{"BRS Type":^13}|{"Dataset":^11}|'
        f'{"NoC@80%":^9}|{"NoC@85%":^9}|{"NoC@90%":^9}|'
        f'{">="+str(n_clicks)+"@85%":^9}|{">="+str(n_clicks)+"@90%":^9}|'
        f'{"SPC,s":^7}|{"Time":^9}|'
    )
    row_width = len(table_header)

    header = f"Eval results for model: {model_name}\n" if model_name is not None else ""
    header += "-" * row_width + "\n"
    header += table_header + "\n" + "-" * row_width

    eval_time = str(timedelta(seconds=int(elapsed_time)))
    table_row = f"|{brs_type:^13}|{dataset_name:^11}|"
    table_row += f"{noc_list[0]:^9.2f}|"
    table_row += f"{noc_list[1]:^9.2f}|" if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f"{noc_list[2]:^9.2f}|" if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f"{over_max_list[1]:^9}|" if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f"{over_max_list[2]:^9}|" if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f"{mean_spc:^7.3f}|{eval_time:^9}|"

    return header, table_row
