import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import torchvision
from torchvision.transforms import Resize
from isegm.model.ops import DistMapsSAM
from segmentation_models_pytorch.losses import DiceLoss
from isegm.inference import utils

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

#from segment_anything import SamPredictor, sam_model_registry

def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

def get_config_path(model_path:Path):
    model_conf_path = str(model_path.name).split("_")
    model_conf_path[-2] = (model_conf_path[-2][0] if model_conf_path[-2][0] !="b" else "b+" )  + ".yaml"
    return  "configs/sam2.1/" + "_".join(model_conf_path[:-1])

def get_model_type(checkpoint_path:str):
    if "base_plus" in checkpoint_path:
        return  str("configs/sam2.1/sam2.1_hiera_b+.yaml")

    elif "large" in checkpoint_path:
        return str("configs/sam2.1/sam2.1_hiera_l.yaml")
    elif "small" in checkpoint_path:
        return str("configs/sam2.1/sam2.1_hiera_s.yaml")

    elif "tiny" in checkpoint_path:
        return str("configs/sam2.1/sam2.1_hiera_t.yaml")

class ISModelSAM2(nn.Module):
    def __init__(self, device='cuda:6', model_path:Path=Path("")):
        super().__init__()
        self._load_model(model_path=model_path, device=device)
        self.dist_maps = DistMapsSAM(norm_radius=5, spatial_scale=1.0, cpu_mode=False, use_disks=True)
       # self.sam_predictor = SAM2TorchImagePredictor(sam2)
        self.prev_mask = None
        self.with_prev_mask = True
        self.binary_prev_mask = False
        self.regularization_distribution = torch.distributions.Gamma(concentration=torch.Tensor([1.7891955953866256]).to(self.device), rate=torch.Tensor([1/0.12104836378138935]).to(self.device))

    def _load_model(self, model_path, device):
        if "hq" in model_path:

            model_config = "configs/sam2.1/sam2.1_hq_hiera_l.yaml"
        else:
            model_config = get_model_type(str(model_path))

        sam2 = build_sam2(model_config, model_path, device=device)
        for n, p in sam2.named_parameters():
            p.requires_grad = False
        sam2.eval()
        sam2.to(device=device)
        self.resize = Resize((sam2.image_size, sam2.image_size))
        self.sam_predictor = SAM2ImagePredictor(sam2)
        self.device = device



    def forward_optimizable_bbox(self, optimize, image_in, gt_mask, bbox_in, args=None, transforms=None, gt_mask_without_transforms=None):
        for MODE in ["MIN" if args.optim_min else "MAX"]:
            alpha = args.lambda_mult
            total_logging_record = {}
            # Convert bbox from xywh to xyxy format
            bbox_in = torch.clone(bbox_in)
            image = image_in[:, :3, :, :]
            input_image = self.resize(image)
            # input_image = image
            # input_image = torch.Tensor(image)

            self.sam_predictor.set_image(image.clone()) # image.shape[2:])
            self.sam_predictor._orig_hw = [image.shape[2:]]


            gt_mask_resized = self.resize(gt_mask)
            gt_mask_resized = (gt_mask_resized > 0.5).float()
            old_h, old_w = image.shape[2:]
            new_h, new_w = self.sam_predictor.model.image_size, self.sam_predictor.model.image_size
            # Scale bbox coordinates
            # _, _, _, bbox_in = self.sam_predictor._prep_prompts(box=bbox_in,
            #     point_coords=None, point_labels=None, mask_logits=None, normalize_coords = True)
            bbox_in = torch.Tensor(bbox_in.flatten())
            bbox_in[..., [0,2]] = bbox_in[..., [0,2]] * (new_w / old_w)
            bbox_in[..., [1,3]] = bbox_in[..., [1,3]] * (new_h / old_h)
            bbox_in = bbox_in.unsqueeze(0).to("cuda")

            #bbox.requires_grad = True
            bbox = torch.nn.Parameter(bbox_in.clone())

            instance_loss = DiceLoss('binary', from_logits=False)

            lr = args.lr_mult * 1 * np.sqrt(image.shape[-1] ** 2 + image.shape[-2] ** 2) / (1024 * np.sqrt(2))
            optimizer = torch.optim.Adam([bbox], lr = lr)

            best_iou = [-2, -2] if MODE == 'MAX' else [2, 2]
            best_params = None
            best_outputs = {}

            if bbox.requires_grad:
                iters = args.n_opt_steps
            else:
                iters = 1
            if args.vis_optim:
                img_numpy = (np.clip(input_image[0].cpu().permute(1, 2, 0).cpu().numpy(), 0, 255)*255) .astype(np.uint8)
                mask_stack = np.dstack([gt_mask_resized[0][0].cpu(), gt_mask_resized[0][0].cpu(), gt_mask_resized[0][0].cpu()])
                img_to_save = img_numpy.copy()
                prediction_stack = None
            for i in range(iters):
                optimizer.zero_grad()
                if args.vis_optim:
                    # Draw bbox in green
                    bbox_numpy = bbox[0].detach().cpu().numpy()
                    bbox_numpy[[0,2]] *= (new_w / old_w)
                    bbox_numpy[[1,3]] *= (new_h / old_h)
                    bbox_numpy[[0,2]] = np.clip(bbox_numpy[[0,2]], 1, new_w - 1).astype(np.int32)
                    bbox_numpy[[1,3]] = np.clip(bbox_numpy[[1,3]], 1, new_h - 1).astype(np.int32)
                    cv2.rectangle(img_to_save,
                                (int(bbox_numpy[0]), int(bbox_numpy[1])),
                                (int(bbox_numpy[2]), int(bbox_numpy[3])),
                                (0,255,30*i), 1)
                res, scores, logits = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                            box=bbox,
                            multimask_output=True,
                            return_logits=True)

                prediction = torch.sigmoid(res)

                prediction = prediction[torch.argmax(scores), ...].unsqueeze(0).unsqueeze(0)


                # loss_start_time = time.time()
                if bbox.requires_grad:
                    main_loss = torch.mean(instance_loss(prediction, gt_mask.to(prediction.device).contiguous())) * (1 if MODE == 'MAX' else - 1)
                    safe_bbox = torch.relu(bbox)
                    #bbox = torch.relu(bbox) # * torch.tensor([new_w, new_h, new_w, new_h]).to(bbox.device)

                    # Calculate area while preserving gradients
                    # bbox_area = (safe_bbox[..., 2] - safe_bbox[..., 0]) * (safe_bbox[..., 3] - safe_bbox[..., 1])
                    # area_regularization = 1.0 / (bbox_area + 1e-6)  # Prevent division by zero

                    ciou_loss = torchvision.ops.ciou_loss.complete_box_iou_loss(safe_bbox, bbox_in)
                    # ciou_loss =  (torch.Tensor([1.0]).to(self.device)  - torch.Tensor([1e-6]).to(self.device )) * torch.sigmoid(ciou_loss)
                    if torch.isnan(ciou_loss).item():
                        ciou_loss = torch.nan_to_num(ciou_loss)
                    reg_value = self.regularization_distribution.log_prob(ciou_loss)
                    reg_grad = torch.autograd.grad(alpha*reg_value, bbox, )
                    reg_grad = (reg_grad[0].norm(2).item() ** 2)

                    torchvision.ops.box_iou(bbox.detach(), bbox_in)
                    dice_grad = torch.autograd.grad(main_loss, bbox, )
                    dice_grad = (dice_grad[0].norm(2).item() ** 2)
                    if ciou_loss != 0  :
                        loss = main_loss+ alpha * reg_value * (-1)
                    else:
                        loss = main_loss

                    #     loss += reg_value

                if args.vis_optim:
                    prediction_stack = nn.functional.interpolate(prediction.detach(), mode='bilinear', align_corners=True, size=mask_stack.shape[:2])
                    prediction_stack = np.dstack([prediction_stack[0][0].cpu(), prediction_stack[0][0].cpu(), prediction_stack[0][0].cpu()])
                curr_params = bbox.detach().cpu().numpy()
                curr_params[..., [0,2]] *= (old_w / new_w)
                curr_params[..., [1,3]] *= (old_h / new_h)
                curr_params[..., [0,2]] = np.clip(curr_params[...,[0,2]], 1, new_w - 1)
                curr_params[..., [1,3]] = np.clip(curr_params[...,[1,3]], 1, new_h - 1)
                # Compute IOU in FULL resolution
                prediction_detached = prediction.detach()
                prediction_cpu = prediction_detached.cpu().detach().numpy()[0, 0] > args.thresh
                curr_iou = utils.get_iou(gt_mask_without_transforms, prediction_cpu)
                curr_biou = utils.get_boundary_iou(gt_mask_without_transforms, prediction_cpu)

                curr_metrics = [curr_iou, curr_biou]
                logging_record = {
                    'iou': curr_metrics[0],
                    'biou': curr_metrics[1],
                    'bbox_coords': bbox.detach().cpu().numpy().tolist(),
                    'normalized_coords': curr_params.tolist(),
                    'input_image_shape': list(image_in.shape),
                    'processed_image_shape': list(input_image.shape),
                    'loss': loss.detach().cpu().item()
                }
                if ((curr_metrics > best_iou and MODE == 'MAX') or (curr_metrics < best_iou and MODE != 'MAX')):
                    best_params = [float(i) for i in curr_params[0]]
                    best_iou    = [float(curr_iou), float(curr_biou)]

                    print(MODE, "IoU/BIoU update: ", best_iou)

                    best_outputs = {'instances': prediction.detach()}
                    best_params = logging_record['normalized_coords']

                    logging_record['updated'] = 1
                else:
                    logging_record['updated'] = 0

                if bbox.requires_grad:
                    loss.backward()
                    optimizer.step()
                    total_norm = float((bbox.grad.data.norm(2).item() ** 2) ** (1/2))
                    logging_record['prompt_grad_norm'] = float(total_norm)

                for key in logging_record:
                    if key not in total_logging_record:
                        total_logging_record[key] = []
                    total_logging_record[key].append(logging_record[key])
        self.sam_predictor.reset_predictor()
        del optimizer
        torch.cuda.empty_cache()
        if args.vis_optim:
            cv2.imwrite(f'sam2_bbox_{i}.png', np.hstack([cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR), 255 * mask_stack, 255 * prediction_stack]))
        return best_outputs, best_params,total_logging_record

    def prepare_input(self, image):
        prev_mask = None
        if self.with_prev_mask:
            prev_mask = image[:, 3:, :, :]
            image = image[:, :3, :, :]
            if self.binary_prev_mask:
                prev_mask = (prev_mask > 0.5).float()
        return image, prev_mask


    def get_coord_features(self, image, prev_mask, points):
        coord_features = self.dist_maps(image, points)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)

        return coord_features
