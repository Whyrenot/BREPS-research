from isegm.model.is_sam_model import ISModelSAM
from segmentation_models_pytorch.losses import DiceLoss
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch
import numpy as np
import cv2
from isegm.model.ops import DistMapsSAM
from isegm.inference import utils
from scribbleprompt.models.sam import ScribblePromptSAM


class ISModelScribblePrompt(ISModelSAM):
    def __init__(self, device="cuda:6", scribble_weights_path=None):
        self.device = device
        super().__init__(device=device, model_path=scribble_weights_path)

        if scribble_weights_path is None:
            raise ValueError("No path to weights")

        self.prev_mask = None
        self.with_prev_mask = True
        self.binary_prev_mask = False

        self.dist_maps = DistMapsSAM(
            norm_radius=5, spatial_scale=1.0,
            cpu_mode=False, use_disks=True
        )

        self._load_model(device, scribble_weights_path)
        self.regularization_distribution = torch.distributions.Gamma(concentration=torch.Tensor([1.7891955953866256]).to(self.device), rate=torch.Tensor([1/0.12104836378138935]).to(self.device))

    def _load_model(self, device, model_path):
        ScribblePromptSAM.weights["v1"] = str(model_path)
        self.sam_predictor = ScribblePromptSAM(version="v1", device=device)

        self.device = device
        self.resize = None

def forward_optimizable_bbox(self, optimize, image_in, gt_mask, bbox_in,
                             args=None, transforms=None, gt_mask_without_transforms=None):
    for MODE in ["MIN" if args.optim_min else "MAX"]:
        bbox_in = torch.clone(bbox_in).to(self.device)
        image = image_in[:, :1, :, :].to(self.device)
        image = image / 255.0 if image.max() > 1 else image

        input_image = self.sam_predictor.prepare_image(image, resize=True, normalize=True)
        gt_mask_resized = F.interpolate(gt_mask.to(self.device),
                                        size=input_image.shape[-2:], mode="nearest")
        gt_mask_resized = (gt_mask_resized > 0.5).float()

        self.sam_predictor.original_size = image.shape[-2:]
        img_features = self.sam_predictor.encoder_forward(image)

        bbox = torch.nn.Parameter(bbox_in.clone(), requires_grad=True)

        instance_loss = DiceLoss('binary', from_logits=False)
        alpha = args.lambda_mult
        lr = args.lr_mult * np.sqrt(image.shape[-1] ** 2 + image.shape[-2] ** 2) / (1024 * np.sqrt(2))
        optimizer = torch.optim.Adam([bbox], lr=lr)

        best_iou = [-2, -2] if MODE == 'MAX' else [2, 2]
        best_params, best_outputs = None, {}

        iters = args.n_opt_steps if bbox.requires_grad else 1

        total_logging_record = {
            'iou': [],
            'biou': [],
            'bbox_coords': [],
            'normalized_coords': [],
            'input_image_shape': [],
            'processed_image_shape': [],
            'loss': []
        }

        if args.vis_optim:
            img_numpy = np.clip(input_image[0].cpu().permute(1, 2, 0).numpy(), 0, 255).astype(np.uint8)
            mask_stack = np.dstack([gt_mask_resized[0][0].cpu()] * 3)
            img_to_save = img_numpy.copy()
            prediction_stack = None

        for i in range(iters):
            optimizer.zero_grad()

            if args.vis_optim:
                bbox_numpy = bbox[0].detach().cpu().numpy()
                bbox_numpy[[0, 2]] = np.clip(bbox_numpy[[0, 2]], 1, input_image.shape[-1] - 1)
                bbox_numpy[[1, 3]] = np.clip(bbox_numpy[[1, 3]], 1, input_image.shape[-2] - 1)
                cv2.rectangle(img_to_save,
                              (int(bbox_numpy[0]), int(bbox_numpy[1])),
                              (int(bbox_numpy[2]), int(bbox_numpy[3])),
                              (0, 255, 30 * i), 2)

            masks, img_features, low_res_masks = self.sam_predictor.predict(
                img=image,
                box=bbox.unsqueeze(0).unsqueeze(0).to(self.device),
                img_features=img_features,
                return_logits=True
            )

            prediction = masks
            main_loss = torch.mean(
                instance_loss(prediction, gt_mask.to(prediction.device).contiguous())
            ) * (1 if MODE == 'MAX' else -1)

            safe_bbox = torch.relu(bbox).to(self.device)
            ciou_loss = torchvision.ops.ciou_loss.complete_box_iou_loss(safe_bbox, bbox_in)
            ciou_loss = torch.nan_to_num(ciou_loss)

            reg_value = self.regularization_distribution.log_prob(ciou_loss)

            loss = main_loss + alpha * reg_value * (-1) if ciou_loss.item() != 0 else main_loss

            if isinstance(gt_mask_without_transforms, np.ndarray):
                gt_mask_without_transforms_tensor = torch.from_numpy(gt_mask_without_transforms).to(self.device)
            else:
                gt_mask_without_transforms_tensor = gt_mask_without_transforms.to(self.device)

            curr_params = bbox.detach().cpu().numpy()
            prediction_cpu = (prediction.detach().cpu().numpy()[0, 0] > args.thresh).astype(np.uint8)

            curr_iou = utils.get_iou_fast(gt_mask_without_transforms_tensor.cpu().numpy(), prediction_cpu)
            curr_biou = utils.get_boundary_iou_fast(gt_mask_without_transforms_tensor.cpu().numpy(), prediction_cpu)

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

            if ((curr_metrics > best_iou and MODE == 'MAX')
                    or (curr_metrics < best_iou and MODE != 'MAX')):
                best_params = [[float(i) for i in curr_params[0]]]
                best_iou = [float(curr_iou), float(curr_biou)]
                print(MODE, "IoU/BIoU update: ", best_iou)
                best_outputs = {'instances': prediction.detach()}
                logging_record['updated'] = 1
            else:
                logging_record['updated'] = 0

            if bbox.requires_grad:
                loss.backward(retain_graph=True)
                optimizer.step()
                total_norm = bbox.grad.data.norm(2).item()
                logging_record['prompt_grad_norm'] = float(total_norm)

            for key, value in logging_record.items():
                if key not in total_logging_record:
                    total_logging_record[key] = []
                total_logging_record[key].append(value)

        self.sam_predictor.reset_image()

        if args.vis_optim:
            prediction_stack = nn.functional.interpolate(
                prediction.detach(), mode='bilinear', align_corners=True, size=mask_stack.shape[:2]
            )
            prediction_stack = np.dstack([prediction_stack[0][0].cpu()] * 3)
            cv2.imwrite(
                f'bbox_{i}.png',
                np.hstack([
                    cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR),
                    255 * mask_stack,
                    255 * prediction_stack
                ])
            )

    return best_outputs, best_params, total_logging_record
