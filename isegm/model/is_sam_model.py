import torch
import torch.nn as nn
import numpy as np
import torchvision
import cv2
from isegm.model.ops import DistMapsSAM
from segmentation_models_pytorch.losses import DiceLoss
from isegm.inference import utils
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide



def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class ISModelSAM(nn.Module):
    def __init__(self, device='cuda:6', model_path=None):
        super().__init__()
        self.dist_maps = DistMapsSAM(norm_radius=5, spatial_scale=1.0, cpu_mode=False, use_disks=True)
        self.prev_mask = None

        self.with_prev_mask = True
        self.binary_prev_mask = False
        self._load_model(device, model_path)
        self.regularization_distribution = torch.distributions.Gamma(concentration=torch.Tensor([1.7891955953866256]).to(self.device), rate=torch.Tensor([1/0.12104836378138935]).to(self.device))

    def _load_model(self,device, model_path ):
        model_type = 'vit_b' if 'vit_b' in str(model_path) else 'vit_h' if 'vit_h' in str(model_path) else 'vit_l'
        sam = sam_model_registry[model_type](checkpoint=model_path)
        for n, p in sam.named_parameters():
            p.requires_grad = False
        sam.eval()
        sam.to(device=device)
        self.device = device
        self.sam_predictor = SamPredictor(sam)
        self.resize = ResizeLongestSide(sam.image_encoder.img_size)



    def forward_optimizable_bbox(self, optimize, image_in, gt_mask, bbox_in, args=None, transforms=None, gt_mask_without_transforms=None):
        for MODE in ["MIN" if args.optim_min else "MAX"]:


            bbox_in = torch.clone(bbox_in)
            image = image_in[:, :3, :, :]
            input_image = self.resize.apply_image_torch(image * 255)
            gt_mask_resized = self.resize.apply_image_torch(gt_mask)
            gt_mask_resized = (gt_mask_resized > 0.5).float()
            self.sam_predictor.set_torch_image(input_image, image.shape[2:])

            old_h, old_w = image.shape[2:]
            new_h, new_w = get_preprocess_shape(old_h, old_w, self.sam_predictor.transform.target_length)

            bbox_in[..., [0,2]] = bbox_in[..., [0,2]] * (new_w / old_w)
            bbox_in[..., [1,3]] = bbox_in[..., [1,3]] * (new_h / old_h)
            bbox_in = bbox_in.to("cuda")

            bbox = torch.nn.Parameter(bbox_in.clone())
            bbox.requires_grad = True
            instance_loss = DiceLoss('binary', from_logits=False)

            alpha = args.lambda_mult
            lr = args.lr_mult * 1 * np.sqrt(image.shape[-1] ** 2 + image.shape[-2] ** 2) / (1024 * np.sqrt(2))
            optimizer = torch.optim.Adam([bbox], lr = lr)

            best_iou = [-2, -2] if MODE == 'MAX' else [2, 2]
            best_params = None
            best_outputs = {}

            total_logging_record = {
                                'iou': [],
                                'biou': [],
                                'bbox_coords': [],
                                'normalized_coords': [],
                                'input_image_shape': [],
                                'processed_image_shape': [],
                                'loss': []
                                }
            if bbox.requires_grad:
                iters = args.n_opt_steps
            else:
                iters = 1
            if args.vis_optim:
                img_numpy = np.clip(input_image[0].cpu().permute(1, 2, 0).cpu().numpy(), 0, 255).astype(np.uint8)
                mask_stack = np.dstack([gt_mask_resized[0][0].cpu(), gt_mask_resized[0][0].cpu(), gt_mask_resized[0][0].cpu()])
                img_to_save = img_numpy.copy()
                prediction_stack = None
            for i in range(iters):

                optimizer.zero_grad()
                if args.vis_optim:
                    bbox_numpy = bbox[0].detach().cpu().numpy()
                    bbox_numpy[[0,2]] *= (new_w / old_w)
                    bbox_numpy[[1,3]] *= (new_h / old_h)
                    bbox_numpy[[0,2]] = np.clip(bbox_numpy[[0,2]], 1, new_w - 1)
                    bbox_numpy[[1,3]] = np.clip(bbox_numpy[[1,3]], 1, new_h - 1)
                    bbox_numpy = np.nan_to_num(bbox_numpy, nan=0.0)
                    cv2.rectangle(img_to_save,
                                (int(bbox_numpy[0]), int(bbox_numpy[1])),
                                (int(bbox_numpy[2]), int(bbox_numpy[3])),
                                (0,255,30*i), 2)

                res, scores, logits = self.sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                            boxes=bbox,
                            multimask_output=True,
                            return_logits=True)
                prediction = torch.sigmoid(res)
                prediction = prediction[0, torch.argmax(scores)][None, None]

                if bbox.requires_grad:
                    main_loss = torch.mean(instance_loss(prediction, gt_mask.to(prediction.device).contiguous())) * (1 if MODE == 'MAX' else - 1)
                    safe_bbox = torch.relu(bbox)
                    ciou_loss = torchvision.ops.ciou_loss.complete_box_iou_loss(safe_bbox, bbox_in)
                    if torch.isnan(ciou_loss).item():
                        ciou_loss = torch.nan_to_num(ciou_loss)
                    reg_value = self.regularization_distribution.log_prob(ciou_loss)
                    reg_grad = torch.autograd.grad(alpha*reg_value, bbox, )
                    reg_grad = (reg_grad[0].norm(2).item() ** 2)

                    dice_grad = torch.autograd.grad(main_loss, bbox, )
                    dice_grad = (dice_grad[0].norm(2).item() ** 2)
                    if ciou_loss != 0 :
                        loss = main_loss+ alpha * reg_value * (-1)
                    else:
                        loss = main_loss



                if args.vis_optim:
                    prediction_stack = nn.functional.interpolate(prediction.detach(), mode='bilinear', align_corners=True, size=mask_stack.shape[:2])
                    prediction_stack = np.dstack([prediction_stack[0][0].cpu(), prediction_stack[0][0].cpu(), prediction_stack[0][0].cpu()])

                curr_params = bbox.detach().cpu().numpy()
                curr_params[..., [0,2]] *= (old_w / new_w)
                curr_params[..., [1,3]] *= (old_h / new_h)
                curr_params[..., [0,2]] = np.clip(curr_params[...,[0,2]], 1, new_w - 1)
                curr_params[..., [1,3]] = np.clip(curr_params[...,[1,3]], 1, new_h - 1)
                prediction_detached = prediction.detach()
                prediction_cpu = prediction_detached.cpu().detach().numpy()[0, 0] > args.thresh

                curr_iou = utils.get_iou_fast(gt_mask_without_transforms, prediction_cpu)
                curr_biou = utils.get_boundary_iou_fast(gt_mask_without_transforms, prediction_cpu)
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
        self.sam_predictor.reset_image()
        if args.vis_optim:
            cv2.imwrite(f'bbox_{i}.png', np.hstack([cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR), 255 * mask_stack, 255 * prediction_stack]))
        return best_outputs, best_params, total_logging_record

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
