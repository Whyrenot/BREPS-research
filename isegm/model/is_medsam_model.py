import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision
from isegm.model.ops import DistMapsSAM
from medsam import SamPredictor
from medsam.build_sam import sam_model_registry
from segmentation_models_pytorch.losses import DiceLoss
from isegm.model.is_sam_model import get_preprocess_shape
from .is_sam_model import ISModelSAM
from isegm.inference import utils
from medsam.utils.transforms import ResizeLongestSide

class ISModelMedSAM(ISModelSAM):
    def __init__(self, device='cuda', model_path=None):

        super().__init__(device=device, model_path=model_path)
        self.dist_maps = DistMapsSAM(norm_radius=5, spatial_scale=1.0, cpu_mode=False, use_disks=True)
        self.prev_mask = None
        self.with_prev_mask = True
        self.binary_prev_mask = False
        self._load_model(device, model_path)
        self.regularization_distribution = torch.distributions.Gamma(concentration=torch.Tensor([1.7891955953866256]).to(self.device), rate=torch.Tensor([1/0.12104836378138935]).to(self.device))

    def _load_model(self, device, model_path):
        model_type = 'vit_b' if 'vit_b' in str(model_path) else 'vit_h' if 'vit_h' in str(model_path) else 'vit_l'
        sam_model = sam_model_registry[model_type](checkpoint=model_path)
        for n, p in sam_model.named_parameters():
            p.requires_grad = False
        sam_model.eval()
        sam_model.to(device=device)
        self.device = device
        self.sam_predictor = SamPredictor(sam_model)
        self.resize = ResizeLongestSide(sam_model.image_encoder.img_size)

    def forward_optimizable_bbox(self, optimize, image_in, gt_mask, bbox_in, args=None, transforms=None, gt_mask_without_transforms=None):
        print(f"[DEBUG] Number of optimization steps (n_opt_steps): {args.n_opt_steps}")
        for MODE in ["MIN" if args.optim_min else "MAX"]:

            per_step_metrics = []
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
            # print("\n[DEBUG] --- BBOX DEBUG INFO ---")
            # print(f"Original bbox_in:\n{bbox_in.cpu().numpy()}")
            # print(f"Scaled bbox:\n{bbox.detach().cpu().numpy()}")
            # print(f"bbox Tensor shape: {bbox.shape}")

            # print(f"Input image shape: {image_in.shape}")
            # print(f"Resized input_image shape: {input_image.shape}")
            # print(f"GT mask shape: {gt_mask.shape}")
            # print("[DEBUG] -----------------------\n")
            bbox = torch.nn.Parameter(bbox_in.clone())
            instance_loss = DiceLoss('binary', from_logits=False)

            alpha = args.lambda_mult
            lr = args.lr_mult * 1 * np.sqrt(image.shape[-1] ** 2 + image.shape[-2] ** 2) / (1024 * np.sqrt(2))
            optimizer = torch.optim.Adam([bbox], lr = lr)

            best_iou = [-2, -2] if MODE == 'MAX' else [2, 2]
            best_params = None
            best_outputs = {}

            if bbox.requires_grad:
                iters = args.n_opt_steps
                print(f"[DEBUG] Starting optimization loop with {iters} steps.")
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
                    cv2.rectangle(img_to_save,
                                (int(bbox_numpy[0]), int(bbox_numpy[1])),
                                (int(bbox_numpy[2]), int(bbox_numpy[3])),
                                (0,255,30*i), 1)

                res, scores, logits = self.sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                            boxes=bbox,
                            multimask_output=True,
                            return_logits=True)
                
                prediction = torch.sigmoid(res)
                mask_weights = scores.softmax(dim=1)
                prediction = torch.sum(prediction * mask_weights[:, :, None, None], dim=1, keepdim=True)

                # if prediction.shape != gt_mask.shape:
                #     prediction = self.sam_predictor.transform.postprocess_masks(
                #         prediction, 
                #         input_size=image.shape[2:],     
                #         original_size=gt_mask.shape[2:]  
                #     )

                if bbox.requires_grad:
                    main_loss = torch.mean(
                        instance_loss(prediction, gt_mask.to(prediction.device))
                    ) * (1 if MODE == 'MAX' else -1)

                    safe_bbox = torch.relu(bbox)
                    ciou_loss = torchvision.ops.complete_box_iou_loss(safe_bbox, bbox_in)  
                    ciou_loss = torch.nan_to_num(ciou_loss)
                    reg_value = ciou_loss.mean()
                    dice_grad = 0
                    reg_grad = 0

                    loss = main_loss + alpha * reg_value * (-1)


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

                logging_record = [*curr_metrics, *bbox.detach().cpu().numpy(),
                                *curr_params, *list(image_in.shape),
                                *list(image.shape),
                                main_loss.detach().cpu().item(), ciou_loss.detach().cpu().item(), reg_value.detach().cpu().item(),
                                dice_grad, reg_grad
                                # total_norm.detach().cpu().item()
                ]

                if ((curr_metrics > best_iou and MODE == 'MAX') or (curr_metrics < best_iou and MODE != 'MAX')):
                    best_params = [[float(i) for i in curr_params[0]]]
                    best_iou    = [float(curr_iou), float(curr_biou)]

                    print(MODE, "IoU/BIoU update: ", best_iou)

                    best_outputs = {'instances': prediction.detach()}
                    logging_record.append(1)
                else:
                    logging_record.append(0)

                if bbox.requires_grad:
                    loss.backward()
                    optimizer.step()
                    total_norm = (bbox.grad.data.norm(2).item() ** 2) ** (1/2)
                    logging_record.insert(-2, total_norm)
                per_step_metrics.append(logging_record)
        self.sam_predictor.reset_image()
        if args.vis_optim:# logger.info(f"Iteration {i} total time: {iter_time:.4f}s")
            cv2.imwrite(f'bbox_{i}.png', np.hstack([cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR), 255 * mask_stack, 255 * prediction_stack]))
        return best_outputs, best_params, np.array(per_step_metrics, dtype=object)
