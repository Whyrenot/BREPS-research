import torch
import torch.nn as nn
import numpy as np
import cv2
from isegm.model.ops import DistMaps, ScaleLayer, BatchImageNormalize, DistMapsDifferentable, DetrministicBilinear2D
from isegm.model.modifiers import LRMult
from segmentation_models_pytorch.losses import DiceLoss
from isegm.inference import utils
from isegm.inference.clicker import Click
from copy import deepcopy


class ISModel(nn.Module):
    def __init__(self, use_rgb_conv=True, with_aux_output=False,
                 norm_radius=260, use_disks=False, cpu_dist_maps=False,
                 clicks_groups=None, with_prev_mask=False, use_leaky_relu=False,
                 binary_prev_mask=False, conv_extend=False, norm_layer=nn.BatchNorm2d,
                 norm_mean_std=([.485, .456, .406], [.229, .224, .225])):
        super().__init__()
        self.with_aux_output = with_aux_output
        self.clicks_groups = clicks_groups

        # Since we need prev mask for interaction location loss for all models
        self.with_prev_mask = True
        
        # Depends on number of channels
        self.with_prev_mask_model_type = with_prev_mask

        self.binary_prev_mask = binary_prev_mask
        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])

        self.coord_feature_ch = 2
        if clicks_groups is not None:
            self.coord_feature_ch *= len(clicks_groups)

        if self.with_prev_mask_model_type:
            self.coord_feature_ch += 1

        if use_rgb_conv:
            rgb_conv_layers = [
                nn.Conv2d(in_channels=3 + self.coord_feature_ch, out_channels=6 + self.coord_feature_ch, kernel_size=1),
                norm_layer(6 + self.coord_feature_ch),
                nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=6 + self.coord_feature_ch, out_channels=3, kernel_size=1)
            ]
            self.rgb_conv = nn.Sequential(*rgb_conv_layers)
        elif conv_extend:
            self.rgb_conv = None
            self.maps_transform = nn.Conv2d(in_channels=self.coord_feature_ch, out_channels=64,
                                            kernel_size=3, stride=2, padding=1)
            self.maps_transform.apply(LRMult(0.1))
        else:
            self.rgb_conv = None
            mt_layers = [
                nn.Conv2d(in_channels=self.coord_feature_ch, out_channels=16, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
                ScaleLayer(init_value=0.05, lr_mult=1)
            ]
            self.maps_transform = nn.Sequential(*mt_layers)

        if self.clicks_groups is not None:
            self.dist_maps = nn.ModuleList()
            for click_radius in self.clicks_groups:
                self.dist_maps.append(DistMaps(norm_radius=click_radius, spatial_scale=1.0,
                                               cpu_mode=cpu_dist_maps, use_disks=use_disks))
        else:
            self.dist_maps = DistMapsDifferentable(norm_radius=norm_radius, spatial_scale=1.0, cpu_mode=cpu_dist_maps, use_disks=use_disks)


    def forward_optimizable(self, optimize, image_in, gt_mask, points_in, args=None, transforms=None, gt_mask_without_transforms=None):

        for MODE in ["MIN" if args.optim_min else "MAX"]:

            print("New click processing")
            per_step_metrics = []
            points = torch.clone(points_in)
            image = torch.clone(image_in)
            image, prev_mask = self.prepare_input(image)
            self.dist_maps.register(image, points)
            instance_loss = DiceLoss('binary', from_logits=False)

            lr = args.lr_mult * 5 * np.sqrt(image.shape[-1] ** 2 + image.shape[-2] ** 2) / (400 * np.sqrt(2))
            optimizer = torch.optim.Adam([self.dist_maps.last_click], lr = lr)

            best_iou = [-2, -2] if MODE == 'MAX' else [2, 2]
            best_mask_loss = 1e50
            best_params = None
            best_outputs = {}

            if not optimize:
                self.dist_maps.last_click.requires_grad = False

            if self.dist_maps.last_click.requires_grad:
                iters = args.n_opt_steps
            else:
                iters = 1

            gt_8_bit = gt_mask[0][0].cpu().numpy().astype(np.uint8)

            if self.with_prev_mask:
                prev_mask_8_bit = ((0.5 * (prev_mask[0] + torch.flip(prev_mask[1], dims=[2])))[0].cpu().numpy() > 0.5).astype(np.uint8)
            else:
                prev_mask_8_bit = 1 - gt_8_bit

            gt_mask_dt_for_positive = cv2.distanceTransform(1 - (gt_8_bit - (gt_8_bit & prev_mask_8_bit)), cv2.DIST_L2, 0)
            gt_mask_dt_for_negative =  cv2.distanceTransform(gt_8_bit | (1 - prev_mask_8_bit), cv2.DIST_L2, 0)

            gt_mask_dt_for_positive = torch.from_numpy(gt_mask_dt_for_positive).to(image.device)
            gt_mask_dt_for_negative =  torch.from_numpy(gt_mask_dt_for_negative).to(image.device)

            for i in range(iters):

                optimizer.zero_grad()
                coord_features, coords_new = self.dist_maps()
                coord_features = coord_features.cuda()
                coords_new = coords_new.cuda()

                # Two distance transforms for interaction location loss
                positive_selected = (coords_new[0][0] * gt_mask_dt_for_positive)
                negative_selected = (coords_new[0][1] * gt_mask_dt_for_negative)
                regularization_loss = positive_selected.mean() + negative_selected.mean()
                curr_mask_loss = regularization_loss.item()

                if args.vis_optim:
                    new_stack =  coord_features[0].permute(1, 2, 0).detach().cpu().numpy() * 255
                    img_numpy = np.clip((image[0].cpu().permute(1, 2, 0).cpu().numpy() + 2) / 4, 0, 1) * 255
                    positive = np.dstack([new_stack[:, :, 0], new_stack[:, :, 0], new_stack[:, :, 0]])
                    negative = np.dstack([new_stack[:, :, 1], new_stack[:, :, 1], new_stack[:, :, 1]])
                    mask_stack = np.dstack([gt_mask[0][0].cpu(), gt_mask[0][0].cpu(), gt_mask[0][0].cpu()])
                    img_to_save = np.where(positive, (0, 255, 0), img_numpy).astype(np.uint8)
                    img_to_save = np.where(negative, (255, 0, 0), img_to_save).astype(np.uint8)

                if image.shape[0] == 2:
                    coord_features = torch.cat([coord_features, torch.flip(coord_features, (3,))])

                if self.with_prev_mask_model_type:
                    coord_features = torch.cat((prev_mask, coord_features), dim=1)
                
                coord_features = self.maps_transform(coord_features)
                outputs = self.backbone_forward(image, coord_features)
                
                if torch.are_deterministic_algorithms_enabled():
                    outputs['instances'] = DetrministicBilinear2D(outputs['instances'], size=image.size()[2:], align_corners=True)
                    if self.with_aux_output:
                        outputs['instances_aux'] = DetrministicBilinear2D(outputs['instances_aux'], size=image.size()[2:], align_corners=True)
                else:
                    outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:], mode='bilinear', align_corners=True)
                    if self.with_aux_output:
                        outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:], mode='bilinear', align_corners=True).detach()
                
                prediction = outputs['instances']
                prediction = torch.sigmoid(prediction)
                
                if args.vis_optim:
                    prediction_stack = np.dstack([prediction[0][0].cpu().detach(), prediction[0][0].cpu().detach(), prediction[0][0].cpu().detach()])
                    cv2.imwrite('clicks.png', np.hstack([cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR), 255 * mask_stack, 255 * prediction_stack]))

                if self.dist_maps.last_click.requires_grad:
                    main_loss = torch.mean(instance_loss(prediction, gt_mask.to(prediction.device))) * (1 if MODE == 'MAX' else - 1)
                    loss = regularization_loss + (main_loss / 1000.)

                # Compute IOU in full resolution
                prediction = nn.functional.interpolate(outputs['instances'].detach(), mode='bilinear', 
                                                    align_corners=True, size=image_in.size()[2:])
                
                curr_params = self.dist_maps.last_click.detach().cpu().numpy()[::-1].copy()
                curr_params[0] = np.clip(curr_params[0], 0, image.shape[2] - 1)
                curr_params[1] = np.clip(curr_params[1], 0, image.shape[3] - 1)

                dummy_click = Click(is_positive=True, coords=(curr_params[0], curr_params[1]))
                for t in reversed(transforms):
                    prediction, dummy_click = t.inv_transform(prediction, dummy_click, side_effects=False)

                prediction_cpu = prediction.cpu().detach().numpy()[0, 0] > args.thresh
                curr_iou = utils.get_iou(gt_mask_without_transforms, prediction_cpu)
                curr_biou = utils.get_boundary_iou(gt_mask_without_transforms, prediction_cpu)

                curr_metrics = [curr_iou, curr_biou]

                logging_record = [*curr_metrics, *dummy_click.coords, *curr_params, *list(image_in.shape), *list(image.shape)]

                if (curr_mask_loss <= best_mask_loss * 1.05) and ((curr_metrics > best_iou and MODE == 'MAX') or (curr_metrics < best_iou and MODE != 'MAX')):
                    best_mask_loss = curr_mask_loss
                    best_iou = deepcopy(curr_metrics)
                    best_params = deepcopy(curr_params)

                    print(MODE, "IoU/BIoU update: ", best_iou)

                    best_outputs = {}
                    for key in outputs.keys():
                        if outputs[key] is not None:
                            best_outputs[key] = outputs[key].detach()

                    # Flag of successful updates
                    logging_record.append(1)
                else:
                    logging_record.append(0)

                if self.dist_maps.last_click.requires_grad: #and (i != iters - 1):
                    loss.backward()
                    optimizer.step()

                per_step_metrics.append(logging_record)

        return best_outputs, best_params, np.array(per_step_metrics)


    def prepare_input(self, image):
        prev_mask = None
        if self.with_prev_mask:
            prev_mask = image[:, 3:, :, :]
            image = image[:, :3, :, :]
            if self.binary_prev_mask:
                prev_mask = (prev_mask > 0.5).float()

        image = self.normalization(image)
        return image, prev_mask

    def backbone_forward(self, image, coord_features=None):
        raise NotImplementedError

    def get_coord_features(self, image, prev_mask, points):
        if self.clicks_groups is not None:
            points_groups = split_points_by_order(points, groups=(2,) + (1, ) * (len(self.clicks_groups) - 2) + (-1,))
            coord_features = [dist_map(image, pg) for dist_map, pg in zip(self.dist_maps, points_groups)]
            coord_features = torch.cat(coord_features, dim=1)
        else:
            coord_features = self.dist_maps(image, points)

        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)

        return coord_features



def split_points_by_order(tpoints: torch.Tensor, groups):
    points = tpoints.cpu().numpy()
    num_groups = len(groups)
    bs = points.shape[0]
    num_points = points.shape[1] // 2

    groups = [x if x > 0 else num_points for x in groups]
    group_points = [np.full((bs, 2 * x, 3), -1, dtype=np.float32)
                    for x in groups]

    last_point_indx_group = np.zeros((bs, num_groups, 2), dtype=np.int)
    for group_indx, group_size in enumerate(groups):
        last_point_indx_group[:, group_indx, 1] = group_size

    for bindx in range(bs):
        for pindx in range(2 * num_points):
            point = points[bindx, pindx, :]
            group_id = int(point[2])
            if group_id < 0:
                continue

            is_negative = int(pindx >= num_points)
            if group_id >= num_groups or (group_id == 0 and is_negative):  # disable negative first click
                group_id = num_groups - 1

            new_point_indx = last_point_indx_group[bindx, group_id, is_negative]
            last_point_indx_group[bindx, group_id, is_negative] += 1

            group_points[group_id][bindx, new_point_indx, :] = point

    group_points = [torch.tensor(x, dtype=tpoints.dtype, device=tpoints.device)
                    for x in group_points]

    return group_points
