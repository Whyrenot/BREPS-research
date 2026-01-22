from loguru import logger
import torch
import torch.nn.functional as F
from torchvision import transforms
from isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, LimitLongestSide
from isegm.inference.bbox_handler import Bbox, BboxHandler
from copy import deepcopy

class BaseBboxPredictor(object):
    def __init__(self, model, device,
                 net_clicks_limit=None,
                 with_flip=False,
                 zoom_in=None,
                 max_size=None,
                 model_name='ritm',
                 **kwargs):
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None
        logger.debug(max_size)
        if isinstance(model, tuple):
            self.net, self.click_models = model
        else:
            self.net = model

        self.to_tensor = transforms.ToTensor()

        self.transforms = [zoom_in] if zoom_in is not None else []
        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))
        if model_name == 'ritm':
            self.transforms.append(SigmoidForPred())
        if with_flip:
            self.transforms.append(AddHorizontalFlip())

    def set_input_image(self, image):
        image_nd = self.to_tensor(image)
        for transform in self.transforms:
            transform.reset()
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])

    def get_single_prediction(self, bbox_handler:BboxHandler, args=None, optimize=True):
        bbox = bbox_handler.get_boxes()[0]


        input_image = self.original_image
        prev_mask = self.prev_prediction
        if hasattr(self.net, 'with_prev_mask') and self.net.with_prev_mask:
            input_image = torch.cat((input_image, prev_mask), dim=1)
        image_nd, gt_mask, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, torch.from_numpy(bbox_handler.gt_mask[None, None, ...]).float(), [bbox]
        )

        pred_logits, last_bbox, metrics_dict = self._get_prediction(optimize, image_nd, clicks_lists, is_image_changed, gt_mask, args, self.transforms, bbox_handler.gt_mask)
        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])
        bbox_handler._remove_last_bbox()
        last_bbox = Bbox(last_bbox)

        for t in reversed(self.transforms):
            prediction, last_click = t.inv_transform(prediction, last_bbox)

        bbox_handler.add_bbox(last_bbox)

        if self.zoom_in is not None and self.zoom_in.check_possible_recalculation():
           # If we deep into recalculate we should skip optimization
           return self.get_prediction(bbox_handler, prev_mask, args, optimize=False)

        self.prev_prediction = prediction

        return prediction.cpu().numpy()[0, 0], metrics_dict

    def get_prediction(self, bbox_handler:BboxHandler, prev_mask=None, args=None, optimize=True):
        bboxes_list = bbox_handler.get_boxes()

        if self.click_models is not None:
            model_indx = min(bbox_handler.bbox_indx_offset + len(bboxes_list), len(self.click_models)) - 1
            if model_indx != self.model_indx:
                self.model_indx = model_indx
                self.net = self.click_models[model_indx]

        input_image = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction
        if hasattr(self.net, 'with_prev_mask') and self.net.with_prev_mask:
            input_image = torch.cat((input_image, prev_mask), dim=1)
        image_nd, gt_mask, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, torch.from_numpy(bbox_handler.gt_mask[None, None, ...]).float(), [bboxes_list]
        )

        pred_logits, last_bbox, metrics_dict = self._get_prediction(optimize, image_nd, clicks_lists, is_image_changed, gt_mask, args, self.transforms, bbox_handler.gt_mask)
        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])
        bbox_handler._remove_last_bbox()
        last_bbox = Bbox(last_bbox)

        for t in reversed(self.transforms):
            prediction, last_click = t.inv_transform(prediction, last_bbox)

        bbox_handler.add_bbox(last_bbox)

        if self.zoom_in is not None and self.zoom_in.check_possible_recalculation():
           # If we deep into recalculate we should skip optimization
           return self.get_prediction(bbox_handler, prev_mask, args, optimize=False)

        self.prev_prediction = prediction

        return prediction.cpu().numpy()[0, 0], metrics_dict

    def _get_prediction(self, optimize, image_nd, bbox_list, is_image_changed, gt_mask=None, args=None, transforms=None, gt_mask_without_transforms=None):
        model_outputs = self.net.forward_optimizable_bbox(optimize, image_nd, gt_mask, torch.Tensor(bbox_list[0]), args, transforms, gt_mask_without_transforms)
        return model_outputs[0]['instances'], model_outputs[1], model_outputs[2]

    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)

    def apply_transforms(self, image_nd, gt_mask, clicks_lists):
        is_image_changed = False
        for t in self.transforms:
            prev_clicks = deepcopy(clicks_lists)
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            gt_mask, _ = t.transform(gt_mask, prev_clicks)
            is_image_changed |= t.image_changed
        gt_mask = (gt_mask > 0.5).float()
        return image_nd, gt_mask, clicks_lists, is_image_changed


    def get_states(self):
        return {
            'transform_states': self._get_transform_states(),
            'prev_prediction': self.prev_prediction.clone()
        }

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.prev_prediction = states['prev_prediction']
