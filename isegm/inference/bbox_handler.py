import numpy as np
from copy import deepcopy
from isegm.inference.utils import mask_to_polygon
from loguru import logger
class BboxHandler(object):
    def __init__(self, gt_mask:np.ndarray|None=None, init_bbox=None, prompt_mask:None|np.ndarray=None, ignore_label=-1, bbox_indx_offset=0):
        self.bbox_indx_offset = bbox_indx_offset
        self._gt_mask:np.ndarray|None
        self._prompt_mask:np.ndarray|None = prompt_mask
        if gt_mask is not None:
            self._gt_mask = gt_mask == 1

            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self._gt_mask = None

        self.reset_boxes()

        if init_bbox is not None:
            self.add_bbox(init_bbox)
        self._norm_params = np.array(self.gt_mask.shape, dtype= np.float64)


    @property
    def gt_mask(self) -> np.ndarray:
        assert self._gt_mask is not None
        return self._gt_mask

    @property
    def prompt_mask(self) -> np.ndarray:
        assert self._prompt_mask is not None
        return self._prompt_mask

    def make_first_bbox(self, from_prompt=False):
        mask = self.gt_mask if not from_prompt else self.prompt_mask
        try:
            bbox = mask_to_polygon(mask)
            y,ym, x, xm = bbox
            bbox = np.array([x, y, xm, ym])
        except ValueError as e:
            logger.error(e)
            logger.warning("cant parse")
            bbox = np.array([0, 0, 1, 1])
        self.bbox_list.append(bbox)

    def make_bbox(self,pred_mask):
        pass

    def get_boxes(self, boxes_limit=None):
        return self.bbox_list[:boxes_limit]

    def add_bbox(self, bbox):

        bbox.indx = self.bbox_indx_offset + self.num_boxes
        self.num_boxes += 1

        self.bbox_list.append(bbox)

    def _remove_last_bbox(self):
        bbox = self.bbox_list.pop()
        self.num_boxes -= 1
        return bbox

    def reset_boxes(self):
        self.num_boxes = 0
        self.bbox_list = []

    def get_state(self):
        return deepcopy(self.bbox_list)

    def set_state(self, state):
        self.reset_boxes()
        for bbox in state:
            self.add_bbox(bbox)

    def __len__(self):
        return len(self.bbox_list)



class Bbox:
    def __init__(self, coords, indx=None):
        if isinstance(coords, Bbox):
            self.coords = coords.coords
            self.indx = coords.indx
        else:
            self.coords = coords
            self.indx = indx
        # add separate xywh - xmin ymin width height
        # with contraints: xmin in [0, image_width], ymin in [0, image_height]
        # 0< width < image_width ,0< height < image_height and xmin + width < image_width and ymin + height < image_height,
    @property
    def coords_and_indx(self):
        return (*self.coords, self.indx)

    def copy(self, **kwargs):
        self_copy = deepcopy(self)
        for k, v in kwargs.items():
            setattr(self_copy, k, v)
        return self_copy
