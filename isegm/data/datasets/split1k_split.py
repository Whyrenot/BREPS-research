
from pathlib import Path

import cv2
from loguru import logger
import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSampleWithPrompt

def get_labels_with_sizes(x):
    obj_sizes = np.bincount(x.flatten())
    labels = np.nonzero(obj_sizes)[0].tolist()
    labels = [x for x in labels if x != 0]
    return labels, obj_sizes[labels].tolist()

class Split1kDataset(ISDataset):
    def __init__(self, dataset_path, args,
        images_dir_name='images', masks_dir_name='masks',
                 **kwargs):
        super(Split1kDataset, self).__init__(**kwargs)
        self.square_image_size = 1024
        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._masks_path = self.dataset_path / masks_dir_name

        # Get all image files
        self.dataset_samples = []
        for img_path in sorted(self._images_path.glob('*.jpg')):
            mask_path = (self._masks_path / img_path.stem).with_suffix('.png')
            if mask_path.exists():
                self.dataset_samples.append(img_path.stem)
            else:
                logger.warning(f"Mask {mask_path} not found for image {img_path}")

        if args.n_samples:
            self.dataset_samples = self.dataset_samples[:args.n_samples]

    def aligned_resize(self, mask, new_height, new_width):
        mask_resized = np.zeros((new_height, new_width), dtype=mask.dtype)
        uniques = np.nonzero(np.bincount(mask.ravel()))[0]
        for val in uniques:
            if val == 0:
                continue
            resized_mask = cv2.resize((mask == val).astype(np.uint8), (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            mask_resized = np.where(resized_mask, val, mask_resized)
        return mask_resized.astype(mask.dtype)

    def get_sample(self, index) -> DSampleWithPrompt:
        sample_name = self.dataset_samples[index]
        image_path = str((self._images_path / sample_name).with_suffix(".jpg"))
        mask_path = str((self._masks_path / sample_name).with_suffix(".png"))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)

        instances_mask = instances_mask.astype(np.int32)
        instances_mask = instances_mask[:, :, 0] * 65536 + instances_mask[:, :, 1] * 256 + instances_mask[:, :, 2]
        instances_mask = self.aligned_resize(instances_mask, self.square_image_size, self.square_image_size)
        object_ids, _ = get_labels_with_sizes(instances_mask)
        image = cv2.resize(image, (self.square_image_size, self.square_image_size))

        return DSampleWithPrompt(image, instances_mask, prompt_positive=None, prompt_negative=None, objects_ids=object_ids, sample_id=index, imname=image_path)
