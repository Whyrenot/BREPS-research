from .interactionset import InteractionDataset, get_labels_with_sizes
from pathlib import Path
from isegm.data.sample import DSampleWithPrompt
import cv2
import numpy as np

class UserStudyNewDataset(InteractionDataset):
    def __init__(self, dataset_path, args,
        images_dir_name='images/', masks_dir_name="masks/",
                 **kwargs):
        super(InteractionDataset, self).__init__(**kwargs)
        self.square_image_size = 1024
        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._masks_path = self.dataset_path /masks_dir_name
        self.dataset_samples = [(x, self._masks_path / x.name) for x in self._images_path.glob("*.*") ]
        self.dataset_samples = [x for x in self.dataset_samples if ("ACDC" not in str(x[0])) and  ("BUID" not in str(x[0]))]
        # if (("BUID" not in str(x.)) or ("ACDC" not in str(x.name)))

    def get_sample(self, index) -> DSampleWithPrompt:

        sample = self.dataset_samples[index]
        image_path, mask_path = sample
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)

        instances_mask = instances_mask.astype(np.int32)
        instances_mask = instances_mask[:, :, 0] * 65536 + instances_mask[:, :, 1] * 256 + instances_mask[:, :, 2]
        instances_mask = self.aligned_resize(instances_mask, self.square_image_size, self.square_image_size)
        object_ids, _ = get_labels_with_sizes(instances_mask)
        image = cv2.resize(image, (self.square_image_size, self.square_image_size))
        return DSampleWithPrompt(image, instances_mask, prompt_positive=None,prompt_negative=None, objects_ids=object_ids,  sample_id=index, imname=image_path, maskname=mask_path)
