import numpy as np
from pathlib import Path
from isegm.data.base import ISDataset
from isegm.data.sample import DSample

class MedScribbleDataset(ISDataset):
    def __init__(self, dataset_path, split='test', **kwargs):
        super().__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / 'images'
        self.masks_dir = self.dataset_path / 'masks'

        self.dataset_samples = []
        for f in self.images_dir.glob('*.npy'):
            name = f.stem.replace('_img', '')
            mask_path = self.masks_dir / f"{name}_mask.npy"
            if mask_path.exists():
                self.dataset_samples.append(name)

    def __len__(self):
        return len(self.dataset_samples)

    def get_sample(self, index) -> DSample:
        name = self.dataset_samples[index]
        image = np.load(self.images_dir / f"{name}_img.npy").astype(np.float32)
        mask = np.load(self.masks_dir / f"{name}_mask.npy").astype(np.int32)

        image = image.astype(np.float32) / 255.0

        objects_ids = [x for x in np.unique(mask) if x != 0]

        sample = DSample(
            image=image,
            encoded_masks=mask,
            objects_ids=objects_ids,
            sample_id=index
        )
        sample.imname = name
        sample.image_name = f"{name}_img.npy"
        sample.mask_name = f"{name}_mask.npy"
        return sample

    def __getitem__(self, index):
        return self.get_sample(index)
