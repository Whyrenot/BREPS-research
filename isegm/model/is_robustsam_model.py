from isegm.model.ops import DistMapsSAM
from .is_sam_model import ISModelSAM
from robust_segment_anything import SamPredictor, sam_model_registry
from robust_segment_anything.utils.transforms import ResizeLongestSide



def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class ISModelRobustSAM(ISModelSAM):
    def __init__(self, device='cuda:5', model_path=None):
        super().__init__(device=device, model_path=model_path)
        self.dist_maps = DistMapsSAM(norm_radius=5, spatial_scale=1.0, cpu_mode=False, use_disks=True)
        self.with_prev_mask = True
        self.binary_prev_mask = False
        self.prev_mask = None

    def _load_model(self, device, model_path):
        model_type = 'vit_b' if 'checkpoint_b' in str(model_path) else 'vit_h' if 'checkpoint_h' in str(model_path) else 'vit_l'
        sam = sam_model_registry[model_type](checkpoint=model_path, opt=None)
        for n, p in sam.named_parameters():
            p.requires_grad = False
        sam.eval()
        sam.to(device=device)
        self.device = device
        self.sam_predictor = SamPredictor(sam)
        self.resize = ResizeLongestSide(sam.image_encoder.img_size)
