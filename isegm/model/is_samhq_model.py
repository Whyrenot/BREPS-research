from isegm.model.ops import DistMapsSAM
from segment_anything_hq import SamPredictor, sam_model_registry
from segment_anything_hq.utils.transforms import ResizeLongestSide
from .is_sam_model import ISModelSAM


class ISModelSAMHQ(ISModelSAM):
    def __init__(self, device='cuda', model_path=None):

        super().__init__(device=device, model_path=model_path)
        self.dist_maps = DistMapsSAM(norm_radius=5, spatial_scale=1.0, cpu_mode=False, use_disks=True)
        self.prev_mask = None
        self.with_prev_mask = True
        self.binary_prev_mask = False

    def _load_model(self, device, model_path):
        model_type = 'vit_b' if 'vit_b' in str(model_path) else 'vit_h' if 'vit_h' in str(model_path) else 'vit_l'
        samhq = sam_model_registry[model_type](checkpoint=model_path)
        for n, p in samhq.named_parameters():
            p.requires_grad = False
        samhq.eval()
        samhq.to(device=device)
        self.device = device
        self.sam_predictor = SamPredictor(samhq)
        self.resize = ResizeLongestSide(samhq.image_encoder.img_size)
