from isegm.model.ops import DistMapsSAM
from .is_sam_model import ISModelSAM
from mobile_sam import sam_model_registry,  SamPredictor
from mobile_sam.utils.transforms import ResizeLongestSide



class ISModelMobileSAM(ISModelSAM):
    def __init__(self, device='cuda', model_path=None):
        super().__init__(device=device, model_path=model_path)
        self.dist_maps = DistMapsSAM(norm_radius=5, spatial_scale=1.0, cpu_mode=False, use_disks=True)
        self.with_prev_mask = True
        self.binary_prev_mask = False
        self.prev_mask = None

    def _load_model(self, device, model_path):
        model_type = 'vit_t'# 'vit_b' if 'vit_b' in str(model_path) else 'vit_h' if 'vit_h' in str(model_path) else 'vit_l'
        mobile_sam = sam_model_registry[model_type](checkpoint=model_path)
        mobile_sam.to(device=device)
        mobile_sam.eval()

        for n, p in mobile_sam.named_parameters():
            p.requires_grad = False
        mobile_sam.to(device=device)
        mobile_sam.eval()
        self.sam_predictor = SamPredictor(mobile_sam)
        self.resize = ResizeLongestSide(mobile_sam.image_encoder.img_size)
        self.device = device
