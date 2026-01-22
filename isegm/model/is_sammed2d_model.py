import torch
from isegm.model.ops import DistMapsSAM
from loguru import logger
from .is_sam_model import ISModelSAM
from sammed2d.build_sam import sam_model_registry
from sammed2d.utils.transforms import ResizeLongestSide
from sammed2d.predictor_sammed import SammedPredictor


class ISModelSAMMed2d(ISModelSAM):
    def __init__(self, device='cuda', model_path=None):

        super().__init__(device=device, model_path=model_path)
        self.dist_maps = DistMapsSAM(norm_radius=5, spatial_scale=1.0, cpu_mode=False, use_disks=True)
        self.prev_mask = None
        self.with_prev_mask = True
        self.binary_prev_mask = False
        self._load_model(device, model_path)
        self.regularization_distribution = torch.distributions.Gamma(concentration=torch.Tensor([1.7891955953866256]).to(self.device), rate=torch.Tensor([1/0.12104836378138935]).to(self.device))

    def _load_model(self, device, model_path):
        class Args:
            pass
        args = Args()
        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = model_path
        sam_model = sam_model_registry['vit_b'](args).to(device)
        logger.debug(f"Loaded SAM-Med2D vit_b with checkpoint {args.sam_checkpoint}")

        for _, p in sam_model.named_parameters():
            p.requires_grad = False

        sam_model.eval()
        self.device = device
        self.img_size = args.image_size
        self.encoder_adapter = args.encoder_adapter
        self.sam_predictor = SammedPredictor(sam_model)
        self.resize = ResizeLongestSide(sam_model.image_encoder.img_size)
