from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from mobile_sam import sam_model_registry, SamPredictor


def get_predictor(models, device):

    predictors = {}

    for name in models:

        if name == "sam_h":

            sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
            model_type = "vit_h"

            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)

            predictors[name] = SamPredictor(sam)
            
        elif name == "sam2_h":

            sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

            sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

            predictors[name] = SAM2ImagePredictor(sam2_model)

        elif name == "mobile_sam":
            sam_checkpoint = "./checkpoints/mobile_sam.pt"
            model_type = "vit_t"

            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            sam.eval()

            predictors[name] = SamPredictor(sam)

    return predictors
