import os
import sys
import logging
import contextlib
import numpy as np
import cv2

os.environ["ULTRALYTICS_VERBOSE"] = "False"
os.environ["YOLO_VERBOSE"] = "False"
os.environ["TQDM_DISABLE"] = "1"

logging.getLogger().setLevel(logging.CRITICAL)

from ultralytics.utils import LOGGER

LOGGER.setLevel(logging.CRITICAL)
LOGGER.disabled = True
LOGGER.info = lambda *a, **k: None
LOGGER.warning = lambda *a, **k: None
LOGGER.debug = lambda *a, **k: None

try:
    from ultralytics.engine import predictor
    predictor.print = lambda *a, **k: None
except:
    pass

try:
    from ultralytics.engine.results import Results
    Results.__str__ = lambda self: ""
    Results.__repr__ = lambda self: ""
except:
    pass

@contextlib.contextmanager
def suppress_ultralytics():
    devnull = open(os.devnull, 'w')
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = devnull, devnull
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        devnull.close()

# ================== WRAPPER ==================
class UltralyticsSAMWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.image = None
        self.orig_shape = None

    def set_image(self, image):
        self.orig_shape = image.shape[:2]
        self.image = image[:, :, ::-1].copy()  # BGR -> RGB

    def predict(self, box=None, multimask_output=False):
        h, w = self.orig_shape

        kwargs = {
            "imgsz": max(h, w),
            "device": self.device,
            "verbose": False
        }

        if box is not None:
            box = np.array(box)

            if box.ndim == 1:
                kwargs["bboxes"] = box.tolist()
            else:
                kwargs["bboxes"] = box.tolist()

        with suppress_ultralytics():
            results = self.model(self.image, stream=False, **kwargs)

        masks = results[0].masks

        if masks is None:
            return [np.zeros((h, w), dtype=bool)], [0.0], None

        masks = masks.data.cpu().numpy()

        resized_masks = []
        for m in masks:
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h)) > 0
            resized_masks.append(m)

        return resized_masks, None, None



# ================== FACTORY ==================
def get_predictors(models, device):
    predictors = {}

    for name in models:

        if name == "sam" or name == "sam1":
            from segment_anything import sam_model_registry, SamPredictor

            sam = sam_model_registry["vit_h"](
                checkpoint="./checkpoints/sam_vit_h_4b8939.pth"
            )
            sam.to(device=device).eval()
            predictors[name] = SamPredictor(sam)

        elif name == "sam2":
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam2_model = build_sam2(
                "configs/sam2.1/sam2.1_hiera_l.yaml",
                "./checkpoints/sam2.1_hiera_large.pt",
                device=device
            ).to(device)

            predictors[name] = SAM2ImagePredictor(sam2_model)

        elif name == "mobilesam":
            from mobile_sam import sam_model_registry, SamPredictor

            sam = sam_model_registry["vit_t"](
                checkpoint="./checkpoints/mobile_sam.pt"
            )
            sam.to(device=device).eval()
            predictors[name] = SamPredictor(sam)

        elif name == "fastsam":
            from ultralytics import FastSAM
            model = FastSAM("./checkpoints/FastSAM-x.pt").to(device)
            predictors[name] = UltralyticsSAMWrapper(model, device)

        elif name == "sam3":
            from ultralytics import SAM
            model = SAM("./checkpoints/sam3.pt").to(device)
            predictors[name] = UltralyticsSAMWrapper(model, device)

    return predictors