import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp

from .PatchEngine import PatchEngine

class CarDetector:
    """
    Класс для инференса модели сегментации.
    """

    def __init__(self, model_path: str, device: str, patch_size: int, overlap: int, threshold: float):
        self.device = device
        self.patch_size = patch_size
        self.threshold = threshold

        self.model = smp.Linknet(
            encoder_name="efficientnet-b1",
            encoder_weights=None,
            classes=1,
            activation=None
        ).to(device)

        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.model.eval()

        self.patch_engine = PatchEngine((patch_size, patch_size), overlap)

    def _preprocess(self, img):
        x = torch.from_numpy(img.astype(np.float32) / 255.0)
        return x.permute(2, 0, 1).unsqueeze(0).to(self.device)

    def detect_patches(self, image_path: str) -> np.ndarray:
        image_raw = cv2.imread(image_path)
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

        patches, coords, (h, w) = self.patch_engine.extract(image_path)

        prob_mask = np.zeros((h, w), dtype=np.float32)
        count_mask = np.zeros((h, w), dtype=np.float32)

        with torch.no_grad():
            for patch, (y, x) in zip(patches, coords):
                pred = torch.sigmoid(self.model(self._preprocess(patch)))
                pred = pred.squeeze().cpu().numpy()

                prob_mask[y:y+self.patch_size, x:x+self.patch_size] += pred
                count_mask[y:y+self.patch_size, x:x+self.patch_size] += 1

        final = np.divide(
            prob_mask,
            count_mask,
            out=np.zeros_like(prob_mask),
            where=count_mask != 0
        )

        final = final[:image_raw.shape[0], :image_raw.shape[1]]

        return (final > self.threshold).astype(np.uint8)

