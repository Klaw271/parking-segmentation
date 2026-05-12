import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp

from .PatchEngine import PatchEngine
from src.parking_monitoring.DataValidator import DataValidator

class CarDetector:
    """
    Класс для инференса модели сегментации.
    """

    def __init__(self, model_path: str, device: str, patch_size: int, overlap: int, threshold: float):
        self.device = device
        self.patch_size = patch_size
        self.threshold = threshold
        self.validator = DataValidator()

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

    def detect_patches(self, image_path: str, content_type: str = None, target_height: int = 704) -> np.ndarray:
        try:
            with open(image_path, 'rb') as f:
                content = f.read()
            
            image_raw = self.validator.validate_image_source(content, content_type)
            image_rgb = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

            original_size = image_rgb.shape[:2]
            scale = target_height / original_size[0]
            new_size = (int(image_rgb.shape[1] * scale), target_height)
            image_rgb = cv2.resize(image_rgb, new_size, interpolation=cv2.INTER_AREA)
            
            patches, coords, (h, w) = self.patch_engine.extract(image_rgb)
            
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

            final = cv2.resize(final, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)

            return (final > self.threshold).astype(np.uint8)

        except FileNotFoundError as e:
            raise ValueError(f"File not found: {e}")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Car detection failed: {e}")

    def detect_full_image(self, image_path: str, content_type: str = None, target_height: int = 512) -> np.ndarray:
        """Инференс на полном изображении без нарезки на патчи."""
        try:
            with open(image_path, 'rb') as f:
                content = f.read()

            image_raw = self.validator.validate_image_source(content, content_type)
            image_rgb = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

            original_size = image_rgb.shape[:2]
            scale = target_height / original_size[0]
            new_size = (int(image_rgb.shape[1] * scale), target_height)
            image_rgb = cv2.resize(image_rgb, new_size, interpolation=cv2.INTER_AREA)
            

            h, w = image_rgb.shape[:2]
            pad_h = (32 - h % 32) % 32
            pad_w = (32 - w % 32) % 32
            padded = image_rgb
            if pad_h or pad_w:
                padded = cv2.copyMakeBorder(image_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            with torch.no_grad():
                pred = torch.sigmoid(self.model(self._preprocess(padded)))
                pred = pred.squeeze().cpu().numpy()

            pred = pred[:h, :w]

            

            pred = cv2.resize(pred, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)

            return (pred > self.threshold).astype(np.uint8)

        except FileNotFoundError as e:
            raise ValueError(f"File not found: {e}")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Full-image car detection failed: {e}")

    def visualize_detection(self, image_path: str, mask: np.ndarray) -> bytes:
        """
        Накладывает маску на изображение и возвращает байты PNG файла.
        """
        # Загружаем оригинальное изображение
        with open(image_path, 'rb') as f:
            img_bgr = self.validator.validate_image_source(f.read())
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Создаем цветовой слой (Cyan)
        colored_mask = np.zeros_like(img_rgb)
        colored_mask[mask == 1] = [0, 255, 255]

        # Смешиваем (blending)
        blended = cv2.addWeighted(img_rgb, 0.7, colored_mask, 0.3, 0)

        # Конвертация в PNG байты
        result_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".png", result_bgr)
        
        return buffer.tobytes()

