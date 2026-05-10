import json
import cv2
import numpy as np
from pydantic import ValidationError
from typing import Dict, Any, Tuple

class DataValidator:
    def __init__(self, max_file_size: int = 10 * 1024 * 1024):
        self.max_file_size = max_file_size
        self.allowed_image_types = ["image/jpeg", "image/png", "image/jpg"]

    def validate_image_source(self, content: bytes, content_type: str = None) -> np.ndarray:
        """
        Полная проверка изображения: формат, размер, целостность.
        """
        # 1. Проверка формата (если передан тип контента)
        if content_type and content_type not in self.allowed_image_types:
            raise ValueError("Invalid image format")

        # 2. Проверка размера
        file_size = len(content)
        if file_size > self.max_file_size:
            raise ValueError(f"File too large. Max: {self.max_file_size // (1024*1024)}MB")
        if file_size == 0:
            raise ValueError("Image file is empty")

        # 3. Декодирование и проверка целостности
        try:
            nparr = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Invalid image: could not decode")
            return img
        except Exception:
            raise ValueError("Invalid image: file is corrupted")

    def verify_consistency(self, img: np.ndarray, annotation: Any):
        """
        Проверка соответствия фактического разрешения картинки данным в JSON.
        """
        h, w = img.shape[:2]
        # 1. Проверка по полям size в JSON
        if h != annotation.size.height or w != annotation.size.width:
            raise ValueError(f"Size mismatch: Image {w}x{h}, JSON claims {annotation.size.width}x{annotation.size.height}")

        # 2. Проверка реальных координат точек (на всякий случай)
        for obj in annotation.objects:
            points = np.array(obj.points.exterior) # или как у вас в модели
            if np.any(points[:, 0] >= w) or np.any(points[:, 1] >= h):
                raise ValueError(f"Annotation points are outside image boundaries (image: {w}x{h})")