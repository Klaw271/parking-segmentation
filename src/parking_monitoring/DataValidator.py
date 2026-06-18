import cv2
import numpy as np
from typing import Any


class DataValidator:
    """
    Валидатор входных данных (изображений и аннотаций).

    Проверяет формат, размер, целостность изображений и соответствие
    координат аннотаций размерам изображения.

    Attributes:
        max_file_size (int): максимальный размер файла в байтах
        allowed_image_types (list): разрешённые MIME-типы изображений
    """

    def __init__(self, max_file_size: int = 10 * 1024 * 1024) -> None:
        """
        Инициализирует валидатор с ограничением размера файла.

        :param max_file_size: максимальный размер файла в байтах (по умолчанию 10 МБ)
        """
        self.max_file_size = max_file_size
        self.allowed_image_types = ["image/jpeg", "image/png", "image/jpg"]

    def validate_image_source(self, content: bytes, content_type: str = None) -> np.ndarray:
        """
        Выполняет полную проверку изображения: формат, размер, целостность.

        :param content: содержимое файла в виде байт-строки
        :param content_type: MIME-тип файла для проверки формата (опционально)
        :return: загруженное изображение формата BGR (H, W, 3) типа uint8
        :raises ValueError: если формат неверный, файл слишком большой, пуст или повреждён
        """
        if content_type and content_type not in self.allowed_image_types:
            raise ValueError("Invalid image format")

        file_size = len(content)
        if file_size > self.max_file_size:
            raise ValueError(f"File too large. Max: {self.max_file_size // (1024*1024)}MB")
        if file_size == 0:
            raise ValueError("Image file is empty")

        try:
            nparr = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Invalid image: could not decode")
            return img
        except Exception:
            raise ValueError("Invalid image: file is corrupted")

    def verify_consistency(self, img: np.ndarray, annotation: Any) -> None:
        """
        Проверяет соответствие фактического разрешения изображения данным в аннотации.

        Сравнивает размеры изображения с данными в JSON аннотации и проверяет,
        что все координаты точек находятся в границах изображения.

        :param img: загруженное изображение формата BGR
        :param annotation: объект аннотации с полями size.height, size.width, objects[].points.exterior
        :raises ValueError: если размеры не совпадают или координаты выходят за границы
        """
        h, w = img.shape[:2]
        if h != annotation.size.height or w != annotation.size.width:
            raise ValueError(f"Size mismatch: Image {w}x{h}, JSON claims {annotation.size.width}x{annotation.size.height}")

        for obj in annotation.objects:
            points = np.array(obj.points.exterior)
            if np.any(points[:, 0] >= w) or np.any(points[:, 1] >= h):
                raise ValueError(f"Annotation points are outside image boundaries (image: {w}x{h})")