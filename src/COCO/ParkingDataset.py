import os
from typing import Optional, Union, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ParkingDataset(Dataset):
    """
    PyTorch Dataset для загрузки изображений и масок припаркованных автомобилей.

    Загружает пары (изображение, маска) из файлов и применяет трансформации.
    Изображения конвертируются в RGB и масштабируются до 256x256, маски бинаризуются.
    Поддерживает как аугментации так и нормализацию.

    Attributes:
        files (list): список имён файлов для загрузки
        image_dir (str): путь к директории с изображениями
        mask_dir (str): путь к директории с масками
        transform: объект трансформации (например, albumentations.Compose)
    """

    def __init__(
        self,
        filelist: Union[str, List[str]],
        image_dir: str = "data/coco_cars/images",
        mask_dir: str = "data/coco_cars/masks",
        transform: Optional[any] = None,
    ) -> None:
        """
        Инициализирует датасет с файлами изображений и масок.

        :param filelist: путь к текстовому файлу со списком имён файлов или список строк
        :param image_dir: путь к директории с изображениями (по умолчанию data/coco_cars/images)
        :param mask_dir: путь к директории с масками (по умолчанию data/coco_cars/masks)
        :param transform: объект трансформации Albumentations для аугментации (опционально)
        :raises FileNotFoundError: если указанный файл списка не найден
        """
        if isinstance(filelist, str):
            with open(filelist, "r", encoding="utf-8-sig") as f:
                self.files = [line.strip() for line in f if line.strip()]
        else:
            self.files = list(filelist)

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self) -> int:
        """
        Возвращает количество элементов в датасете.

        :return: длина списка файлов
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Загружает одну пару (изображение, маска) по индексу и применяет трансформации.

        Процесс:
        1. Загружает изображение и конвертирует из BGR в RGB
        2. Масштабирует до 256x256
        3. Загружает соответствующую маску
        4. Бинаризует маску
        5. Применяет трансформации (если заданы)
        6. Нормализует пиксели изображения на [0, 1]
        7. Переводит в тензоры PyTorch

        :param idx: индекс элемента в датасете
        :return: кортеж (изображение, маска) где оба - тензоры формата torch.float32
                 изображение: (3, 256, 256), маска: (1, 256, 256)
        :raises FileNotFoundError: если изображение или маска не найдены
        """
        fname = self.files[idx]

        image_path = os.path.join(self.image_dir, fname)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))

        mask_name = os.path.splitext(fname)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        mask = cv2.resize(mask, (256, 256))
        mask = (mask > 0).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = image.astype(np.float32) / 255.0

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask