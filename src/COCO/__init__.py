"""
Модуль для работы с датасетом COCO Cars и обучения моделей сегментации.

Содержит:
- ParkingDataset: PyTorch Dataset для загрузки изображений и масок
- train.py: скрипт обучения модели Linknet
- Утилиты для конвертации и обработки датасета
"""

from .ParkingDataset import ParkingDataset

__all__ = ["ParkingDataset"]
