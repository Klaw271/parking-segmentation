"""
Пакет для анализа занятости парковочных мест с использованием семантической сегментации.

Модули:
- parking_monitoring: основные компоненты системы (детектор, анализаторы, pipeline)
- api: REST API endpoints на основе FastAPI
- COCO: работа с датасетом COCO Cars и обучение моделей
- self_made_dataset: утилиты для работы с собственным датасетом

Основная точка входа - ParkingPipeline класс из parking_monitoring модуля.
"""

from .parking_monitoring import ParkingPipeline
from .api import app

__all__ = [
    "ParkingPipeline",
    "app",
]
