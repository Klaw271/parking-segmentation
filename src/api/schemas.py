from pydantic import BaseModel
from typing import List


class DetectResponse(BaseModel):
    """
    Ответ API при детекции автомобилей на изображении.

    Attributes:
        car_pixel_ratio (float): доля пикселей, определённых как часть автомобиля (0-1)
        mask_shape (list): размеры маски детекции [высота, ширина]
    """
    car_pixel_ratio: float
    mask_shape: List[int]


class Full_pipelineResponse(BaseModel):
    """
    Ответ API полного анализа занятости парковки.

    Содержит метрики качества изображения, полный подсчёт мест и результаты анализа занятости.

    Attributes:
        quality (dict): словарь с метриками качества (edge_percent, is_good_quality)
        total_spots (int): всего парковочных мест на изображении
        occupied (int): количество занятых парковочных мест
        free (int): количество свободных парковочных мест
        occupancy_percent (float): процент занятости от 0 до 100
        status (list): список булевых значений занятости каждого места (True - занято, False - свободно)
    """
    quality: dict
    total_spots: int
    occupied: int
    free: int
    occupancy_percent: float
    status: List[bool]


class PointExterior(BaseModel):
    """
    Координаты точек полигона парковочного места.

    Attributes:
        exterior (list): координаты внешних точек полигона, каждая точка - [x, y]
        interior (list): координаты внутренних точек (отверстий) полигона (опционально)
    """
    exterior: List[List[int]]
    interior: List[List[int]] = []


class ParkingObject(BaseModel):
    """
    Объект парковочного места в аннотации.

    Представляет одно парковочное место как полигон с классификацией и типом геометрии.

    Attributes:
        classTitle (str): класс объекта, обычно 'parking_slot'
        geometryType (str): тип геометрии, обычно 'polygon'
        points (PointExterior): координаты полигона
    """
    classTitle: str = "parking_slot"
    geometryType: str = "polygon"
    points: PointExterior


class ImageSize(BaseModel):
    """
    Размеры изображения.

    Attributes:
        height (int): высота изображения в пикселях
        width (int): ширина изображения в пикселях
    """
    height: int
    width: int


class SuperviselyAnnotation(BaseModel):
    """
    Полная аннотация изображения парковки в формате Supervisely.

    Содержит метаданные изображения, список парковочных мест и дополнительные теги.

    Attributes:
        size (ImageSize): размеры изображения
        objects (list): список объектов (парковочные места)
        description (str): описание изображения (опционально)
        tags (list): список тегов изображения (опционально)
    """
    size: ImageSize
    objects: List[ParkingObject]
    description: str = ""
    tags: List[str] = []