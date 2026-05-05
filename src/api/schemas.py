from pydantic import BaseModel
from typing import List


class DetectResponse(BaseModel):
    car_pixel_ratio: float
    mask_shape: List[int]

class Full_pipelineResponse(BaseModel):
    quality: dict
    total_spots: int
    occupied: int
    free: int
    occupancy_percent: float
    status: List[bool]

class PointExterior(BaseModel):
    # Каждая точка — это список из двух чисел [x, y]
    exterior: List[List[int]] 
    interior: List[List[int]] = []

class ParkingObject(BaseModel):
    classTitle: str = "parking_slot"
    geometryType: str = "polygon"
    points: PointExterior

class ImageSize(BaseModel):
    height: int
    width: int

class SuperviselyAnnotation(BaseModel):
    size: ImageSize
    objects: List[ParkingObject]
    description: str = ""
    tags: List[str] = []