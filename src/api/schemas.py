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
