import numpy as np
import cv2
import json
from typing import List


class OccupancyAnalyzer:
    """
    Работа с полигонами и определение занятости.
    """

    @staticmethod
    def load_polygons(json_path: str) -> List[np.ndarray]:
        with open(json_path) as f:
            data = json.load(f)

        polygons = []
        for obj in data.get("objects", []):
            if obj.get("classTitle") == "parking_slot" and obj.get("geometryType") == "polygon":
                points = obj["points"]["exterior"]
                polygons.append(np.array(points, dtype=np.int32))
                
        return polygons
    @staticmethod
    def check_occupancy(mask: np.ndarray, polygons: List[np.ndarray]) -> List[bool]:
        results = []

        for poly in polygons:
            spot_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.fillPoly(spot_mask, [poly], 1)

            content = cv2.bitwise_and(mask, mask, mask=spot_mask)

            x, y, w, h = cv2.boundingRect(poly)
            roi = content[y:y+h, x:x+w]

            if roi.size == 0:
                results.append(False)
                continue

            grid = cv2.resize(roi, (5, 5), interpolation=cv2.INTER_AREA)
            grid = (grid > 0.4).astype(np.uint8)

            total = np.sum(grid)
            inner = np.sum(grid[1:4, 1:4])

            edges = (
                np.sum(grid[0, :]) > 0,
                np.sum(grid[4, :]) > 0,
                np.sum(grid[:, 0]) > 0,
                np.sum(grid[:, 4]) > 0
            )

            if (inner > 0 or sum(edges) > 1) and (total / 25 >= 0.2):
                results.append(True)
            else:
                results.append(False)

        return results

