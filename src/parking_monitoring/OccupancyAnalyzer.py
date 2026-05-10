import numpy as np
import cv2
import json
from typing import Any, Dict, List

from pydantic import ValidationError


class OccupancyAnalyzer:
    """
    Работа с полигонами и определение занятости.
    """

    @staticmethod
    def load_polygons(json_path: str, schema_class=None) -> List[np.ndarray]:
        # 1. Читаем файл в бинарном режиме для передачи в валидатор
        try:
            with open(json_path, 'rb') as f:
                content = f.read()
        except FileNotFoundError:
            raise ValueError(f"File not found: {json_path}")
        
        # 2. Проверка расширения, пустоты и синтаксиса
        data = OccupancyAnalyzer.validate_json_source(content, json_path)

        # 3. Валидация структуры
        if schema_class:
            annotations = OccupancyAnalyzer.validate_annotation_structure(data, schema_class)

        # 4. Основная логика извлечения
        polygons = []
        for obj in data.get("objects", []):
            if obj.get("classTitle") == "parking_slot" and obj.get("geometryType") == "polygon":
                # Добавляем базовую проверку наличия ключей в объекте
                if "points" in obj and "exterior" in obj["points"]:
                    points = obj["points"]["exterior"]
                    polygons.append(np.array(points, dtype=np.int32))
                    
                
        return polygons, annotations

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

    @staticmethod
    def validate_json_source(content: bytes, filename: str = "") -> Dict[str, Any]:
        """
        Проверка JSON: расширение, пустота, синтаксис.
        """
        # 1. Проверка расширения
        if filename and not filename.endswith(".json"):
            raise ValueError("File must have .json extension")

        # 2. Проверка на пустоту
        if len(content) == 0:
            raise ValueError("Empty JSON file")

        # 3. Проверка синтаксиса
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format: syntax error")

    @staticmethod
    def validate_annotation_structure(data: Dict[str, Any], schema_class) -> Any:
        """
        Проверка соответствия схеме Pydantic и наличия объектов.
        """
        try:
            annotation = schema_class.model_validate(data)
            if not annotation.objects:
                raise ValueError("The parking zones list is empty. Provide at least one polygon.")
            return annotation
        except ValidationError as e:
            raise ValueError(f"Annotation schema mismatch: {e.errors()}")