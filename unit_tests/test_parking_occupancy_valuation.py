import io
import json
import cv2
import numpy as np

def test_occupancy_valuation_endpoint_success(client):
    """Сценарий 3:  Успешная загрузка изображения, разметки парковки и оценка занятости парковочных мест"""
    
    # 1. Изображение
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 100
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = io.BytesIO(img_encoded.tobytes())

    # 2. Правильная структура Supervisely JSON
    # Убедись, что координаты x и y не перепутаны (Supervisely: [x, y])
    parking_data = {
        "description": "",
        "tags": [],
        "size": {"height": 720, "width": 1280},
        "objects": [
            {
                "id": 12345,
                "classId": 54321,
                "classTitle": "parking_slot", # Проверь, как называется класс в твоем коде!
                "geometryType": "polygon",
                "points": {
                    "exterior": [[100, 100], [200, 100], [200, 200], [100, 200]],
                    "interior": []
                }
            }
        ]
    }
    json_bytes = io.BytesIO(json.dumps(parking_data).encode('utf-8'))

    response = client.post(
        "/full_pipeline",
        files={
            "image_file": ("parking.jpg", img_bytes, "image/jpeg"),
            "ann_file": ("annotation.json", json_bytes, "application/json")
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["total_spots"] > 0, f"Expected at least 1 spot, got {data['total_spots']}. Response: {data}"
    

