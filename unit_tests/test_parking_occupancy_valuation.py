import io
import json
import cv2
import numpy as np

def test_occupancy_valuation_endpoint_success(client):
    """Сценарий 3.1:  Успешная загрузка изображения, разметки парковки и оценка занятости парковочных мест"""
    
    # Изображение
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 100
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = io.BytesIO(img_encoded.tobytes())

    # Структура Supervisely JSON
    parking_data = {
        "description": "",
        "tags": [],
        "size": {"height": 720, "width": 1280},
        "objects": [
            {
                "id": 12345,
                "classId": 54321,
                "classTitle": "parking_slot",
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
        "/full_pipeline?fail_if_low_quality=false",
        files={
            "image_file": ("parking.jpg", img_bytes, "image/jpeg"),
            "ann_file": ("annotation.json", json_bytes, "application/json")
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["total_spots"] > 0, f"Expected at least 1 spot, got {data['total_spots']}. Response: {data}"
    
def test_pipeline_unsupported_geometry(client):
    """Сценарий 3.2:  Проверка обработки объектов с типом геометрии 'point' вместо 'polygon'"""
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    _, img_encoded = cv2.imencode('.jpg', img)
    
    parking_data = {
        "size": {"height": 720, "width": 1280},
        "objects": [{
            "classTitle": "parking_slot",
            "geometryType": "point", 
            "points": {"exterior": [[10, 10]], "interior": []}
        }]
    }
    
    response = client.post(
        "/full_pipeline?fail_if_low_quality=false",
        files={
            "image_file": ("img.jpg", io.BytesIO(img_encoded.tobytes()), "image/jpeg"),
            "ann_file": ("ann.json", io.BytesIO(json.dumps(parking_data).encode()), "application/json")
        }
    )
    # Система должна проигнорировать этот объект и вернуть 0 мест
    assert response.status_code == 200
    assert response.json()["total_spots"] == 0

def test_pipeline_polygon_out_of_bounds(client):
    """Сценарий 3.3:  Полигон частично выходит за границы кадра"""
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    _, img_encoded = cv2.imencode('.jpg', img)
    
    parking_data = {
        "size": {"height": 720, "width": 1280},
        "objects": [{
            "classTitle": "parking_slot",
            "geometryType": "polygon",
            # Одна точка выходит за границу (1500 при ширине 1280)
            "points": {"exterior": [[100, 100], [1500, 100], [1500, 200], [100, 200]], "interior": []}
        }]
    }
    
    response = client.post(
        "/full_pipeline?fail_if_low_quality=false",
        files={
            "image_file": ("border.jpg", io.BytesIO(img_encoded.tobytes()), "image/jpeg"),
            "ann_file": ("ann.json", io.BytesIO(json.dumps(parking_data).encode()), "application/json")
        }
    )
    # Система должна выдать ошибку 400
    assert response.status_code in [400]

def test_pipeline_many_slots(client):
    """Сценарий 3.4: Обработка большого количества парковочных мест (50+)"""
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    _, img_encoded = cv2.imencode('.jpg', img)
    
    # Генерируем 50 маленьких квадратов
    objects = []
    for i in range(50):
        x = (i * 20) % 1200
        y = (i * 10) % 600
        objects.append({
            "classTitle": "parking_slot",
            "geometryType": "polygon",
            "points": {"exterior": [[x, y], [x+10, y], [x+10, y+10], [x, y+10]], "interior": []}
        })
    
    parking_data = {"size": {"height": 720, "width": 1280}, "objects": objects}
    
    response = client.post(
        "/full_pipeline?fail_if_low_quality=false",
        files={
            "image_file": ("stress.jpg", io.BytesIO(img_encoded.tobytes()), "image/jpeg"),
            "ann_file": ("ann.json", io.BytesIO(json.dumps(parking_data).encode()), "application/json")
        }
    )
    assert response.status_code == 200
    assert response.json()["total_spots"] == 50