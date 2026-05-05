import io
import cv2
import numpy as np


def test_detection_endpoint_success(client):
    """Сценарий 2.1: Успешная загрузка изображения и обнаружение автомобилей"""
    
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = io.BytesIO(img_encoded.tobytes())

    # Отправляем POST запрос с файлом
    response = client.post(
        "/detect",
        files={"image_file": ("test.jpg", img_bytes, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    # Проверяем наличие ключей
    assert "car_pixel_ratio" in data
    assert "mask_shape" in data
    
def test_no_cars_detected(client):
    """Сценарий 2.2:  Изображение без машин (пустая дорога)"""
    # Создаем серый шум или просто однотонный фон
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 150
    _, img_encoded = cv2.imencode('.jpg', img)
    
    response = client.post(
        "/detect",
        files={"image_file": ("empty.jpg", io.BytesIO(img_encoded.tobytes()), "image/jpeg")}
    )
    assert response.status_code == 200
    # Проверяем, что маска пустая или коэффициент занятости пикселей близок к нулю
    car_pixel_ratio = response.json()["car_pixel_ratio"]
    assert car_pixel_ratio < 0.001, f"Detected too many false positives: {car_pixel_ratio}"