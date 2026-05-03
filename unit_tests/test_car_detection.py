import io


def test_detection_endpoint_success(client):
    """Сценарий 2: Успешная загрузка изображения и обнаружение автомобилей"""
    # Создаем фиктивное изображение в памяти (белый квадрат 100x100)
    import numpy as np
    import cv2
    
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