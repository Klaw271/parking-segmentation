import io


def test_quality_endpoint_success(client):
    """Сценарий 1: Успешная загрузка изображения и получение метрик качества"""
    # Создаем фиктивное изображение в памяти (белый квадрат 100x100)
    import numpy as np
    import cv2
    
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = io.BytesIO(img_encoded.tobytes())

    # Отправляем POST запрос с файлом
    response = client.post(
        "/quality",
        files={"image_file": ("test.jpg", img_bytes, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    # Проверяем наличие ключей
    assert "edge_percent" in data
    assert "is_good_quality" in data
    

