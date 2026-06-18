"""
Тесты для проверки анализа качества изображения.

Включают:
- Успешный анализ изображения
- Анализ черного изображения (без границ)
"""

import io
import numpy as np
import cv2


def test_quality_endpoint_success(client):
    """Сценарий 1.1: Успешная загрузка изображения и получение метрик качества."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = io.BytesIO(img_encoded.tobytes())

    response = client.post(
        "/quality",
        files={"image_file": ("test.jpg", img_bytes, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "edge_percent" in data
    assert "is_good_quality" in data


def test_quality_black_image(client):
    """Сценарий 1.2: Анализ полностью черного изображения."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = io.BytesIO(img_encoded.tobytes())

    response = client.post(
        "/quality",
        files={"image_file": ("black.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["edge_percent"] == 0
    assert data["is_good_quality"] is False
