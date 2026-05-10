import io
import numpy as np
import cv2
import json

def test_error_handling_invalid_image_format(client):
    """Сценарий 4.1: Обработка ошибок — передача текстового файла вместо изображения"""
    fake_file = io.BytesIO(b"this is not an image")
    
    response = client.post(
        "/quality",
        files={"image_file": ("test.txt", fake_file, "text/plain")}
    )

    assert response.status_code == 400
    error_data = response.json()
    
    # Исправленное утверждение:
    assert "detail" in error_data 
    assert error_data["detail"] == "Invalid image format"

def test_error_handling_invalid_json_format(client):
    """Сценарий 4.2: Обработка ошибок — передача текстового файла вместо json"""
    
    # 1. Готовим корректную картинку (так как эндпоинт full_pipeline ждет оба файла)
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = io.BytesIO(img_encoded.tobytes())

    # 2. Готовим "неправильный" JSON
    fake_json = io.BytesIO(b"this is not a json")

    # 3. Отправляем запрос на full_pipeline
    response = client.post(
        "/full_pipeline",
        files={
            "image_file": ("test.jpg", img_bytes, "image/jpeg"),
            "ann_file": ("test.json", fake_json, "application/json")
        }
    )

    # 4. Проверяем результат
    # Если validate_json или json.load упадут ожидаем 400 (Bad Input)
    assert response.status_code == 400
    
    error_data = response.json()
    assert "detail" in error_data
    print(f"\nCaught expected error: {error_data['detail']}")    

def test_error_handling_empty_json(client):
    """Сценарий 4.3: Обработка ошибок — передача пустого json файла"""
    
    # 1. Готовим корректную картинку (так как эндпоинт full_pipeline ждет оба файла)
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = io.BytesIO(img_encoded.tobytes())

    # 2. Готовим пустой JSON
    invalid_json_content = b"" # Совсем пустой файл
    json_bytes = io.BytesIO(invalid_json_content)

    # 3. Отправляем запрос на full_pipeline
    response = client.post(
        "/full_pipeline",
        files={
            "image_file": ("test.jpg", img_bytes, "image/jpeg"),
            "ann_file": ("test.json", json_bytes, "application/json")
        }
    )

    # 4. Проверяем результат
    # Если validate_json или json.load упадут, мы ожидаем 400 (Bad Input)
    assert response.status_code == 400
    
    error_data = response.json()
    assert "detail" in error_data
    print(f"\nCaught expected error: {error_data['detail']}")

def test_invalid_endpoint(client):
    """Сценарий 4.4: Обработка ошибок — вызвов несуществующего эндпоинта"""
    response = client.get("/no_such_endpoint")
    assert response.status_code == 404
    
def test_pipeline_size_mismatch(client):
    """Сценарий 4.5: Обработка ошибок — несоответствие размеров в JSON и реальном изображении"""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    _, img_encoded = cv2.imencode('.jpg', img)
    
    parking_data = {
        "size": {"height": 720, "width": 1280}, # Ожидаем большое
        "objects": []
    }
    
    response = client.post(
        "/full_pipeline",
        files={
            "image_file": ("small.jpg", io.BytesIO(img_encoded.tobytes()), "image/jpeg"),
            "ann_file": ("ann.json", io.BytesIO(json.dumps(parking_data).encode()), "application/json")
        }
    )
    # Код должен либо выдать ошибку 400
    assert response.status_code == 400
    
def test_pipeline_no_parking_zones(client):
    """Сценарий 4.6: Обработка ошибок — передача пустого списка объектов разметки"""
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    _, img_encoded = cv2.imencode('.jpg', img)
    
    parking_data = {
        "size": {"height": 720, "width": 1280},
        "objects": []
    }
    
    response = client.post(
        "/full_pipeline",
        files={
            "image_file": ("img.jpg", io.BytesIO(img_encoded.tobytes()), "image/jpeg"),
            "ann_file": ("ann.json", io.BytesIO(json.dumps(parking_data).encode()), "application/json")
        }
    )
    assert response.status_code == 400
    
def test_large_image_rejection(client):
    """Сценарий 4.7: Обработка ошибок — попытка загрузить слишком тяжелое изображение (имитация)"""
    # Создаем "тяжелый" файл
    large_file = io.BytesIO(b"\0" * (15 * 1024 * 1024))
    
    response = client.post(
        "/detect",
        files={"image_file": ("huge.jpg", large_file, "image/jpeg")}
    )
    
    assert response.status_code == 400

def test_corrupted_image(client):
    """Сценарий 4.8: Обработка ошибок — Передача поврежденного файла изображения"""
    corrupted_data = io.BytesIO(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00\x48") # Только заголовок
    
    response = client.post(
        "/detect",
        files={"image_file": ("corrupt.jpg", corrupted_data, "image/jpeg")}
    )
    assert response.status_code == 400
    assert "Invalid image" in response.json()["detail"]