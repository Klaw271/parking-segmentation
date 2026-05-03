import io
import numpy as np
import cv2

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
    # Если validate_json или json.load упадут, мы ожидаем 400 (Bad Input) или 500
    assert response.status_code in [400, 500]
    
    error_data = response.json()
    assert "detail" in error_data
    print(f"\nCaught expected error: {error_data['detail']}")    

def test_error_handling_empty_json(client):
    """Сценарий 4.3: Обработка ошибок — передача пустого/некорректного json файла"""
    
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
    # Если validate_json или json.load упадут, мы ожидаем 400 (Bad Input) или 500
    assert response.status_code in [400, 500]
    
    error_data = response.json()
    assert "detail" in error_data
    print(f"\nCaught expected error: {error_data['detail']}")

def test_invalid_endpoint(client):
    """Сценарий 4.4: Обработка ошибок — вызвов несуществующего эндпоинта"""
    response = client.get("/no_such_endpoint")
    assert response.status_code == 404