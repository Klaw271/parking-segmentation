import json

import cv2
from fastapi import UploadFile, HTTPException
import numpy as np
from pydantic import ValidationError

from src.api.schemas import SuperviselyAnnotation

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

def validate_image(image_file: UploadFile):
    # Проверка формата
    if image_file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # Проверка размера файла (защита от "тяжелых" файлов)
    image_file.file.seek(0, 2)  # Переходим в конец файла
    file_size = image_file.file.tell()  # Получаем размер
    image_file.file.seek(0)  # Возвращаемся в начало
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Проверка целостности (декодирование)
    try:
        # Читаем байты, не закрывая файл
        content = image_file.file.read()
        image_file.file.seek(0)
        
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image: could not decode")
            
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image: file is corrupted")

async def validate_json(file: UploadFile) -> SuperviselyAnnotation:
    if not file.filename.endswith(".json"):
        raise HTTPException(400, "File must be .json")
    
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(400, "Empty JSON file")

    try:
        data = json.loads(content)
        await file.seek(0)
        
        annotation = SuperviselyAnnotation.model_validate(data)
        
        if not annotation.objects:
            raise HTTPException(
                status_code=400, 
                detail="Validation Error: The parking zones list is empty. Provide at least one polygon."
            )
            
        return annotation
        
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON format")
    except ValidationError as e:
        raise HTTPException(400, detail=e.errors())

def validate_image_and_json_size(img_bytes: bytes, annotation: SuperviselyAnnotation):
    """
    Проверка соответствия фактического размера изображения 
    данным, указанным в JSON-аннотации.
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data for size check")
    
    h, w = img.shape[:2]
    
    if h != annotation.size.height or w != annotation.size.width:
        raise HTTPException(
            status_code=400, 
            detail=f"Size mismatch: Image is {w}x{h}, but JSON claims {annotation.size.width}x{annotation.size.height}"
        )