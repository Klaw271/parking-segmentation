from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse, JSONResponse
import tempfile
import shutil
import os
import cv2
import io
import numpy as np

from src.parking_monitoring.ImageQualityAnalyzer import ImageQualityAnalyzer
from src.parking_monitoring.CarDetector import CarDetector
from src.parking_monitoring.OccupancyAnalyzer import OccupancyAnalyzer

from src.api.schemas import (
    Full_pipelineResponse,
    DetectResponse
)

from src.api.validators import validate_image, validate_image_and_json_size, validate_json

import torch


app = FastAPI(title="Parking Detection API")


# =========================
# ИНИЦИАЛИЗАЦИЯ (1 раз)
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

detector = CarDetector(
    model_path="src/models/best_linknet.pth",
    device=device,
    patch_size=640,
    overlap=320,
    threshold=0.3
)

quality_analyzer = ImageQualityAnalyzer()
parking_analyzer = OccupancyAnalyzer()


# =========================
# HEALTHCHECK
# Проверка доступности сервиса
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}

# =========================
# QUALITY CHECK (METRIC API)
# Оценка качества изображения на основе edge-density
# Используется для фильтрации плохих входных данных
# =========================
@app.post("/quality")
async def check_quality(
    image_file: UploadFile = File(...)
):
    """
    Input:
        image_file (jpg/png)

    Output:
        edge_percent: float
        is_good_quality: bool
    """
    validate_image(image_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_tmp:
        shutil.copyfileobj(image_file.file, img_tmp)
        img_path = img_tmp.name

    try:
        result = quality_analyzer.analyze(img_path, visualize=False)
        return result

    finally:
        os.remove(img_path)

# =========================
# QUALITY VISUALIZATION
# Визуализация границ + overlay на изображении
# Используется для debug / анализа качества данных
# =========================
@app.post("/quality/image")
async def quality_image(image_file: UploadFile = File(...)):
    """
    Input:
        image_file

    Output:
        PNG изображение:
        - original
        - filtered edges
        - binary edges
        - overlay
    """
    validate_image(image_file)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(image_file.file, tmp)
        path = tmp.name

    try:
        img = cv2.imread(path)

        filtered, edges = quality_analyzer.process(img)
        result_img = quality_analyzer.visualize_image(img, filtered, edges)

        _, buffer = cv2.imencode(".png", result_img)

        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/png"
        )

    finally:
        os.remove(path)

# =========================
# CAR DETECTION (CORE ML API)
# Patch-based сегментация автомобилей
# Возвращает плотность машин на изображении
# =========================
@app.post("/detect", response_model=DetectResponse)
async def detect_cars(
    image_file: UploadFile = File(...)
):
    """
    Input:
        image_file

    Output:
        mask_shape: [H, W]
        car_pixel_ratio: float (доля пикселей машин)
    """
    validate_image(image_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_tmp:
        shutil.copyfileobj(image_file.file, img_tmp)
        img_path = img_tmp.name

    try:
        mask = detector.detect_patches(img_path)

        # можно вернуть статистику по маске
        coverage = float(mask.mean())

        return {
            "mask_shape": mask.shape,
            "car_pixel_ratio": coverage
        }

    finally:
        os.remove(img_path)

# =========================
# CAR DETECTION VISUALIZATION
# Overlay сегментационной маски на оригинальное изображение
# =========================        
@app.post("/detect/image")
async def detect_cars_image(
    image_file: UploadFile = File(...)
):
    """
    Output:
        PNG image with:
        - original image
        - semi-transparent car mask (cyan overlay)
    """
    validate_image(image_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_tmp:
        shutil.copyfileobj(image_file.file, img_tmp)
        img_path = img_tmp.name

    try:
        # --- читаем изображение ---
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # --- предсказание ---
        mask = detector.detect_patches(img_path)

        # =========================
        # OVERLAY МАСКИ
        # =========================

        overlay = img_rgb.copy()

        # создаём цветную маску (например, cyan)
        colored_mask = np.zeros_like(img_rgb)
        mask_bin = (mask > 0.5).astype(np.uint8)
        colored_mask[mask_bin == 1] = [0, 255, 255]

        # накладываем полупрозрачно
        blended = cv2.addWeighted(img_rgb, 0.7, colored_mask, 0.3, 0)

        # обратно в BGR для OpenCV encoding
        result_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

        _, buffer = cv2.imencode(".png", result_bgr)

        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/png"
        )

    finally:
        os.remove(img_path)

# =========================
# FULL PIPELINE (BATCH ANALYSIS API)
# Полный CV pipeline:
# 1. Проверка качетсва изображения
# 2. Сегментация автомобилей
# 3. Анализ занятости парковочного пространства
# =========================
@app.post("/full_pipeline", response_model=Full_pipelineResponse)
async def full_pipeline(
    image_file: UploadFile = File(...),
    ann_file: UploadFile = File(...)
):
    """
    Input:
        image_file
        ann_file (Supervisely JSON polygons)

    Output:
        quality
        total_spots
        occupied
        free
        occupancy_percent
        status: List[bool]
    """
    validate_image(image_file)
    
    annotation = await validate_json(ann_file)
    
    image_content = await image_file.read()
    validate_image_and_json_size(image_content, annotation)
    await image_file.seek(0)

    # --- временные файлы ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_tmp:
        shutil.copyfileobj(image_file.file, img_tmp)
        img_path = img_tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as ann_tmp:
        shutil.copyfileobj(ann_file.file, ann_tmp)
        ann_path = ann_tmp.name

    try:
        # 1. Качество
        quality = quality_analyzer.analyze(img_path, visualize=False)

        # 2. Детекция
        mask = detector.detect_patches(img_path)

        # 3. Полигоны
        polygons = parking_analyzer.load_polygons(ann_path)

        # 4. Анализ
        status = parking_analyzer.check_occupancy(mask, polygons)

        total = len(status)
        occupied = sum(status)

        return {
            "quality": quality,
            "total_spots": total,
            "occupied": occupied,
            "free": total - occupied,
            "occupancy_percent": (occupied / total * 100) if total else 0,
            "status": status
        }

    finally:
        # --- cleanup ---
        os.remove(img_path)
        os.remove(ann_path)

# =========================
# FULL PIPELINE VISUALIZATION
# End-to-end визуализация результата анализа:
# - сегментационная маска
# - полигоны парковочных мест
# - статус занятости парковочных мест
# =========================
@app.post("/full_pipeline_visualize")
async def full_pipeline_visualize(
    image_file: UploadFile = File(...),
    ann_file: UploadFile = File(...)
):
    """
    Input:
        image_file
        ann_file (Supervisely JSON polygons)
    Output:
        PNG visualization:
        - car segmentation
        - parking slots
        - occupancy coloring
    """
    validate_image(image_file)
    
    annotation = await validate_json(ann_file)
    
    image_content = await image_file.read()
    validate_image_and_json_size(image_content, annotation)
    await image_file.seek(0)

    # --- сохраняем временные файлы ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_tmp:
        shutil.copyfileobj(image_file.file, img_tmp)
        img_path = img_tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as ann_tmp:
        shutil.copyfileobj(ann_file.file, ann_tmp)
        ann_path = ann_tmp.name

    try:
        # --- загрузка изображения ---
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # --- детекция ---
        mask = detector.detect_patches(img_path)

        # --- полигоны ---
        polygons = parking_analyzer.load_polygons(ann_path)

        # --- анализ занятости ---
        status = parking_analyzer.check_occupancy(mask, polygons)

        # =========================
        # ВИЗУАЛИЗАЦИЯ
        # =========================

        # 1. Маска (cyan)
        overlay_mask = img_rgb.copy()
        overlay_mask[mask == 1] = [0, 255, 255]  # cyan

        combined = cv2.addWeighted(img_rgb, 0.7, overlay_mask, 0.3, 0)

        # 2. Полигоны
        poly_overlay = combined.copy()

        for i, poly in enumerate(polygons):
            color = (255, 0, 0) if status[i] else (0, 255, 0)  # red/green

            # заливка
            cv2.fillPoly(poly_overlay, [poly], color)

            # контур
            cv2.polylines(combined, [poly], True, color, 2)

        # прозрачность
        final_img = cv2.addWeighted(poly_overlay, 0.3, combined, 0.7, 0)

        # =========================
        # КОНВЕРТАЦИЯ В PNG
        # =========================

        final_bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)

        _, buffer = cv2.imencode(".png", final_bgr)

        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/png"
        )

    finally:
        os.remove(img_path)
        os.remove(ann_path)
    
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )
    
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Bad input",
            "detail": str(exc)
        }
    )