import contextlib
import io
import os
import shutil
import tempfile
from typing import Union, Tuple, Iterator

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import StreamingResponse, JSONResponse

from src.parking_monitoring.ParkingPipeline import ParkingPipeline
from src.parking_monitoring.ImageQualityAnalyzer import ImageQualityAnalyzer
from src.parking_monitoring.CarDetector import CarDetector
from src.parking_monitoring.OccupancyAnalyzer import OccupancyAnalyzer
from src.api.schemas import Full_pipelineResponse, DetectResponse

app = FastAPI(title="Parking Detection API")

# =========================
# ИНИЦИАЛИЗАЦИЯ
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

detector = CarDetector(
    model_path="src/models/best_linknet_finetuned.pth",
    device=device,
    patch_size=320,
    overlap=160,
    threshold=0.3
)
quality_analyzer = ImageQualityAnalyzer()
parking_analyzer = OccupancyAnalyzer()
# Инициализируем pipeline как экземпляр класса
pipeline = ParkingPipeline(model_path="src/models/best_linknet_finetuned.pth")

# =========================
# УНИВЕРСАЛЬНЫЙ МЕНЕДЖЕР ФАЙЛОВ
# =========================
@contextlib.contextmanager
def temp_files(*upload_files: UploadFile) -> Iterator[Union[str, Tuple[str, ...]]]:
    paths = []
    try:
        for file in upload_files:
            suffix = os.path.splitext(file.filename)[1] or ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                paths.append(tmp.name)
        
        yield paths[0] if len(paths) == 1 else tuple(paths)
    finally:
        for path in paths:
            if os.path.exists(path):
                os.remove(path)

# =========================
# API ENDPOINTS
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/quality")
async def check_quality(image_file: UploadFile = File(...)):
    with temp_files(image_file) as path:
        return quality_analyzer.analyze(path, visualize=False, content_type=image_file.content_type)

@app.post("/quality/image")
async def quality_image(image_file: UploadFile = File(...)):
    with temp_files(image_file) as path:
        image_bytes = quality_analyzer.get_visualized_report(path, content_type=image_file.content_type)
        return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")

@app.post("/detect", response_model=DetectResponse)
async def detect_cars(image_file: UploadFile = File(...)):
    with temp_files(image_file) as path:
        try:
            mask = detector.detect_patches(path, content_type=image_file.content_type)
            if mask.size == 0:
                 raise HTTPException(status_code=400, detail="Invalid image format")
                 
            return {
                "mask_shape": mask.shape,
                "car_pixel_ratio": float(mask.mean())
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

@app.post("/detect_cars_image")
async def detect_cars_image(image_file: UploadFile = File(...)):
    with temp_files(image_file) as path:
        mask = detector.detect_patches(path)
        image_bytes = detector.visualize_detection(path, mask)
        return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")

@app.post("/full_pipeline", response_model=Full_pipelineResponse)
async def full_pipeline(
    image_file: UploadFile = File(...),
    ann_file: UploadFile = File(...),
    fail_if_low_quality: bool = True,
):
    with temp_files(image_file, ann_file) as (img_path, ann_path):
        result = pipeline.run(
            img_path,
            ann_path,
            visualize=False,
            content_type=image_file.content_type,
            fail_if_low_quality=fail_if_low_quality,
        )
        if isinstance(result, dict) and result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("message", "Unknown error"))
        return result

@app.post("/full_pipeline_visualize")
async def full_pipeline_visualize(
    image_file: UploadFile = File(...),
    ann_file: UploadFile = File(...),
    fail_if_low_quality: bool = True,
):
    with temp_files(image_file, ann_file) as (img_path, ann_path):
        data = pipeline.run(
            img_path,
            ann_path,
            visualize=False,
            content_type=image_file.content_type,
            fail_if_low_quality=fail_if_low_quality,
        )
        
        if isinstance(data, dict) and data.get("status") == "error":
            raise HTTPException(status_code=400, detail=data.get("message", "Unknown error"))

        image_bytes = pipeline.get_visualized_image(
            img_path,
            data["mask"],
            data["polygons"],
            data["status"],
        )
        return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")

# =========================
# EXCEPTION HANDLERS
# =========================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url)}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )
    
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Bad input", "detail": str(exc)}
    )