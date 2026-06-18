import contextlib
import io
import os
import shutil
import tempfile
from typing import Union, Tuple, Iterator

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import StreamingResponse, JSONResponse
import numpy as np

from src.parking_monitoring.ParkingPipeline import ParkingPipeline
from src.parking_monitoring.ImageQualityAnalyzer import ImageQualityAnalyzer
from src.parking_monitoring.CarDetector import CarDetector
from src.parking_monitoring.OccupancyAnalyzer import OccupancyAnalyzer
from src.api.schemas import Full_pipelineResponse, DetectResponse

app = FastAPI(title="Parking Detection API")

# Инициализация компонентов
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
pipeline = ParkingPipeline(model_path="src/models/best_linknet_finetuned.pth")


@contextlib.contextmanager
def temp_files(*upload_files: UploadFile) -> Iterator[Union[str, Tuple[str, ...]]]:
    """
    Контекст-менеджер для создания и автоматической очистки временных файлов.

    Сохраняет загруженные файлы в временную директорию и гарантирует их удаление
    при выходе из контекста, даже если произойдёт исключение.

    :param upload_files: переменное количество объектов UploadFile для сохранения
    :return: путь к одному файлу (если был один) или кортеж путей (если было несколько)
    :raises Exception: любое исключение из блока with пропускается после очистки файлов
    """
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


@app.get("/health")
def health() -> dict:
    """
    Проверка доступности API и статуса компонентов.

    :return: словарь со статусом 'ok'
    """
    return {"status": "ok"}


@app.post("/quality")
async def check_quality(image_file: UploadFile = File(...)) -> dict:
    """
    Анализирует качество загруженного изображения.

    Вычисляет процент пикселей границ и определяет, достаточно ли качество
    для надёжного анализа припаркованных автомобилей.

    :param image_file: загруженный файл изображения (JPEG или PNG)
    :return: словарь с ключами 'edge_percent' (float) и 'is_good_quality' (bool)
    :raises HTTPException: если формат файла неверный или произошла ошибка обработки
    """
    with temp_files(image_file) as path:
        return quality_analyzer.analyze(path, visualize=False, content_type=image_file.content_type)


@app.post("/quality/image")
async def quality_image(image_file: UploadFile = File(...)) -> StreamingResponse:
    """
    Возвращает визуализацию анализа качества изображения.

    Генерирует композитный отчёт 2x2 (исходное, фильтр, границы, наложение)
    и отправляет как PNG изображение.

    :param image_file: загруженный файл изображения
    :return: PNG изображение со сравнением результатов анализа
    :raises HTTPException: при ошибке обработки файла
    """
    with temp_files(image_file) as path:
        image_bytes = quality_analyzer.get_visualized_report(path, content_type=image_file.content_type)
        return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")


@app.post("/detect", response_model=DetectResponse)
async def detect_cars(image_file: UploadFile = File(...)) -> DetectResponse:
    """
    Детектирует автомобили на загруженном изображении.

    Выполняет инференс модели семантической сегментации и возвращает статистику.

    :param image_file: загруженный файл изображения
    :return: объект DetectResponse с маской и ratio автомобилей
    :raises HTTPException: если формат файла неверный или детекция не удалась
    """
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
async def detect_cars_image(image_file: UploadFile = File(...)) -> StreamingResponse:
    """
    Возвращает визуализацию результатов детекции автомобилей.

    Накладывает маску детекции (cyan цвет) на исходное изображение и отправляет как PNG.

    :param image_file: загруженный файл изображения
    :return: PNG изображение с наложением маски детекции
    :raises HTTPException: при ошибке обработки файла
    """
    with temp_files(image_file) as path:
        mask = detector.detect_patches(path)
        image_bytes = detector.visualize_detection(path, mask)
        return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")


@app.post("/full_pipeline", response_model=Full_pipelineResponse)
async def full_pipeline(
    image_file: UploadFile = File(...),
    ann_file: UploadFile = File(...),
    fail_if_low_quality: bool = True,
) -> Full_pipelineResponse:
    """
    Выполняет полный анализ занятости парковки и возвращает результаты.

    Последовательно проверяет качество, детектирует автомобили, загружает полигоны
    парковочных мест из аннотации и определяет занятость каждого места.

    :param image_file: загруженный файл изображения парковки
    :param ann_file: загруженный JSON файл с полигонами парковочных мест
    :param fail_if_low_quality: прерывать ли при низком качестве (по умолчанию True)
    :return: объект Full_pipelineResponse с метриками качества и занятости
    :raises HTTPException: при ошибке валидации данных или анализа (код 400)
    """
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
) -> StreamingResponse:
    """
    Выполняет полный анализ и возвращает визуализацию результатов.

    Возвращает PNG изображение с наложением маски детекции и цветных полигонов
    (красные - занятые места, зелёные - свободные).

    :param image_file: загруженный файл изображения парковки
    :param ann_file: загруженный JSON файл с полигонами парковочных мест
    :param fail_if_low_quality: прерывать ли при низком качестве (по умолчанию True)
    :return: PNG изображение с результатами анализа
    :raises HTTPException: при ошибке валидации данных или анализа (код 400)
    """
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


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Обработчик ошибок для несуществующих endpoints.

    :param request: объект запроса
    :param exc: исключение
    :return: JSON ответ с кодом 404 и описанием ошибки
    """
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url)}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Глобальный обработчик необработанных исключений.

    :param request: объект запроса
    :param exc: исключение
    :return: JSON ответ с кодом 500 и описанием ошибки
    """
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """
    Обработчик ошибок валидации входных данных.

    :param request: объект запроса
    :param exc: исключение ValueError
    :return: JSON ответ с кодом 400 и описанием ошибки
    """
    return JSONResponse(
        status_code=400,
        content={"error": "Bad input", "detail": str(exc)}
    )