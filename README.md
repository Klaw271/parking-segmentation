# 🚗 Parking Segmentation - Система анализа занятости парковочных мест

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136.1-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

Система компьютерного зрения для автоматического анализа занятости парковочных мест на основе семантической сегментации. Использует нейросеть Linknet с энкодером EfficientNet-B1 для детекции припаркованных автомобилей.

## 📋 Содержание

- [Обзор](#обзор)
- [Возможности](#возможности)
- [Требования](#требования)
- [Установка](#установка)
- [Использование](#использование)
- [API endpoints](#api-endpoints)
- [Структура проекта](#структура-проекта)
- [Архитектура](#архитектура)
- [Тестирование](#тестирование)
- [Примеры](#примеры)
- [Документация кода](#документация-кода)

## 🎯 Обзор

Система анализирует изображения парковок и определяет занятость каждого парковочного места. Полный pipeline включает:

1. **Проверка качества изображения** - анализ границ для оценки пригодности
2. **Детекция автомобилей** - семантическая сегментация припаркованных машин
3. **Анализ занятости** - определение занятости каждого места на основе полигонов
4. **Визуализация результатов** - наложение масок и полигонов на исходное изображение

## ✨ Возможности

- ✅ **Детекция автомобилей** с нейросетью Linknet (IoU > 0.85)
- ✅ **Анализ качества** изображения (выделение границ)
- ✅ **Определение занятости** парковочных мест
- ✅ **REST API** на FastAPI с документацией Swagger
- ✅ **Две режима детекции**: с нарезкой на патчи (точнее) или на полное изображение (быстрее)
- ✅ **Аугментация данных** через Albumentations
- ✅ **Unit-тесты** для всех компонентов (18+ тестов)
- ✅ **Полная документация** кода на русском языке

## 📦 Требования

### Системные требования
- **Python**: 3.9 или выше
- **Память**: минимум 4 ГБ для инференса, 8+ ГБ для обучения
- **GPU**: опционально (NVIDIA CUDA для ускорения)

### Зависимости
Все зависимости указаны в `requirements.txt`:
- FastAPI - веб-фреймворк
- PyTorch - глубокое обучение
- segmentation_models_pytorch - архитектуры сегментации
- OpenCV - обработка изображений
- Pydantic - валидация данных
- pytest - тестирование

## 🚀 Установка

### 1. Клонирование репозитория
```bash
git clone <repository-url>
cd parking-segmentation
```

### 2. Создание виртуального окружения
```bash
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
```

### 3. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 4. Загрузка предобученной модели
```bash
# Модель должна быть расположена здесь:
# src/models/best_linknet_finetuned.pth
# 
# Если нет, скачайте её или обучите на собственных данных (см. COCO/train.py)
```

## 💻 Использование

### Как Python библиотека

```python
from src.parking_monitoring import ParkingPipeline

# Инициализация pipeline
pipeline = ParkingPipeline(
    model_path="src/models/best_linknet_finetuned.pth",
    fail_if_low_quality=True
)

# Запуск анализа
results = pipeline.run(
    image_path="data/parking.jpg",
    json_path="data/parking_annotation.json",
    visualize=True,
    fail_if_low_quality=False
)

# Результаты
print(f"Всего мест: {results['total_spots']}")
print(f"Занято: {results['occupied']}")
print(f"Свободно: {results['free']}")
print(f"Занятость: {results['occupancy_percent']:.1f}%")
```

### Запуск REST API

```bash
# Запуск сервера
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Документация доступна на: http://localhost:8000/docs
```

### Примеры curl запросов

```bash
# Проверка здоровья
curl http://localhost:8000/health

# Анализ качества
curl -X POST http://localhost:8000/quality \
  -F "image_file=@parking.jpg"

# Детекция автомобилей
curl -X POST http://localhost:8000/detect \
  -F "image_file=@parking.jpg"

# Полный анализ занятости
curl -X POST http://localhost:8000/full_pipeline \
  -F "image_file=@parking.jpg" \
  -F "ann_file=@parking.json" \
  -F "fail_if_low_quality=false"
```

## 🔌 API endpoints

### Health Check
```http
GET /health
```
**Ответ**: `{"status": "ok"}`

### Анализ качества
```http
POST /quality
Content-Type: multipart/form-data

image_file: <JPEG/PNG>
```
**Ответ**:
```json
{
  "edge_percent": 5.2,
  "is_good_quality": true
}
```

### Детекция автомобилей
```http
POST /detect
Content-Type: multipart/form-data

image_file: <JPEG/PNG>
```
**Ответ**:
```json
{
  "car_pixel_ratio": 0.15,
  "mask_shape": [720, 1280]
}
```

### Полный анализ занятости
```http
POST /full_pipeline?fail_if_low_quality=false
Content-Type: multipart/form-data

image_file: <JPEG/PNG>
ann_file: <JSON>
```
**Ответ**:
```json
{
  "quality": {"edge_percent": 5.2, "is_good_quality": true},
  "total_spots": 50,
  "occupied": 18,
  "free": 32,
  "occupancy_percent": 36.0,
  "status": [true, false, true, ...]
}
```

### Визуализация
```http
POST /full_pipeline_visualize?fail_if_low_quality=false
POST /quality/image
POST /detect_cars_image
```
**Ответ**: PNG изображение с визуализацией

## 📁 Структура проекта

```
parking-segmentation/
├── src/
│   ├── __init__.py
│   ├── parking_monitoring/          # Основной модуль анализа
│   │   ├── __init__.py
│   │   ├── CarDetector.py          # Детектор автомобилей
│   │   ├── ImageQualityAnalyzer.py # Анализатор качества
│   │   ├── OccupancyAnalyzer.py    # Анализатор занятости
│   │   ├── DataValidator.py        # Валидатор данных
│   │   ├── PatchEngine.py          # Нарезка на патчи
│   │   └── ParkingPipeline.py      # Главный pipeline
│   │
│   ├── api/                         # REST API
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI приложение
│   │   └── schemas.py              # Pydantic модели
│   │
│   ├── COCO/                        # Работа с датасетом COCO
│   │   ├── __init__.py
│   │   ├── ParkingDataset.py       # PyTorch Dataset
│   │   ├── train.py                # Обучение модели
│   │   ├── finetune_balanced.py    # Fine-tuning
│   │   └── ...
│   │
│   └── self_made_dataset/           # Собственный датасет
│       ├── __init__.py
│       ├── BalancedParkingDataset.py
│       └── ...
│
├── unit_tests/                      # Unit-тесты
│   ├── conftest.py
│   ├── test_health.py
│   ├── test_car_detection.py
│   ├── test_quality_alaysis.py
│   ├── test_errors.py
│   ├── test_parking_occupancy_valuation.py
│   └── run_tests.py
│
├── data/                            # Данные для тестирования
│   ├── test/                        # Тестовые изображения
│   └── ...
│
├── requirements.txt                 # Зависимости
├── README.md                        # Этот файл
├── TESTING.md                       # Документация тестирования
└── .gitignore
```

## 🏗️ Архитектура

### Компоненты системы

```
┌─────────────────┐
│  Изображение    │
│  + Аннотация    │
└────────┬────────┘
         │
         ▼
┌────────────────────────┐
│ DataValidator          │ ◄─── Проверка формата, размера, целостности
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ ImageQualityAnalyzer   │ ◄─── Анализ границ (edge_percent)
└────────┬───────────────┘
         │
         ├─► Плохое качество? ──► STOP (опционально)
         │
         ▼
┌────────────────────────┐
│ CarDetector            │ ◄─── Linknet (EfficientNet-B1)
│ - detect_patches()     │       Детекция припаркованных машин
│ - detect_full_image()  │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ OccupancyAnalyzer      │ ◄─── Анализ занятости по полигонам
│ - check_occupancy()    │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ Результаты             │
│ - total_spots          │
│ - occupied             │
│ - occupancy_percent    │
└────────────────────────┘
```

### Алгоритм определения занятости

Для каждого парковочного места используется комбинированный критерий:

1. **Общее соотношение пересечения** (≥ 30% И внутренняя часть ≥ 12%)
2. **Прямое соотношение** (≥ 45%)
3. **Анализ сетки 5×5** (проверка внутренних пикселей и краев)

Это обеспечивает устойчивость к помехам и частичным перекрытиям.

## 🧪 Тестирование

### Запуск всех тестов
```bash
pytest unit_tests/ -v
```

### Запуск с отчетом покрытия
```bash
pytest unit_tests/ --cov=src --cov-report=html
```

### Запуск конкретного теста
```bash
pytest unit_tests/test_health.py::test_root_health -v
```

### Использование скрипта запуска
```bash
python unit_tests/run_tests.py
python unit_tests/run_tests.py --coverage
```

Подробнее см. [TESTING.md](TESTING.md)

### Покрытие тестами

- ✅ **18+ unit-тестов** всех компонентов
- ✅ **5 сценариев**: здоровье, детекция, качество, ошибки, занятость
- ✅ **Граничные случаи**: пустые изображения, поврежденные файлы, размеры вне границ

## 📚 Примеры

### Пример 1: Анализ парковки из Python

```python
from src.parking_monitoring import ParkingPipeline
import json

# Инициализация
pipeline = ParkingPipeline(
    model_path="src/models/best_linknet_finetuned.pth"
)

# Запуск
results = pipeline.run(
    image_path="data/parking.jpg",
    json_path="data/parking.json",
    visualize=False
)

# Вывод результатов
print(f"Анализ завершен!")
print(f"├─ Всего мест: {results['total_spots']}")
print(f"├─ Занято: {results['occupied']}")
print(f"├─ Свободно: {results['free']}")
print(f"└─ Занятость: {results['occupancy_percent']:.1f}%")

# Сохранение результатов
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Пример 2: Формат аннотации (Supervisely JSON)

```json
{
  "size": {
    "height": 720,
    "width": 1280
  },
  "objects": [
    {
      "classTitle": "parking_slot",
      "geometryType": "polygon",
      "points": {
        "exterior": [
          [100, 100],
          [200, 100],
          [200, 200],
          [100, 200]
        ],
        "interior": []
      }
    }
  ]
}
```

### Пример 3: Использование отдельных компонентов

```python
import cv2
import numpy as np
from src.parking_monitoring import (
    CarDetector, ImageQualityAnalyzer, OccupancyAnalyzer
)

# Загрузка изображения
img = cv2.imread("parking.jpg")

# 1. Проверка качества
quality_analyzer = ImageQualityAnalyzer()
quality = quality_analyzer.analyze("parking.jpg", visualize=False)
print(f"Качество: {'Хорошее' if quality['is_good_quality'] else 'Плохое'}")

# 2. Детекция автомобилей
detector = CarDetector("src/models/best_linknet_finetuned.pth", "cpu")
mask = detector.detect_patches("parking.jpg")

# 3. Анализ занятости
occupancy = OccupancyAnalyzer()
polygons = [...]  # Загруженные полигоны
status = occupancy.check_occupancy(mask, polygons)
print(f"Занято: {sum(status)}/{len(status)}")
```

## 📖 Документация кода

### Стандарты документации

Весь код документирован в соответствии со следующими стандартами:

- **Язык**: Русский
- **Формат docstring**: NumPy-style (`:param`, `:return`, `:raises`)
- **Type hints**: Все функции имеют type hints
- **Комментарии**: Только для не очевидной логики

### Примеры из кода

```python
def detect_patches(self, image_path: str, content_type: str = None, target_height: int = 704) -> np.ndarray:
    """
    Детектирует автомобили с использованием нарезки на патчи с перекрытием.

    Нарезает изображение на перекрывающиеся патчи, выполняет инференс на каждом,
    затем собирает результаты в единую маску с учётом перекрытий.

    :param image_path: путь к файлу изображения
    :param content_type: MIME-тип файла для валидации (опционально)
    :param target_height: целевая высота изображения для масштабирования (по умолчанию 704)
    :return: бинаризованная маска размером (H, W) с значениями 0 или 1
    :raises ValueError: если файл не найден, повреждён или детекция не удалась
    """
```

### IDE поддержка

Благодаря полной документации, IDE корректно показывает:
- ✅ Подсказки параметров при вводе
- ✅ Описание функций при наведении
- ✅ Типы параметров и возвращаемых значений
- ✅ Возможные исключения

## 🤝 Интеграция

### Обучение модели

Для обучения модели на собственных данных:

```bash
python src/COCO/train.py
```

Параметры настраиваются в файле `train.py`:
- EPOCHS: количество эпох
- batch_size: размер батча
- learning rate: начальная скорость обучения

### Fine-tuning на новых данных

```bash
python src/COCO/finetune_balanced.py
```

## 📝 Лицензия

MIT License - см. LICENSE файл

## 📞 Поддержка

### Часто встречающиеся проблемы

**Q: Модель не найдена**
```
A: Убедитесь, что файл находится в src/models/best_linknet_finetuned.pth
```

**Q: CUDA out of memory**
```
A: Используйте режим CPU или уменьшите размер патча в CarDetector
```

**Q: Низкое качество результатов**
```
A: Проверьте качество аннотаций (JSON) и соответствие размеров
```

## 🔗 Полезные ссылки

- [FastAPI документация](https://fastapi.tiangolo.com/)
- [PyTorch документация](https://pytorch.org/docs/)
- [OpenCV документация](https://docs.opencv.org/)
- [segmentation_models_pytorch](https://smp.readthedocs.io/)

## 📝 История изменений

### v1.0.0 (Текущая версия)
- ✅ Полный pipeline анализа занятости
- ✅ REST API с FastAPI
- ✅ Unit-тесты (18+ тестов)
- ✅ Полная документация кода
- ✅ Поддержка GPU/CPU
- ✅ Две режима детекции

---

**Разработано для автоматического анализа парковочных мест**