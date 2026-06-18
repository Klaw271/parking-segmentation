"""
REST API модуль для системы анализа занятости парковочных мест.

Предоставляет FastAPI endpoints для следующих операций:
- /health - проверка доступности API
- /quality - анализ качества изображения
- /quality/image - визуализация анализа качества
- /detect - детекция автомобилей
- /detect_cars_image - визуализация результатов детекции
- /full_pipeline - полный анализ занятости парковки
- /full_pipeline_visualize - полный анализ с визуализацией

Все endpoints поддерживают загрузку файлов (multipart/form-data) и возвращают
результаты в JSON или PNG формате в зависимости от endpoint.

Пример использования:
    curl -X POST http://localhost:8000/quality \\
      -F "image_file=@parking.jpg"
"""

