"""
Модуль для мониторинга и анализа занятости парковочных мест.

Содержит компоненты для полного pipeline обработки изображений парковок:
- CarDetector: детекция автомобилей на основе семантической сегментации
- ImageQualityAnalyzer: анализ качества входных изображений
- OccupancyAnalyzer: определение занятости парковочных мест
- DataValidator: валидация входных данных
- PatchEngine: разбиение изображений на перекрывающиеся патчи
- ParkingPipeline: оркестрирует все компоненты в единый workflow

Пример использования:
    from src.parking_monitoring import ParkingPipeline

    pipeline = ParkingPipeline("path/to/model.pth")
    results = pipeline.run("image.jpg", "annotations.json")
    print(f"Occupancy: {results['occupancy_percent']:.1f}%")
"""
