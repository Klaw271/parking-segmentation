# Используем официальный легкий образ Python
FROM python:3.11-slim

# Устанавливаем расширенный список зависимостей для OpenCV и работы с графикой
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Сначала обновляем pip
RUN pip install --no-cache-dir --upgrade pip

# Устанавливаем torch отдельно, указывая индекс PyTorch (для CPU версии, так проще для Docker)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Копируем файл с зависимостями
COPY requirements.txt .

# Устанавливаем библиотеки (используем --no-cache-dir для уменьшения размера образа)
RUN pip install --no-cache-dir -r requirements.txt

# 2. КОПИРУЕМ ТЕСТЫ
COPY unit_tests/ ./unit_tests/

# Копируем исходный код и веса моделей
COPY src/ ./src/
COPY src/models/ ./models/

# Запуск unit-тестов при сборке образа
RUN pytest unit_tests/

# Выставляем порт, на котором работает FastAPI
EXPOSE 8000

# Команда для запуска сервера при старте контейнера
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]