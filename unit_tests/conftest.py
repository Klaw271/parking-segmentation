"""
Конфигурация для pytest - общие фикстуры и настройки.

Создаёт клиент для тестирования FastAPI приложения.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="session")
def client():
    """
    Создает экземпляр TestClient для всего сеанса тестирования.

    Использование context manager гарантирует выполнение startup/shutdown событий.

    :return: TestClient для тестирования API endpoints
    """
    with TestClient(app) as c:
        yield c
