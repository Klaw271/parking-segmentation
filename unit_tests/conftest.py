import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture(scope="session")
def client():
    """
    Создает экземпляр TestClient для всего сеанса тестирования.
    Использование context manager гарантирует выполнение startup/shutdown событий.
    """
    with TestClient(app) as c:
        yield c