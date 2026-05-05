def test_root_health(client):
    """Сценарий 5: Проверка доступности корневого эндпоинта"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()