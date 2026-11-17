from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_valid():
    payload = {
        "school": "GP",
        "sex": "F",
        "age": 17,
        "studytime": 2,
        "failures": 1,
        "famsup": "yes",
        "higher": "yes",
        "internet": "yes",
        "goout": 3,
        "absences": 6,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["riesgo_predicho"] in [0, 1]
    assert 0.0 <= data["probabilidad_riesgo"] <= 1.0
    assert data["risk_level"] in ["Bajo", "Medio", "Alto"]


def test_predict_invalid():
    bad_payload = {
        "school": "GP",
        "sex": "F",
        "age": 10,          # Edad fuera de rango (15-22)
        "studytime": 2,
        "failures": 0,
        "famsup": "yes",
        "higher": "yes",
        "internet": "yes",
        "goout": 3,
        "absences": -5,     # Faltas negativas inválidas
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422