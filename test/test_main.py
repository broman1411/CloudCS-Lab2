# -*- coding: utf-8 -*-
import pytest
from fastapi.testclient import TestClient
from keycloak.uma_permissions import AuthStatus
from typing import Any


@pytest.fixture
def init_test_client(monkeypatch) -> TestClient:
    def mock_make_inference(*args, **kwargs) -> dict[str, float]:
        return {"churn": 0.847}

    def mock_load_model(*args, **kwargs) -> None:
        return None

    def mock_keycloak_openid(*args, **kwargs) -> Any:
        class FakedKeycloakOpenID:
            @staticmethod
            def well_known(*args, **kwargs):
                return {"token_endpoint": "fakedendpoint"}

            @staticmethod
            def has_uma_access(token: str, *args, **kwargs) -> AuthStatus:
                if token == "Ok":
                    return AuthStatus(True, True, set())
                elif token == "Not_logged":
                    return AuthStatus(False, False, set())
                elif token == "Not_authorized":
                    return AuthStatus(True, False, set())
                else:
                    return AuthStatus(False, False, set())
        return FakedKeycloakOpenID

    def mock_get_keycloak_data(*args, **kwargs):
        return FakedKeycloakOpenID(), "fakedendpoint"

    monkeypatch.setenv("MODEL_PATH", "faked/model.pkl")
    monkeypatch.setenv("KEYCLOAK_URL", "fakeurl")
    monkeypatch.setenv("CLIENT_ID", "fakeid")
    monkeypatch.setenv("CLIENT_SECRET", "fakesecret")
    monkeypatch.setattr("model_utils.make_inference", mock_make_inference)
    monkeypatch.setattr("model_utils.load_model", mock_load_model)
    monkeypatch.setattr("keycloak.KeycloakOpenID", mock_keycloak_openid)
    monkeypatch.setattr("keycloak_utils.get_keycloak_data", mock_get_keycloak_data)

    from main import app
    return TestClient(app)


def test_healthcheck(init_test_client) -> None:
    response = init_test_client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_token_correctness(init_test_client) -> None:
    """Тест с правильным токеном (авторизован)"""
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer Ok"},
        json={
            "Age": 35.0,
            "Support_Calls": 3.0,
            "Payment_Delay": 12.5,
            "Total_Spend": 1250.75,
            "Subscription_Type": "Premium",
            "Contract_Length": "Annual"
        }
    )
    assert response.status_code == 200
    assert response.json() == {"churn": 0.847}


def test_token_not_correctness(init_test_client):
    """Тест с невалидным токеном (не залогинен)"""
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer Not_logged"},
        json={
            "Age": 35.0,
            "Support_Calls": 3.0,
            "Payment_Delay": 12.5,
            "Total_Spend": 1250.75,
            "Subscription_Type": "Premium",
            "Contract_Length": "Annual"
        }
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid authentication credentials"


def test_access_denied(init_test_client):
    """Тест с токеном без прав (залогинен, но не авторизован)"""
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer Not_authorized"},
        json={
            "Age": 35.0,
            "Support_Calls": 3.0,
            "Payment_Delay": 12.5,
            "Total_Spend": 1250.75,
            "Subscription_Type": "Premium",
            "Contract_Length": "Annual"
        }
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Access denied"


def test_token_absent(init_test_client):
    """Тест без токена"""
    response = init_test_client.post(
        "/predictions",
        json={
            "Age": 35.0,
            "Support_Calls": 3.0,
            "Payment_Delay": 12.5,
            "Total_Spend": 1250.75,
            "Subscription_Type": "Premium",
            "Contract_Length": "Annual"
        }
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"


def test_inference(init_test_client):
    """Тест корректного предсказания"""
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer Ok"},
        json={
            "Age": 60.0,
            "Support_Calls": 9.0,
            "Payment_Delay": 28.0,
            "Total_Spend": 200.0,
            "Subscription_Type": "Basic",
            "Contract_Length": "Monthly"
        }
    )
    assert response.status_code == 200
    assert response.json()["churn"] == 0.847