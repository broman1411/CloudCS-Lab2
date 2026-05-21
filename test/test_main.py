# -*- coding: utf-8 -*-
import sys
import os
import pytest
from fastapi.testclient import TestClient
from keycloak.uma_permissions import AuthStatus
from typing import Any
from unittest.mock import MagicMock, AsyncMock

from fastapi import Depends, FastAPI, HTTPException, status, Request


@pytest.fixture
def init_test_client(monkeypatch) -> TestClient:
    def mock_make_inference(*args, **kwargs) -> dict[str, float]:
        return {"churn": 1.0}


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

    # --- ИЗМЕНЕНИЕ 1: импорт Request ---
    from fastapi import Request  # ← добавляем импорт

    class MockOauth2ClientCredentials:
        def __init__(self, tokenUrl: str):
            self.tokenUrl = tokenUrl

        # --- ИЗМЕНЕНИЕ 2: корректная сигнатура с Request ---
        async def __call__(self, request: Request) -> str:
            # Извлекаем токен из заголовка Authorization
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                return auth_header[7:]  # убираем "Bearer "
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated"
            )

    monkeypatch.setenv("MODEL_PATH", "faked/model.pkl")
    monkeypatch.setenv("KEYCLOAK_URL", "fakeurl")
    monkeypatch.setenv("CLIENT_ID", "fakeid")
    monkeypatch.setenv("CLIENT_SECRET", "fakesecret")
    monkeypatch.setattr("src.model_utils.make_inference", mock_make_inference)
    monkeypatch.setattr("src.model_utils.load_model", mock_load_model)
    monkeypatch.setattr("keycloak.KeycloakOpenID", mock_keycloak_openid)

    # Подменяем модуль fastapi_utils
    mock_fastapi_utils = MagicMock()
    mock_fastapi_utils.Oauth2ClientCredentials = MockOauth2ClientCredentials
    sys.modules['fastapi_utils'] = mock_fastapi_utils
    sys.modules['src.fastapi_utils'] = mock_fastapi_utils

    from src.main import app
    return TestClient(app)


# ================= Тесты Healthcheck =================

def test_healthcheck(init_test_client) -> None:
    """Тест эндпоинта healthcheck"""
    response = init_test_client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ================= Тесты аутентификации =================

def test_token_correctness(init_test_client) -> None:
    """Тест успешной аутентификации и авторизации"""
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer Ok"},
        json={
            "Age": 22.0,
            "Support_Calls": 7.0,
            "Payment_Delay": 50.0,
            "Total_Spend": 100.0,
            "Subscription_Type": "Basic",
            "Contract_Length": "Monthly"
        }
    )
    # Так как мы мокаем check_token, а он возвращает ошибки для не-Ok токенов,
    # но сейчас у нас mock_token, нужно проверить что вернет
    print(f"\nDEBUG test_token_correctness: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert "churn" in response.json()
    assert response.json()["churn"] == 1.0


def test_token_not_logged_in(init_test_client):
    """Тест ошибки при невалидном токене (не залогинен)"""
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer Not_logged"},
        json={
            "Age": 22.0,
            "Support_Calls": 7.0,
            "Payment_Delay": 50.0,
            "Total_Spend": 100.0,
            "Subscription_Type": "Basic",
            "Contract_Length": "Monthly"
        }
    )
    print(f"\nDEBUG test_token_not_logged_in: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 401
    assert response.json() == {
        "detail": "Invalid authentication credentials"
    }


def test_access_denied(init_test_client):
    """Тест ошибки при отсутствии прав доступа"""
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer Not_authorized"},
        json={
            "Age": 22.0,
            "Support_Calls": 7.0,
            "Payment_Delay": 50.0,
            "Total_Spend": 100.0,
            "Subscription_Type": "Basic",
            "Contract_Length": "Monthly"
        }
    )
    print(f"\nDEBUG test_access_denied: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 403
    assert response.json() == {
        "detail": "Access denied"
    }


def test_token_absent(init_test_client):
    """Тест ошибки при отсутствии токена"""
    response = init_test_client.post(
        "/predictions",
        json={
            "Age": 22.0,
            "Support_Calls": 7.0,
            "Payment_Delay": 50.0,
            "Total_Spend": 100.0,
            "Subscription_Type": "Basic",
            "Contract_Length": "Monthly"
        }
    )
    print(f"\nDEBUG test_token_absent: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 401
    assert response.json() == {
        "detail": "Not authenticated"
    }


# ================= Тесты валидации входных данных =================

def test_invalid_subscription_type(init_test_client):
    """Тест ошибки валидации при неверном типе подписки"""
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer Ok"},
        json={
            "Age": 22.0,
            "Support_Calls": 7.0,
            "Payment_Delay": 50.0,
            "Total_Spend": 100.0,
            "Subscription_Type": "Invalid_Type",
            "Contract_Length": "Monthly"
        }
    )
    assert response.status_code == 422
    assert "Subscription_Type" in str(response.json())


def test_invalid_contract_length(init_test_client):
    """Тест ошибки валидации при неверной длительности контракта"""
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer Ok"},
        json={
            "Age": 22.0,
            "Support_Calls": 7.0,
            "Payment_Delay": 50.0,
            "Total_Spend": 100.0,
            "Subscription_Type": "Basic",
            "Contract_Length": "Invalid_Length"
        }
    )
    assert response.status_code == 422
    assert "Contract_Length" in str(response.json())


def test_missing_required_fields(init_test_client):
    """Тест ошибки при отсутствии обязательных полей"""
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer Ok"},
        json={
            "Age": 22.0,
            "Support_Calls": 7.0,
        }
    )
    assert response.status_code == 422

# ================= Тесты успешного предсказания =================

def test_inference_basic(init_test_client):
    """Тест успешного предсказания с тарифом Basic"""
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer Ok"},
        json={
            "Age": 22.0,
            "Support_Calls": 7.0,
            "Payment_Delay": 50.0,
            "Total_Spend": 100.0,
            "Subscription_Type": "Basic",
            "Contract_Length": "Monthly"
        }
    )
    print(f"\nDEBUG test_inference_basic: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()["churn"] == 1.0


def test_inference_standard(init_test_client):
    """Тест успешного предсказания с тарифом Standard"""
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer Ok"},
        json={
            "Age": 22.0,
            "Support_Calls": 7.0,
            "Payment_Delay": 50.0,
            "Total_Spend": 100.0,
            "Subscription_Type": "Standard",
            "Contract_Length": "Monthly"
        }
    )
    assert response.status_code == 200
    assert response.json()["churn"] == 1.0


def test_inference_premium(init_test_client):
    """Тест успешного предсказания с тарифом Premium"""
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer Ok"},
        json={
            "Age": 22.0,
            "Support_Calls": 7.0,
            "Payment_Delay": 50.0,
            "Total_Spend": 100.0,
            "Subscription_Type": "Premium",
            "Contract_Length": "Monthly"
        }
    )
    assert response.status_code == 200
    assert response.json()["churn"] == 1.0