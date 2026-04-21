# -*- coding: utf-8 -*-
import pytest
import pandas as pd
from model_utils import make_inference, load_model
from sklearn.pipeline import Pipeline
from pickle import dumps


def test_make_inference(monkeypatch):
    """Тест функции make_inference"""
    def mock_predict_proba(_, data: pd.DataFrame) -> list[list[float]]:
        # Проверяем, что колонки созданы с подчёркиваниями
        expected_columns = [
            'Age', 'Support_Calls', 'Payment_Delay', 
            'Total_Spend', 'Subscription_Type', 'Contract_Length'
        ]
        assert list(data.columns) == expected_columns
        assert data.iloc[0]['Age'] == 60.0
        assert data.iloc[0]['Support_Calls'] == 9.0
        return [[0.15, 0.85]]

    in_model = Pipeline([])
    monkeypatch.setattr(Pipeline, "predict_proba", mock_predict_proba)

    data = {
        "Age": 60.0,
        "Support_Calls": 9.0,
        "Payment_Delay": 28.0,
        "Total_Spend": 200.0,
        "Subscription_Type": "Basic",
        "Contract_Length": "Monthly"
    }
    
    result = make_inference(in_model, data)
    assert result == {"churn": 0.85}


def test_make_inference_rounding(monkeypatch):
    """Тест округления до 3 знаков"""
    def mock_predict_proba(_, data: pd.DataFrame) -> list[list[float]]:
        return [[0.1, 0.123456789]]

    in_model = Pipeline([])
    monkeypatch.setattr(Pipeline, "predict_proba", mock_predict_proba)

    data = {
        "Age": 35.0,
        "Support_Calls": 1.0,
        "Payment_Delay": 5.0,
        "Total_Spend": 900.0,
        "Subscription_Type": "Premium",
        "Contract_Length": "Annual"
    }
    
    result = make_inference(in_model, data)
    assert result == {"churn": 0.123}


def test_load_model(tmp_path):
    """Тест функции load_model"""
    import pickle
    test_model = Pipeline([('test', None)])
    p = tmp_path / "model.pkl"
    with open(p, "wb") as f:
        pickle.dump(test_model, f)
    
    loaded = load_model(str(p))
    assert isinstance(loaded, Pipeline)