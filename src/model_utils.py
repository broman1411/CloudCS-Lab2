# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.pipeline import Pipeline
from pickle import load


def make_inference(in_model: Pipeline, in_data: dict) -> dict[str, float]:
    """Return the result of predictions for in_data using in_model."""
    df = pd.DataFrame(in_data, index=[0])
    churn = float(in_model.predict_proba(df)[0][1])
    return {"churn": round(churn, 3)}


def load_model(path: str) -> Pipeline:
    """Return the model being read which stored on the path."""
    with open(path, "rb") as file:
        model: Pipeline = load(file)
    return model