# -*- coding: utf-8 -*-
import os
from model_utils import load_model, make_inference
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi_utils import Oauth2ClientCredentials
from pydantic import BaseModel
from keycloak_utils import get_keycloak_data


class Instance(BaseModel):
    Age: float
    Support_Calls: float
    Payment_Delay: float
    Total_Spend: float
    Subscription_Type: str
    Contract_Length: str


app = FastAPI()
keycloak_openid, token_endpoint = get_keycloak_data()
oauth2_scheme = Oauth2ClientCredentials(tokenUrl=token_endpoint)

model_path: str = os.getenv("MODEL_PATH")
if model_path is None:
    raise ValueError("The environment variable $MODEL_PATH is empty!")


async def get_token_status(token: str) -> dict:
    # Декодируем токен без проверки в Keycloak
    import jwt
    try:
        decoded = jwt.decode(token, options={"verify_signature": False})
        return {"active": True, "scope": decoded.get("scope", "")}
    except Exception as e:
        print(f"Token decode error: {e}")
        return {"active": False, "scope": ""}


async def check_token(token: str = Depends(oauth2_scheme)) -> None:
    print("---")
    print(token[:50] + "..." if len(token) > 50 else token)
    print("---")
    
    token_info = await get_token_status(token)
    
    is_active = token_info.get("active", False)
    scopes = token_info.get("scope", "").split()
    has_doinfer = "doInfer" in scopes
    
    print(f"Active: {is_active}, Scopes: {scopes}, Has doInfer: {has_doinfer}")
    
    if not is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not has_doinfer:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied - missing doInfer scope",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predictions")
async def predictions(instance: Instance,
                      token: str = Depends(check_token)) -> dict[str, float]:
    return make_inference(load_model(model_path), instance.model_dump())