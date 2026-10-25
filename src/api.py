"""FastAPI forecasting + health-scoring service called by the n8n workflow.
Run:  uvicorn src.api:app --port 8000   (after `python -m src.train_eval`)"""
import os
import numpy as np, pandas as pd, torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .data import add_features
from .models import CNNLSTM
from .health import health_index

app = FastAPI(title="Wind turbine forecasting & health API")
_ck, _model = None, None


def load(path=os.environ.get("MODEL_PATH", "models/cnn_lstm.pt")):
    global _ck, _model
    if _model is None:
        if not os.path.exists(path):
            raise HTTPException(503, "model not trained yet: run python -m src.train_eval")
        _ck = torch.load(path, weights_only=False); _model = CNNLSTM(_ck["n_features"])
        _model.load_state_dict(_ck["state"]); _model.eval()
    return _ck, _model


class Reading(BaseModel):
    timestamp: str
    wind_speed: float
    wind_direction: float = 0.0
    temperature: float = 25.0
    pressure: float = 1010.0
    humidity: float = 60.0
    power: float = 0.0


class ForecastRequest(BaseModel):
    turbine_id: str
    readings: list[Reading]


class HealthRequest(BaseModel):
    turbine_id: str
    vibration_rms: float
    gearbox_temp: float
    power_residual: float = 0.0


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    ck, m = load()
    df = pd.DataFrame([r.model_dump() for r in req.readings])
    df.index = pd.to_datetime(df.pop("timestamp")); df = add_features(df)[ck["columns"]]
    if len(df) < ck["lookback"]:
        raise HTTPException(422, f"need at least {ck['lookback']} hourly readings")
    x = ((df.values[-ck["lookback"]:] - ck["x_mean"]) / ck["x_std"]).astype("float32")[None]
    with torch.no_grad():
        y = float(m(torch.tensor(x))) * ck["y_std"] + ck["y_mean"]
    return {"turbine_id": req.turbine_id, "horizon_h": 1, "wind_speed_forecast": round(y, 3)}


@app.post("/health-score")
def health(req: HealthRequest):
    out = health_index(req.model_dump()); out["turbine_id"] = req.turbine_id
    out["anomaly"] = out["action"] != "nominal"; return out
