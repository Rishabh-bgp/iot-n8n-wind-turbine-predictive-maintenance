"""Call the API in-process with FastAPI's TestClient."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastapi.testclient import TestClient
from src.api import app
from src.data import synthetic_scada

c = TestClient(app)
df = synthetic_scada(48).reset_index().rename(columns={"index": "timestamp"})
df["timestamp"] = df["timestamp"].astype(str)
r = c.post("/forecast", json={"turbine_id": "WTG-01", "readings": df.to_dict("records")}); print(r.status_code, r.json())
r = c.post("/health-score", json={"turbine_id": "WTG-01", "vibration_rms": 6.0, "gearbox_temp": 80}); print(r.status_code, r.json())
