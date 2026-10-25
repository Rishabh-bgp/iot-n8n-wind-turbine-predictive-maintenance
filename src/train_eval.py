"""Compare persistence, ARIMA, LSTM, CNN-LSTM and an inverse-RMSE weighted ensemble;
save the CNN-LSTM for the forecasting API used by the n8n workflow."""
import argparse, json, os, warnings
import numpy as np, torch
from .data import load_scada_csv, synthetic_scada, add_features, make_windows, chrono_split, Scaler
from .metrics import regression_metrics
from .models import LSTMNet, CNNLSTM
from .trainutil import fit, predict
from .health import health_index, simulate_degradation


def arima_forecast(series, n_train, order=(2, 0, 1)):
    from statsmodels.tsa.arima.model import ARIMA
    warnings.filterwarnings("ignore")
    res = ARIMA(series[:n_train], order=order).fit()
    full = res.apply(series)                                  # same params, filtered over the whole series
    return full.predict(start=n_train, end=len(series) - 1)   # one-step-ahead predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/synthetic_scada.csv" if os.path.exists("data/synthetic_scada.csv") else None, help="SCADA-style CSV (default: committed synthetic CSV)"); ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results"); a = ap.parse_args()
    if a.quick: a.epochs = 5
    raw = load_scada_csv(a.csv) if a.csv else synthetic_scada(24 * 90)
    df = add_features(raw); cols = list(df.columns)
    X, y = make_windows(df, lookback=a.lookback)
    tr, va, te = chrono_split(len(X))
    sc = Scaler().fit(X[tr]); Xs = sc.transform(X).astype("float32")
    ym, ys = float(y[tr].mean()), float(y[tr].std()); yn = ((y - ym) / ys).astype("float32")
    preds_va, preds_te, res = {}, {}, {}
    ws = df["wind_speed"].values; off = a.lookback          # window i targets ws[off+i]
    ar = arima_forecast(ws, off + va.start)
    preds_va["ARIMA"], preds_te["ARIMA"] = ar[: va.stop - va.start], ar[va.stop - va.start:]
    models = {}
    for name, M in {"LSTM": LSTMNet, "CNN-LSTM": CNNLSTM}.items():
        m = fit(M(X.shape[2]), Xs[tr], yn[tr], Xs[va], yn[va], epochs=a.epochs, patience=10); models[name] = m
        preds_va[name] = predict(m, Xs[va]) * ys + ym; preds_te[name] = predict(m, Xs[te]) * ys + ym
    rm = {k: regression_metrics(y[va], v)["RMSE"] for k, v in preds_va.items()}
    w = {k: (1 / r) / sum(1 / x for x in rm.values()) for k, r in rm.items()}
    preds_te["Ensemble"] = sum(w[k] * preds_te[k] for k in w)
    res["persistence"] = regression_metrics(y[te], X[te][:, -1, cols.index("wind_speed")])
    for k, p in preds_te.items(): res[k] = regression_metrics(y[te], p)
    res["ensemble_weights"] = w
    hs = [health_index(s) for s in simulate_degradation()]
    res["health_demo"] = {"first_alert_step": next((i for i, h in enumerate(hs) if h["action"] != "nominal"), None),
                          "final_health_index": hs[-1]["health_index"]}
    os.makedirs("models", exist_ok=True)
    torch.save({"state": models["CNN-LSTM"].state_dict(), "n_features": X.shape[2], "columns": cols,
                "lookback": a.lookback, "x_mean": sc.m, "x_std": sc.s, "y_mean": ym, "y_std": ys}, "models/cnn_lstm.pt")
    os.makedirs(a.out, exist_ok=True); json.dump(res, open(f"{a.out}/results.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
