"""Simplified turbine health index + alert logic (paper section VIII), computed from
vibration RMS, gearbox oil temperature and power-curve residual. Thresholds are illustrative."""
import numpy as np

LIMITS = {"vibration_rms": (2.0, 7.1), "gearbox_temp": (55.0, 85.0), "power_residual": (0.0, 0.25)}
WEIGHTS = {"vibration_rms": 0.5, "gearbox_temp": 0.3, "power_residual": 0.2}
ALERT, CRITICAL = 0.6, 0.4


def health_index(sample):
    """Returns 1 (healthy) .. 0 (failed) plus per-signal sub-scores and a recommended action."""
    subs = {}
    for k, (good, bad) in LIMITS.items():
        v = float(sample.get(k, good))
        subs[k] = float(np.clip(1 - (abs(v) - good) / (bad - good), 0, 1))
    hi = sum(WEIGHTS[k] * subs[k] for k in subs)
    action = "critical_shutdown_review" if hi < CRITICAL else "schedule_maintenance" if hi < ALERT else "nominal"
    return {"health_index": round(hi, 4), "sub_scores": subs, "action": action}


def simulate_degradation(n=500, seed=0):
    rng = np.random.default_rng(seed); t = np.linspace(0, 1, n)
    return [{"vibration_rms": 2 + 5 * t[i] ** 2 + rng.normal(0, .3), "gearbox_temp": 60 + 22 * t[i] + rng.normal(0, 1),
             "power_residual": 0.2 * t[i] + abs(rng.normal(0, .02))} for i in range(n)]
