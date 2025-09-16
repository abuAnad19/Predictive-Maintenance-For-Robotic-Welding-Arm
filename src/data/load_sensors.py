from pathlib import Path
import numpy as np
import pandas as pd
from src.config import SENSORS_RAW_DIR, SENSOR_WINDOW, SENSOR_STEP, SENSOR_FEATURES, RANDOM_STATE

def _window_array(arr, window, step):
    starts = np.arange(0, len(arr) - window + 1, step)
    return np.stack([arr[s:s+window] for s in starts], axis=0)

def _synth_sensor_df(n_samples=5000, n_faults=3, seed=RANDOM_STATE):
    """
    Generate synthetic sensor data with guaranteed balanced faults.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)

    vib = 0.5*np.sin(2*np.pi*0.01*t) + 0.1*rng.standard_normal(n_samples)
    cur = 5 + 0.5*np.sin(2*np.pi*0.03*t) + 0.2*rng.standard_normal(n_samples)
    tmp = 40 + 0.05*t + 0.5*rng.standard_normal(n_samples)
    label = np.zeros(n_samples, dtype=int)

    # Inject faults at evenly spaced intervals
    fault_positions = np.linspace(500, n_samples-500, n_faults, dtype=int)
    for start in fault_positions:
        vib[start:start+500] += 0.5
        cur[start:start+500] += 0.5
        tmp[start:start+500] += 5
        label[start:start+500] = 1

    return pd.DataFrame({"vibration": vib, "current": cur, "temp": tmp, "label": label})

def load_sensor_windows():
    csvs = list(Path(SENSORS_RAW_DIR).glob("*.csv"))
    if csvs:
        df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    else:
        df = _synth_sensor_df()

    X = df[SENSOR_FEATURES].values
    y = df["label"].values
    Xw = _window_array(X, SENSOR_WINDOW, SENSOR_STEP)           # [N, W, F]
    yw = _window_array(y, SENSOR_WINDOW, SENSOR_STEP)[:, -1]    # label at window end
    return Xw, yw
