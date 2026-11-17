"""Short instructor answers to the three student tasks."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg import BANDS, spectrum

PREDICT_EVERY_SECONDS = 0.25


def extract_features(window):
    frequencies, psd = spectrum(window)
    df = frequencies[1] - frequencies[0]
    powers = [psd[(low <= frequencies) & (frequencies < high)].sum(axis=0) * df
              for low, high in BANDS]
    return np.log(np.stack(powers, axis=1) + 1e-20).ravel()


def fit_candidate(X, y, C):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
    return model.fit(X, y)


def select_candidate(results):
    return min(results, key=lambda row: row["val_loss"])


def flight_command(probabilities):
    smoothed = float(np.mean(probabilities[-3:]))
    if smoothed >= 0.6:
        return smoothed, 0.65
    if smoothed <= 0.4:
        return smoothed, -0.65
    return smoothed, 0.0
