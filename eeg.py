"""Shared EEG settings and feature extraction for training and flight."""
import numpy as np
from scipy.signal import welch

CHANNELS = ("C3", "Cz", "C4")
BANDS = ((8, 12), (13, 30))  # mu, beta (Hz)
SFREQ = 160
WINDOW_SECONDS = 2
WINDOW = SFREQ * WINDOW_SECONDS
FEET, HANDS = 0, 1


def spectrum(window):
    """Return frequencies (Hz) and power density (V²/Hz), one column per channel."""
    return welch(window, fs=SFREQ, nperseg=SFREQ, axis=0)


def extract_features(window):
    """Return six log band powers in channel-first order.

    Use spectrum(window), then sum PSD bins with low <= f < high for each
    band and channel. Multiply by the bin width, add 1e-20, and take the
    natural log. Order: C3 mu, C3 beta, Cz mu, Cz beta, C4 mu, C4 beta.
    """
    # TODO 1: Integrate the PSD in each band and return the flattened log powers.
    raise NotImplementedError("TODO 1 in eeg.py: extract six features")
