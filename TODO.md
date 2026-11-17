# Student TODO checklist

- **Setup** — Run `python prepare_data.py` from the repo root.
- **1 · Features** ([`eeg.py`](eeg.py)) — Use the provided Welch spectrum to compute mu and beta power for C3, Cz, and C4. Take the log and return six values in channel order.
- **2a · Fit** ([`training/train.py`](training/train.py)) — Fit a pipeline with `StandardScaler` and `LogisticRegression(C=C)` on the training data.
- **2b · Select** ([`training/train.py`](training/train.py)) — Return the candidate with the lowest validation log loss. Run `python -m training.train` and open `artifacts/evaluation.png`.
- **3a · Cadence** ([`game/play.py`](game/play.py)) — Set the time between predictions in seconds. Try `0.25` first.
- **3b · Flight** ([`game/play.py`](game/play.py)) — Average recent probabilities from the current cue. Return the smoothed probability and a velocity: +0.65 for up, −0.65 for down, or 0 for hover. Try averaging three predictions with thresholds of 0.4 and 0.6.
- **Try it** — Run `python -m game.play`. Change either the cadence, averaging length, or thresholds and explain the effect on flight.
