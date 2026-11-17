# EEG → Drone workshop

Train a simple decoder on recorded EEG from someone imagining movement of both
hands or both feet. Use its predictions to fly a simulated drone through a
reserved recording segment.

## Setup

Use Python 3.11–3.13. Run these commands from the repo root.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python prepare_data.py
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate
it with `.venv\Scripts\activate`. The first setup downloads dependencies and
three EDF recordings. Run `prepare_data.py` again to rebuild the prepared files
from the cached recordings.

## 1. Walk through the data

`prepare_data.py` reads the original EDFs under `data/raw/`, picks C3, Cz and
C4, and applies a causal 7–30 Hz bandpass per split. It takes two-second windows
within labeled trials for training and evaluation. Inspect the MNE filter call
and the `SPLITS` table before starting.

| Split | Subject 1 recordings | Role |
| --- | --- | --- |
| Train | Run 6, 15 trials | Fit the scaler and classifier |
| Validation | Run 10, 15 trials | Choose regularization strength |
| Test | Run 14 before 68 s, 8 trials | Score the selected model |
| Drone | Run 14 after 68 s, 7 trials | Replay EEG for flight |

The boundary in run 14 is in a rest interval. Windows stay inside one trial and
one split. This tests transfer to a different run from the same person.
`data/prepared/splits.json` records the boundaries. Train, validation, and test
each have a compressed NumPy file with `X` (windows × samples × channels, volts),
`y` (feet = 0, hands = 1), and `trial_ids`. `drone.npz` holds continuous filtered
EEG at 160 Hz and task start/stop times for the rest gate.

## 2. Fill in three short exercises

The [student TODO checklist](TODO.md) lists every edit in order.

1. **Features — `eeg.py`.** Turn a 320 × 3 window into six log band powers
   using the provided spectrum helper. Both training and flight call your
   function.
2. **Model — `training/train.py`.** Fit a scaler and logistic regression
   together. Choose the candidate with the lowest validation log loss. The
   provided code sweeps three values of `C`, scores the test set, saves the
   fitted pipeline, and plots loss, accuracy, and the test confusion matrix.

   ```sh
   python -m training.train
   ```

   Open `artifacts/evaluation.png`. Identify which value of `C` you selected
   and which mistakes the decoder makes on test.

3. **Flight — `game/play.py`.** Pick how often to classify. Average recent
   probabilities from the current cue and choose when to fly up, down, or hover.
   Start with a prediction every 250 ms, a mean of the last three probabilities,
   and thresholds of 0.4 and 0.6. Change one choice after the first flight and
   explain what happened. The provided game paces recorded EEG, slices the window,
   calls the model, and applies your velocity. The cue gate hovers during rest
   and clears the prediction history between cues.

   ```sh
   python -m game.play
   ```

   Space pauses and Esc exits. Add `--headless` for fast replay in a terminal
   without a desktop. Inspect `artifacts/flight.csv` for each task prediction
   and the resulting command.

## Instructor demo

```sh
python -m training.train --reference --out artifacts/workshop-reference
python -m game.play --reference --model artifacts/workshop-reference/model.joblib \
  --log artifacts/workshop-reference/flight.csv
```

Students can use the reference model to work on flight first. To do that, run
the second command without `--reference` after filling in the flight TODOs.
Flight loads the scaler and classifier from the saved model and uses the same
feature function that trained them. Use joblib files from a trusted source.

For a 60-minute session, spend 10 minutes on the demo and data, 15 on features,
15 on training and plots, 15 on flight, and 5 on the exit questions. The
[instructor code](instructor/solution.py), [questions](docs/QUESTIONS.md), and
[answer key](docs/ANSWER_KEY.md) are included. `python -m pytest -q` runs
small playback checks without downloading EEG.

Data source: [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/),
Schalk (2009), [DOI 10.13026/C28G6P](https://doi.org/10.13026/C28G6P).
The original data are downloaded from PhysioNet and covered by the
[Open Data Commons Attribution License](https://physionet.org/content/eegmmidb/view-license/1.0.0/).
