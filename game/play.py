"""Replay reserved EEG into a trained classifier and fly the drone."""
import argparse
import csv
from pathlib import Path

import joblib
import numpy as np

from eeg import CHANNELS, HANDS, SFREQ, WINDOW, extract_features
from game.drone import Drone, show_game

# TODO 3a: Choose the number of seconds between classifier calls, e.g. 0.25.
PREDICT_EVERY_SECONDS = None


def flight_command(probabilities):
    """Smooth P(hands), then return (smoothed, velocity).

    probabilities contains predictions so far in the current task period.
    Try averaging the last 3, then choose two thresholds. Positive velocity
    goes up, negative goes down, and zero hovers. Use a speed of 0.65 m/s.
    Try changing cadence, averaging length, and thresholds after the first flight.
    """
    # TODO 3b: Average recent predictions and choose up, down, or hover.
    raise NotImplementedError("TODO 3b in game/play.py: turn probabilities into movement")


class Replay:
    """Advance to the playback cursor using only EEG that has arrived."""

    def __init__(self, data, classifier, features, command, interval):
        if interval is None:
            raise NotImplementedError("TODO 3a in game/play.py: choose a prediction interval")
        if not np.isfinite(interval) or interval < 1 / SFREQ:
            raise ValueError("Prediction interval must be at least one EEG sample")
        self.samples = data["samples"]
        self.intervals = data["task_intervals"]
        self.delay = int(data["onset_delay"])
        self.step = round(interval * SFREQ)
        self.classifier, self.features, self.command = classifier, features, command
        self.hands_column = list(classifier.classes_).index(HANDS)
        self.drone = Drone()
        self.cursor = 0
        self.probabilities, self.log = [], []
        self.smoothed = None
        # Cue ends are events too, so the drone hovers as soon as rest begins.
        events = set(range(self.step, len(self.samples) + 1, self.step))
        self.events = iter(sorted(events.union(self.intervals[:, 1])))
        self.next_event = next(self.events, None)

    def advance(self, target):
        while self.next_event is not None and self.next_event <= target:
            now = self.next_event
            self.drone.advance((now - self.cursor) / SFREQ)
            self.cursor = now
            active = any(start + self.delay + WINDOW <= now < stop
                         for start, stop in self.intervals)
            if not active:
                self.probabilities.clear()
                self.smoothed = None
                self.drone.velocity = 0.0
            elif now % self.step == 0:
                self.predict(now)
            self.next_event = next(self.events, None)
        self.drone.advance((target - self.cursor) / SFREQ)
        self.cursor = target

    def predict(self, now):
        window = self.samples[now - WINDOW:now]
        probability = float(self.classifier.predict_proba(self.features(window)[None, :])[0, self.hands_column])
        self.probabilities.append(probability)
        self.smoothed, self.drone.velocity = self.command(self.probabilities)
        self.log.append({"time_seconds": now / SFREQ, "p_hands": probability,
                         "smoothed": self.smoothed, "velocity": self.drone.velocity,
                         "altitude": self.drone.altitude})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--data", type=Path, default=Path("data/prepared/drone.npz"))
    parser.add_argument("--log", type=Path, default=Path("artifacts/flight.csv"))
    parser.add_argument("--reference", action="store_true", help="Use the instructor's flight policy")
    parser.add_argument("--headless", action="store_true", help="Run the full replay without a window or delays")
    args = parser.parse_args()
    try:
        # Load only local models you trained or received from a trusted source.
        saved = joblib.load(args.model)
        if not isinstance(saved, dict) or not {"classifier", "reference_features"} <= saved.keys():
            raise ValueError("Model file uses an older or unknown format. Retrain with python -m training.train")
        features, command, interval = extract_features, flight_command, PREDICT_EVERY_SECONDS
        if saved["reference_features"] or args.reference:
            from instructor import solution
            if saved["reference_features"]:
                features = solution.extract_features
            if args.reference:
                command, interval = solution.flight_command, solution.PREDICT_EVERY_SECONDS
        with np.load(args.data) as data:
            if int(data["sfreq"]) != SFREQ or tuple(data["channels"]) != CHANNELS:
                raise ValueError("Flight data does not match the expected EEG channels or sample rate")
            replay = Replay(data, saved["classifier"], features, command, interval)
        if args.headless:
            replay.advance(len(replay.samples))
        else:
            show_game(replay, SFREQ)
    except (FileNotFoundError, NotImplementedError, ValueError) as error:
        parser.exit(1, f"{error}\nSee setup and tasks in README.md.\n")

    if replay.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(replay.log[0]))
            writer.writeheader()
            writer.writerows(replay.log)
        up = sum(row["velocity"] > 0 for row in replay.log)
        down = sum(row["velocity"] < 0 for row in replay.log)
        print(f"{len(replay.log)} task predictions · {up} up · {down} down · "
              f"{len(replay.log) - up - down} hover. Log saved to {args.log}.")
    else:
        print("Replay ended before any eligible predictions.")


if __name__ == "__main__":
    main()
