import numpy as np
import pytest

from eeg import SFREQ, WINDOW
from game.play import Replay


class Classifier:
    classes_ = [1, 0]  # Exercises the label lookup, including reversed class order.

    def predict_proba(self, features):
        p = features[0, 0]
        return np.array([[p, 1 - p]])


def make_replay(interval):
    data = {
        "samples": np.full((12 * SFREQ, 3), 0.8),
        "task_intervals": np.array([[0, 5 * SFREQ], [6 * SFREQ, 11 * SFREQ]]),
        "onset_delay": 0,
    }

    def features(window):
        assert window.shape == (WINDOW, 3)
        return np.array([window[-1, 0]])

    def command(history):
        # Expose the length of the per-cue history in the returned velocity.
        return history[-1], len(history) / 20

    return Replay(data, Classifier(), features, command, interval)


def test_cadence_is_independent_of_display_and_stops_at_rest():
    whole, incremental = make_replay(0.5), make_replay(0.5)
    whole.advance(12 * SFREQ)
    for cursor in range(7, 12 * SFREQ, 7):
        incremental.advance(cursor)
    incremental.advance(12 * SFREQ)

    assert [row["time_seconds"] for row in whole.log] == [
        2, 2.5, 3, 3.5, 4, 4.5, 8, 8.5, 9, 9.5, 10, 10.5,
    ]
    for a, b in zip(whole.log, incremental.log):
        assert a == pytest.approx(b)
    assert whole.drone.altitude == pytest.approx(incremental.drone.altitude)
    assert whole.drone.velocity == 0
    assert whole.log[0]["velocity"] == whole.log[6]["velocity"] == 0.05


def test_student_cadence_and_past_only_windows():
    replay = make_replay(0.25)
    replay.samples[2 * SFREQ:] = 0.1
    replay.advance(int(2.25 * SFREQ))
    assert [row["time_seconds"] for row in replay.log] == [2, 2.25]
    assert [row["p_hands"] for row in replay.log] == [0.8, 0.1]
