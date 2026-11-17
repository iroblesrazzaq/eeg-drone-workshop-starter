"""Download PhysioNet subject 1 and prepare four reproducible splits."""
import argparse
import json
from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci

from eeg import CHANNELS, FEET, HANDS, SFREQ, WINDOW

SUBJECT = 1
# Review together: run, start and stop in seconds. The cut is during rest.
SPLITS = {"train": (6, 0, None), "val": (10, 0, None),
          "test": (14, 0, 68), "drone": (14, 68, None)}
ONSET_DELAY = SFREQ // 2
WINDOW_STEP = SFREQ // 2


def read_recording(path):
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    eegbci.standardize(raw)
    raw.pick(list(CHANNELS)).reorder_channels(list(CHANNELS))
    if raw.info["sfreq"] != SFREQ:
        raise ValueError(f"Expected {SFREQ} Hz in {path}")
    labels = {"T1": HANDS, "T2": FEET}
    cues = [(round(onset * SFREQ), round((onset + duration) * SFREQ), labels[cue])
            for onset, duration, cue in zip(raw.annotations.onset, raw.annotations.duration,
                                            raw.annotations.description)
            if cue in labels]
    return raw.get_data().T, cues


def split_recording(samples, cues, start, end):
    # Filter each split separately. Forward IIR uses only past and present samples.
    filtered = mne.filter.filter_data(
        samples[start:end].T, SFREQ, 7, 30, method="iir", phase="forward",
        iir_params={"order": 4, "ftype": "butter", "output": "sos"}, verbose="ERROR",
    ).T.astype(np.float32)
    local_cues = [(a - start, b - start, y) for a, b, y in cues if start <= a < b <= end]
    if len({label for _, _, label in local_cues}) != 2:
        raise ValueError("Each split needs both hands and feet cues")
    return filtered, np.asarray(local_cues, dtype=int)


def make_windows(samples, cues):
    windows, labels, trials = [], [], []
    for trial_id, (start, stop, label) in enumerate(cues):
        for index in range(start + ONSET_DELAY, stop - WINDOW + 1, WINDOW_STEP):
            windows.append(samples[index:index + WINDOW])
            labels.append(label)
            trials.append(trial_id)
    return np.stack(windows), np.asarray(labels), np.asarray(trials)


def prepare(data_dir):
    raw_dir, out = data_dir / "raw", data_dir / "prepared"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    run_ids = sorted({split[0] for split in SPLITS.values()})
    paths = eegbci.load_data(SUBJECT, run_ids, path=raw_dir, update_path=False)
    runs = {run: read_recording(path) for run, path in zip(run_ids, paths)}
    manifest = {}
    for name, (run, start, stop) in SPLITS.items():
        samples, cues = runs[run]
        end = len(samples) if stop is None else round(stop * SFREQ)
        signal, local_cues = split_recording(samples, cues, round(start * SFREQ), end)
        if name == "drone":
            # The game receives only task start/stop times for its rest gate.
            np.savez_compressed(out / "drone.npz", samples=signal, sfreq=SFREQ,
                                channels=CHANNELS, task_intervals=local_cues[:, :2],
                                onset_delay=ONSET_DELAY)
        else:
            X, y, trial_ids = make_windows(signal, local_cues)
            np.savez_compressed(out / f"{name}.npz", X=X, y=y, trial_ids=trial_ids)
        manifest[name] = {"subject": SUBJECT, "run": run, "start_seconds": start,
                          "stop_seconds": end / SFREQ, "trials": len(local_cues)}
        print(f"{name:5} run {run:2} · {len(local_cues):2} trials · {out / (name + '.npz')}")
    (out / "splits.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    prepare(parser.parse_args().data_dir)
