"""Fit and select a classifier, score the test split, and save it for flight."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss

from eeg import extract_features
from training.plots import save_evaluation

C_VALUES = (0.1, 1.0, 10.0)


def fit_candidate(X, y, C):
    """Fit StandardScaler → LogisticRegression(C=C).

    Import make_pipeline, StandardScaler, and LogisticRegression from sklearn.
    Return the fitted pipeline. X and y are training data.
    """
    # TODO 2a: Fit the scaler and classifier together using X and y.
    raise NotImplementedError("TODO 2a in training/train.py: fit the pipeline")


def select_candidate(results):
    """Return the candidate with the lowest validation loss."""
    # TODO 2b: Select the row with the lowest val_loss.
    raise NotImplementedError("TODO 2b in training/train.py: select by validation loss")


def load_features(data_dir, split, features):
    with np.load(data_dir / f"{split}.npz") as data:
        return np.stack([features(window) for window in data["X"]]), data["y"]


def train(data_dir, out, features, fit, select, *, reference=False):
    X_train, y_train = load_features(data_dir, "train", features)
    X_val, y_val = load_features(data_dir, "val", features)
    results = []
    for C in C_VALUES:
        model = fit(X_train, y_train, C)
        results.append(dict(C=C, model=model,
                            train_loss=log_loss(y_train, model.predict_proba(X_train)),
                            val_loss=log_loss(y_val, model.predict_proba(X_val))))

    chosen = select(results)
    model = chosen["model"]
    X_test, y_test = load_features(data_dir, "test", features)
    test_predictions = model.predict(X_test)
    matrix = confusion_matrix(y_test, test_predictions, labels=[0, 1]).tolist()
    metrics = {
        "selected_C": chosen["C"],
        "train_accuracy": float(model.score(X_train, y_train)),
        "val_accuracy": float(model.score(X_val, y_val)),
        "test_accuracy": float(model.score(X_test, y_test)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, test_predictions)),
        "test_confusion_matrix": matrix,
        "candidates": [{key: row[key] for key in ("C", "train_loss", "val_loss")}
                       for row in results],
    }
    out.mkdir(parents=True, exist_ok=True)
    save_evaluation(metrics, out / "evaluation.png")
    joblib.dump({"classifier": model, "reference_features": reference}, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--reference", action="store_true", help="Run the instructor solution")
    args = parser.parse_args()
    features, fit, select = extract_features, fit_candidate, select_candidate
    if args.reference:
        from instructor import solution
        features, fit, select = solution.extract_features, solution.fit_candidate, solution.select_candidate
    try:
        metrics = train(args.data_dir, args.out, features, fit, select, reference=args.reference)
    except (FileNotFoundError, NotImplementedError) as error:
        parser.exit(1, f"{error}\nSee setup and tasks in README.md.\n")
    for split in ("train", "val", "test"):
        print(f"{split:5} accuracy: {metrics[f'{split}_accuracy']:.1%}")
    print(f"Selected C={metrics['selected_C']:g}. Model and evaluation saved to {args.out}/")


if __name__ == "__main__":
    main()
