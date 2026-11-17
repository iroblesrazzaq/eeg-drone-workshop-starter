"""Provided plots. Use the metrics in train.py to explain the results."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_evaluation(metrics, path):
    fig, (loss, accuracy, confusion) = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    candidates = metrics["candidates"]
    values = [row["C"] for row in candidates]
    for split, label in (("train", "Train"), ("val", "Validation")):
        loss.plot(values, [row[f"{split}_loss"] for row in candidates], "o-", label=label)
    loss.axvline(metrics["selected_C"], color="0.7", linestyle="--")
    loss.set(xscale="log", xlabel="C (inverse regularization strength)", ylabel="Log loss",
             title="Choose C on validation")
    loss.legend(frameon=False)

    names = ("Train", "Validation", "Test")
    scores = [metrics[f"{name}_accuracy"] for name in ("train", "val", "test")]
    bars = accuracy.bar(names, scores, color=("#5d8396", "#5d8396", "#800028"))
    accuracy.bar_label(bars, labels=[f"{s:.0%}" for s in scores], padding=3)
    accuracy.set(ylim=(0, 1.12), ylabel="Window accuracy", title="Selected model")

    matrix = np.asarray(metrics["test_confusion_matrix"])
    confusion.imshow(matrix, cmap="Blues", vmin=0)
    for (row, col), count in np.ndenumerate(matrix):
        confusion.text(col, row, str(count), ha="center", va="center",
                       color="white" if count > matrix.max() * 0.65 else "black")
    confusion.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Feet", "Hands"],
                  yticklabels=["Feet", "Hands"], xlabel="Predicted", ylabel="Actual",
                  title="Test confusion matrix")
    fig.savefig(path, dpi=150)
    plt.close(fig)
