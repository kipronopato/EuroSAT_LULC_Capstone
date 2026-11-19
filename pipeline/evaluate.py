"""Evaluation shared by the baseline and the CNNs: macro F1, per-class precision/recall, confusion matrix."""
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from pipeline.config import CLASSES, IMBALANCE_RATIO

# The four classes made rarest in the training set; "minority macro F1" averages over these
MINORITY = CLASSES[-4:]


def scores(y_true, proba):
    pred = proba.argmax(1)
    f1_per = f1_score(y_true, pred, average=None, labels=range(len(CLASSES)), zero_division=0)
    top3 = np.argsort(-proba, axis=1)[:, :3]
    return {
        "macro_f1":          f1_score(y_true, pred, average="macro"),
        "minority_macro_f1": float(np.mean(f1_per[-len(MINORITY):])),
        "accuracy":          accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "top3_accuracy":     float(np.mean([t in row for t, row in zip(y_true, top3)])),
    }


def per_class_table(y_true, proba):
    p, r, f, n = precision_recall_fscore_support(y_true, proba.argmax(1), labels=range(len(CLASSES)),
                                                 zero_division=0)
    return pd.DataFrame({"class": CLASSES, "precision": p, "recall": r, "f1": f, "support": n})


def confusion_figure(y_true, proba, title):
    cm = confusion_matrix(y_true, proba.argmax(1), labels=range(len(CLASSES)), normalize="true")
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j] >= 0.005:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual (rows sum to 1 = recall)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def log_evaluation(mlflow, y_true, proba, split):
    """Log metrics, the per-class table and the confusion matrix for one split to the active run."""
    m = scores(y_true, proba)
    mlflow.log_metrics({f"{split}_{k}": v for k, v in m.items()})
    table = per_class_table(y_true, proba)
    mlflow.log_table(table, f"{split}_per_class.json")
    for _, row in table.iterrows():
        mlflow.log_metric(f"{split}_recall_{row['class']}", row["recall"])
        mlflow.log_metric(f"{split}_f1_{row['class']}", row["f1"])
    fig = confusion_figure(y_true, proba, f"{split} confusion matrix (row-normalised)")
    mlflow.log_figure(fig, f"{split}_confusion_matrix.png")
    plt.close(fig)
    with tempfile.TemporaryDirectory() as tmp:   # raw probabilities, for the write-up notebook
        np.save(Path(tmp) / "proba.npy", proba)
        mlflow.log_artifact(str(Path(tmp) / "proba.npy"), artifact_path=f"predictions_{split}")
    return m


def describe_imbalance(y_train):
    counts = np.bincount(y_train, minlength=len(CLASSES))
    return {"train_imbalance_ratio": float(counts.max() / max(counts.min(), 1)),
            "target_imbalance_ratio": IMBALANCE_RATIO, "n_train": int(len(y_train))}
