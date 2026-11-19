"""Stage 3 - conventional remote-sensing baseline: random forest on band statistics + NDVI.

Runs logged to MLflow:
    rf_unweighted          spatial split, no imbalance handling
    rf_balanced            spatial split, class_weight="balanced"
    rf_balanced_randsplit  same model on a RANDOM split with the same class sizes, to measure how
                           much near-identical neighbouring patches inflate test scores

Run:  python -m pipeline.baseline
"""
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from pipeline.config import BANDS, CLASSES, EXPERIMENT, SEED, TRACKING, WORK_DIR
from pipeline.data import band_features, feature_names
from pipeline.evaluate import describe_imbalance, log_evaluation


def all_features():
    """Features for every patch, cached because percentiles over 27k x 4096 x 13 values take a while."""
    cache = WORK_DIR / "features.npy"
    if cache.exists():
        return np.load(cache)
    patches = np.load(WORK_DIR / "patches.npy", mmap_mode="r")
    F = np.concatenate([band_features(np.asarray(patches[i:i + 1000])) for i in range(0, len(patches), 1000)])
    np.save(cache, F)
    return F


def random_split_like(splits, labels):
    """A random (non-spatial) split with exactly the same per-class sizes as the spatial one."""
    rng = np.random.default_rng(SEED)
    out = pd.Series("unused", index=splits.index)
    for c in range(len(CLASSES)):
        idx = rng.permutation(np.flatnonzero(labels == c))
        n = splits.loc[labels == c, "split"].value_counts()
        n_test, n_val, n_train = n.get("test", 0), n.get("val", 0), n.get("train", 0)
        out.iloc[idx[:n_test]] = "test"
        out.iloc[idx[n_test:n_test + n_val]] = "val"
        out.iloc[idx[n_test + n_val:n_test + n_val + n_train]] = "train"
    return out


def run(name, F, y, split, class_weight, tags):
    tr, va, te = (split == "train").to_numpy(), (split == "val").to_numpy(), (split == "test").to_numpy()
    with mlflow.start_run(run_name=name):
        model = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt",
                                       class_weight=class_weight, n_jobs=-1, random_state=SEED)
        model.fit(F[tr], y[tr])
        mlflow.set_tags({"family": "baseline", "model": "random_forest", **tags})
        mlflow.log_params({"model": "random_forest", "bands": "ms", "features": "band_stats+ndvi",
                           "loss": "weighted" if class_weight else "none", "n_estimators": 400,
                           "n_features": F.shape[1], **describe_imbalance(y[tr])})
        val = log_evaluation(mlflow, y[va], model.predict_proba(F[va]), "val")
        test = log_evaluation(mlflow, y[te], model.predict_proba(F[te]), "test")
        imp = pd.Series(model.feature_importances_, index=feature_names(BANDS)).sort_values(ascending=False)
        mlflow.log_dict(imp.round(5).to_dict(), "feature_importance.json")
        print(f"{name:24} val macro-F1={val['macro_f1']:.3f}  test macro-F1={test['macro_f1']:.3f}  "
              f"minority F1={test['minority_macro_f1']:.3f}  acc={test['accuracy']:.3f}")
        return test


def main():
    mlflow.set_tracking_uri(TRACKING)
    mlflow.set_experiment(EXPERIMENT)

    F = all_features()
    y = pd.read_csv(WORK_DIR / "metadata.csv")["label"].to_numpy()
    splits = pd.read_csv(WORK_DIR / "splits.csv")

    run("rf_unweighted", F, y, splits["split"], None, {"split": "spatial"})
    spatial = run("rf_balanced", F, y, splits["split"], "balanced", {"split": "spatial"})
    rand = run("rf_balanced_randsplit", F, y, random_split_like(splits, y), "balanced", {"split": "random"})
    print(f"Random split inflates test macro-F1 by {rand['macro_f1'] - spatial['macro_f1']:+.3f}")


if __name__ == "__main__":
    main()
