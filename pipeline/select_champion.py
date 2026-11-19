"""Leaderboard, quality gate, registry and export of the research champion.

* The champion is chosen on VALIDATION macro F1 among CNNs trained on the imbalanced set
  (test scores are only reported, never used to choose).
* Quality gate: the champion must reach MIN_VAL_MACRO_F1, otherwise nothing is exported and the
  script exits with status 1 (so CI stops, as in fraud_mlops/train.py).
* The champion is registered in MLflow with the alias "champion" and exported with metadata
  under artifacts/research_champion/. The packaged models used by the API live under model/.

Run:  python -m pipeline.select_champion
"""
import json
import shutil
import sys

import keras
import mlflow

from pipeline.config import (
    BANDS, CLASSES, EXPERIMENT, REG_NAME, REPORT_DIR, RESEARCH_MODEL_DIR, RGB_MAX_REFLECTANCE, TRACKING,
)

MIN_VAL_MACRO_F1 = 0.80

COLUMNS = ["tags.mlflow.runName", "params.model", "params.bands", "params.loss", "params.train_set",
           "tags.split", "metrics.val_macro_f1", "metrics.test_macro_f1", "metrics.test_minority_macro_f1",
           "metrics.test_accuracy", "metrics.test_balanced_accuracy", "metrics.test_top3_accuracy",
           "metrics.train_minutes"]


def leaderboard():
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], filter_string="attributes.status = 'FINISHED'")
    runs = runs[runs["tags.mlflow.runName"].notna() & runs["metrics.val_macro_f1"].notna()]
    runs = runs.sort_values("start_time").drop_duplicates("tags.mlflow.runName", keep="last")   # latest re-run wins
    board = runs[["run_id", "tags.family", "tags.local_model_path"] + [c for c in COLUMNS if c in runs]]
    board.columns = [c.split(".")[-1] for c in board.columns]
    return board.rename(columns={"runName": "run"}).sort_values("val_macro_f1", ascending=False)


def main():
    mlflow.set_tracking_uri(TRACKING)
    board = leaderboard()
    REPORT_DIR.mkdir(exist_ok=True)
    board.drop(columns=["run_id", "local_model_path"]).to_csv(REPORT_DIR / "leaderboard.csv", index=False)
    print(board.drop(columns=["run_id", "family", "local_model_path"]).to_string(index=False, float_format="%.3f"))

    eligible = board[(board["family"] == "cnn") & (board["train_set"] == "imbalanced")]
    if eligible.empty:
        sys.exit("No CNN runs found - run pipeline.train first.")
    best = eligible.iloc[0]
    print(f"\nChampion: {best['run']}  val macro-F1={best['val_macro_f1']:.3f}  test macro-F1={best['test_macro_f1']:.3f}")

    if best["val_macro_f1"] < MIN_VAL_MACRO_F1:
        print(f"FAILED quality gate: val macro-F1 below {MIN_VAL_MACRO_F1}. Not exporting.")
        sys.exit(1)

    model = keras.models.load_model(best["local_model_path"])
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("champion", "true")
        info = mlflow.tensorflow.log_model(model, name="model")
    version = mlflow.register_model(info.model_uri, REG_NAME)
    mlflow.MlflowClient().set_registered_model_alias(REG_NAME, "champion", version.version)

    RESEARCH_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(best["local_model_path"], RESEARCH_MODEL_DIR / "model.keras")
    metadata = {
        "model_name": REG_NAME,
        "version": version.version,
        "run": best["run"],
        "run_id": best["run_id"],
        "architecture": best["model"],
        "loss": best["loss"],
        "input": {
            "bands": best["bands"],
            "shape": list(model.input_shape[1:]),
            "description": ("8-bit true colour (B04, B03, B02), reflectance / %d * 255 clipped" % RGB_MAX_REFLECTANCE
                            if best["bands"] == "rgb" else "raw Sentinel-2 L1C reflectance, 13 bands"),
            "band_order": BANDS,
        },
        "classes": CLASSES,
        "metrics": {k: round(float(best[k]), 4) for k in
                    ("val_macro_f1", "test_macro_f1", "test_minority_macro_f1", "test_accuracy", "test_top3_accuracy")},
    }
    (RESEARCH_MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"PASSED: {REG_NAME} v{version.version} (@champion) exported to {RESEARCH_MODEL_DIR}")


if __name__ == "__main__":
    main()
