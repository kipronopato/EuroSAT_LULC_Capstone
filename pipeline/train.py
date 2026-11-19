"""Stage 4 - train one CNN configuration and log everything to MLflow.

Examples (run from capstone_eurosat/):
    python -m pipeline.train --model cnn    --bands rgb --loss ce
    python -m pipeline.train --model cnn    --bands ms  --loss weighted_ce
    python -m pipeline.train --model effnet --bands rgb --loss focal
    python -m pipeline.train --model effnet --bands rgb --loss weighted_ce --train-set full   # balanced reference

Every run uses flip/rotation augmentation, early-stops on VALIDATION macro F1, and reports on the
untouched test set. The trained model is saved to artifacts/models/<run_name>.keras.
"""
import argparse
import time

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from pipeline.config import CLASSES, EXPERIMENT, SEED, TRACKING, WORK_DIR
from pipeline.data import load_split, make_dataset
from pipeline.evaluate import describe_imbalance, log_evaluation
from pipeline.models import custom_cnn, efficientnet_b0, unfreeze_top


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "effnet"], required=True)
    p.add_argument("--bands", choices=["rgb", "ms"], default="rgb")
    p.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default="ce")
    p.add_argument("--train-set", choices=["imbalanced", "full"], default="imbalanced")
    p.add_argument("--epochs", type=int, default=25, help="max epochs (custom CNN)")
    p.add_argument("--head-epochs", type=int, default=100, help="max epochs for the head on frozen features (EfficientNet)")
    p.add_argument("--ft-epochs", type=int, default=12, help="max fine-tuning epochs (EfficientNet)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ft-lr", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=64, help="EfficientNet input size (patches upsampled if > 64)")
    p.add_argument("--unfreeze", type=int, default=40, help="EfficientNet layers unfrozen for fine-tuning")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--limit", type=int, default=0, help="debug: use only this many patches per split")
    a = p.parse_args()
    if a.model == "effnet" and a.bands != "rgb":
        p.error("EfficientNet's ImageNet weights expect 3-band RGB")
    return a


class MLflowEpochLogger(keras.callbacks.Callback):
    def __init__(self, prefix, offset=0):
        super().__init__()
        self.prefix, self.offset = prefix, offset

    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({f"{self.prefix}{k}": float(v) for k, v in (logs or {}).items()},
                           step=self.offset + epoch)


def make_loss(name):
    if name == "focal":
        return keras.losses.CategoricalFocalCrossentropy(gamma=2.0)
    return keras.losses.CategoricalCrossentropy()


def compile_model(model, lr, loss):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=make_loss(loss),
                  metrics=["accuracy", keras.metrics.F1Score(average="macro", name="macro_f1")])


def fit(model, train_ds, val_ds, epochs, class_weight, patience, prefix, offset=0):
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, patience // 2)),
        MLflowEpochLogger(prefix, offset),
    ]
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight,
                     callbacks=callbacks, verbose=2).history


def curves_figure(histories):
    h = {k: sum((hh.get(k, []) for hh in histories), []) for k in ("loss", "val_loss", "macro_f1", "val_macro_f1")}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, (metric, title) in enumerate([("loss", "Loss"), ("macro_f1", "Macro F1")]):
        ax[i].plot(h[metric], label="Train")
        ax[i].plot(h[f"val_{metric}"], label="Validation")
        if len(histories) > 1:
            ax[i].axvline(len(histories[0]["loss"]) - 0.5, ls="--", c="grey", label="start fine-tuning")
        ax[i].set_title(title)
        ax[i].set_xlabel("Epoch")
        ax[i].legend()
    fig.tight_layout()
    return fig


def main():
    a = parse_args()
    keras.utils.set_random_seed(SEED)
    run_name = f"{a.model}_{a.bands}_{a.loss}" + ("_fulltrain" if a.train_set == "full" else "")

    # Test data is only loaded after training, to keep peak memory low on an 8 GB laptop
    X_tr, y_tr = load_split("train", a.train_set, a.bands)
    X_va, y_va = load_split("val", bands=a.bands)
    sub = lambda X, y: (X, y)
    if a.limit:                                   # debug: random subset of each split (rows are sorted by class)
        rng = np.random.default_rng(SEED)
        def sub(X, y):
            i = rng.choice(len(y), size=min(a.limit, len(y)), replace=False)
            return X[i], y[i]
        X_tr, y_tr = sub(X_tr, y_tr)
        X_va, y_va = sub(X_va, y_va)
    n_channels = X_tr.shape[-1]

    class_weight = None
    if a.loss == "weighted_ce":
        w = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=y_tr)
        class_weight = dict(enumerate(w))

    train_ds = make_dataset(X_tr, y_tr, a.batch, augment=True, shuffle=True, seed=SEED)
    val_ds   = make_dataset(X_va, y_va, 256)

    mlflow.set_tracking_uri(TRACKING)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"family": "cnn", "model": a.model, "split": "spatial"})
        mlflow.log_params({**vars(a), "n_channels": n_channels, **describe_imbalance(y_tr),
                           "augmentation": "flip_lr+flip_ud+rot90"})
        if class_weight:
            mlflow.log_dict({CLASSES[k]: round(v, 4) for k, v in class_weight.items()}, "class_weights.json")

        start = time.time()
        if a.model == "cnn":
            adapt_ds = make_dataset(X_tr, y_tr, 512).map(lambda x, t: x)     # training data only
            model = custom_cnn(n_channels, adapt_ds)
            compile_model(model, a.lr, a.loss)
            histories = [fit(model, train_ds, val_ds, a.epochs, class_weight, a.patience, "epoch_")]
        else:
            model, base, extractor, head = efficientnet_b0(a.img_size)
            # Stage 1: frozen base. Its features never change, so compute them once and train the head on them.
            feats = lambda X, y: tf.data.Dataset.from_tensor_slices(
                (extractor.predict(make_dataset(X, y, 256), verbose=0), tf.one_hot(y, len(CLASSES))))
            f_tr = feats(X_tr, y_tr).shuffle(len(y_tr), seed=SEED).batch(a.batch)
            f_va = feats(X_va, y_va).batch(256)
            compile_model(head, a.lr, a.loss)
            h1 = fit(head, f_tr, f_va, a.head_epochs, class_weight, 2 * a.patience, "frozen_")
            unfreeze_top(base, a.unfreeze)                              # stage 2: partial fine-tuning, with augmentation
            compile_model(model, a.ft_lr, a.loss)
            h2 = fit(model, train_ds, val_ds, a.ft_epochs, class_weight, a.patience, "finetune_", len(h1["loss"]))
            histories = [h1, h2]
        mlflow.log_metric("train_minutes", (time.time() - start) / 60)
        mlflow.log_metric("n_params", model.count_params())

        fig = curves_figure(histories)
        mlflow.log_figure(fig, "training_curves.png")
        plt.close(fig)

        val  = log_evaluation(mlflow, y_va, model.predict(val_ds, verbose=0), "val")
        del train_ds, X_tr
        X_te, y_te = sub(*load_split("test", bands=a.bands))
        test = log_evaluation(mlflow, y_te, model.predict(make_dataset(X_te, y_te, 256), verbose=0), "test")

        out = WORK_DIR / "models" / f"{run_name}.keras"
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(out)
        mlflow.log_artifact(str(out), artifact_path="keras_model")
        mlflow.set_tag("local_model_path", str(out))

        print(f"\n{run_name}: val macro-F1={val['macro_f1']:.3f} | test macro-F1={test['macro_f1']:.3f} "
              f"minority F1={test['minority_macro_f1']:.3f} acc={test['accuracy']:.3f} "
              f"({(time.time() - start) / 60:.1f} min, run {run.info.run_id[:8]})")


if __name__ == "__main__":
    main()
