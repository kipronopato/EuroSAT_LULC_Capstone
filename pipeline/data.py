"""Loading splits, RGB conversion, hand-crafted features and tf.data pipelines."""
import numpy as np
import pandas as pd

from pipeline.config import CLASSES, NIR, RED, RGB_IDX, RGB_MAX_REFLECTANCE, WORK_DIR


def load_split(split, train_set="imbalanced", bands="ms"):
    """Return (X, y) for 'train', 'val' or 'test'. X is uint16 (N, 64, 64, 13) for bands='ms'
    or uint8 true colour (N, 64, 64, 3) for bands='rgb' (converted in chunks to save memory).

    train_set='full' returns the whole (roughly balanced) training pool instead of the
    imbalanced subsample; it is only used for the balanced reference run.
    """
    meta   = pd.read_csv(WORK_DIR / "metadata.csv")
    splits = pd.read_csv(WORK_DIR / "splits.csv")
    if split == "train" and train_set == "full":
        mask = splits["train_full"].to_numpy()
    else:
        mask = (splits["split"] == split).to_numpy()
    idx = splits.loc[mask, "idx"].to_numpy()
    y = meta.loc[idx, "label"].to_numpy()
    if bands == "ms":
        return DiskPatches(idx), y
    patches = np.load(WORK_DIR / "patches.npy", mmap_mode="r")
    return np.concatenate([to_rgb8(patches[idx[i:i + 1000]]) for i in range(0, len(idx), 1000)]), y


class DiskPatches:
    """13-band patches left on disk (artifacts/patches.npy) and read batch by batch, because a
    full 13-band split does not fit alongside everything else in 8 GB of RAM."""

    def __init__(self, idx):
        self.idx = np.asarray(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):                    # subsetting returns another lazy view
        return DiskPatches(self.idx[i])

    @property
    def shape(self):
        return (len(self.idx), 64, 64, 13)

    @staticmethod
    def read(rows):
        patches = np.load(WORK_DIR / "patches.npy", mmap_mode="r")
        return np.asarray(patches[np.asarray(rows)])


def rgb_reflectance_to_8bit(rgb):
    return np.clip(rgb.astype(np.float32) / RGB_MAX_REFLECTANCE * 255.0, 0, 255).astype(np.uint8)


def to_rgb8(x13):
    """13-band reflectance (..., 13) -> 8-bit true-colour (..., 3). Used in training AND in the API."""
    return rgb_reflectance_to_8bit(x13[..., RGB_IDX])


def band_features(X):
    """Conventional remote-sensing features: per-band statistics plus NDVI statistics."""
    X = X.astype(np.float32)
    flat = X.reshape(len(X), -1, X.shape[-1])
    feats = [flat.mean(1), flat.std(1), np.percentile(flat, 10, axis=1), np.percentile(flat, 90, axis=1)]
    ndvi = (X[..., NIR] - X[..., RED]) / (X[..., NIR] + X[..., RED] + 1e-6)
    ndvi = ndvi.reshape(len(X), -1)
    feats.append(np.stack([ndvi.mean(1), ndvi.std(1),
                           np.percentile(ndvi, 10, axis=1), np.percentile(ndvi, 90, axis=1)], axis=1))
    return np.concatenate(feats, axis=1)


def feature_names(bands):
    names = [f"{b}_{s}" for s in ("mean", "std", "p10", "p90") for b in bands]
    return names + ["NDVI_mean", "NDVI_std", "NDVI_p10", "NDVI_p90"]


def make_dataset(X, y, batch_size, augment=False, shuffle=False, seed=42):
    """tf.data pipeline. Labels are one-hot so the same pipeline serves CE, weighted CE and focal loss.

    Augmentation uses all eight flip/rotate orientations. Unlike the chest X-rays in Lab 07,
    a top-down satellite patch has no 'correct' orientation, so every flip is a valid image.
    """
    import tensorflow as tf

    if isinstance(X, DiskPatches):
        # Shuffle/batch the row indices, read each batch from disk, then augment image by image
        ds = tf.data.Dataset.from_tensor_slices((X.idx, tf.one_hot(y, len(CLASSES))))
        if shuffle:
            ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
        read = lambda rows, t: (tf.ensure_shape(tf.numpy_function(DiskPatches.read, [rows], tf.uint16),
                                                [None, 64, 64, 13]), t)
        ds = ds.batch(batch_size).map(read).unbatch()
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, tf.one_hot(y, len(CLASSES))))
        if shuffle:
            ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)

    def prep(x, t):
        x = tf.cast(x, tf.float32)
        if augment:
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_flip_up_down(x)
            x = tf.image.rot90(x, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        return x, t

    return ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
