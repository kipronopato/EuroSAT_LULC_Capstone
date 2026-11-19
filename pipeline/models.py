"""Keras models. Preprocessing lives INSIDE each model, so the saved file takes raw pixels
(8-bit RGB or 13-band reflectance) and the API cannot drift from what training saw."""
import keras
from keras import layers

from pipeline.config import CLASSES


def custom_cnn(n_channels, normaliser_data, dropout=0.5):
    """Three convolution + pooling blocks (32/64/128 filters) -> global average pooling -> dropout.

    `normaliser_data` is the TRAINING set only; the Normalization layer learns per-band mean and
    variance from it, so validation and test statistics never leak into preprocessing.
    """
    norm = layers.Normalization(axis=-1, name="band_normalisation")
    norm.adapt(normaliser_data)

    inputs = keras.Input(shape=(64, 64, n_channels), name="patch")
    x = norm(inputs)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name=f"custom_cnn_{n_channels}band")


def efficientnet_b0(img_size, dropout=0.3):
    """EfficientNet-B0 pretrained on ImageNet. Keras' EfficientNet rescales 0-255 RGB internally,
    so the model takes 8-bit true colour (upsampled to `img_size` if larger than 64).

    Returns (model, base, extractor, head): `extractor` maps a patch to its pooled ImageNet
    features and `head` maps features to class probabilities, so the frozen stage can train the
    head on features computed once instead of re-running the frozen base every epoch.
    """
    base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                             input_shape=(img_size, img_size, 3))
    base.trainable = False

    inputs = keras.Input(shape=(64, 64, 3), name="patch")
    x = inputs if img_size == 64 else layers.Resizing(img_size, img_size, interpolation="bilinear")(inputs)
    x = base(x, training=False)          # keep BatchNorm statistics frozen, even when fine-tuning
    features = layers.GlobalAveragePooling2D()(x)
    extractor = keras.Model(inputs, features, name="feature_extractor")

    head = keras.Sequential([keras.Input(shape=features.shape[1:]), layers.Dropout(dropout),
                             layers.Dense(len(CLASSES), activation="softmax", name="probabilities")], name="head")
    model = keras.Model(inputs, head(features), name="efficientnet_b0")
    return model, base, extractor, head


def unfreeze_top(base, n_layers):
    """Partial fine-tuning: unfreeze the last `n_layers` of the base, but leave BatchNorm frozen."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers[-n_layers:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
