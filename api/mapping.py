"""Land-cover MAP of a larger image from the patch classifier, by overlapping sliding windows.

The models classify one 64x64 patch (640 m x 640 m of Sentinel-2 at 10 m). To label every pixel of a bigger
image we slide that window across it with overlap, classify every window, and give each pixel the
centre-weighted average of the probabilities of all windows that cover it; its class is the argmax.

This is patch-level mapping, not true semantic segmentation: boundaries are only as sharp as the stride
(16-32 px, i.e. 160-320 m), and a window straddling two land covers yields a blend.
"""
import numpy as np
from PIL import Image

PATCH = 64

# Conventional land-cover colours, chosen so neighbouring classes stay distinguishable
CLASS_COLOURS = {
    "AnnualCrop":           "#f2c84b",
    "PermanentCrop":        "#e07b39",
    "Pasture":              "#b5d99c",
    "HerbaceousVegetation": "#6aa84f",
    "Forest":               "#1e5631",
    "Residential":          "#d93f3f",
    "Industrial":           "#8e44ad",
    "Highway":              "#7c7c7c",
    "River":                "#4fa3e0",
    "SeaLake":              "#1f4e9c",
}

DETAIL_STRIDE = {"fast": 32, "normal": 16, "fine": 8}   # stride at model resolution (64 px windows)
MAX_WINDOWS   = 12_000                                   # stride grows automatically past this
MAX_PIXELS    = 2048 * 2048                              # after rescaling to model resolution


def _rgb(hex_colour):
    return tuple(int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))


def _resize(x, scale):
    """Resize an (H, W, C) array by `scale`, band by band (works for uint8 RGB and 13-band reflectance)."""
    if scale == 1:
        return x
    h, w = x.shape[:2]
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    bands = [np.asarray(Image.fromarray(x[..., c].astype(np.float32), mode="F").resize(size, Image.BILINEAR))
             for c in range(x.shape[-1])]
    return np.stack(bands, -1).astype(x.dtype if x.dtype == np.uint8 else np.float32)


def _window_weights():
    """Centre-weighted window: a pixel trusts the windows it sits in the middle of more than those at whose edge it lies."""
    w = np.hanning(PATCH + 2)[1:-1]
    return (np.outer(w, w) + 1e-3).astype(np.float32)


def predict_map(x, predict_fn, n_classes, tile=PATCH, detail="normal", batch=32):
    """x: (H, W, C) image. tile: how many source pixels make up one 64-px model patch (64 for 10 m imagery).
    Returns (class index map (H, W), confidence map (H, W), info dict)."""
    H, W = x.shape[:2]
    scale = PATCH / tile
    xs = _resize(x, scale)
    h, w = xs.shape[:2]
    if h * w > MAX_PIXELS:
        raise ValueError(f"Image too large: {h}x{w} px at model resolution (max {MAX_PIXELS // 2048} x 2048)")
    if min(h, w) < PATCH:
        raise ValueError(f"Image is smaller than one {tile}x{tile} px window - use single-patch classification")

    stride = DETAIL_STRIDE.get(detail, 16)
    while ((h - PATCH) // stride + 2) * ((w - PATCH) // stride + 2) > MAX_WINDOWS:
        stride *= 2

    # Pad (reflect) so windows fit exactly and the borders are covered as well as the centre
    ph = (-(h - PATCH)) % stride
    pw = (-(w - PATCH)) % stride
    xp = np.pad(xs, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    hp, wp = xp.shape[:2]
    ys = range(0, hp - PATCH + 1, stride)
    xs_ = range(0, wp - PATCH + 1, stride)
    corners = [(i, j) for i in ys for j in xs_]

    acc  = np.zeros((hp, wp, n_classes), np.float32)
    wsum = np.zeros((hp, wp), np.float32)
    kern = _window_weights()
    for b in range(0, len(corners), batch):
        chunk = corners[b:b + batch]
        patches = np.stack([xp[i:i + PATCH, j:j + PATCH] for i, j in chunk]).astype(np.float32)
        proba = np.asarray(predict_fn(patches), dtype=np.float32)
        for (i, j), p in zip(chunk, proba):
            acc[i:i + PATCH, j:j + PATCH] += kern[..., None] * p
            wsum[i:i + PATCH, j:j + PATCH] += kern

    proba = acc[:h, :w] / wsum[:h, :w, None]
    cls  = proba.argmax(-1).astype(np.uint8)
    conf = proba.max(-1)
    if (h, w) != (H, W):                                   # back to the uploaded image's size
        cls  = np.asarray(Image.fromarray(cls).resize((W, H), Image.NEAREST))
        conf = np.asarray(Image.fromarray(conf, mode="F").resize((W, H), Image.BILINEAR))
    return cls, conf, {"windows": len(corners), "stride_px": round(stride / scale), "tile_px": tile}


def colourise(cls, classes):
    lut = np.array([_rgb(CLASS_COLOURS.get(c, "#000000")) for c in classes], np.uint8)
    return lut[cls]


def class_stats(cls, conf, classes, pixel_area_m2=None):
    """Share of the image (and area, when the pixel size is known) per class, largest first."""
    total = cls.size
    rows = []
    for k, name in enumerate(classes):
        mask = cls == k
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "class": name,
            "colour": CLASS_COLOURS.get(name, "#000000"),
            "fraction": round(n / total, 4),
            "pixels": n,
            "area_ha": round(n * pixel_area_m2 / 1e4, 1) if pixel_area_m2 else None,
            "mean_confidence": round(float(conf[mask].mean()), 3),
        })
    return sorted(rows, key=lambda r: -r["fraction"])
