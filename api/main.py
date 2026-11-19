"""EuroSAT land-use / land-cover web platform: the classification API plus the web page that calls it.

Run locally (from the project root):  uvicorn api.main:app --reload
    http://127.0.0.1:8000/       web page: upload an image, see the classification
    http://127.0.0.1:8000/docs   interactive API docs

POST /predict classifies one 64x64 patch; POST /map labels every part of a larger image (api/mapping.py).

The packaged serving configuration uses both models for 13-band EuroSAT GeoTIFFs and the RGB model
for PNG/JPEG and 3-band GeoTIFFs. See model/serving.json for ensemble and augmentation settings.
"""
import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import threading

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from api.mapping import CLASS_COLOURS, class_stats, colourise, predict_map
from pipeline.data import rgb_reflectance_to_8bit, to_rgb8

ROOT       = Path(__file__).resolve().parent.parent
MODEL_DIR  = ROOT / "model"
WEB_DIR    = ROOT / "web"
SAMPLE_DIR = ROOT / "samples"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_MAP_BYTES    = 60 * 1024 * 1024

# rasterio wheels bundle their own PROJ database; another install's PROJ_LIB (e.g. PostGIS) breaks CRS look-ups
try:
    import rasterio as _rio
    _proj = Path(_rio.__file__).parent / "proj_data"
    if _proj.exists():
        os.environ["PROJ_LIB"] = os.environ["PROJ_DATA"] = str(_proj)
except ImportError:
    pass

metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
CLASSES  = metadata["classes"]
BANDS    = metadata["input"]["bands"]          # champion input: "rgb" or "ms"

import keras  # noqa: E402  (imported after metadata so a missing model fails fast with a clear error)


def _load(folder):
    meta = json.loads((folder / "metadata.json").read_text())
    if meta["classes"] != CLASSES:
        raise RuntimeError(f"{folder} has a different class order from the champion")
    return {"model": keras.models.load_model(folder / "model.keras", compile=False), "meta": meta}


# One entry per input type the platform can serve: "ms" (13 bands) and/or "rgb"
MODELS = {BANDS: _load(MODEL_DIR)}
if "rgb" not in MODELS and (MODEL_DIR / "rgb" / "model.keras").exists():
    MODELS["rgb"] = _load(MODEL_DIR / "rgb")

# How each input type is predicted: which models are averaged (an "ms" upload can also feed its true-colour
# bands to the RGB model) and whether each averages the 8 flips/rotations (test-time augmentation).
# Chosen on the validation set; see model/serving.json. Default: one model, no TTA.
_serving_file = MODEL_DIR / "serving.json"
SERVING = json.loads(_serving_file.read_text()) if _serving_file.exists() else {}
PIPELINES = {}
for key in MODELS:
    members = [m for m in SERVING.get(key, {}).get("members", [{"model": key, "tta": False}]) if m["model"] in MODELS]
    PIPELINES[key] = members or [{"model": key, "tta": False}]


def _dihedral(x):
    """The 8 flips/rotations of a batch (N, H, W, C) - every orientation is valid for a top-down image."""
    views = []
    for k in range(4):
        r = np.rot90(x, k, axes=(1, 2))
        views += [r, r[:, :, ::-1]]
    return views


def infer(key, x, tta=True):
    """Class probabilities for a batch x in `key`'s input format ("ms": 13 bands, "rgb": 8-bit RGB).
    tta=False skips test-time augmentation (used where speed matters more)."""
    probs = []
    for m in PIPELINES[key]:
        xm = x if m["model"] == key else to_rgb8(x)          # only ms -> rgb conversion is ever needed
        xm = np.asarray(xm, np.float32)
        views = _dihedral(xm) if (tta and m["tta"]) else [xm]
        stacked = np.concatenate([np.ascontiguousarray(v) for v in views])     # one call instead of 8
        model = MODELS[m["model"]]["model"]
        # direct call is fastest for a few patches; predict() (compiled, batched) for map-sized inputs
        out = (np.asarray(model(stacked, training=False)) if len(stacked) <= 64
               else model.predict(stacked, verbose=0, batch_size=256))
        probs.append(out.reshape(len(views), len(xm), -1).mean(0))
    return np.mean(probs, axis=0)


def describe(key):
    """What answers an upload of this type, and how well it scored, for the response and the web page."""
    runs = [MODELS[m["model"]]["meta"]["run"] + (" +TTA" if m["tta"] else "") for m in PIPELINES[key]]
    scores = SERVING.get(key, {}).get("metrics") or MODELS[key]["meta"]["metrics"]
    return {"input": key, "run": " + ".join(runs), "ensemble": len(runs) > 1,
            "test_macro_f1": scores["test_macro_f1"], "val_macro_f1": scores.get("val_macro_f1")}

app = FastAPI(title="EuroSAT Land-Cover Platform", version="2.0",
              description="Upload a Sentinel-2 patch or an RGB satellite image and get its land-use / land-cover "
                          "class (one of ten EuroSAT classes).")


def _need(bands):
    if bands not in MODELS:
        need = "a 13-band GeoTIFF" if "ms" in MODELS else "an RGB PNG/JPEG"
        raise HTTPException(422, f"No {bands.upper()} model is deployed - upload {need}")
    return bands


def _square_64(img):
    """Centre-crop to a square (so photos are not squashed), then resize to the model's 64x64."""
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return img if img.size == (64, 64) else img.resize((64, 64), Image.BILINEAR)


def read_patch(data: bytes, filename: str):
    """Bytes of an uploaded file -> (model key, model-ready array (64, 64, C), 8-bit RGB preview)."""
    name = filename.lower()
    if name.endswith((".tif", ".tiff")):
        from rasterio.io import MemoryFile
        with MemoryFile(data) as mem, mem.open() as src:
            x = np.moveaxis(src.read(), 0, -1)
        if x.shape[:2] != (64, 64):
            raise HTTPException(422, f"GeoTIFF must be 64x64 pixels, got {x.shape[0]}x{x.shape[1]}")
        if x.shape[-1] == 13:
            rgb = to_rgb8(x)
            key = "ms" if "ms" in MODELS else _need("rgb")
            return key, (x if key == "ms" else rgb), rgb
        if x.shape[-1] == 3:                      # already true colour (R, G, B)
            rgb = x if x.dtype == np.uint8 else rgb_reflectance_to_8bit(x)
            return _need("rgb"), rgb, rgb
        raise HTTPException(422, f"GeoTIFF has {x.shape[-1]} bands; expected 13 (Sentinel-2) or 3 (RGB)")

    if name.endswith((".png", ".jpg", ".jpeg")):
        _need("rgb")
        rgb = np.asarray(_square_64(Image.open(io.BytesIO(data)).convert("RGB")))
        return "rgb", rgb, rgb

    raise HTTPException(415, "Upload a .tif (13-band Sentinel-2) or a .png/.jpg RGB image")


def _png_data_url(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).resize((256, 256), Image.NEAREST).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    """The champion's metadata (unchanged for existing clients) plus every model the platform serves."""
    return {**metadata, "served": {k: v["meta"] for k, v in MODELS.items()},
            "pipelines": {k: describe(k) for k in MODELS}, "colours": CLASS_COLOURS}


@app.post("/predict")
async def predict(file: UploadFile = File(..., description="Sentinel-2 GeoTIFF (64x64, 13 bands) or RGB .png/.jpg")):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES // 2**20} MB)")
    try:
        key, x, rgb = read_patch(data, file.filename or "")
    except HTTPException:
        raise
    except Exception as e:                       # unreadable / corrupt file
        raise HTTPException(422, f"Could not read image: {e}")

    proba = np.asarray(await run_in_threadpool(infer, key, x[None]), dtype=float)[0]
    order = np.argsort(-proba)
    return {
        "predicted_class": CLASSES[order[0]],
        "confidence":      round(float(proba[order[0]]), 4),
        "top3":            [{"class": CLASSES[i], "probability": round(float(proba[i]), 4)} for i in order[:3]],
        "probabilities":   [{"class": CLASSES[i], "probability": round(float(proba[i]), 4)} for i in order],
        "model":           describe(key),
        "model_version":   MODELS[key]["meta"]["version"],
        "preview":         _png_data_url(rgb),
    }


def read_scene(data: bytes, filename: str):
    """A larger upload -> (model key, image (H, W, C), 8-bit RGB view, metres per pixel or None)."""
    name = filename.lower()
    if name.endswith((".tif", ".tiff")):
        from rasterio.io import MemoryFile
        with MemoryFile(data) as mem, mem.open() as src:
            x = np.moveaxis(src.read(), 0, -1)
            try:                                    # metres per pixel, only if the CRS is projected (UTM etc.)
                res = abs(src.res[0]) if src.crs is not None and src.crs.is_projected else None
            except Exception:
                res = None
        if x.shape[-1] == 13:
            if np.issubdtype(x.dtype, np.floating) and np.nanmax(x) <= 2:   # reflectance exported as 0-1
                x = x * 10000.0
            rgb = to_rgb8(x)
            key = "ms" if "ms" in MODELS else _need("rgb")
            return key, (x if key == "ms" else rgb), rgb, res
        if x.shape[-1] in (3, 4):
            rgb = x[..., :3]
            rgb = rgb if rgb.dtype == np.uint8 else rgb_reflectance_to_8bit(rgb)
            return _need("rgb"), rgb, rgb, res
        raise HTTPException(422, f"GeoTIFF has {x.shape[-1]} bands; expected 13 (Sentinel-2) or 3 (RGB)")
    if name.endswith((".png", ".jpg", ".jpeg")):
        _need("rgb")
        rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        return "rgb", rgb, rgb, None
    raise HTTPException(415, "Upload a .tif (13-band Sentinel-2) or a .png/.jpg RGB image")


def _fit(img, long_side=1024, resample=Image.NEAREST):
    s = long_side / max(img.size)
    return img if s >= 1 else img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), resample)


def _data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/map")
async def land_cover_map(
    file: UploadFile = File(..., description="A larger Sentinel-2 GeoTIFF (13 bands) or RGB satellite image"),
    resolution_m: float | None = Form(None, gt=0.1, le=100,
                                      description="Ground size of one pixel in metres. Read from the GeoTIFF when "
                                                  "it is georeferenced; otherwise 10 m (Sentinel-2) is assumed."),
    detail: str = Form("normal", pattern="^(fast|normal|fine)$"),
):
    """Classify every part of a larger image: sliding 640 m windows -> a colour-coded land-cover map."""
    data = await file.read()
    if len(data) > MAX_MAP_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_MAP_BYTES // 2**20} MB)")
    return await run_in_threadpool(make_map, data, file.filename or "", resolution_m, detail)


_demo_cache = {}
_demo_lock = threading.Lock()


@app.get("/map/demo")
def demo_map():
    """The demo landscape's map, computed once and cached (used by the home page and 'Try the demo')."""
    with _demo_lock:
        return _demo()


def _demo():
    if "result" not in _demo_cache:
        path = SAMPLE_DIR / "mosaic_demo.tif"
        if not path.exists():
            raise HTTPException(404, "Demo scene not found - run python -m pipeline.make_mosaic")
        _demo_cache["result"] = {**make_map(path.read_bytes(), path.name, None, "normal"), "demo": True}
    return _demo_cache["result"]


def make_map(data, filename, resolution_m, detail):
    try:
        key, x, rgb, file_res = read_scene(data, filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"Could not read image: {e}")

    res = resolution_m or file_res or 10.0
    tile = int(np.clip(round(640 / res), 16, 1024))      # source pixels covering one 640 m model patch
    tta = False   # overlapping windows already average many views; TTA cost 8x for no map gain (demo scene)
    try:
        cls, conf, info = predict_map(x, lambda b: infer(key, b, tta=tta), len(CLASSES),
                                      tile=tile, detail=detail)
    except ValueError as e:
        raise HTTPException(422, str(e))

    H, W = cls.shape
    colour = Image.fromarray(colourise(cls, CLASSES))
    grid = _fit(Image.fromarray(cls), 256)                 # small class grid for hover look-ups in the browser
    return {
        "width": W, "height": H,
        "resolution_m": res, "resolution_source": "user" if resolution_m else ("file" if file_res else "assumed"),
        **info,
        "model": describe(key), "tta": tta,
        "classes": class_stats(cls, conf, CLASSES, pixel_area_m2=res * res),
        "legend": [{"class": c, "colour": CLASS_COLOURS[c]} for c in CLASSES],
        "mean_confidence": round(float(conf.mean()), 3),
        "image":      _data_url(_fit(Image.fromarray(rgb), resample=Image.BILINEAR)),
        "map":        _data_url(_fit(colour)),
        "confidence": _data_url(_fit(Image.fromarray((conf * 255).astype(np.uint8)), resample=Image.BILINEAR)),
        "grid": {"width": grid.width, "height": grid.height,
                 "data": base64.b64encode(np.asarray(grid).tobytes()).decode()},
    }


@app.get("/samples")
def list_samples():
    """Example patches (one per class) the web page offers as 'try one'."""
    return [{"class": p.stem, "thumbnail": f"/samples/{p.stem}.png",
             "file": f"/samples/{p.stem}.tif" if (p.with_suffix(".tif")).exists() else f"/samples/{p.name}"}
            for p in sorted(SAMPLE_DIR.glob("*.png")) if not p.stem.startswith("mosaic")]


if SAMPLE_DIR.exists():
    app.mount("/samples", StaticFiles(directory=SAMPLE_DIR), name="samples")
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        """The web app. Asset URLs carry the files' modification time, so browsers never mix an old
        script or stylesheet with a new page after an update."""
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        for name in ("style.css", "app.js"):
            v = int((WEB_DIR / name).stat().st_mtime)
            html = html.replace(f"/static/{name}", f"/static/{name}?v={v}")
        return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@app.middleware("http")
async def revalidate_static(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith(("/static/", "/samples/")):
        response.headers["Cache-Control"] = "no-cache"          # cache, but check for a newer version first
    return response


@app.on_event("startup")
def warm_up():
    """Trace each model once and precompute the demo map in the background, so the first visitor
    does not wait ~15 s and the server answers immediately."""
    def work():
        for key in MODELS:
            infer(key, np.zeros((1, 64, 64, 13 if key == "ms" else 3), np.float32))
        try:
            demo_map()
        except HTTPException:
            pass
    threading.Thread(target=work, daemon=True).start()
