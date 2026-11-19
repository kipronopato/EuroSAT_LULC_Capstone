from pathlib import Path

from fastapi.testclient import TestClient

from api.main import CLASSES, MODELS, app

client = TestClient(app)
SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def post(path):
    return client.post("/predict", files={"file": (path.name, path.read_bytes())})


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_passes_quality_gate():
    info = client.get("/model-info").json()
    assert info["classes"] == CLASSES and len(CLASSES) == 10
    assert info["metrics"]["val_macro_f1"] >= 0.80


def test_predict_geotiff_returns_class_and_top3():
    result = post(SAMPLES / "Forest.tif").json()
    assert result["predicted_class"] in CLASSES
    assert len(result["top3"]) == 3
    probs = [t["probability"] for t in result["top3"]]
    assert probs == sorted(probs, reverse=True)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["top3"][0]["class"] == result["predicted_class"]


def test_most_samples_are_classified_correctly():
    """Smoke test of model quality: the champion should get most of the 10 sample patches right."""
    correct = sum(post(p).json()["predicted_class"] == p.stem for p in SAMPLES.glob("*.tif") if p.stem in CLASSES)
    assert correct >= 7


def test_png_is_served_by_the_rgb_model():
    """An RGB PNG goes to the RGB model; without one it is rejected with a clear message."""
    response = post(SAMPLES / "Forest.png")
    assert response.status_code == (200 if "rgb" in MODELS else 422)
    if "rgb" in MODELS:
        assert response.json()["model"]["input"] == "rgb"


def test_geotiff_is_served_by_the_13_band_model_when_deployed():
    result = post(SAMPLES / "River.tif").json()
    assert result["model"]["input"] == ("ms" if "ms" in MODELS else "rgb")


def test_full_probability_list_and_preview():
    result = post(SAMPLES / "Industrial.tif").json()
    probs = result["probabilities"]
    assert sorted(p["class"] for p in probs) == sorted(CLASSES)
    assert abs(sum(p["probability"] for p in probs) - 1) < 0.01
    assert result["preview"].startswith("data:image/png;base64,")


def test_large_non_square_photo_is_cropped_and_resized():
    if "rgb" not in MODELS:
        return
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.open(SAMPLES / "SeaLake.png").convert("RGB").resize((640, 480)).save(buf, format="JPEG")
    response = client.post("/predict", files={"file": ("photo.jpg", buf.getvalue())})
    assert response.status_code == 200


def test_rejects_oversized_upload():
    response = client.post("/predict", files={"file": ("big.png", b"0" * (11 * 1024 * 1024))})
    assert response.status_code == 413


def test_web_page_and_samples_are_served():
    page = client.get("/")
    assert page.status_code == 200 and "EuroSAT" in page.text and 'data-page="map"' in page.text
    assert client.get("/static/app.js").status_code == 200
    samples = client.get("/samples").json()
    assert len(samples) == 10
    assert client.get(samples[0]["file"]).status_code == 200


def test_rejects_unsupported_file_type():
    response = client.post("/predict", files={"file": ("notes.txt", b"hello")})
    assert response.status_code == 415


def test_rejects_corrupt_image():
    response = client.post("/predict", files={"file": ("broken.tif", b"not really a tiff")})
    assert response.status_code == 422


def test_rejects_missing_file():
    assert client.post("/predict").status_code == 422


# ---------------------------------------------------------------- land-cover map (POST /map)
def _map(path, **form):
    return client.post("/map", files={"file": (path.name, path.read_bytes())}, data=form)


def _grid(result):
    import base64
    import numpy as np
    g = result["grid"]
    return np.frombuffer(base64.b64decode(g["data"]), np.uint8).reshape(g["height"], g["width"])


def test_map_of_demo_scene_matches_ground_truth():
    """The demo mosaic is 48 held-out test patches with known classes; most of the map must be right."""
    import numpy as np
    from PIL import Image
    result = _map(SAMPLES / "mosaic_demo.tif", detail="fast")
    assert result.status_code == 200
    r = result.json()
    assert (r["width"], r["height"]) == (512, 384)
    assert r["resolution_source"] == "file" and r["resolution_m"] == 10
    grid = _grid(r)
    truth = np.asarray(Image.open(SAMPLES / "mosaic_truth.png").resize(grid.shape[::-1], Image.NEAREST))
    assert (grid == truth).mean() >= 0.70
    assert abs(sum(c["fraction"] for c in r["classes"]) - 1) < 0.01
    # 384 x 512 px at 10 m = 1966 ha
    assert abs(sum(c["area_ha"] for c in r["classes"]) - 1966) < 5
    for key in ("image", "map", "confidence"):
        assert r[key].startswith("data:image/png;base64,")


def test_map_of_rgb_image_uses_rgb_model_and_user_resolution():
    if "rgb" not in MODELS:
        return
    r = _map(SAMPLES / "mosaic_demo.png", detail="fast", resolution_m="10").json()
    assert r["model"]["input"] == "rgb"
    assert r["resolution_source"] == "user" and r["tile_px"] == 64


def test_map_rejects_image_smaller_than_one_window():
    response = _map(SAMPLES / "Forest.png", resolution_m="5")      # 64 px at 5 m = half a window
    assert response.status_code == 422


def test_map_rejects_bad_detail_option():
    assert _map(SAMPLES / "mosaic_demo.png", detail="ultra").status_code == 422


def test_demo_map_is_cached():
    first = client.get("/map/demo")
    assert first.status_code == 200 and first.json()["demo"] is True
    assert client.get("/map/demo").json()["map"] == first.json()["map"]
