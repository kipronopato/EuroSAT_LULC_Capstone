import io
import os

import numpy as np
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="EuroSAT Land Cover", page_icon="🛰️")

# Where the FastAPI server lives (override with the API_URL environment variable)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.title("Land-cover classifier")
st.write("Upload a 64x64 Sentinel-2 patch: a 13-band GeoTIFF from EuroSAT_MS, or an RGB PNG/JPEG. "
         "Example patches are in the `samples/` folder.")

try:
    info = requests.get(f"{API_URL}/model-info", timeout=5).json()
    st.caption(f"Deployed model: **{info['run']}** (v{info['version']}, input: {info['input']['bands']}) - "
               f"test macro F1 {info['metrics']['test_macro_f1']:.3f}")
except requests.exceptions.RequestException:
    st.warning(f"API not reachable at {API_URL}. Start it with `uvicorn api.main:app`.")

upload = st.file_uploader("Image patch", type=["tif", "tiff", "png", "jpg", "jpeg"])

if upload is not None:
    data = upload.getvalue()
    col1, col2 = st.columns([1, 2])

    with col1:
        if upload.name.lower().endswith((".tif", ".tiff")):
            from rasterio.io import MemoryFile
            with MemoryFile(data) as mem, mem.open() as src:
                x = src.read()
            preview = x[[3, 2, 1]] if x.shape[0] == 13 else x[:3]          # B04, B03, B02
            preview = np.clip(np.moveaxis(preview, 0, -1) / 3000 * 255, 0, 255).astype(np.uint8)
            st.image(Image.fromarray(preview).resize((192, 192), Image.NEAREST), caption="True-colour preview")
        else:
            st.image(Image.open(io.BytesIO(data)).resize((192, 192), Image.NEAREST), caption="Uploaded patch")

    with col2:
        try:
            response = requests.post(f"{API_URL}/predict", files={"file": (upload.name, data)}, timeout=30)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.HTTPError:
            st.error(f"The API rejected the file: {response.json().get('detail')}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the API. Is it running?\n\n{e}")
        else:
            st.success(f"**{result['predicted_class']}** - confidence {result['confidence']:.1%}")
            for item in result["top3"]:
                st.write(f"{item['class']}: {item['probability']:.1%}")
                st.progress(item["probability"])
