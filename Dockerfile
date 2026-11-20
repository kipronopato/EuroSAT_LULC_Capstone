FROM python:3.12-slim

WORKDIR /app

# Rasterio's Linux wheel needs the system Expat library on the slim image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt \
    && python -c "import rasterio"

# Package the API, web page, shared preprocessing, and pretrained serving models and samples.
COPY pipeline/__init__.py pipeline/config.py pipeline/data.py pipeline/
COPY api/     api/
COPY web/     web/
COPY model/   model/
COPY samples/ samples/

EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
