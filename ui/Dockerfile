FROM python:3.12-slim

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir streamlit requests pillow numpy rasterio \
    && python -c "import rasterio"
COPY ui/app.py app.py

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
