FROM python:3.12-slim

WORKDIR /app

# Install the API's packages first (Docker caches this layer)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Package the API, web page, shared preprocessing, and pretrained serving models and samples.
COPY pipeline/__init__.py pipeline/config.py pipeline/data.py pipeline/
COPY api/     api/
COPY web/     web/
COPY model/   model/
COPY samples/ samples/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
