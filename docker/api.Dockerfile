# Same Python version you develop with (3.11, as shown in /metrics)
FROM python:3.11-slim

WORKDIR /app

# XGBoost needs the OpenMP runtime library on Linux
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install packages first, so Docker can reuse this layer when only code changes
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the code and the exported model
COPY src/ src/
COPY model_store/ model_store/

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]