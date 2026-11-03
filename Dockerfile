# PEARLSAQI Flask API — the same setup as the Render deployment.
#
#   docker compose up --build        → http://localhost:5000/health

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app.py .
COPY src ./src
COPY models ./models
COPY database ./database

EXPOSE 5000

CMD gunicorn app:app --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:$PORT
