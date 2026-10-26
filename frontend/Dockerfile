FROM python:3.11-slim

WORKDIR /app/frontend

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CAMELS_MIP_DATA_DIR=/app \
    PORT=8080

COPY frontend/requirements.txt /app/frontend/requirements.txt
RUN pip install --no-cache-dir -r /app/frontend/requirements.txt

COPY leaderboards /app/leaderboards
COPY splits /app/splits
COPY frontend /app/frontend

EXPOSE 8080

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8080"]
