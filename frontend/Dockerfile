FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CAMELS_MIP_DATA_DIR=/app \
    HYDROMIP_DATA_DIR=/app \
    PYTHONPATH=/app \
    PORT=8080

COPY frontend/requirements.txt /app/frontend/requirements.txt
RUN pip install --no-cache-dir -r /app/frontend/requirements.txt

COPY hydromip /app/hydromip
COPY mips /app/mips
COPY metrics /app/metrics
COPY leaderboards /app/leaderboards
COPY frontend /app/frontend

EXPOSE 8080

CMD ["uvicorn", "frontend.backend.app:app", "--host", "0.0.0.0", "--port", "8080"]
