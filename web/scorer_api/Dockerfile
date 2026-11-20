FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PYTHONPATH=/app \
    HYDROMIP_REPO_ROOT=/app \
    HYDROMIP_QOBS_PATH=/app/data/qobs_531.npz \
    HYDROMIP_LEADERBOARDS_DIR=/app/mips/camels_us_daily/leaderboards

RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 hydromip
WORKDIR /app

COPY web/scorer_api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY hydromip /app/hydromip
COPY metrics /app/metrics
COPY web /app/web
COPY mips/camels_us_daily/protocol.json /app/mips/camels_us_daily/protocol.json
COPY mips/camels_us_daily/splits /app/mips/camels_us_daily/splits
COPY mips/camels_us_daily/leaderboards /app/mips/camels_us_daily/leaderboards
COPY data/qobs_531.npz /app/data/qobs_531.npz

RUN chown -R hydromip:hydromip /app
USER hydromip

EXPOSE 8080
CMD ["uvicorn", "web.scorer_api.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
