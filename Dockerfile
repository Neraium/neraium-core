FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

COPY requirements.txt ./requirements.txt
COPY apps/api/requirements.txt ./apps/api/requirements.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY . .

RUN adduser --disabled-password --gecos "" appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn apps.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
