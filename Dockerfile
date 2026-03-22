FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    NERAIUM_LOG_LEVEL=INFO

COPY pyproject.toml README.md ./
COPY neraium_core ./neraium_core
COPY apps ./apps
COPY docker/entrypoint.sh /app/docker/entrypoint.sh

RUN python -m pip install --upgrade pip && \
    python -m pip install . && \
    chmod +x /app/docker/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["python", "-m", "apps.api.main"]
