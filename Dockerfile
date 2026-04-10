FROM python:3.11-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir .
RUN chmod +x /app/docker/entrypoint.sh

ENV NERAIUM_UI_API_BASE_URL=http://127.0.0.1:8000
ENV NERAIUM_UI_WS_ENDPOINT=/ws/telemetry

ENTRYPOINT ["/app/docker/entrypoint.sh"]
