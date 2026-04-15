FROM python:3.11-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir .
RUN chmod +x /app/docker/entrypoint.sh

# Run as a non-root user for security.
RUN addgroup --system neraium && adduser --system --ingroup neraium --no-create-home neraium
RUN mkdir -p /data && chown neraium:neraium /data
USER neraium

ENV NERAIUM_UI_API_BASE_URL=http://127.0.0.1:8000
ENV NERAIUM_UI_WS_ENDPOINT=/ws/telemetry

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT','8000') + '/health')" || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
