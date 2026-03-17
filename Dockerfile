FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY neraium_core ./neraium_core
COPY apps ./apps

RUN python -m pip install --upgrade pip && \
    python -m pip install .

EXPOSE 8000

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
