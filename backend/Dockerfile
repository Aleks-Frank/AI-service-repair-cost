FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /service

COPY requirements.txt pyproject.toml ./
COPY rag ./rag
COPY llm/estimator ./llm/estimator
COPY backend/pyproject.toml backend/requirements.lock.txt ./backend/
COPY backend/app ./backend/app
RUN pip install --no-cache-dir -r backend/requirements.lock.txt . ./backend \
    && useradd --create-home appuser \
    && mkdir -p /service/backend/data \
    && chown -R appuser:appuser /service/backend/data

USER appuser
EXPOSE 8000
CMD ["python", "-m", "app"]
