# Образ ML-сервиса: Python + PyTorch (CUDA 13) + sentence-transformers.
# Веса моделей в образ не кладём — они монтируются из кэша Hugging Face (см. docker-compose.yml),
# иначе образ весил бы на 4,5 ГБ больше и пересобирался бы при каждом обновлении модели.
FROM python:3.11-slim

# Не создавать .pyc и сразу выводить логи (без буферизации).
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /srv

# torch ставим отдельным слоем с официального индекса PyTorch: он быстрее PyPI,
# а отдельный слой не пересобирается при изменении остальных зависимостей.
# Для сервера без GPU замените cu130 на cpu.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN --mount=type=cache,target=/root/.cache/pip pip install torch --index-url ${TORCH_INDEX}

COPY ml_service/requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY ml_service ./ml_service

EXPOSE 8000
CMD ["uvicorn", "ml_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
