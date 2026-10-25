# Образ LLM-сервиса: Python + PyTorch (CUDA 13) + transformers + bitsandbytes (4-битная загрузка).
# Веса модели в образ не кладём — они монтируются из кэша Hugging Face (см. docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# Кэш скомпилированных ядер triton: контейнер работает не от root, поэтому нужна папка с правом записи.
ENV TRITON_CACHE_DIR=/tmp/triton HOME=/tmp

# gcc нужен triton: torch 2.14 выполняет часть операций Qwen3-VL ядрами triton,
# а triton при первом запуске компилирует небольшой C-модуль для работы с CUDA.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# torch с индекса PyTorch (из кэша сборки) + torchvision той же сборки: он нужен процессору Qwen3-VL.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN --mount=type=cache,target=/root/.cache/pip pip install torch torchvision --index-url ${TORCH_INDEX}

COPY llm_service/requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY llm_service ./llm_service

EXPOSE 8000
CMD ["uvicorn", "llm_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
