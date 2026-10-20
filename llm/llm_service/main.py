"""LLM-сервис: мультимодальная модель Qwen3-VL за HTTP-API.

Эндпоинты:
    POST /generate — системный промпт + текст + фотографии -> ответ модели (текст);
    GET  /health   — состояние сервиса и загруженная модель.

Модель загружается один раз при старте контейнера в 4-битном режиме (bitsandbytes, NF4):
Qwen3-VL-4B занимает ~3 ГБ видеопамяти вместо ~9 ГБ в bf16 и помещается на GPU 12 ГБ
вместе с ML-сервисом (эмбеддинги + reranker).

Сервис ничего не знает о сметах и справочнике — он только «отвечает на промпт».
Вся логика (что спросить, как проверить ответ) находится в пакете estimator.

Запуск без Docker (для отладки):
    uvicorn llm_service.main:app --port 8002
"""

import base64
import binascii
import io
import logging
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
# Каждый HTTP-запрос к Hugging Face (проверка версий файлов модели) логируется на уровне INFO — это шум.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("llm_service")

# Настройки из переменных окружения (их задаёт docker-compose.yml).
MODEL_ID = os.getenv("LLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
# Границы числа пикселей одной картинки для процессора Qwen: больше пикселей — больше токенов и памяти.
MIN_IMAGE_PIXELS = int(os.getenv("MIN_IMAGE_PIXELS", "65536"))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "524288"))
# Лимит длины входа в токенах: защищает видеопамять от слишком больших заявок.
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "10000"))
MAX_IMAGES = 4


# ---------------------------------------------------------------------------
# Схемы запросов и ответов
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Запрос к модели: инструкции, текст и до 4 фотографий в base64 (JPEG/PNG/WebP)."""

    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)
    images: list[str] = Field(default_factory=list, max_length=MAX_IMAGES)
    # Верхняя граница 8192 — с запасом для Thinking-моделей, которые рассуждают перед ответом.
    max_new_tokens: int = Field(default=1700, ge=1, le=8192)


class GenerateResponse(BaseModel):
    """Ответ модели и размеры входа/выхода в токенах."""

    model: str
    text: str
    input_tokens: int
    output_tokens: int


class HealthResponse(BaseModel):
    """Состояние сервиса."""

    status: str
    model: str
    gpu_memory_gb: float


# ---------------------------------------------------------------------------
# Загрузка модели
# ---------------------------------------------------------------------------


def load_model() -> dict[str, Any]:
    """Загружает Qwen3-VL в 4-битном режиме и её процессор (токенизатор + обработка изображений).

    Принимает: ничего (параметры из переменных окружения).
    Возвращает: словарь {"model", "processor", "torch"}.
    """
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("GPU недоступен: LLM-сервис рассчитан на видеокарту NVIDIA")

    # bf16 поддерживается на RTX 30xx/40xx; на старых картах (T4) — fp16.
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # NF4 — 4-битный формат весов, подобранный под нормальное распределение весов нейросети;
    # double_quant дополнительно сжимает константы квантования (~0,4 бита на параметр экономии).
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    logger.info("Загрузка %s (4 бита, %s)", MODEL_ID, compute_dtype)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization,
        dtype=compute_dtype,
        device_map={"": 0},  # вся модель на первую видеокарту
        attn_implementation="sdpa",  # встроенное в PyTorch быстрое внимание, без сборки FlashAttention
        low_cpu_mem_usage=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_IMAGE_PIXELS, max_pixels=MAX_IMAGE_PIXELS)
    logger.info("Модель загружена, занято видеопамяти: %.2f ГБ", torch.cuda.memory_allocated() / 2**30)
    return {"model": model, "processor": processor, "torch": torch}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Загружает модель при старте сервиса.

    Принимает: app — экземпляр FastAPI.
    Возвращает: асинхронный контекст (код до yield — старт, после — остановка).
    """
    if not getattr(app.state, "llm", None):
        app.state.llm = load_model()
    yield


app = FastAPI(title="LLM-сервис AI-сметчика", description="Qwen3-VL: анализ фото и выбор работ", lifespan=lifespan)

# Одна видеокарта: генерации выполняем строго по очереди.
_gpu_lock = threading.Lock()


def _decode_image(data: str, number: int) -> Any:
    """Декодирует картинку из base64 в RGB-изображение PIL.

    Принимает: data — строка base64; number — номер фото (для текста ошибки).
    Возвращает: объект PIL.Image в режиме RGB.
    Исключения: 422, если данные не являются картинкой.
    """
    from PIL import Image

    try:
        with Image.open(io.BytesIO(base64.b64decode(data, validate=True))) as image:
            return image.convert("RGB")
    except (binascii.Error, OSError) as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Фото {number}: не удалось прочитать изображение") from error


@app.post("/generate", response_model=GenerateResponse)
def generate(body: GenerateRequest, request: Request) -> GenerateResponse:
    """Генерирует ответ модели на промпт с фотографиями.

    Генерация жадная (do_sample=False): при одинаковом входе ответ одинаковый —
    для сметы важна воспроизводимость, а не «творчество».

    Принимает: body — промпты, фото в base64, лимит длины ответа.
    Возвращает: GenerateResponse с текстом ответа.
    Исключения: 413 — вход длиннее MAX_INPUT_TOKENS; 507 — не хватило видеопамяти.
    """
    llm = getattr(request.app.state, "llm", None)
    if not llm:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Модель ещё загружается")
    model, processor, torch = llm["model"], llm["processor"], llm["torch"]

    # Сообщение в формате чата Qwen: перед каждой картинкой подпись «Фото N.»,
    # чтобы модель могла ссылаться на номера фотографий в ответе.
    content: list[dict[str, Any]] = []
    for number, data in enumerate(body.images, start=1):
        content += [{"type": "text", "text": f"Фото {number}."}, {"type": "image", "image": _decode_image(data, number)}]
    content.append({"type": "text", "text": body.user_prompt})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": body.system_prompt}]},
        {"role": "user", "content": content},
    ]

    with _gpu_lock:
        # apply_chat_template превращает сообщения и картинки в токены модели.
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        input_tokens = int(inputs["input_ids"].shape[-1])
        if input_tokens > MAX_INPUT_TOKENS:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"Вход {input_tokens} токенов, лимит {MAX_INPUT_TOKENS}",
            )
        inputs = inputs.to(model.device)
        generated = None
        try:
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=body.max_new_tokens, do_sample=False, use_cache=True)
            # Модель возвращает вход + продолжение; берём только новые токены.
            output = generated[0, input_tokens:]
            text = processor.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        except torch.cuda.OutOfMemoryError as error:
            raise HTTPException(
                status.HTTP_507_INSUFFICIENT_STORAGE,
                "Не хватило видеопамяти: уменьшите число фото или MAX_IMAGE_PIXELS",
            ) from error
        finally:
            # Освобождаем промежуточные тензоры сразу, чтобы видеопамять не копилась между запросами.
            del inputs, generated
            torch.cuda.empty_cache()

    return GenerateResponse(model=MODEL_ID, text=text, input_tokens=input_tokens, output_tokens=int(output.shape[-1]))


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Сообщает, что модель загружена, и сколько видеопамяти она занимает.

    Принимает: request — текущий HTTP-запрос.
    Возвращает: HealthResponse.
    """
    llm = getattr(request.app.state, "llm", None)
    if not llm:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Модель ещё загружается")
    memory = llm["torch"].cuda.memory_allocated() / 2**30
    return HealthResponse(status="ok", model=MODEL_ID, gpu_memory_gb=round(memory, 2))
