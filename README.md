# AI-сметчик

Серверная AI-система для первичной оценки ремонтных заявок управляющей компании «Территория».
По фото и описанию заявки формирует черновик сметы: **работы** из справочника ПЗР со стоимостью и
**материалы** из прайсов поставщиков с ценами. Смету проверяет и утверждает сметчик.

## Как устроено

```
Битрикс24 / API ─► backend (FastAPI): приём заявки, очередь, статусы
                      │
                      ▼  llm/estimator — конвейер сметы
   фото + текст ─► llm_service (Qwen3-VL): дефекты и поисковые запросы
                ─► RAG по справочнику ПЗР: кандидаты работ ───────────┐ ml_service (bge-m3 + reranker)
                ─► llm_service: выбор работ из кандидатов + материалы  │ pzr-db (справочник ПЗР)
                ─► RAG по прайсам: кандидаты товаров ──────────────────┘ prices-db (прайсы поставщиков)
                ─► llm_service: какой товар подходит
                ─► код: тот же товар у всех поставщиков, самый дешёвый в наличии, расчёт сумм
                      ▼
                 смета (JSON) ─► Битрикс24
```

Модель только выбирает из найденного; цены, количества и суммы считает код по данным из базы.

| Папка / сервис | Назначение |
|---|---|
| `backend/` → контейнер `backend` | API приёма заявок, очередь, интеграция с Битрикс24 |
| `llm/estimator/` | Конвейер «заявка → смета» (Python-модуль, вызывается бэкендом или из CLI) |
| `llm/llm_service/` → контейнер `llm_service` | Qwen3-VL-8B в 4 битах: `/generate` |
| `rag/` | Загрузка справочника и прайсов, гибридный поиск, подбор поставщика; `rag/price_lists/` — парсеры прайсов (модуль на формат) |
| `ml_service/` → контейнер `ml_service` | Эмбеддинги и reranker: `/embed`, `/rerank` |
| контейнеры `pzr-db`, `prices-db` | PostgreSQL + pgvector: справочник ПЗР (`pzr_pricing`) и прайсы (`supplier_prices`) |
| `data/` (не в git) | `PZR/` — справочник ПЗР, `shop/` — прайсы поставщиков; передаются отдельно |
| `data_sources.toml` | Какие файлы из `data/` каким парсером загружать в базы |
| `.github/workflows/` | CI/CD: проверки на pull request, деплой на сервер после слияния в `main` |

## Требования

- Видеокарта **NVIDIA, 12 ГБ** (для `llm_service`), драйвер с поддержкой CUDA 13.
- **Docker** с доступом к GPU: Linux — Docker Engine + NVIDIA Container Toolkit; Windows — Docker Desktop.
- **Python 3.11+** — для загрузки данных и CLI.
- ~40 ГБ на диске: образы ~15 ГБ, веса моделей ~20 ГБ, база прайсов ~5 ГБ.

---

## Запуск локально — Linux

```bash
# 1. Docker + доступ к GPU (один раз)
curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER      # затем перелогиниться
sudo apt-get install -y nvidia-container-toolkit     # после подключения репозитория NVIDIA (docs.nvidia.com/datacenter/cloud-native)
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all ubuntu nvidia-smi        # проверка: таблица с видеокартой

# 2. Python-окружение
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -e backend

# 3. Настройки
cp .env.example .env
cp backend/.env.example backend/.env                # вписать API_KEY=<любая строка>

# 4. Дальше — «Общие шаги»
```

## Запуск локально — Windows (PowerShell)

```powershell
# 1. Установить драйвер NVIDIA и Docker Desktop; проверка:
docker run --rm --gpus all ubuntu nvidia-smi

# 2. Python-окружение (Python 3.11+ с python.org)
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # если запрещено: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt -e backend

# 3. Настройки
copy .env.example .env
copy backend\.env.example backend\.env     # вписать API_KEY=<любая строка>
# В .env обязательно добавить путь к кэшу моделей (в Windows нет переменной HOME):
#   HF_CACHE_DIR=C:/Users/<имя>/.cache/huggingface

# 4. Дальше — «Общие шаги» (пути можно писать через \ или /)
```

## Общие шаги

**Дополнить `.env`** (остальное — значения по умолчанию):
```ini
LLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
DEVICE=cpu                 # эмбеддинги и reranker на CPU, видеокарта — под LLM
RERANK_TOP_N=10
SELECTION_MAX_TOKENS=3000
```

**Запустить сервисы** (первый запуск — сборка образов и скачивание весов ~20 ГБ):
```bash
docker compose up -d --build db prices_db ml_service llm_service    # без бэкенда
docker compose up -d --build                                       # вместе с бэкендом
docker compose ps                                                  # все healthy (llm_service ~1 мин)
```

**Загрузить данные** (хранятся в Docker-томах `pgdata` и `prices_pgdata`). Что и из какого файла
грузить, записано в [`data_sources.toml`](data_sources.toml): справочник ПЗР (`data/PZR/`) и прайсы
поставщиков (`data/shop/`) с указанием парсера. Одна команда загружает всё, повторный запуск —
только изменившиеся файлы:
```bash
python -m rag.cli load-all            # или в Docker, без Python на машине: docker compose run --rm loader
python -m rag.cli load-all --prune    # плюс удалить из базы поставщиков, убранных из списка
python -m rag.cli list-suppliers
```
Новый поставщик — строка в `data_sources.toml`; новый формат файла — модуль в `rag/price_lists/`
(один модуль — один формат: `onec` — выгрузка 1С, `shop_json` — каталог JSON, `shop_export` — выгрузка
магазина .csv/.xlsx). Каталоги магазинов — ~400 тыс. позиций: на CPU загрузка идёт часами, на GPU — минуты.
Загрузка на GPU: `docker compose stop llm_service`, затем `DEVICE=cuda docker compose up -d ml_service`
(PowerShell: `$env:DEVICE="cuda"; docker compose up -d ml_service`); после загрузки — снова
`docker compose up -d ml_service llm_service` (без `DEVICE=cuda`).

**Получить смету:**
```bash
# Напрямую через конвейер, без бэкенда: результат в results/<дата>/estimate.json и estimate.csv
python -m llm.estimator.cli data/photos/electric/burned_socket.jpg -d "Обгорела розетка, белая, с заземлением. Заменить 1 розетку."

# Через бэкенд (как из Битрикса); сервер запущен в Docker или командой `cd backend && python -m app`
python backend/scripts/demo.py data/photos/electric/burned_socket.jpg --description "Обгорела розетка, заменить 1 розетку."
```
Количество в описании лучше писать цифрами («заменить **1** розетку»). Без чисел объём оценивается
по фото или подставляется минимум (1 единица расценки) — такие строки помечены «≈, проверьте».
Документация API бэкенда: http://127.0.0.1:8000/docs (ключ в заголовке `X-API-Key`).

---

## Развёртывание на сервере

На сервере всё работает в Docker, включая бэкенд и загрузку данных — Python на самой машине не нужен.
Нужен Linux-сервер с GPU NVIDIA (12 ГБ — минимум, 16+ ГБ — можно `DEVICE=cuda` для поиска).

**Что попадает на сервер:**

| Что | Откуда | Нужно постоянно? |
|---|---|---|
| Код, `docker-compose.yml`, `data_sources.toml` | `git clone` / `git pull` | да |
| Веса моделей (~20 ГБ) | скачиваются сами при первом запуске | да (кэш Hugging Face) |
| Файлы данных `data/PZR`, `data/shop` (~0,5 ГБ) | копируются вручную: `rsync -av data/ сервер:/opt/ai-smetchik/data/` | только для загрузки в базы; хранить, чтобы перезагрузить при обновлении |
| Базы (`pgdata`, `prices_pgdata`, ~5 ГБ) | создаются на сервере командой `docker compose run --rm loader` | да |

**Первый запуск:**
1. Docker Engine + NVIDIA Container Toolkit (как в разделе Linux), `git clone` в `/opt/ai-smetchik`,
   скопировать папку `data/`.
2. `.env` и `backend/.env`, как выше. В `backend/.env` — надёжный `API_KEY`, `BITRIX_OUTBOUND_TOKEN`,
   `BITRIX_WEBHOOK_URL` и параметры записи результата в Битрикс24.
3. `docker compose up -d --build`, затем `docker compose run --rm loader` — загрузка ПЗР и всех прайсов.
   Контейнеры перезапускаются сами (`restart: unless-stopped`), в том числе после перезагрузки сервера.
4. Доступ снаружи: все порты открыты только на `127.0.0.1`. Для вебхуков Битрикс24 бэкенд публикуется
   через обратный прокси с HTTPS (nginx или Caddy → `127.0.0.1:8000`). Базы и сервисы моделей наружу не открывать.

**Обновления (CI/CD, [`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml)):**
- на каждый pull request — тесты бэкенда, проверка импорта кода и `docker-compose.yml`;
- после слияния в `main` — автоматически по SSH на сервере: `git pull`, `docker compose up -d --build`,
  `docker compose run --rm loader` (грузит только изменившиеся файлы данных). Секреты для SSH описаны в начале файла.
- **Новый прайс:** скопировать файл в `data/shop/` на сервере (и строку в `data_sources.toml`, если поставщик новый),
  затем `docker compose run --rm loader` — или дождаться ближайшего деплоя.

**Резервные копии:** том `requests_data` (заявки и результаты — восстановить нельзя, копировать регулярно);
`pgdata` и `prices_pgdata` восстанавливаются повторной загрузкой из `data/` (`loader --force`).

---

## Полезные команды

| Команда | Что делает |
|---|---|
| `docker compose ps`, `docker compose logs -f <сервис>` | Состояние и логи сервисов |
| `docker compose down` | Остановить (данные сохраняются); `down -v` — удалить и данные |
| `python -m rag.cli search "замена розетки"` | Поиск расценок ПЗР |
| `python -m rag.cli search-prices "лампа E27"` | Поиск товаров в прайсах |
| `python -m rag.cli offers <артикул> --qty 3` | Один товар у всех поставщиков, выбор самого дешёвого |
| `python -m llm.estimator.cli … --no-materials` | Смета только по работам (~30 с) |
| `cd backend && python -m pytest -q` | Тесты бэкенда |

| Проблема | Решение |
|---|---|
| `could not select device driver "nvidia"` | Linux: установить NVIDIA Container Toolkit; Windows: обновить драйвер и Docker Desktop |
| Ошибка про путь `/.cache/huggingface` (Windows) | Добавить `HF_CACHE_DIR` в `.env` |
| Сервис долго в состоянии `starting` | `docker compose logs <сервис>` — обычно идёт первое скачивание весов |
| HTTP 507 «не хватило видеопамяти» | Закрыть программы на GPU, уменьшить `MAX_IMAGE_PIXELS` или взять `Qwen/Qwen3-VL-4B-Instruct` |
| `503 API_KEY is not configured` | Задать `API_KEY` в `backend/.env` и перезапустить бэкенд |
