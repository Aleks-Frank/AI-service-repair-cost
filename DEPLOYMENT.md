# Запуск my-feature на Ubuntu / облачной ВМ

Инструкция подготовлена для ветки `my-feature`, исходный коммит
`98d456fe5a654144acb3e26ddf2f800ee23fc140`. Все команды, кроме SSH-туннеля,
выполняйте в терминале сервера из корня репозитория.

## 1. Что потребуется

- Docker Engine и Docker Compose v2.24+ (используется необязательный `env_file`).
- NVIDIA GPU, драйвер и NVIDIA Container Toolkit. LLM-сервис явно требует CUDA GPU;
  `DEVICE=cpu` меняет только ML-сервис и не позволяет запустить весь стек на CPU.
- В существующих Dockerfile выбран PyTorch `cu130`: драйвер должен быть совместим
  с этой сборкой. Оба модельных сервиса используют одну GPU; нужен запас VRAM.
- Доступ к Docker Hub, индексам Python/PyTorch и Hugging Face для первой загрузки.
- Справочник ПЗР `.xlsx`, а для подбора материалов — прайсы `.xls/.xlsx`.
  Папка `data/` не входит в Git: перенесите свои файлы на сервер.

Проверьте инфраструктуру:

```bash
docker version
docker compose version
nvidia-smi
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

Если GPU не видна из последней команды, сначала настройте NVIDIA Container Toolkit.
Полный запуск с моделями здесь не проверялся: в среде подготовки нет GPU и Docker Engine.

## 2. Нужная ветка и изменения Docker

Для новой копии:

```bash
git clone --branch my-feature --single-branch https://github.com/Aleks-Frank/AI-service-repair-cost.git
cd AI-service-repair-cost
```

Для существующей копии сначала посмотрите `git status`, затем:

```bash
git fetch origin
git switch my-feature
git pull --ff-only origin my-feature
```

Если локальной ветки ещё нет: `git switch --track origin/my-feature`.
Перед переключением сохраните свои изменения коммитом или stash.

Если изменения Docker получены отдельным архивом, распакуйте его в корень проекта
после обновления ветки. Архив заменяет только перечисленные в нём файлы;
свои изменения в них предварительно сохраните. `.env` и данные в архив не входят.

```bash
unzip /путь/к/my-feature-docker.zip -d /tmp/my-feature-docker
cp -r /tmp/my-feature-docker/. .
```

## 3. Настройки

```bash
python3 scripts/prepare_docker.py
```

Команда создаёт `.env`, `backend/.env`, `data/import` и `data/hf-cache`.
Для новой установки генерирует пароль PostgreSQL и API-ключ.
Существующие `.env` не перезаписываются.

Если `.env` уже был создан, проверьте в нём:

```dotenv
HF_CACHE_DIR=./data/hf-cache
FRONTEND_BIND_HOST=127.0.0.1
FRONTEND_PORT=8080
```

На Linux `UID` и `GID` в `.env` должны совпадать с выводом `id -u` и `id -g`;
каталог кэша должен быть доступен этому пользователю для записи.
Не оставляйте Windows-путь `C:/Users/...` из старого шаблона.

Адреса `localhost` в корневом `.env` предназначены для запуска Python на хосте.
Compose переопределяет адреса внутри backend именами `db`, `prices_db`,
`ml_service`, `llm_service`; вручную менять их не требуется.

API-ключ находится в `backend/.env`, строка `API_KEY=...`. Его нужно ввести на странице.
Не добавляйте `.env` в Git. Для уже созданных volumes PostgreSQL смена
`POSTGRES_PASSWORD` в файле сама по себе не меняет пароль в базе.

## 4. Базы и модели

docker compose up — запуск с выводом логов; Ctrl+C остановит контейнеры.
docker compose up -d — запуск в фоне.
docker compose up -d --build — пересобрать образы и запустить; нужно после изменений файлов, которые входят в образы.

```bash
docker compose config --quiet
docker compose build
docker compose up -d db prices_db ml_service llm_service
docker compose up -d tunnel
docker compose logs -f ml_service llm_service
```

Дождитесь загрузки моделей. Выход из просмотра логов — `Ctrl+C`, контейнеры
продолжат работать. При медленной первой загрузке healthcheck может временно
показывать `unhealthy`; проверьте логи и дождитесь готовности.

```bash
docker compose ps
curl -f http://127.0.0.1:8001/health
curl -f http://127.0.0.1:8002/health
```

## 5. Таблицы и справочники

Одного запуска Postgres недостаточно: нужны таблицы pgvector и данные.
Создание отсутствующих таблиц без удаления существующих:

```bash
docker compose run --rm --no-deps backend python -c 'from rag.db import get_session_factory, init_db; from rag.prices_db import get_prices_session_factory, init_prices_db; init_db(get_session_factory().kw["bind"]); init_prices_db(get_prices_session_factory().kw["bind"])'
```

Положите настоящий справочник в `data/import/pzr.xlsx` и прайс поставщика
в `data/import/materials.xls` (либо используйте своё имя и расширение).
Этот каталог смонтирован в backend как `/service/import` только для чтения.

Проверка структуры и загрузка справочника работ:

```bash
docker compose run --rm --no-deps backend python -m rag.cli validate /service/import/pzr.xlsx
docker compose run --rm --no-deps backend python -m rag.cli load /service/import/pzr.xlsx
```

Прайс материалов:

```bash
docker compose run --rm --no-deps backend python -m rag.cli validate-prices /service/import/materials.xls
docker compose run --rm --no-deps backend python -m rag.cli load-prices /service/import/materials.xls --supplier ros_elektro
```

Укажите фактический код поставщика. `load` заменяет справочник работ,
`load-prices` заменяет позиции выбранного поставщика. На существующей базе
сначала убедитесь, что хотите обновить данные. Если материалы пока не нужны,
оставьте их таблицы пустыми: неподобранные товары будут отмечены в смете.

Проверьте поиск:

```bash
docker compose run --rm --no-deps backend python -m rag.cli search 'смена розетки' --top-k 3
docker compose run --rm --no-deps backend python -m rag.cli list-suppliers
```

## 6. API и веб-интерфейс

```bash
docker compose up -d backend frontend
docker compose ps
curl -f http://127.0.0.1:8000/health
curl -f http://127.0.0.1:8080/healthz
```

На самом сервере веб-интерфейс доступен на http://127.0.0.1:8080.
По умолчанию порт привязан к localhost: открывать его в облачном firewall не требуется.

При работе с ВМ через VS Code Remote SSH на вкладке **Ports / Порты**
перенаправьте `8080`. Либо выполните **на своём компьютере**:

```bash
ssh -N -L 8080:127.0.0.1:8080 user1@IP_СЕРВЕРА
```

Если хост уже настроен в SSH config, например `vm1gpu`:

```bash
ssh -N -L 8080:127.0.0.1:8080 vm1gpu
```

Откройте http://localhost:8080, раскройте «Настройки подключения к API» и введите
`API_KEY` из `backend/.env`. Добавьте описание/фото и нажмите «Рассчитать стоимость».

Для Swagger отдельно перенаправьте `8000` и откройте http://localhost:8000/docs.
Авторизация — **Authorize**, ключ из `backend/.env`.

## Связь контейнеров и публичный доступ

Nginx в `frontend` отдаёт страницу на 8080 и пересылает `/api/v1/` в `backend:8000`.
Backend обращается к `db`, `prices_db`, `ml_service`, `llm_service`.
Nginx передаёт введённый пользователем `X-API-Key`; ключ не зашит в образ или JavaScript.
Backend возвращает `202` и ID, страница опрашивает заявку до `result_ready`.

Для публичного доступа по внешнему IP или домену нужен HTTPS перед интерфейсом:
не передавайте API-ключ по открытому HTTP через интернет. При внешнем reverse proxy
сохраните `FRONTEND_BIND_HOST=127.0.0.1`, направьте его на `127.0.0.1:8080`
и согласуйте лимит тела запроса 10 МиБ.

## Логи, остановка и обновление

```bash
docker compose logs --tail=100 backend frontend
docker compose logs --tail=100 ml_service llm_service
docker compose logs -f --tail=100 tunnel
docker compose down
```

`down` сохраняет volumes и кэш моделей. Не добавляйте `-v`, если хотите сохранить данные.
Для обновления после коммита изменений:

```bash
git pull --ff-only origin my-feature
docker compose up -d --build
```
тунель 
docker run -d \
  --name smeta-tunnel \
  --restart unless-stopped \
  --network ai-smetchik_default \
  cloudflare/cloudflared:latest \
  tunnel --url http://frontend:8080

docker start smeta-tunnel
docker logs --tail=100 -f smeta-tunnel

| Симптом | Что проверить |
| --- | --- |
| `could not select device driver ... gpu` | NVIDIA Container Toolkit и GPU в Docker |
| `Permission denied` в `/hf-cache` | Права на `data/hf-cache`, UID/GID в `.env` |
| `401 invalid_credentials` | API-ключ в настройках страницы |
| `503 authentication_not_configured` | Непустой `API_KEY` в `backend/.env`; пересоздайте backend |
| `rag_database_failed` | Таблицы в обеих БД; шаг 5 |
| Список работ пуст | Загрузка ПЗР, результат `rag.cli search` |
| `llm_failed` / CUDA out of memory | Логи LLM и свободная VRAM с учётом ML-сервиса |
| `413` | До 10 МиБ на весь JSON: base64 увеличивает размер фото примерно на треть |
