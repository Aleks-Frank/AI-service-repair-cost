# Смета AI — одностраничный frontend

Откройте `index.html` через локальный HTTP-сервер:

```bash
cd repair-estimate-frontend
python3 -m http.server 5173
```

Интерфейс: http://localhost:5173

## Подключение

Внизу страницы есть поле адреса API. По умолчанию используется origin страницы. Для реального запуска рекомендуется проксировать `/api/` через Nginx на backend на том же домене.

**Важно:** frontend использует примерный контракт `POST /api/v1/estimates` с JSON `{description, images: [base64...]}` и `GET /api/v1/estimates/{id}`. Его необходимо адаптировать к фактическим маршрутам и полям вашего backend. Не помещайте `X-API-Key` в публичный JavaScript: используйте защищённую серверную прослойку с авторизацией и лимитами.
