# Смета AI — веб-интерфейс

Статические HTML/CSS/JavaScript, без Node.js, npm и сборщика. Nginx в Docker
отдаёт страницу и проксирует `/api/v1/` на `backend:8000` без изменения пути.

Запуск всего проекта описан в [DEPLOYMENT.md](../DEPLOYMENT.md).

После запуска откройте http://localhost:8080 и в «Настройках подключения к API»
введите значение `API_KEY` из `backend/.env`. Ключ передаётся в заголовке
`X-API-Key`, не включается в сборку, не сохраняется в localStorage или cookies.
Для удалённого подключения используйте SSH-туннель или HTTPS.

Контракт соответствует backend ветки `my-feature`:

```text
POST /api/v1/requests
X-API-Key: <ключ>
Content-Type: application/json

{"data":{"description":"Заменить 1 розетку","images":["<base64 без data:-префикса>"]}}

202 -> {"request_id":"<uuid>","status":"queued","duplicate":false}

GET /api/v1/requests/<uuid>
X-API-Key: <ключ>

200 -> {"id":"<uuid>","status":"result_ready","result":{...}}
```

Страница опрашивает статус до часа, обрабатывает `processing_failed`,
показывает работы, материалы, замечания и частичные итоги из `result`.
Неизвестные суммы отображаются как «Не рассчитано».

Ограничения backend: до 4 фото JPG/PNG/WEBP, до 5 МиБ и 16 Мп на фотографию,
до 4000 символов описания, до 10 МиБ на весь JSON с base64.

`python -m http.server` отдаст статические файлы, но для API нужен Nginx-прокси;
для работы с реальным backend используйте сервис `frontend` в Docker Compose.
