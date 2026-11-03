'use strict';

const $ = id => document.getElementById(id);
const selected = [];
const MAX_IMAGES = 4;
const MAX_IMAGE_BYTES = 5 * 1024 * 1024;
const MAX_REQUEST_BYTES = 10 * 1024 * 1024;
const READY = new Set(['result_ready', 'sending', 'delivered', 'delivery_failed']);
const STATUS_LABELS = {
  queued: 'Заявка в очереди…',
  received: 'Заявка принята…',
  processing: 'Анализируем фотографии и подбираем расценки…',
};
const ERRORS = {
  invalid_credentials: 'Неверный API-ключ. Проверьте настройки подключения.',
  authentication_not_configured: 'На backend не настроен API_KEY.',
  invalid_image: 'Нужны JPG, PNG или WEBP до 5 МиБ и 16 мегапикселей.',
  invalid_agent_input: 'Добавьте описание до 4000 символов или до 4 фотографий.',
  llm_failed: 'Ошибка LLM-сервиса. Проверьте его логи.',
  ml_failed: 'Ошибка сервиса эмбеддингов. Проверьте его логи.',
  rag_database_failed: 'Ошибка базы RAG. Проверьте создание таблиц и загрузку справочников.',
  agent_input_too_long: 'Вход модели слишком большой. Уменьшите число фото или описание.',
};

const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[char]));
const fmt = value => value == null || !Number.isFinite(Number(value))
  ? 'Не рассчитано'
  : new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 2 }).format(Number(value)) + ' ₽';
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function showError(message) {
  $('formError').innerHTML = '<div class="error">' + esc(message) + '</div>';
}

function addFiles(files) {
  $('formError').innerHTML = '';
  for (const file of files) {
    if (selected.length >= MAX_IMAGES) {
      showError('Можно загрузить не более 4 фотографий.');
      break;
    }
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)
        || file.size > MAX_IMAGE_BYTES) {
      showError('Поддерживаются JPG, PNG и WEBP до 5 МиБ на файл.');
      continue;
    }
    selected.push(file);
  }
  renderThumbs();
}

function renderThumbs() {
  $('thumbs').innerHTML = '';
  selected.forEach((file, index) => {
    const url = URL.createObjectURL(file);
    const element = document.createElement('div');
    element.className = 'thumb';
    const img = document.createElement('img');
    img.src = url;
    img.onload = img.onerror = () => URL.revokeObjectURL(url);
    img.alt = 'Загруженное фото';
    const button = document.createElement('button');
    button.textContent = '×';
    button.setAttribute('aria-label', 'Удалить фото');
    button.onclick = () => { selected.splice(index, 1); renderThumbs(); };
    element.append(img, button);
    $('thumbs').append(element);
  });
}

function loading(message, id = '') {
  $('output').className = 'empty';
  $('output').innerHTML = '<div class="loader"></div><strong>' + esc(message)
    + '</strong><p>Обработка может занять несколько минут. Не закрывайте страницу.</p>'
    + (id ? '<p class="muted">ID заявки: ' + esc(id) + '</p>' : '');
}

function resultLine(item, material = false) {
  const quantity = material ? item.buy_qty : item.quantity;
  const unit = material ? item.price_unit : item.unit;
  const price = material ? item.price_rub : item.unit_price_rub;
  const details = quantity == null ? 'Объём не определён' : quantity + ' ' + (unit ?? '');
  return '<div class="line"><span>' + esc(item.name)
    + '<div class="muted">' + esc(details) + ' · ' + esc(fmt(price))
    + (item.note ? '<br>' + esc(item.note) : '') + '</div></span><span>'
    + esc(fmt(item.amount_rub)) + '</span></div>';
}

function showResult(record) {
  const data = record.result;
  if (!data || typeof data !== 'object') throw new Error('Backend не вернул результат заявки.');
  const works = data.works ?? [];
  const materials = data.materials ?? [];
  const notices = [
    ...(data.warnings ?? []),
    ...(data.unknowns ?? []).map(text => 'Уточнить: ' + text),
    ...(data.assumptions ?? []).map(text => 'Допущение: ' + text),
    ...(data.unmatched_works ?? []).map(text => 'Работа без расценки: ' + text),
    ...(data.materials_not_found ?? []).map(item => 'Материал не подобран: ' + item.query),
  ];
  $('output').className = '';
  $('output').innerHTML = '<span class="pill">Предварительная смета</span>'
    + '<p class="muted">ID заявки: ' + esc(record.id) + '</p>'
    + (data.summary ? '<p>' + esc(data.summary) + '</p>' : '')
    + '<div class="muted">Итого по рассчитанным позициям</div>'
    + '<div class="total">' + esc(fmt(data.total_rub)) + '</div>'
    + '<div class="muted">Работы: ' + esc(fmt(data.works_total_rub))
    + ' · Материалы: ' + esc(fmt(data.materials_total_rub)) + '</div>'
    + '<h3>Работы</h3>'
    + (works.length ? works.map(item => resultLine(item)).join('')
      : '<p class="muted">Работы из справочника не подобраны.</p>')
    + (materials.length ? '<h3>Материалы</h3>'
      + materials.map(item => resultLine(item, true)).join('') : '')
    + (data.calculation_complete_for_selected_works === false
      ? '<p class="error">Расчёт работ неполный: проверьте объёмы и неподобранные позиции.</p>' : '')
    + (notices.length ? '<h3>Замечания</h3><ul class="muted">'
      + [...new Set(notices)].map(text => '<li>' + esc(text) + '</li>').join('') + '</ul>' : '')
    + '<p class="muted">Фактическую стоимость и состав работ должен проверить специалист.</p>';
}

function fileBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(',')[1]);
    reader.onerror = () => reject(new Error('Не удалось прочитать фотографию.'));
    reader.readAsDataURL(file);
  });
}

async function api(path, key, options = {}) {
  const response = await fetch('/api/v1/requests' + path, {
    ...options,
    headers: { 'Content-Type': 'application/json', 'X-API-Key': key },
    cache: 'no-store',
    signal: AbortSignal.timeout(60000),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    if (response.status === 413) throw new Error('Размер запроса превышает 10 МиБ. Уменьшите фото.');
    const detail = data?.detail;
    throw new Error(ERRORS[detail?.code] ?? detail?.message ?? ('Ошибка API: HTTP ' + response.status));
  }
  if (!data) throw new Error('API вернул ответ без JSON. Проверьте настройки Nginx.');
  return data;
}

async function waitForResult(id, key) {
  const deadline = Date.now() + 60 * 60 * 1000;
  while (Date.now() < deadline) {
    const record = await api('/' + encodeURIComponent(id), key);
    if (record.status === 'processing_failed') {
      throw new Error(ERRORS[record.last_error] ?? ('Ошибка обработки: ' + (record.last_error ?? 'неизвестна')));
    }
    // delivery_failed относится к Битрикс24: сама смета уже готова.
    if (READY.has(record.status)) return record;
    loading(STATUS_LABELS[record.status] ?? 'Ожидаем результат…', id);
    await sleep(2500);
  }
  throw new Error('Время ожидания истекло. Заявка может продолжать обрабатываться.');
}

$('drop').onclick = () => $('files').click();
$('drop').onkeydown = event => {
  if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); $('files').click(); }
};
$('files').onchange = event => { addFiles(event.target.files); event.target.value = ''; };
$('drop').ondragover = event => { event.preventDefault(); $('drop').classList.add('over'); };
$('drop').ondragleave = () => $('drop').classList.remove('over');
$('drop').ondrop = event => {
  event.preventDefault(); $('drop').classList.remove('over'); addFiles(event.dataTransfer.files);
};

$('submit').onclick = async () => {
  const description = $('description').value.trim();
  const key = $('apiKey').value.trim();
  if (!selected.length && !description) { showError('Добавьте хотя бы фотографию или описание.'); return; }
  if (description.length > 4000) { showError('Описание должно быть не длиннее 4000 символов.'); return; }
  if (!key) {
    $('connection').open = true;
    $('apiKey').focus();
    showError('Укажите API-ключ из backend/.env в настройках подключения.');
    return;
  }
  $('formError').innerHTML = '';
  $('submit').disabled = true;
  loading('Отправляем заявку…');
  let id = '';
  try {
    const images = await Promise.all(selected.map(fileBase64));
    const body = JSON.stringify({ data: { description, images } });
    if (new Blob([body]).size > MAX_REQUEST_BYTES) {
      throw new Error('Фото в base64 и описание занимают больше 10 МиБ. Уменьшите размер или число фото.');
    }
    const receipt = await api('', key, { method: 'POST', body });
    id = receipt.request_id;
    if (!id) throw new Error('API не вернул request_id.');
    loading('Заявка принята…', id);
    showResult(await waitForResult(id, key));
  } catch (error) {
    $('output').className = 'empty';
    $('output').innerHTML = '<div class="symbol">!</div><strong>Не удалось получить смету</strong>'
      + '<p>Проверьте ключ API и логи сервисов.</p>'
      + (id ? '<p class="muted">ID заявки: ' + esc(id) + '</p>' : '');
    showError(error.message + (id ? '\nID заявки: ' + id : ''));
  } finally {
    $('submit').disabled = false;
  }
};
