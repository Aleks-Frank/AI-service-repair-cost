"""Классификатор типа обращения по тексту заявки («Водоснабжение», «Освещение, электричество», …).

Модель — TF-IDF по буквенным n-граммам (устойчив к опечаткам и словоформам) + логистическая
регрессия; обучается на выгрузке задач Битрикс24 (колонки «Описание» и «Тип обращения»):
    python -m rag.cli train-classifier "<выгрузка>.xlsx"
    python -m rag.cli classify "Не горит лампа в подъезде"

Качество на отложенной выборке (20 % из 26,9 тыс. задач за 05–10.2026): верный тип первым —
86 % (91 % на задачах с АВР), среди выбранных по правилу COVERAGE — 96 % (98 % с АВР).

В конвейер сметы не подключён: фильтр поиска работ по разделам ПЗР по типу обращения проверен
на 50 заявках и не помог (совпало 44, лучше 1, хуже 3) — поиск и так находит работы нужного
раздела. Возможное применение — отсеивать заявки, по которым сметы обычно не бывает
(лифты, домофоны, уборка, отсутствие воды).
"""

import pickle
import re
from collections import Counter
from pathlib import Path

# Правило выбора типов: копим вероятность до COVERAGE, но не больше MAX_TYPES типов.
COVERAGE = 0.9
MAX_TYPES = 3
# Типы с меньшим числом примеров в обучении не используются (случайные значения поля).
MIN_EXAMPLES = 100


def clean_description(text: str) -> str:
    """Оставляет из описания задачи Битрикс24 только текст проблемы.

    В выгрузке описание — «Описание проблемы: … Комментарий: Подъезд … ФИО … телефон … ссылки».
    Всё после «Комментарий:» (контакты, ссылки на файлы) к сути заявки не относится.

    Принимает: text — описание задачи (или просто текст заявки).
    Возвращает: текст проблемы без префикса, служебной части и ссылок.
    """
    text = re.sub(r"^\s*Описание проблемы:\s*", "", str(text or ""))
    text = text.split("Комментарий:")[0]
    return re.sub(r"\s+", " ", re.sub(r"https?://\S+", " ", text)).strip()


class RequestClassifier:
    """Обученный классификатор типа обращения (TF-IDF + логистическая регрессия)."""

    def __init__(self, vectorizer, model):
        """Принимает: vectorizer — TfidfVectorizer; model — LogisticRegression. Возвращает: ничего."""
        self._vectorizer = vectorizer
        self._model = model

    @classmethod
    def load(cls, path: Path) -> "RequestClassifier":
        """Загружает модель из файла, сохранённого train().

        Файл — pickle: загружать только свой (pickle исполняет код при чтении).

        Принимает: path — путь к файлу модели.
        Возвращает: RequestClassifier.
        """
        with path.open("rb") as file:
            vectorizer, model = pickle.load(file)
        return cls(vectorizer, model)

    def predict(self, text: str) -> list[tuple[str, float]]:
        """Вероятности типов обращения для текста заявки.

        Принимает: text — текст заявки.
        Возвращает: список (тип, вероятность) по убыванию вероятности.
        """
        probabilities = self._model.predict_proba(self._vectorizer.transform([clean_description(text)]))[0]
        return sorted(zip(self._model.classes_, map(float, probabilities)), key=lambda pair: -pair[1])

    def likely_types(self, text: str) -> list[tuple[str, float]]:
        """Самые вероятные типы по правилу COVERAGE / MAX_TYPES.

        Принимает: text — текст заявки.
        Возвращает: 1–MAX_TYPES пар (тип, вероятность).
        """
        chosen, total = [], 0.0
        for request_type, probability in self.predict(text)[:MAX_TYPES]:
            chosen.append((request_type, round(probability, 3)))
            total += probability
            if total >= COVERAGE:
                break
        return chosen


def train(export_path: Path, model_path: Path) -> dict:
    """Обучает классификатор на выгрузке задач Битрикс24 и сохраняет его.

    Принимает:
        export_path — .xlsx с колонками «Описание» и «Тип обращения»;
        model_path — куда сохранить модель (папка создаётся).
    Возвращает: сводку {"examples", "types", "holdout_top1", "holdout_coverage"}.
    """
    import random

    from openpyxl import load_workbook
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    sheet = load_workbook(export_path, read_only=True, data_only=True).worksheets[0]
    sheet.reset_dimensions()
    rows = sheet.iter_rows(values_only=True)
    header = [str(value) for value in next(rows)]
    text_column, type_column = header.index("Описание"), header.index("Тип обращения")
    examples = [(clean_description(row[text_column]), row[type_column]) for row in rows if row[type_column]]
    counts = Counter(request_type for _, request_type in examples)
    examples = [(text, t) for text, t in examples if counts[t] >= MIN_EXAMPLES and len(text) >= 5]

    def fit(data: list[tuple[str, str]]) -> RequestClassifier:
        """Обучает модель на парах (текст, тип). Возвращает: RequestClassifier."""
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                     max_features=200_000, sublinear_tf=True)
        model = LogisticRegression(max_iter=2000, C=5).fit(vectorizer.fit_transform([t for t, _ in data]),
                                                           [label for _, label in data])
        return RequestClassifier(vectorizer, model)

    # Оценка на отложенных 20 %, затем итоговая модель — на всех данных.
    shuffled = examples[:]
    random.Random(0).shuffle(shuffled)
    split = int(len(shuffled) * 0.8)
    check = fit(shuffled[:split])
    holdout = shuffled[split:]
    top1 = sum(check.predict(text)[0][0] == label for text, label in holdout) / len(holdout)
    coverage = sum(label in {t for t, _ in check.likely_types(text)} for text, label in holdout) / len(holdout)

    final = fit(examples)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump((final._vectorizer, final._model), file)
    return {"examples": len(examples), "types": len(final._model.classes_),
            "holdout_top1": round(top1, 3), "holdout_coverage": round(coverage, 3)}
