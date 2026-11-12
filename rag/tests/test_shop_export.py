"""Тесты разбора выгрузок каталогов магазинов (rag/price_lists/shop_export.py) — без базы."""

from openpyxl import Workbook

from rag.price_lists import shop_export


def write_xlsx(path, header, rows):
    """Сохраняет таблицу в .xlsx (как выгрузка магазина: первая строка — заголовки)."""
    book = Workbook()
    sheet = book.active
    sheet.append(header)
    for row in rows:
        sheet.append(row)
    book.save(path)


def test_html_removed_from_spec_columns(tmp_path):
    """HTML в характеристиках («Комплектация», «Преимущества» у «Кувалды») не попадает в поиск."""
    path = tmp_path / "kuvalda.xlsx"
    write_xlsx(path, ["Название", "Артикул", "Цена", "Наличие", "Категория", "Комплектация", "Мощность", "Описание"], [
        ["Прожектор светодиодный 50 Вт", "327828", "990", "Да", "Освещение",
         '<div class="kit-list__header">Прожектор, крепёж</div>', "50 Вт", "<p>Для улицы</p>"],
    ])
    assert shop_export.is_shop_export(path)
    material, = shop_export.parse(path).materials
    assert "<" not in material.details and "kit-list" not in material.details
    assert "Комплектация: Прожектор, крепёж" in material.details and "Мощность: 50 Вт" in material.details
    assert material.description == "Для улицы"


def test_availability_and_duplicates(tmp_path):
    """«Наличие: Нет» — остаток 0; повтор артикула (товар в двух разделах) пропускается."""
    path = tmp_path / "shop.xlsx"
    write_xlsx(path, ["Название", "Артикул", "Цена", "Наличие"], [
        ["Лампа E27", "1", "100", "Нет"],
        ["Лампа E27", "1", "100", "Да"],
    ])
    result = shop_export.parse(path)
    assert [m.stock for m in result.materials] == [0]
    assert result.skipped == [(3, "duplicate_article")]


def test_price_unit_from_price_note(tmp_path):
    """«Цена указана: за 1 метр» (лента у «Светодиод66») — цена за метр, а не за катушку."""
    path = tmp_path / "svetodiod66.xlsx"
    write_xlsx(path, ["Название", "Артикул", "Цена", "Наличие", "Цена указана"], [
        ["Лента светодиодная 12V 9,6W/m 50м", "Б0059924", "100", "Да", "за 1 метр"],
        ["Лента светодиодная 24V 5м", "L2", "90", "Да", "за 1м"],
        ["Комплект ленты 5м с блоком питания", "K1", "1500", "Да", "за комплект"],
        ["Лампа светодиодная T8 G13", "T8", "150", "Да", ""],
        ["Шинопровод 2000 мм", "SH2", "1400", "Да", "за 2м"],
        ["Гибкий неон 50м", "N1", "220", "Да", "за 1 метр неона"],
    ])
    units = {m.article: m.unit for m in shop_export.parse(path).materials}
    assert units == {"Б0059924": "м", "L2": "м", "K1": "компл", "T8": "шт", "SH2": "шт", "N1": "м"}
    assert all("Цена указана" not in m.details for m in shop_export.parse(path).materials)


def test_sale_multiple_makes_pack(tmp_path):
    """Цена за метр + «Продажа: упаковка 5м» — покупка упаковкой 5 м, цена за метр не меняется."""
    path = tmp_path / "svetodiod66.xlsx"
    write_xlsx(path, ["Название", "Артикул", "Цена", "Наличие", "Цена указана", "Продажа"], [
        ["Светодиодная лента ЭРА 12V 5м", "E1", "105", "Да", "за 1м", "упаковка 5м"],
        ["Гибкий неон 220V 50м", "N1", "240", "Да", "за 1м", "кратно 50м"],
        ["Лента светодиодная 12V катушка", "L0", "130", "Да", "", "катушка 5 метров"],
        ["Лента светодиодная 24V", "L1", "300", "Да", "за 1 метр", "от 1 метра"],
        ["Профиль алюминиевый 2м", "P1", "500", "Да", "", "1шт = 2000мм"],
    ])
    items = {m.article: m for m in shop_export.parse(path).materials}
    tape = items["E1"]
    assert (tape.unit, tape.price, tape.pack_size, tape.pack_unit) == ("уп", 525, 5, "м")
    assert tape.pricing == ("м", 5, True) and tape.base_price == 105
    assert items["N1"].price == 240 * 50 and items["N1"].pack_size == 50
    assert items["L0"].unit == "уп" and items["L0"].base_price == 130  # пустое «Цена указана» у ленты — за метр
    assert items["L1"].unit == "м" and items["L1"].price == 300        # «от 1 метра» — без кратности
    assert items["P1"].unit == "шт" and items["P1"].price == 500       # отрезок-штука
