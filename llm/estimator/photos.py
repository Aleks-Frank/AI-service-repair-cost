"""Фотографии заявки: принадлежность заявке, отсутствующие файлы и снимки «после работ».

Ошибки разложения фото (чужие снимки, пропуски, фото «после» как исходное состояние) искажали
выводы моделей. Здесь соответствие проверяется явно, ещё до вызова модели:
    - файл должен принадлежать заявке: имя начинается с «<заявка>_» (a03_1.png — только для a03);
    - отсутствующий файл не пропускается молча — предупреждение;
    - снимок «после работ» в модель не отправляется: состояние задаётся только явной меткой
      (state = "after" в данных заявки), порядок файлов доказательством «до/после» не считается;
    - пустой список фото допустим — смета строится по тексту;
    - manifest фиксирует, какой файл стал «изображением N» промпта (для трассировки и аудита).
Файлы не изменяются и не удаляются.
"""

from pathlib import Path

PHOTO_STATES = ("before", "after", "unknown")


def resolve_photos(request_name: str, entries: list, base_dir: Path) -> tuple[list[Path], list[dict], list[str]]:
    """Проверяет фото одной заявки и возвращает те, что можно отправить модели.

    Принимает:
        request_name — имя заявки (a03, b12, r05);
        entries — список путей (строки) или словарей {"file": путь, "state": before/after/unknown};
            пути — относительно base_dir;
        base_dir — папка набора данных.
    Возвращает: (пути фото для модели в порядке нумерации, manifest
        [{"image_number", "file", "state"}], предупреждения).
    """
    paths: list[Path] = []
    manifest: list[dict] = []
    warnings: list[str] = []
    for entry in entries:
        file, state = (entry, "unknown") if isinstance(entry, str) else (entry.get("file", ""), entry.get("state", "unknown"))
        if state not in PHOTO_STATES:
            state = "unknown"
        path = base_dir / file
        if not path.name.startswith(f"{request_name}_"):
            warnings.append(f"{request_name}: фото {file} не принадлежит заявке (ожидается имя «{request_name}_…») — исключено.")
            continue
        if not path.is_file():
            warnings.append(f"{request_name}: фото {file} не найдено — заявка обрабатывается без него.")
            continue
        if state == "after":
            warnings.append(f"{request_name}: фото {file} помечено «после работ» — не используется как исходное состояние.")
            continue
        paths.append(path)
        manifest.append({"image_number": len(paths), "file": str(file), "state": state})
    return paths, manifest, warnings
