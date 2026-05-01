"""
Генерирует комбинированный файл предсказания на текущую торговую дату.

Скрипт читает за сегодня файлы предсказаний двух моделей из settings.yaml
(`model_1`, `model_2`), извлекает из каждого направление (`up`/`down`/`skip`),
вырезает строку `Предсказанное направление: ...` и склеивает оставшееся
содержимое в один файл `<predict_path>/YYYY-MM-DD.txt`. В конец добавляется
комбинированное направление:

- оба `up`   → `up`
- оба `down` → `down`
- иначе      → `skip`

Папки моделей вычисляются от `predict_path` комбинации:
`<predict_path>.parent / <model_folder>` (имя папки модели = `model.replace(":", "_")`).

Защита от перезаписи: если файл прогноза за сегодня уже существует и создан
после `time_start` (из settings.yaml) — скрипт ничего не делает (это
"настоящий" прогноз). Если создан раньше — перезаписывает (это тестовый
прогон или старый файл). Если `time_start` не задан — всегда перезаписывает
с предупреждением в лог.

Скрипт всегда возвращает 0, чтобы сбой не останавливал внешний пайплайн.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from pathlib import Path
import sys

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
TICKER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(TICKER_DIR))
from config_loader import load_settings_for
LOG_DIR = SCRIPT_DIR / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

VALID_DIRECTIONS = {"up", "down", "skip"}
DIRECTION_LINE_RE = re.compile(
    r"^[ \t]*Предсказанное направление[ \t]*:[ \t]*(\S+)[ \t]*\r?\n?",
    re.MULTILINE,
)


def cleanup_old_logs(log_dir: Path, max_files: int = 3) -> None:
    """Оставляет только последние max_files логов sentiment_to_predict."""
    log_files = sorted(
        log_dir.glob("sentiment_to_predict_*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in log_files[max_files:]:
        try:
            old.unlink()
        except Exception as exc:
            print(f"Не удалось удалить старый лог {old}: {exc}")


def setup_logging() -> logging.Logger:
    """Настраивает логирование в новый файл и в консоль."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = LOG_DIR / f"sentiment_to_predict_{timestamp}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    cleanup_old_logs(LOG_DIR)
    return logging.getLogger(__name__)


def model_folder_name(model: str) -> str:
    """Имя папки модели: `gemma3:12b` -> `gemma3_12b`."""
    return model.replace(":", "_")


def predict_file_date(path: Path) -> date | None:
    """Возвращает дату из имени YYYY-MM-DD.txt или None для чужого формата."""
    try:
        return datetime.strptime(path.stem, "%Y-%m-%d").date()
    except ValueError:
        return None


def should_delete_existing_predict_file(out_file: Path, today: date, time_start: str) -> bool:
    """Удаляем только сегодняшний файл прогноза, созданный сегодня до time_start."""
    if predict_file_date(out_file) != today:
        return False

    cutoff = datetime.combine(today, datetime.strptime(time_start, "%H:%M:%S").time())
    file_mtime = datetime.fromtimestamp(out_file.stat().st_mtime)
    return file_mtime.date() == today and file_mtime < cutoff


def load_settings(path: Path | None = None) -> dict:
    """Загружает настройки combine из единого {ticker}/settings.yaml."""
    return load_settings_for(__file__, "combine")


def parse_predict_file(content: str) -> tuple[str, str | None]:
    """Возвращает (текст_с_заменённой_строкой_направления, direction|None).

    Строка `Предсказанное направление: <X>` заменяется на `Направление: <X>`
    (само направление сохраняется в блоке, но уже без ключевых слов). Если
    значение направления не валидно — строка удаляется, direction=None.
    """
    match = DIRECTION_LINE_RE.search(content)
    if not match:
        return content.rstrip() + "\n", None
    direction_raw = match.group(1).strip().lower()
    direction = direction_raw if direction_raw in VALID_DIRECTIONS else None
    if direction is not None:
        replacement = f"Направление: {direction}\n"
        cleaned = content[: match.start()] + replacement + content[match.end() :]
    else:
        cleaned = content[: match.start()] + content[match.end() :]
    return cleaned.rstrip() + "\n", direction


def combine_directions(d1: str | None, d2: str | None) -> str:
    """up+up → up; down+down → down; иначе → skip."""
    if d1 == "up" and d2 == "up":
        return "up"
    if d1 == "down" and d2 == "down":
        return "down"
    return "skip"


def write_combined_predict(
    out_file: Path,
    date_str: str,
    blocks: list[tuple[str, str]],
    direction: str,
) -> None:
    """Атомарно записывает комбинированный файл прогноза."""
    parts: list[str] = [f"Дата: {date_str}\n"]
    for label, body in blocks:
        parts.append(f"\n--- {label} ---\n")
        parts.append(body if body.endswith("\n") else body + "\n")
    parts.append(f"\nПредсказанное направление: {direction}\n")
    content = "".join(parts)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = out_file.with_suffix(out_file.suffix + ".tmp")
    tmp_file.write_text(content, encoding="utf-8")
    tmp_file.replace(out_file)


def read_model_block(
    label: str, model_path: Path, date_str: str, logger: logging.Logger
) -> tuple[str, str | None]:
    """Читает файл предсказания модели за сегодня и возвращает (block_text, direction)."""
    file = model_path / f"{date_str}.txt"
    if not file.exists():
        logger.warning("Файл %s не найден — направление модели = skip.", file)
        return f"Status: file_missing\nNote: {file} не найден\n", None
    try:
        content = file.read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("Не удалось прочитать %s: %s", file, exc)
        return f"Status: read_error\nNote: {exc}\n", None

    cleaned, direction = parse_predict_file(content)
    if direction is None:
        logger.warning(
            "В %s не нашёл валидной строки 'Предсказанное направление' — направление = skip.",
            file,
        )
    else:
        logger.info("%s: direction=%s", label, direction)
    return cleaned, direction


def main() -> int:
    """Основной сценарий генерации комбинированного прогноза на текущую дату."""
    logger = setup_logging()
    try:
        settings = load_settings()
        model_1 = settings.get("model_1")
        model_2 = settings.get("model_2")
        predict_path_raw = settings.get("predict_path")
        if not model_1 or not model_2:
            logger.error("В settings.yaml должны быть указаны model_1 и model_2.")
            return 0
        if not predict_path_raw:
            logger.error("В settings.yaml должен быть указан predict_path.")
            return 0

        folder_1 = model_folder_name(str(model_1))
        folder_2 = model_folder_name(str(model_2))

        predict_path = Path(predict_path_raw)
        base_dir = predict_path.parent
        model_predict_1 = base_dir / folder_1
        model_predict_2 = base_dir / folder_2

        today = date.today()
        date_str = today.strftime("%Y-%m-%d")
        out_file = predict_path / f"{date_str}.txt"

        time_start = settings.get("time_start")
        if out_file.exists():
            if not time_start:
                logger.warning(
                    "В settings.yaml не задан time_start — файл %s будет перезаписан без проверки.",
                    out_file,
                )
            elif should_delete_existing_predict_file(out_file, today, time_start):
                out_file.unlink()
                logger.info(
                    "Файл %s создан сегодня до %s (тестовый) — удаляем перед созданием.",
                    out_file,
                    time_start,
                )
            else:
                logger.info("Файл %s уже существует и создан после %s — пропуск.", out_file, time_start)
                return 0

        blocks: list[tuple[str, str]] = []
        directions: list[str | None] = []
        for label, model_path in [
            (folder_1, model_predict_1),
            (folder_2, model_predict_2),
        ]:
            block_text, direction = read_model_block(label, model_path, date_str, logger)
            blocks.append((label, block_text))
            directions.append(direction)

        combined = combine_directions(directions[0], directions[1])
        write_combined_predict(out_file, date_str, blocks, combined)
        logger.info(
            "Записан комбинированный файл: %s (direction=%s, dir_1=%s, dir_2=%s)",
            out_file,
            combined,
            directions[0],
            directions[1],
        )
        return 0
    except Exception as exc:
        logger.exception("Необработанная ошибка sentiment_to_predict (combine)")
        try:
            today = date.today()
            date_str = today.strftime("%Y-%m-%d")
            settings = load_settings()
            out_file = (
                Path(settings.get("predict_path", str(SCRIPT_DIR / "predict")))
                / f"{date_str}.txt"
            )
            error_block = f"Status: error\nNote: {type(exc).__name__}: {exc}\n"
            write_combined_predict(
                out_file, date_str, [("error", error_block)], "skip"
            )
        except Exception as write_exc:
            logger.error("Не удалось записать файл предсказания с ошибкой: %s", write_exc)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
