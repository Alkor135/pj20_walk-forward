"""
Генерирует файл предсказания направления цены на текущую торговую дату.

Читает sentiment_scores.pkl, берет строку за сегодня (одна дата - одна строка),
применяет правила из rules.yaml и пишет текстовый файл <predict_path>/YYYY-MM-DD.txt
в формате:

    Дата: 2026-04-09
    Sentiment: -4.00
    Action: invert
    Status: ok
    Предсказанное направление: up

Файл пишется всегда, включая нештатные ситуации. Направление = skip во всех
не-ok случаях; причина записывается в Status, подробности - в Note.

Скрипт всегда возвращает 0, чтобы сбой sentiment по одной модели не останавливал
внешний пайплайн.
"""

from __future__ import annotations

import logging
import pickle
from datetime import date, datetime
from pathlib import Path
import sys

import pandas as pd
import yaml

MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_DIR.parent))
from config_loader import load_settings_for
LOG_DIR = MODEL_DIR / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

VALID_ACTIONS = {"follow", "invert"}


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


def load_yaml(path: Path) -> dict:
    """Читает YAML-файл как dict."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(path: Path | None = None) -> dict:
    """Загружает настройки модели из единого {ticker}/settings.yaml."""
    return load_settings_for(__file__, "model")


def resolve_sentiment_pkl(settings: dict, base_dir: Path = MODEL_DIR) -> Path:
    """Возвращает путь к PKL с sentiment-оценками."""
    sentiment_path = Path(settings.get("sentiment_output_pkl", "sentiment_scores.pkl"))
    return sentiment_path if sentiment_path.is_absolute() else base_dir / sentiment_path


def load_rules(path: Path) -> list[dict]:
    """Загружает и валидирует rules.yaml для прогноза."""
    data = load_yaml(path)
    rules = data.get("rules") or []
    if not isinstance(rules, list) or not rules:
        raise ValueError(f"В {path} нет списка 'rules' или он пустой")
    for i, rule in enumerate(rules):
        for key in ("min", "max", "action"):
            if key not in rule:
                raise ValueError(f"Правило #{i} без поля '{key}': {rule}")
        if rule["action"] not in VALID_ACTIONS:
            raise ValueError(
                f"Правило #{i}: action должен быть одним из {sorted(VALID_ACTIONS)}, "
                f"получено {rule['action']!r}"
            )
        if float(rule["min"]) > float(rule["max"]):
            raise ValueError(f"Правило #{i}: min > max ({rule})")
    return rules


def match_action(sentiment: float, rules: list[dict]) -> str | None:
    """Возвращает action первого подходящего правила."""
    for rule in rules:
        if float(rule["min"]) <= sentiment <= float(rule["max"]):
            return rule["action"]
    return None


def resolve_direction(sentiment: float, action: str) -> str:
    """Преобразует sentiment и action в направление up/down."""
    if action == "follow":
        return "up" if sentiment >= 0 else "down"
    if action == "invert":
        return "down" if sentiment >= 0 else "up"
    return "skip"


def get_today_sentiment(pkl_path: Path, today: date) -> float | None:
    """Возвращает sentiment за указанную дату из PKL."""
    if not pkl_path.exists():
        raise FileNotFoundError(f"Файл sentiment PKL не найден: {pkl_path}")
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    df = pd.DataFrame(data)
    if "source_date" not in df.columns or "sentiment" not in df.columns:
        raise ValueError(
            f"PKL не содержит обязательные колонки 'source_date'/'sentiment': {pkl_path}"
        )

    df["source_date"] = pd.to_datetime(df["source_date"], errors="coerce").dt.date
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    df = df.dropna(subset=["source_date", "sentiment"])

    today_rows = df[df["source_date"] == today]
    if today_rows.empty:
        return None
    if len(today_rows) > 1:
        raise ValueError(
            f"В pkl несколько строк за {today}: ожидалась одна. "
            "Перегенерируй pkl: sentiment_analysis.py хранит одну строку на дату."
        )
    return float(today_rows["sentiment"].iloc[0])


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


def write_predict(
    out_file: Path,
    date_str: str,
    direction: str,
    status: str,
    sentiment: float | None = None,
    action: str | None = None,
    note: str = "",
) -> None:
    """Атомарно записывает текстовый файл прогноза."""
    sentiment_label = f"{sentiment:.2f}" if sentiment is not None else "n/a"
    action_label = action if action is not None else "n/a"
    lines = [
        f"Дата: {date_str}",
        f"Sentiment: {sentiment_label}",
        f"Action: {action_label}",
        f"Status: {status}",
    ]
    if note:
        lines.append(f"Note: {note}")
    lines.append(f"Предсказанное направление: {direction}")
    content = "\n".join(lines) + "\n"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = out_file.with_suffix(out_file.suffix + ".tmp")
    tmp_file.write_text(content, encoding="utf-8")
    tmp_file.replace(out_file)


def main() -> int:
    """Основной сценарий генерации прогноза на текущую дату."""
    logger = setup_logging()

    try:
        settings = load_settings()
        predict_path = Path(settings["predict_path"])
        predict_path.mkdir(parents=True, exist_ok=True)

        today = date.today()
        date_str = today.strftime("%Y-%m-%d")
        out_file = predict_path / f"{date_str}.txt"

        if out_file.exists():
            if should_delete_existing_predict_file(out_file, today, settings["time_start"]):
                out_file.unlink()
                logger.info(
                    "Файл %s создан сегодня до %s (тестовый) - удаляем перед созданием.",
                    out_file,
                    settings["time_start"],
                )
            else:
                logger.info("Файл %s уже существует - пропуск.", out_file)
                return 0

        rules = load_rules(MODEL_DIR / "rules.yaml")
        pkl_path = resolve_sentiment_pkl(settings)

        try:
            sentiment = get_today_sentiment(pkl_path, today)
        except FileNotFoundError as exc:
            logger.error("pkl_missing: %s", exc)
            write_predict(out_file, date_str, "skip", "pkl_missing", note=str(exc))
            return 0
        except ValueError as exc:
            msg = str(exc)
            status = "pkl_duplicate" if "несколько строк" in msg else "error"
            logger.error("%s: %s", status, msg)
            write_predict(out_file, date_str, "skip", status, note=msg)
            return 0

        if sentiment is None:
            logger.info("В pkl нет записи за %s.", today)
            write_predict(
                out_file,
                date_str,
                "skip",
                "no_pkl_row",
                note=f"в sentiment_scores.pkl нет строки за {date_str}",
            )
            return 0

        action = match_action(sentiment, rules)
        if action is None:
            logger.info(
                "%s: sentiment=%.2f не попал ни в один диапазон rules.yaml.",
                today,
                sentiment,
            )
            write_predict(
                out_file,
                date_str,
                "skip",
                "no_rule_match",
                sentiment=sentiment,
                note="sentiment вне всех диапазонов rules.yaml",
            )
            return 0

        direction = resolve_direction(sentiment, action)
        logger.info("%s: sentiment=%.2f, action=%s, direction=%s", today, sentiment, action, direction)

        if direction == "skip":
            note = f"не удалось определить направление для action={action!r}"
            write_predict(
                out_file,
                date_str,
                "skip",
                "error",
                sentiment=sentiment,
                action=action,
                note=note,
            )
            return 0

        write_predict(out_file, date_str, direction, "ok", sentiment=sentiment, action=action)
        logger.info("Записан файл предсказания: %s", out_file)
        return 0

    except Exception as exc:
        logger.exception("Необработанная ошибка sentiment_to_predict")
        try:
            today = date.today()
            date_str = today.strftime("%Y-%m-%d")
            settings = load_settings()
            out_file = Path(settings.get("predict_path", str(MODEL_DIR / "predict"))) / f"{date_str}.txt"
            write_predict(out_file, date_str, "skip", "error", note=f"{type(exc).__name__}: {exc}")
        except Exception as write_exc:
            logger.error("Не удалось записать файл предсказания с ошибкой: %s", write_exc)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
