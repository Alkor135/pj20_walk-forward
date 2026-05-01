"""Собирает sentiment-оценки новостей из markdown-файлов через локальную модель Ollama.

Скрипт читает настройки из единого `rts/settings.yaml`, строит жесткий
промпт для модели, вызывает локальный `/api/generate`, строго парсит ответ
как одно число и сохраняет результаты в PKL.

Дополнительно скрипт:
- пересчитывает новые и измененные markdown-файлы по `content_hash`;
- добавляет рыночные признаки из дневной SQLite-базы котировок;
- сохраняет `raw_response` для последующего аудита качества ответов модели.
"""

from __future__ import annotations

import hashlib
import logging
import math
import pickle
import re
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd
import requests
import tiktoken
import typer
import yaml
from tqdm import tqdm

TICKER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TICKER_DIR.parent))
from config_loader import load_settings_for

app = typer.Typer(help="Собирает sentiment-оценки новостей через локальную модель Ollama.")

# DEFAULT_PROMPT_TEMPLATE = (
#     "Оцени влияние новости на {ticker} по шкале от -10 до +10.\n"
#     "-10 = сильно негативно, 0 = нейтрально, +10 = сильно позитивно.\n\n"
#     "Текст новости:\n{news_text}\n\n"
#     "Ответ: только одно целое число от -10 до +10."
# )

DEFAULT_PROMPT_TEMPLATE = (
    "Оцени влияние на {ticker} от -10 до +10.\n\n"
    "Текст новости:\n\n{news_text}\n\n"
    "Верни только одно число от -10 до +10 без пояснений."
)

DEFAULT_TOKEN_LIMIT = 16000
STRICT_NUMBER_REGEX = re.compile(r"^\s*([+-]?\d+(?:[.,]\d+)?)\s*$")


def cleanup_old_logs(log_dir: Path, max_files: int = 3) -> None:
    """Удаляет старые log-файлы, оставляя только несколько самых свежих."""
    log_files = sorted(log_dir.glob("sentiment_analysis_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old_file in log_files[max_files:]:
        try:
            old_file.unlink()
        except OSError as exc:
            print(f"Не удалось удалить старый лог {old_file}: {exc}")


def setup_logging(ticker_label: str, verbose: bool = False) -> None:
    """Настраивает файловое и консольное логирование для запуска скрипта."""
    level = logging.DEBUG if verbose else logging.INFO
    log_dir = Path(__file__).resolve().parent / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"sentiment_analysis_{timestamp}.txt"
    cleanup_old_logs(log_dir, max_files=3)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.info("[%s] Запуск sentiment_analysis. Лог: %s", ticker_label, log_file)


def load_settings() -> dict:
    """Загружает настройки модели из единого {ticker}/settings.yaml."""
    return load_settings_for(__file__, "model")


def find_md_files(md_dir: Path) -> list[Path]:
    """Возвращает отсортированный список markdown-файлов в каталоге новостей."""
    return sorted(path for path in md_dir.rglob("*.md") if path.is_file())


def read_markdown(path: Path) -> str:
    """Читает markdown-файл как UTF-8 текст и убирает лишние пробелы по краям."""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def compute_content_hash(path: Path) -> str:
    """Считает SHA-256 хэш содержимого файла для контроля изменений."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_prompt(ticker: str, prompt_template: str, news_text: str) -> str:
    """Подставляет тикер и текст новости в шаблон промпта."""
    return prompt_template.format(ticker=ticker, news_text=news_text)


def get_token_count(text: str) -> int:
    """Оценивает число токенов в тексте через tiktoken."""
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def warn_if_token_limit_exceeded(prompt: str, token_limit: int, file_name: str) -> int:
    """Логирует предупреждение, если промпт превышает заданный порог токенов."""
    prompt_tokens = get_token_count(prompt)
    if prompt_tokens > token_limit:
        logging.warning(
            "Промпт для %s содержит %s токенов, превышает порог %s. Возможны обрезание или плохой ответ.",
            file_name,
            prompt_tokens,
            token_limit,
        )
    return prompt_tokens


def round_half_away_from_zero(value: float) -> int:
    """Округляет число до ближайшего целого по правилу half away from zero."""
    if value >= 0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def parse_sentiment_strict(response: str) -> Optional[int]:
    """Строго парсит ответ модели как одно число и возвращает целый sentiment."""
    if not response:
        return None
    match = STRICT_NUMBER_REGEX.fullmatch(response)
    if not match:
        return None
    value = match.group(1).replace(",", ".")
    try:
        score = float(value)
    except ValueError:
        return None
    rounded = round_half_away_from_zero(score)
    return max(min(rounded, 10), -10)


def extract_date_from_path(path: Path) -> Optional[str]:
    """Извлекает дату формата YYYY-MM-DD из пути к markdown-файлу."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})", str(path))
    return match.group(1) if match else None


def parse_ollama_processor_status(ps_output: str, model: str) -> str:
    """Извлекает CPU/GPU-размещение модели из вывода `ollama ps`."""
    processor_pattern = re.compile(r"((?:\d+%/\d+%\s+CPU/GPU)|(?:\d+%\s+(?:CPU|GPU)))\s+\d+")
    for line in ps_output.splitlines():
        if not line.strip().startswith(model):
            continue
        match = processor_pattern.search(line)
        if match:
            return match.group(1)
    return "not loaded"


def get_ollama_processor_status(model: str) -> str:
    """Возвращает CPU/GPU-размещение модели из `ollama ps` без остановки пайплайна при ошибке."""
    try:
        completed = subprocess.run(
            ["ollama", "ps"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except Exception as exc:
        return f"unavailable ({exc.__class__.__name__})"
    return parse_ollama_processor_status(completed.stdout, model)


def run_ollama(model: str, prompt: str, keepalive: Optional[str] = None, timeout: int = 60) -> str:
    """Вызывает Ollama HTTP API с детерминированными параметрами генерации."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "seed": 42,
        },
    }
    if keepalive:
        payload["keep_alive"] = keepalive
    response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    return (response.json().get("response") or "").strip()


def load_existing_results(path: Path) -> pd.DataFrame:
    """Загружает ранее сохраненные результаты из PKL, если файл существует."""
    if not path.exists():
        return pd.DataFrame()
    with path.open("rb") as file_obj:
        return pd.DataFrame(pickle.load(file_obj))


def should_process_file(md_file: Path, existing_df: pd.DataFrame) -> bool:
    """Определяет, нужно ли пересчитывать файл по наличию и content_hash."""
    if existing_df.empty:
        return True
    md_file_path = str(md_file.resolve())
    matches = existing_df[existing_df["file_path"] == md_file_path]
    if matches.empty:
        return True
    current_hash = compute_content_hash(md_file)
    stored_hash = matches.iloc[-1].get("content_hash")
    return stored_hash != current_hash


def _resolve_with_gdrive_suffix(path: Path) -> Optional[Path]:
    """Возвращает path или его дубликат вида `stem (N).ext` (Google Drive sync).

    Why: на машинах с синхронизацией через Google Drive оригинальный файл
    иногда отсутствует, а рядом лежит копия с суффиксом `(1)` / `(2)` и т.п.
    How to apply: если path существует — возвращаем как есть; иначе ищем
    в той же папке самый свежий файл с подходящим суффиксом.
    """
    if path.exists():
        return path
    parent = path.parent
    if not parent.exists():
        return None
    candidates = sorted(
        parent.glob(f"{path.stem} (*){path.suffix}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def attach_market_features(df: pd.DataFrame, quotes_path: Path) -> pd.DataFrame:
    """Добавляет к sentiment-таблице рыночные признаки из SQLite-базы котировок."""
    if df.empty:
        return df
    resolved = _resolve_with_gdrive_suffix(quotes_path)
    if resolved is None:
        logging.warning("Файл котировок не найден: %s. Пропускаю добавление рыночных признаков.", quotes_path)
        return df
    if resolved != quotes_path:
        logging.warning("Файл котировок %s отсутствует, использую дубликат %s.", quotes_path.name, resolved.name)
        quotes_path = resolved

    with sqlite3.connect(str(quotes_path)) as conn:
        quotes_df = pd.read_sql_query(
            "SELECT TRADEDATE, OPEN, CLOSE FROM Futures",
            conn,
            parse_dates=["TRADEDATE"],
        )

    quotes_df = quotes_df.dropna(subset=["TRADEDATE", "OPEN", "CLOSE"]).sort_values("TRADEDATE").reset_index(drop=True)
    quotes_df["body"] = quotes_df["CLOSE"] - quotes_df["OPEN"]
    quotes_df["date_only"] = quotes_df["TRADEDATE"].dt.date

    quote_dates = np.array(quotes_df["date_only"].tolist())
    bodies = quotes_df["body"].to_numpy()
    opens = quotes_df["OPEN"].to_numpy()

    def body_for(date_value):
        if date_value is None:
            return None
        idx = np.searchsorted(quote_dates, date_value)
        if idx < len(quote_dates) and quote_dates[idx] == date_value:
            return float(bodies[idx])
        return None

    def next_body_for(date_value):
        if date_value is None:
            return None
        idx = np.searchsorted(quote_dates, date_value, side="right")
        if idx < len(quote_dates):
            return float(bodies[idx])
        return None

    def next_open_to_open_for(date_value):
        if date_value is None:
            return None
        idx = np.searchsorted(quote_dates, date_value, side="right")
        if idx + 1 < len(opens):
            return float(opens[idx + 1] - opens[idx])
        return None

    def parse_date(value):
        if value is None:
            return None
        try:
            return datetime.strptime(str(value), "%Y-%m-%d").date()
        except ValueError:
            return None

    result_df = df.copy()
    result_df["date"] = result_df["source_date"].apply(parse_date)
    result_df["body"] = result_df["date"].apply(body_for)
    result_df["next_body"] = result_df["date"].apply(next_body_for)
    result_df["next_open_to_open"] = result_df["date"].apply(next_open_to_open_for)
    return result_df


def save_results(path: Path, df: pd.DataFrame) -> None:
    """Сохраняет итоговый датафрейм результатов в PKL-файл."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_obj:
        pickle.dump(df, file_obj)
    logging.info("Сохранено записей: %s в %s", len(df), path)


def has_failed_sentiments(df: pd.DataFrame) -> bool:
    """Проверяет, остались ли записи без распарсенного sentiment."""
    return not df.empty and "sentiment" in df.columns and df["sentiment"].isna().any()


def drop_failed_sentiments(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет записи с sentiment=None, чтобы следующий проход пересчитал их из markdown."""
    if df.empty or "sentiment" not in df.columns:
        return df
    return df[df["sentiment"].notna()].reset_index(drop=True)


@app.command()
def main(
    output_pkl: Optional[Path] = typer.Option(
        None,
        help="Файл для сохранения sentiment-оценок. Если не задан, берется из settings.yaml.",
    ),
    model: Optional[str] = typer.Option(
        None,
        help="Локальная модель Ollama. По умолчанию берётся из settings.yaml:sentiment_model.",
    ),
    keepalive: str = typer.Option("5m", help="Удерживать модель Ollama загруженной между запросами."),
    token_limit: int = typer.Option(DEFAULT_TOKEN_LIMIT, help="Порог токенов для предупреждения о длинном prompt."),
    prompt_template: str = typer.Option(DEFAULT_PROMPT_TEMPLATE, help="Шаблон промпта для модели."),
    use_cache: Optional[bool] = typer.Option(
        None,
        "--use-cache/--no-use-cache",
        help="Использовать PKL-кэш и пропускать неизмененные файлы. Если не задано, берется из settings.yaml.",
    ),
    max_retry_passes: int = typer.Option(
        3,
        help="Сколько дополнительных проходов сделать для записей с sentiment=None.",
    ),
    save_every: int = typer.Option(
        10,
        help="Сохранять PKL-чекпоинт каждые N обработанных файлов (0 — отключить).",
    ),
    verbose: bool = typer.Option(False, help="Включить подробный лог."),
) -> None:
    """Запускает полный пайплайн расчёта sentiment-оценок."""
    settings = load_settings()
    ticker = settings.get("ticker", "")
    setup_logging(ticker, verbose)

    if model is None:
        model = settings.get("sentiment_model", "gemma4:e2b")
    logging.info("Модель sentiment: %s", model)

    if not isinstance(use_cache, bool):
        use_cache = bool(settings.get("use_cache", True))
    logging.info("Использовать кэш: %s", use_cache)
    if not isinstance(max_retry_passes, int):
        max_retry_passes = 3
    ollama_timeout = int(settings.get("ollama_timeout_seconds", 60))
    logging.info("Таймаут Ollama: %s сек.", ollama_timeout)

    md_path = Path(settings.get("md_path", "."))
    if output_pkl is None:
        output_pkl = Path(settings.get("sentiment_output_pkl", "sentiment_scores.pkl"))
    if not output_pkl.is_absolute():
        output_pkl = TICKER_DIR / output_pkl

    if not md_path.exists():
        raise typer.BadParameter(f"Папка markdown-файлов не найдена: {md_path}")

    files = find_md_files(md_path)
    if not files:
        typer.echo("В папке не найдено markdown-файлов.")
        raise typer.Exit(code=1)

    logging.info("Найдено markdown-файлов: %s в %s", len(files), md_path)

    df = pd.DataFrame()
    retry_limit = max(0, max_retry_passes)
    for retry_pass in range(retry_limit + 1):
        if retry_pass:
            logging.info("Повторный проход %s/%s для строк с sentiment=None", retry_pass, retry_limit)

        existing_df = load_existing_results(output_pkl) if use_cache else pd.DataFrame()

        rows_by_path: dict[str, dict] = {}
        if not existing_df.empty:
            for row in existing_df.to_dict("records"):
                rows_by_path[row["file_path"]] = row

        processed_since_checkpoint = 0
        progress = tqdm(
            files,
            desc=f"[{ticker}] проход {retry_pass + 1}/{retry_limit + 1}",
            unit="file",
            dynamic_ncols=True,
        )
        for md_file in progress:
            md_file_path = str(md_file.resolve())
            if use_cache and not should_process_file(md_file, existing_df):
                logging.info("[%s] Пропуск неизменённого файла: %s", ticker, md_file.name)
                continue

            processor_status = get_ollama_processor_status(model)
            logging.info(
                "[%s] Обработка файла: %s | модель=%s | процессор=%s",
                ticker,
                md_file.name,
                model,
                processor_status,
            )
            news_text = read_markdown(md_file)
            prompt = build_prompt(ticker, prompt_template, news_text)
            prompt_tokens = warn_if_token_limit_exceeded(prompt, token_limit, md_file.name)
            content_hash = compute_content_hash(md_file)

            try:
                raw_response = run_ollama(model=model, prompt=prompt, keepalive=keepalive, timeout=ollama_timeout)
                sentiment = parse_sentiment_strict(raw_response)
            except Exception as exc:
                logging.error("Ошибка обработки %s: %s", md_file.name, exc)
                raw_response = str(exc)
                sentiment = None

            rows_by_path[md_file_path] = {
                "file_path": md_file_path,
                "content_hash": content_hash,
                "source_date": extract_date_from_path(md_file),
                "ticker": ticker,
                "model": model,
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "raw_response": raw_response,
                "sentiment": sentiment,
                "processed_at": datetime.now(timezone.utc),
            }

            logging.info(
                "[%s] Результат %s: sentiment=%s, prompt_tokens=%s",
                ticker,
                md_file.name,
                sentiment,
                prompt_tokens,
            )

            processed_since_checkpoint += 1
            if save_every > 0 and processed_since_checkpoint >= save_every:
                checkpoint_df = pd.DataFrame(rows_by_path.values())
                save_results(output_pkl, checkpoint_df)
                logging.info(
                    "Чекпоинт: сохранено %s записей в %s (без рыночных признаков).",
                    len(checkpoint_df),
                    output_pkl,
                )
                processed_since_checkpoint = 0

        df = pd.DataFrame(rows_by_path.values())

        if not df.empty and "source_date" in df.columns:
            before = len(df)
            df = (
                df.sort_values(["source_date", "processed_at"], kind="stable")
                .drop_duplicates(subset="source_date", keep="last")
                .reset_index(drop=True)
            )
            if len(df) < before:
                logging.info("Дедуп по source_date: %s -> %s строк", before, len(df))

        path_db_day_str = settings.get("path_db_day", "")
        if path_db_day_str:
            df = attach_market_features(df, Path(path_db_day_str))

        save_results(output_pkl, df)
        if not has_failed_sentiments(df):
            break

        failed_count = int(df["sentiment"].isna().sum())
        cleaned_df = drop_failed_sentiments(df)
        save_results(output_pkl, cleaned_df)
        logging.warning(
            "Удалено строк с sentiment=None перед повторным проходом: %s. Они будут пересчитаны из markdown.",
            failed_count,
        )
        if retry_pass >= retry_limit:
            df = cleaned_df
            logging.error("Достигнут лимит повторных проходов; строк для пересчёта в следующем запуске: %s.", failed_count)
    typer.echo(f"Готово: {len(df)} записей сохранено в {output_pkl}")

    console_cols = [
        "source_date",
        "ticker",
        "model",
        "content_hash",
        "sentiment",
        "body",
        "next_body",
        "next_open_to_open",
        "prompt_tokens",
    ]
    console_df = df[[col for col in console_cols if col in df.columns]]
    typer.echo("\nРезультаты:")
    typer.echo(console_df.to_string(index=False))


if __name__ == "__main__":
    app()
