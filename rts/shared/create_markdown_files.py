"""
Генерирует markdown-файлы с новостями, сгруппированными по торговым сессиям.
Источник — SQLite БД (rss_news_*.db), фильтрация по провайдерам и ключевым словам "нефт"/"газ".
Торговые интервалы: 21:00 предыдущего дня — 20:59:59 текущего (из таблицы Futures).
Для каждого интервала создаётся .md файл с попавшими в него новостями.
Перед созданием удаляет последний существующий markdown-файл (пересчёт крайнего дня).
Конфигурация через единый settings.yaml тикера, логирование с ротацией (3 файла).
"""

import pandas as pd
from pathlib import Path
import sys
import sqlite3
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, time
import yaml

# --- Загрузка настроек из единого {ticker}/settings.yaml ---
TICKER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TICKER_DIR))
from config_loader import load_settings_for

settings = load_settings_for(__file__, "shared")

ticker = settings['ticker']
ticker_lc = ticker.lower()
num_mds = settings['num_mds']  # Количество последних интервалов (дней) для сохранения в markdown файлы
num_dbs = settings['num_dbs']
time_start = settings['time_start']
time_end = settings['time_end']
path_db_day = Path(settings['path_db_day'])
db_news_dir = Path(settings['db_news_dir'])
md_path = Path(settings['md_path'])
provider = settings['provider']

# Создание папки для логов
log_dir = TICKER_DIR / 'log'
log_dir.mkdir(parents=True, exist_ok=True)

# Имя файла лога с датой и временем запуска (один файл на запуск!)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = log_dir / f'create_markdown_files_{timestamp}.txt'

# Настройка логирования: ТОЛЬКО один файл + консоль
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),  # один файл
        logging.StreamHandler()                           # консоль
    ]
)

# Ручная очистка старых логов (оставляем только 3 самых новых)
def cleanup_old_logs(log_dir: Path, max_files: int = 3):
    """Удаляет старые лог-файлы, оставляя max_files самых новых."""
    log_files = sorted(log_dir.glob("create_markdown_files_*.txt"))
    if len(log_files) > max_files:
        for old_file in log_files[:-max_files]:
            try:
                old_file.unlink()
                print(f"Удалён старый лог: {old_file.name}")
            except Exception as e:
                print(f"Не удалось удалить {old_file}: {e}")

# Вызываем очистку ПЕРЕД началом логирования
cleanup_old_logs(log_dir, max_files=3)
logging.info(f"🚀 Запуск скрипта. Лог-файл: {log_file}")

def read_news_dbs_to_df(db_dir: Path, num_dbs: int | None = None) -> pd.DataFrame:
    """
    Читает несколько файлов SQLite БД с новостями из директории db_dir
    в один DataFrame и сортирует по полю loaded_at.

    Ожидаемый формат файлов: rss_news_YYYY_MM.db
    Колонки в таблице: loaded_at, date, title, provider.
    """
    db_files = sorted(db_dir.glob("rss_news_*.db"))

    if num_dbs is not None and num_dbs > 0:
        db_files = db_files[-num_dbs:]  # последние num_dbs файлов

    all_rows = []

    for db_file in db_files:
        try:
            with sqlite3.connect(db_file) as conn:
                df_part = pd.read_sql_query(
                    "SELECT loaded_at, date, title, provider FROM news",
                    conn
                )
                df_part["source_db"] = db_file.name  # опционально: откуда строка
                all_rows.append(df_part)
            logging.info(f"Успешно прочитан файл БД: {db_file}")
        except Exception as e:
            logging.error(f"Ошибка чтения БД {db_file}: {e}")

    if not all_rows:
        logging.warning("Не удалось прочитать ни одного файла БД новостей")
        return pd.DataFrame(columns=["loaded_at", "date", "title", "provider", "source_db"])

    df_all = pd.concat(all_rows, ignore_index=True)

    # Выбор строк новостей по провайдерам (investing, prime, interfax).
    if provider=='investing':
        df_all = df_all[df_all['provider'].str.contains('investing', case=False, na=False)]
    elif provider=='prime_interfax':
        df_all = df_all[df_all['provider'].str.contains('interfax|prime', case=False, na=False)]

    # Фильтрация по ключевым словам в заголовке: "нефть" или "газ"
    keywords_pattern = r'нефт|газ'
    df_all = df_all[df_all['title'].str.contains(keywords_pattern, case=False, na=False)]
    logging.info(f"Отфильтровано {len(df_all)} новостей по ключевым словам 'нефт' или 'газ'")

    # Приводим loaded_at к datetime и сортируем по времени загрузки в БД
    df_all["loaded_at"] = pd.to_datetime(df_all["loaded_at"])
    df_all = df_all.sort_values(["loaded_at", "provider", "title"]).reset_index(drop=True)

    return df_all

def build_trade_intervals(
    db_path: str,
    time_start: str = '21:00:00',
    time_end: str = '20:59:59',
    table_name: str = "Futures"
):
    """
    Читает отсортированную колонку TRADEDATE из SQLite-БД и строит интервалы:
    (prev_date + time_start, curr_date + time_end).

    Пример результата:
    (
        (datetime(2025, 6, 2, 21, 0), datetime(2025, 6, 3, 20, 59, 59)),
        (datetime(2025, 6, 3, 21, 0), datetime(2025, 6, 4, 20, 59, 59)),
        ...
    )
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT TRADEDATE FROM {table_name} ORDER BY TRADEDATE")
        rows = cur.fetchall()

    # Берём только список дат (str)
    dates = [r[0] for r in rows]

    # Нужно минимум две даты для построения хотя бы одного интервала
    if len(dates) < 2:
        return tuple()

    intervals = []

    for prev_date_str, curr_date_str in zip(dates[:-1], dates[1:]):
        # Склеиваем дату и время и переводим в datetime
        start_dt = datetime.fromisoformat(f"{prev_date_str} {time_start}")
        end_dt = datetime.fromisoformat(f"{curr_date_str} {time_end}")
        intervals.append((start_dt, end_dt))

    return tuple(intervals)

def should_delete_latest_markdown_file(
    file_mtime: datetime,
    now: datetime | None = None,
) -> bool:
    """Удаляем крайний md только если он старее сегодняшних 21:00."""
    now = now or datetime.now()
    today_21 = datetime.combine(now.date(), time(21, 0, 0))
    return file_mtime < today_21

def create_markdown_files_from_intervals(
    df_news: pd.DataFrame,
    intervals: tuple,
    md_dir: Path,
    ticker: str,
) -> None:
    """
    По каждому интервалу (start_dt, end_dt) из intervals
    создаёт markdown-файл с заголовками новостей из df_news.title,
    у которых loaded_at попадает в этот интервал.

    Имя файла: YYYY-MM-DD.md, где дата берётся из end_dt элемента пары.
    Перед созданием новых файлов:
    - удаляется самый поздний существующий md-файл по дате в имени;
    - новые файлы создаются только если их ещё нет.
    """
    md_dir.mkdir(parents=True, exist_ok=True)

    # ==== 1. Удаляем самый последний md-файл по дате в имени ====
    md_files = sorted(md_dir.glob("*.md"))
    if md_files:
        # Ожидаемый формат имени: YYYY-MM-DD.md
        def extract_date(p: Path):
            try:
                return datetime.fromisoformat(p.stem).date()
            except ValueError:
                return None

        dated_files = [(extract_date(p), p) for p in md_files]
        dated_files = [(d, p) for d, p in dated_files if d is not None]
        if dated_files:
            last_date, last_path = max(dated_files, key=lambda x: x[0])
            try:
                last_mtime = datetime.fromtimestamp(last_path.stat().st_mtime)
                if should_delete_latest_markdown_file(last_mtime):
                    last_path.unlink()
                    logging.info(f"Удалён последний markdown-файл: {last_path}")
                else:
                    logging.info(
                        f"Последний markdown-файл создан после 21:00 текущего дня, "
                        f"оставляем без изменений: {last_path} (mtime={last_mtime})"
                    )
            except Exception as e:
                logging.error(f"Ошибка удаления markdown-файла {last_path}: {e}")

    # ==== 2. Создаём только отсутствующие файлы ====

    # Убедимся, что loaded_at в datetime
    if not pd.api.types.is_datetime64_any_dtype(df_news["loaded_at"]):
        df_news = df_news.copy()
        df_news["loaded_at"] = pd.to_datetime(df_news["loaded_at"])

    for start_dt, end_dt in intervals:
        # Имя файла по дате конца интервала
        date_str = end_dt.date().isoformat()
        filename = f"{date_str}.md"
        filepath = md_dir / filename

        # Если файл уже существует — пропускаем
        if filepath.exists():
            logging.info(f"Markdown-файл уже существует, пропуск: {filepath}")
            continue

        # Фильтрация новостей по интервалу
        mask = (df_news["loaded_at"] >= start_dt) & (df_news["loaded_at"] <= end_dt)
        df_slice = df_news.loc[mask].sort_values("loaded_at")

        if df_slice.empty:
            continue  # нет новостей — файл не создаём

        lines = []
        for _, row in df_slice.iterrows():
            title = str(row["title"])
            lines.append(title)
            lines.append("")  # пустая строка-разделитель

        content = "\n".join(lines)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info(f"Создан markdown-файл: {filepath}")
        except Exception as e:
            logging.error(f"Ошибка записи markdown-файла {filepath}: {e}")

if __name__ == "__main__":
    df_news = read_news_dbs_to_df(db_news_dir, num_dbs=num_dbs)
    with pd.option_context(  # Печать широкого и длинного датафрейма
            "display.width", 1000,
            "display.max_columns", 30,
            "display.max_colwidth", 100
    ):
        print("Датафрейм с результатом:")
        print(df_news)

    intervals = build_trade_intervals(
        db_path=path_db_day,  # из settings.yaml
        time_start=time_start,
        time_end=time_end,
        table_name="Futures"  # Таблица в БД котировок
    )
    for it in intervals[:5]:
        print(it)

    create_markdown_files_from_intervals(
        df_news=df_news,
        intervals=intervals[-num_mds:],  # например, только последние num_mds интервалов
        md_dir=md_path,
        ticker=ticker,
    )
    print(f"\nMarkdown файлы созданы в {md_path}")
