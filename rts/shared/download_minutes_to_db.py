"""
Скрипт скачивает минутные данные из MOEX ISS API и сохраняет их в базу данных SQLite.
Если в базе данных уже есть данные, он проверяет их полноту и докачивает недостающие данные.
Если данных нет, он загружает все доступные данные, начиная с указанной даты.
Минутные данные за текущую сессию на MOEX ISS API доступны с 15 минутной задержкой.
"""

from pathlib import Path
import sys
import sqlite3
from datetime import datetime, timedelta, date, time
import requests
import pandas as pd
import logging
import yaml

# --- Загрузка настроек из единого {ticker}/settings.yaml ---
TICKER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TICKER_DIR))
from config_loader import load_settings_for

settings = load_settings_for(__file__, "shared")

# ==== Параметры ====
ticker = settings['ticker']
ticker_lc = ticker.lower()

# Начальная дата для загрузки минутных данных
start_date = datetime.strptime(settings['start_date_download_minutes'], "%Y-%m-%d").date()

# Путь к базе данных с минутными барами фьючерсов
path_db_minute = Path(settings['path_db_minute'])

# --- Настройка логирования ---
log_dir = TICKER_DIR / 'log'
log_dir.mkdir(parents=True, exist_ok=True)

# Имя файла лога с датой и временем запуска (один файл на запуск)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = log_dir / f'download_minutes_to_db_{timestamp}.txt'

# Настройка логирования: файл + консоль
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Очистка старых логов (оставляем только 3 самых новых)
def cleanup_old_logs(log_dir: Path, prefix: str, max_files: int = 3):
    """Удаляет старые лог-файлы, оставляя max_files самых новых."""
    log_files = sorted(log_dir.glob(f"{prefix}_*.txt"))
    if len(log_files) > max_files:
        for old_file in log_files[:-max_files]:
            try:
                old_file.unlink()
                logger.info(f"Удалён старый лог: {old_file.name}")
            except Exception as e:
                logger.warning(f"Не удалось удалить {old_file}: {e}")

cleanup_old_logs(log_dir, prefix="download_minutes_to_db")


def request_moex(session, url, retries = 5, timeout = 10):
    """Функция запроса данных с повторными попытками"""
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Ошибка запроса {url} (попытка {attempt + 1}): {e}")
            if attempt == retries - 1:
                return None

def create_tables(connection: sqlite3.Connection) -> None:
    """ Функция создания таблиц в БД если их нет"""
    try:
        with connection:
            connection.execute('''CREATE TABLE if not exists Futures (
                            TRADEDATE         TEXT PRIMARY KEY UNIQUE NOT NULL,
                            SECID             TEXT NOT NULL,
                            OPEN              REAL NOT NULL,
                            LOW               REAL NOT NULL,
                            HIGH              REAL NOT NULL,
                            CLOSE             REAL NOT NULL,
                            VOLUME            INTEGER NOT NULL,
                            LSTTRADE          DATE NOT NULL)'''
                           )
        logger.info('Таблицы в БД созданы')
    except sqlite3.OperationalError as exception:
        logger.error(f"Ошибка при создании БД: {exception}")

def get_info_future(session, security):
    """Запрашивает у MOEX информацию по инструменту"""
    url = f'https://iss.moex.com/iss/securities/{security}.json'
    j = request_moex(session, url)

    if not j:
        return pd.Series(["", "2130-01-01"])  # Гарантируем, что всегда 2 значения

    data = [{k: r[i] for i, k in enumerate(j['description']['columns'])} for r in j['description']['data']]
    df = pd.DataFrame(data)

    shortname = df.loc[df['name'] == 'SHORTNAME', 'value'].values[0] \
        if 'SHORTNAME' in df['name'].values else ""
    lsttrade = df.loc[df['name'] == 'LSTTRADE', 'value'].values[0] \
        if 'LSTTRADE' in df['name'].values else df.loc[df['name'] == 'LSTDELDATE', 'value'].values[0] \
        if 'LSTDELDATE' in df['name'].values else "2130-01-01"

    return pd.Series([shortname, lsttrade])  # Гарантируем возврат 2 значений

def get_minute_candles(
        session, 
        ticker: str, 
        start_date: date, 
        from_str: str = None, 
        till_str: str = None
        ) -> pd.DataFrame:
    """Получает все минутные данные по фьючерсу за указанную дату с учетом пагинации"""
    if from_str is None:
        from_str = datetime.combine(start_date, time(0, 0)).isoformat()
    if till_str is None:
        till_str = datetime.combine(start_date, time(23, 59, 59)).isoformat()

    all_data = []
    start = 0
    page_size = 500  # MOEX ISS API возвращает до 500 записей за запрос

    # MOEX ISS отдаёт минутные свечи с задержкой ~15 мин: если till попадает
    # в это окно, пустой ответ — штатная ситуация, не ошибка. Окно расширено
    # до 20 мин с запасом; недостающие минуты добиваются из QUIK tail-fill.
    try:
        till_dt = datetime.fromisoformat(till_str)
        within_delay_window = (datetime.now() - till_dt) < timedelta(minutes=20)
    except ValueError:
        within_delay_window = False

    while True:
        url = (
            f'https://iss.moex.com/iss/engines/futures/markets/forts/securities/'
            f'{ticker}/candles.json?'
            f'interval=1&from={from_str}&till={till_str}'
            f'&start={start}'
        )
        logger.info(f"Запрос минутных данных (start={start}): {url}")

        j = request_moex(session, url)
        if not j or 'candles' not in j or not j['candles'].get('data'):
            if within_delay_window:
                logger.warning(
                    f"Нет минутных данных для {ticker} на {start_date}: "
                    f"окно till={till_str} внутри 20-мин зоны задержки MOEX, добьём из QUIK tail-fill"
                )
            else:
                logger.error(f"Нет минутных данных для {ticker} на {start_date}")
            break

        data = [{k: r[i] for i, k in enumerate(j['candles']['columns'])} for r in j['candles']['data']]
        if not data:
            break

        all_data.extend(data)
        start += page_size

        if len(data) < page_size:
            break

    if not all_data:
        if within_delay_window:
            logger.warning(f"Нет данных для {ticker} на {start_date} (окно в зоне задержки MOEX)")
        else:
            logger.error(f"Нет данных для {ticker} на {start_date}")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    df = df.rename(columns={
        'begin': 'TRADEDATE',
        'open': 'OPEN',
        'close': 'CLOSE',
        'high': 'HIGH',
        'low': 'LOW',
        'volume': 'VOLUME'
    })

    df['SECID'] = ticker

    df = df.dropna(subset=['OPEN', 'LOW', 'HIGH', 'CLOSE', 'VOLUME'])
    logger.info(df.to_string(max_rows=6, max_cols=18))

    return df[['TRADEDATE', 'SECID', 'OPEN', 'LOW', 'HIGH', 'CLOSE', 'VOLUME']].reset_index(drop=True)

def save_to_db(df: pd.DataFrame, connection: sqlite3.Connection) -> None:
    """Сохраняет DataFrame в таблицу Futures"""
    if df.empty:
        logger.error("DataFrame пуст, данные не сохранены")
        return

    try:
        with connection:
            df.to_sql('Futures', connection, if_exists='append', index=False)
        logger.info(f"Сохранено {len(df)} записей в таблицу Futures")
    except sqlite3.Error as e:
        logger.error(f"Ошибка при сохранении данных в БД: {e}")

def get_future_date_results(
        session,
        start_date: date,
        ticker: str,
        connection: sqlite3.Connection,
        cursor: sqlite3.Cursor) -> None:
    """Получает данные по фьючерсам с MOEX ISS API и сохраняет их в базу данных."""
    today_date = datetime.now().date()  # Текущая дата
    while start_date <= today_date:
        date_str = start_date.strftime('%Y-%m-%d')
        # Проверяем количество записей в БД за дату
        cursor.execute("SELECT COUNT(*) FROM Futures WHERE DATE(TRADEDATE) = ?", (date_str,))
        count = cursor.fetchone()[0]

        if count == 0:
            # Нет минутных данных в БД, запрашиваем данные о торгуемых фьючерсах на дату
            # За текущую дату торгуемые тикеры доступны после окончания Единой Торговой Сессии (ЕТС), 
            # после 23:50.
            request_url = (
                f'https://iss.moex.com/iss/history/engines/futures/markets/forts/securities.json?'
                f'date={date_str}&assetcode={ticker}'
            )

            j = request_moex(session, request_url)
            if j is None:
                logger.error(f"Ошибка получения данных для {start_date}. Прерываем процесс, чтобы повторить попытку в следующий запуск.")
                break
            elif 'history' not in j or not j['history'].get('data'):
                # History API не вернул данных — возможно, торги ещё идут.
                # Fallback: берём SECID и LSTTRADE из последней записи в БД.
                cursor.execute("SELECT SECID, LSTTRADE FROM Futures ORDER BY TRADEDATE DESC LIMIT 1")
                last_row = cursor.fetchone()
                if last_row is None:
                    logger.info(f"Нет данных по торгуемым фьючерсам {ticker} за {start_date}, БД пуста — пропускаем")
                    start_date += timedelta(days=1)
                    continue

                last_secid = last_row[0]
                last_lsttrade = datetime.strptime(last_row[1], '%Y-%m-%d').date() if isinstance(last_row[1], str) else last_row[1]

                if last_lsttrade <= start_date:
                    # Контракт из БД истёк — ролловер.
                    # Пробуем взять ticker_close из settings.yaml как новый контракт.
                    ticker_close = settings.get('ticker_close')
                    if ticker_close and ticker_close != last_secid:
                        # Проверяем, что ticker_close — действующий контракт
                        _, tc_lsttrade_str = get_info_future(session, ticker_close)
                        try:
                            tc_lsttrade = datetime.strptime(tc_lsttrade_str, '%Y-%m-%d').date()
                        except (ValueError, TypeError):
                            tc_lsttrade = None

                        if tc_lsttrade and tc_lsttrade > start_date:
                            logger.info(f"Ролловер: {last_secid} истёк {last_lsttrade}, "
                                        f"используем ticker_close={ticker_close} (LSTTRADE={tc_lsttrade})")
                            current_ticker = ticker_close
                            lasttrade = tc_lsttrade

                            minute_df = get_minute_candles(session, current_ticker, start_date)
                            minute_df['LSTTRADE'] = lasttrade
                            if not minute_df.empty:
                                save_to_db(minute_df, connection)

                            start_date += timedelta(days=1)
                            continue

                    # ticker_close отсутствует, совпадает со старым или тоже истёк — ждём history API
                    if not ticker_close:
                        logger.info(f"Контракт {last_secid} истёк {last_lsttrade}, "
                                    f"ticker_close не задан в settings.yaml — ждём history API")
                    elif ticker_close == last_secid:
                        logger.warning(f"ticker_close={ticker_close} совпадает с истёкшим контрактом "
                                       f"{last_secid} — обновите settings.yaml")
                    else:
                        logger.info(f"ticker_close={ticker_close} тоже истёк — ждём history API")
                    start_date += timedelta(days=1)
                    continue

                # Контракт ещё активен, используем его для загрузки минутных данных
                logger.info(f"History API пуст за {start_date}, fallback на {last_secid} (LSTTRADE={last_lsttrade})")
                current_ticker = last_secid
                lasttrade = last_lsttrade

                minute_df = get_minute_candles(session, current_ticker, start_date)
                minute_df['LSTTRADE'] = lasttrade
                if not minute_df.empty:
                    save_to_db(minute_df, connection)

                start_date += timedelta(days=1)
                continue

            data = [{k: r[i] for i, k in enumerate(j['history']['columns'])} for r in j['history']['data']]
            df = pd.DataFrame(data).dropna(subset=['OPEN', 'LOW', 'HIGH', 'CLOSE'])
            if len(df) == 0:
                start_date += timedelta(days=1)
                continue

            df[['SHORTNAME', 'LSTTRADE']] = df.apply(
                lambda x: get_info_future(session, x['SECID']), axis=1, result_type='expand'
            )
            df["LSTTRADE"] = pd.to_datetime(df["LSTTRADE"], errors='coerce').dt.date.fillna('2130-01-01')
            df = df[df['LSTTRADE'] > start_date].dropna(subset=['OPEN', 'LOW', 'HIGH', 'CLOSE'])
            df = df[df['LSTTRADE'] == df['LSTTRADE'].min()].reset_index(drop=True)
            df = df.drop(columns=[
                'OPENPOSITIONVALUE', 'VALUE', 'SETTLEPRICE', 'SWAPRATE', 'WAPRICE',
                'SETTLEPRICEDAY', 'NUMTRADES', 'SHORTNAME', 'CHANGE', 'QTY'
            ], errors='ignore')

            current_ticker = df.loc[0, 'SECID']
            lasttrade = df.loc[0, 'LSTTRADE']

            # Получаем минутные данные
            minute_df = get_minute_candles(session, current_ticker, start_date)
            minute_df['LSTTRADE'] = lasttrade
            if not minute_df.empty:
                save_to_db(minute_df, connection)

        else:
            # Есть минутные данные за дату, проверяем полноту
            cursor.execute("SELECT MAX(TRADEDATE) FROM Futures WHERE DATE(TRADEDATE) = ?", (date_str,))
            max_time_str = cursor.fetchone()[0]
            max_dt = datetime.strptime(max_time_str, '%Y-%m-%d %H:%M:%S')

            threshold_time = time(23, 49, 0)
            is_today = start_date == today_date

            if not is_today and max_dt.time() >= threshold_time:
                logger.info(f"Минутные данные за {start_date} полные, пропускаем дату {start_date}")
                start_date += timedelta(days=1)
                continue

            # Неполные минутные данные или сегодняшний день (после 19:05), докачиваем
            cursor.execute("SELECT SECID, LSTTRADE FROM Futures WHERE DATE(TRADEDATE) = ? LIMIT 1", (date_str,))
            row = cursor.fetchone()
            current_ticker = row[0]
            lasttrade = datetime.strptime(row[1], '%Y-%m-%d').date() if isinstance(row[1], str) else row[1]

            from_dt = max_dt + timedelta(minutes=1)
            from_str = from_dt.isoformat()

            if is_today:
                till_dt = datetime.now()
            else:
                till_dt = datetime.combine(start_date, time(23, 59, 59))
            till_str = till_dt.isoformat()

            minute_df = get_minute_candles(session, current_ticker, start_date, from_str, till_str)
            minute_df['LSTTRADE'] = lasttrade
            if not minute_df.empty:
                save_to_db(minute_df, connection)

        start_date += timedelta(days=1)

def fill_today_tail_from_quik(
        csv_path: Path,
        connection: sqlite3.Connection,
        cursor: sqlite3.Cursor,
        today_date: date) -> None:
    """
    Добирает последние минутные бары за сегодня из CSV, который пишет
    lua-экспортёр QUIK (trade/quik_export_minutes.lua).

    ISS отдаёт минутные свечи с задержкой ≥15 минут, поэтому при запуске
    в 21:00 бар 20:59 из ISS недоступен. QUIK видит его в реальном времени,
    данные пишутся в CSV и здесь мержатся в sqlite.

    Добавляются только минуты, которые строго позже max(TRADEDATE) в БД
    за сегодня и не позже 20:59 (время закрытия дневного бара по settings.yaml).
    Используется INSERT OR IGNORE: существующие ISS-бары не перезаписываются.

    Любая ошибка (нет файла, устарел, повреждён, QUIK не запущен) приводит к
    молчаливому пропуску — пайплайн run_all.py не должен падать из-за tail-fill.
    """
    if not csv_path.exists():
        logger.info(f"QUIK tail-fill: CSV не найден ({csv_path}), пропускаем")
        return

    # Проверка свежести: если файл не обновлялся > 10 минут, QUIK скорее всего не пишет
    mtime = datetime.fromtimestamp(csv_path.stat().st_mtime)
    if (datetime.now() - mtime) > timedelta(minutes=10):
        logger.info(f"QUIK tail-fill: CSV устарел (mtime={mtime}), пропускаем")
        return

    today_str = today_date.strftime('%Y-%m-%d')
    cursor.execute(
        "SELECT SECID, LSTTRADE, MAX(TRADEDATE) FROM Futures WHERE DATE(TRADEDATE) = ?",
        (today_str,)
    )
    row = cursor.fetchone()
    if not row or row[2] is None:
        logger.info("QUIK tail-fill: в БД нет сегодняшних ISS-баров, пропускаем")
        return

    current_ticker = row[0]
    lasttrade = row[1]
    max_iss_dt = datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S')

    cutoff = datetime.combine(today_date, time(20, 59))
    if max_iss_dt >= cutoff:
        logger.info(f"QUIK tail-fill: ISS уже покрывает хвост до {max_iss_dt}, не требуется")
        return

    try:
        df_csv = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"QUIK tail-fill: не удалось прочитать {csv_path}: {e}")
        return

    required_cols = {'SECID', 'TRADEDATE', 'OPEN', 'LOW', 'HIGH', 'CLOSE', 'VOLUME'}
    if not required_cols.issubset(df_csv.columns):
        logger.error(f"QUIK tail-fill: в CSV нет нужных колонок, есть только {list(df_csv.columns)}")
        return

    df_csv['TRADEDATE'] = pd.to_datetime(df_csv['TRADEDATE'], errors='coerce')
    df_csv = df_csv.dropna(subset=['TRADEDATE'])

    mask = (
        (df_csv['SECID'] == current_ticker)
        & (df_csv['TRADEDATE'].dt.date == today_date)
        & (df_csv['TRADEDATE'] > max_iss_dt)
        & (df_csv['TRADEDATE'] <= cutoff)
    )
    df_tail = df_csv.loc[mask].copy()
    if df_tail.empty:
        logger.info(f"QUIK tail-fill: в CSV нет новых баров после {max_iss_dt} для {current_ticker}")
        return

    df_tail['TRADEDATE'] = df_tail['TRADEDATE'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_tail['LSTTRADE'] = lasttrade
    df_tail = df_tail[['TRADEDATE', 'SECID', 'OPEN', 'LOW', 'HIGH', 'CLOSE', 'VOLUME', 'LSTTRADE']]

    rows_to_insert = [tuple(r) for r in df_tail.itertuples(index=False, name=None)]
    with connection:
        connection.executemany(
            "INSERT OR IGNORE INTO Futures "
            "(TRADEDATE, SECID, OPEN, LOW, HIGH, CLOSE, VOLUME, LSTTRADE) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows_to_insert
        )
    logger.info(f"QUIK tail-fill: добавлено {len(rows_to_insert)} минут "
                f"за {today_date} ({current_ticker}, {max_iss_dt.time()} → {cutoff.time()})")


def main(
        ticker: str = ticker,
        path_db: Path = path_db_minute,
        start_date: date = start_date) -> None:
    """
    Основная функция: подключается к базе данных, создает таблицы и загружает данные по фьючерсам.
    """
    try:
        # Создание директории под БД, если не существует
        path_db.parent.mkdir(parents=True, exist_ok=True)

        # Подключение к базе данных
        connection = sqlite3.connect(str(path_db), check_same_thread=True)
        cursor = connection.cursor()

        # Проверяем наличие таблицы Futures
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Futures'")
        exist_table = cursor.fetchone()
        # Если таблица Futures не существует, создаем её
        if exist_table is None:
            create_tables(connection)

        # Проверяем, есть ли записи в таблице Futures
        cursor.execute("SELECT EXISTS (SELECT 1 FROM Futures) as has_rows")
        exists_rows = cursor.fetchone()[0]
        # Если таблица Futures не пустая
        if exists_rows:
            # Находим максимальную дату
            cursor.execute("SELECT MAX(DATE(TRADEDATE)) FROM Futures")
            max_trade_date = cursor.fetchone()[0]
            if max_trade_date:
                # Устанавливаем start_date на максимальную дату для проверки полноты
                start_date = datetime.strptime(max_trade_date, "%Y-%m-%d").date()
                logger.info(f"Начальная дата для загрузки минутных данных: {start_date}")

        with requests.Session() as session:
            get_future_date_results(session, start_date, ticker, connection, cursor)

        # Добивка последних минут сегодняшней сессии из QUIK CSV.
        # Best-effort: любая ошибка здесь не должна валить run_all.py.
        quik_csv_raw = settings.get('quik_csv_path')
        if quik_csv_raw:
            try:
                fill_today_tail_from_quik(
                    Path(quik_csv_raw),
                    connection,
                    cursor,
                    datetime.now().date()
                )
            except Exception as e:
                logger.error(f"QUIK tail-fill упал, игнорируем: {e}")

    except Exception as e:
        logger.error(f"Ошибка в main: {e}")

    finally:
        # VACUUM и закрытие соединения (с проверкой, что объекты были созданы)
        if 'cursor' in locals():
            cursor.execute("VACUUM")
            logger.info("VACUUM выполнен: база данных оптимизирована")
            cursor.close()
        if 'connection' in locals():
            connection.close()
            logger.info(f"Соединение с минутной БД {path_db} по фьючерсам {ticker} закрыто.")


if __name__ == '__main__':
    main(ticker, path_db_minute, start_date)
