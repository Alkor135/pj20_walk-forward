"""
Строит сырую групповую статистику по значениям sentiment.

Скрипт читает настройки из единого `rts/settings.yaml`, загружает
`sentiment_scores.pkl`, берёт из него `sentiment` и `next_body` и моделирует
базовую follow-стратегию:
- `LONG`, если `sentiment >= 0`;
- `SHORT`, если `sentiment < 0`;

Для каждого значения sentiment скрипт считает:
- `count_pos` — количество прибыльных сделок;
- `count_neg` — количество убыточных сделок;
- `total_pnl` — суммарный P/L;
- `trades` — число сделок в группе.

Результат выводится в консоль и сохраняется в XLSX в папку `group_stats`
рядом со скриптом.
"""

import pickle
from datetime import date
from pathlib import Path
import sys
from typing import Optional

import pandas as pd
import typer
import yaml

TICKER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TICKER_DIR.parent))
from config_loader import load_settings_for


def resolve_sentiment_pkl(settings: dict) -> Path:
    """Возвращает абсолютный путь к PKL-файлу с sentiment-оценками."""
    sentiment_path = Path(settings.get("sentiment_output_pkl", "sentiment_scores.pkl"))
    return sentiment_path if sentiment_path.is_absolute() else TICKER_DIR / sentiment_path


def load_sentiment(path: Path) -> pd.DataFrame:
    """Загружает PKL, приводит типы колонок и проверяет обязательные поля."""
    if not path.exists():
        raise typer.BadParameter(f"Файл sentiment PKL не найден: {path}")
    with path.open("rb") as f:
        data = pickle.load(f)
    df = pd.DataFrame(data)
    required = {"source_date", "sentiment", "next_body"}
    missing = required - set(df.columns)
    if missing:
        raise typer.BadParameter(
            f"PKL не содержит обязательные колонки: {missing}. "
            "Запусти sentiment_analysis.py, чтобы дополнить pkl колонкой next_body."
        )
    df["source_date"] = pd.to_datetime(df["source_date"], errors="coerce").dt.date
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    df["next_body"] = pd.to_numeric(df["next_body"], errors="coerce")
    return df.dropna(subset=["source_date", "sentiment", "next_body"])


def index_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Переиндексирует данные по source_date и проверяет уникальность дат."""
    if df["source_date"].duplicated().any():
        dups = df.loc[df["source_date"].duplicated(keep=False), "source_date"].unique()
        raise typer.BadParameter(
            f"В pkl несколько строк за одну дату: {sorted(dups)[:5]}... "
            "Перегенерируй pkl: sentiment_analysis.py теперь хранит одну строку на дату."
        )
    return (
        df.set_index("source_date")[["sentiment", "next_body"]]
        .sort_index()
    )

app = typer.Typer(help="Сырая группировка sentiment-сделок по значению настроения.")


def _parse_date(value) -> Optional[date]:
    """Преобразует входное значение в объект date или возвращает None."""
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    return pd.to_datetime(str(value)).date()


def resolve_group_stats_output_xlsx(settings: dict, output_dir: Path) -> Path:
    """Возвращает путь к итоговому XLSX-файлу групповой статистики."""
    filename = str(settings.get("group_stats_output_xlsx", "sentiment_group_stats.xlsx"))
    return output_dir / filename


def build_follow_trades(aggregated: pd.DataFrame, quantity: int) -> pd.DataFrame:
    """Строит список сделок follow-стратегии с P/L, рассчитанным по next_body."""
    rows = []
    for source_date, row in aggregated.iterrows():
        sentiment = float(row["sentiment"])
        next_body = float(row["next_body"])
        direction = "LONG" if sentiment >= 0 else "SHORT"
        pnl = next_body * quantity if direction == "LONG" else -next_body * quantity
        rows.append(
            {
                "source_date": source_date,
                "sentiment": sentiment,
                "direction": direction,
                "next_body": next_body,
                "pnl": pnl,
            }
        )
    return pd.DataFrame(rows)


def group_by_sentiment(trades: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует сделки по значениям sentiment и считает сводную статистику."""
    grouped = (
        trades.groupby("sentiment")
        .agg(
            count_pos=("pnl", lambda s: int((s > 0).sum())),
            count_neg=("pnl", lambda s: int((s < 0).sum())),
            total_pnl=("pnl", "sum"),
            trades=("pnl", "size"),
        )
        .reset_index()
    )
    full = pd.DataFrame({"sentiment": [float(s) for s in range(-10, 11)]})
    grouped = full.merge(grouped, on="sentiment", how="left").fillna(
        {"count_pos": 0, "count_neg": 0, "total_pnl": 0.0, "trades": 0}
    )
    for col in ("count_pos", "count_neg", "trades"):
        grouped[col] = grouped[col].astype(int)
    return grouped.sort_values("sentiment").reset_index(drop=True)


@app.command()
def main(
    quantity: Optional[int] = typer.Option(
        None,
        help="Количество контрактов на сделку. По умолчанию — quantity_test из settings.yaml.",
    ),
    date_from: Optional[str] = typer.Option(
        None,
        "--date-from",
        help="Нижняя граница окна (YYYY-MM-DD). Переопределяет settings.yaml:stats_date_from.",
    ),
    date_to: Optional[str] = typer.Option(
        None,
        "--date-to",
        help="Верхняя граница окна (YYYY-MM-DD). Переопределяет settings.yaml:stats_date_to.",
    ),
) -> None:
    """Запускает полный расчет групповой статистики и сохраняет итог в XLSX."""
    # --- Загрузка настроек модели из единого {ticker}/settings.yaml ---
    settings = load_settings_for(__file__, "model")

    ticker = settings.get("ticker", "")

    sentiment_pkl = resolve_sentiment_pkl(settings)
    if quantity is None:
        quantity = int(settings.get("quantity_test", 1))

    # Окно дат: CLI приоритет над settings.yaml
    d_from = _parse_date(date_from if date_from is not None else settings.get("stats_date_from"))
    d_to = _parse_date(date_to if date_to is not None else settings.get("stats_date_to"))

    df = load_sentiment(sentiment_pkl)
    aggregated = index_by_date(df)

    if d_from is not None:
        aggregated = aggregated[aggregated.index >= d_from]
    if d_to is not None:
        aggregated = aggregated[aggregated.index <= d_to]

    if aggregated.empty:
        typer.echo("После фильтра по дате не осталось записей. Проверьте окно.")
        raise typer.Exit(code=1)

    trades = build_follow_trades(aggregated, quantity)
    if trades.empty:
        typer.echo("Нет торгуемых дней после фильтрации периода.")
        raise typer.Exit(code=1)

    grouped = group_by_sentiment(trades)

    output_dir = Path(__file__).resolve().parent / "group_stats"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_xlsx = resolve_group_stats_output_xlsx(settings, output_dir)
    grouped.to_excel(output_xlsx, index=False)

    period = f"{aggregated.index.min()} .. {aggregated.index.max()}"
    with pd.option_context(
        "display.width", 1000,
        "display.max_columns", 10,
        "display.max_colwidth", 30,
        "display.float_format", "{:,.2f}".format,
    ):
        typer.echo(
            f"\n{ticker}: follow-статистика по значениям sentiment | период: {period}"
        )
        typer.echo(grouped.to_string(index=False))

    typer.echo(f"\nИтого сделок: {len(trades)}")
    typer.echo(f"Суммарный P/L (чистый follow): {trades['pnl'].sum():.2f}")
    typer.echo(f"XLSX сохранён: {output_xlsx}")
    typer.echo(
        "\nПодсказка: total_pnl > 0 -> в rules.yaml ставь 'follow', "
        "< 0 -> 'invert', ~0 или мало сделок -> 'skip'."
    )


if __name__ == "__main__":
    app()
