"""
Walk-Forward бэктест sentiment-стратегии.

Для каждого OOS-блока скрипт строит правила на предыдущих N месяцах
и применяет их только к следующим M дням. Основной `rules.yaml` не меняется:
текущие правила окна пишутся во временный `walk_forward/rules_tmp.yaml`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
from typing import Optional

import pandas as pd
import typer

MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_DIR.parent))

from config_loader import load_settings_for
from rules_recommendation import build_rules_recommendation, render_rules_yaml
from sentiment_backtest import (
    _max_drawdown,
    build_backtest,
    build_qs_report,
    build_report,
    index_by_date,
    load_sentiment,
    resolve_sentiment_pkl,
)
from sentiment_group_stats import build_follow_trades, group_by_sentiment


app = typer.Typer(help="Walk-Forward бэктест sentiment-стратегии.")


def _parse_date(value) -> Optional[date]:
    """Преобразует входное значение в date или None."""
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    return pd.to_datetime(str(value)).date()


def _minus_months(value: date, months: int) -> date:
    """Возвращает дату на указанное число календарных месяцев раньше."""
    return (pd.Timestamp(value) - pd.DateOffset(months=months)).date()


def _filter_by_date(
    aggregated: pd.DataFrame,
    date_from: Optional[date],
    date_to: Optional[date],
) -> pd.DataFrame:
    """Фильтрует индексированный по датам DataFrame по включительным границам."""
    filtered = aggregated.sort_index()
    if date_from is not None:
        filtered = filtered[filtered.index >= date_from]
    if date_to is not None:
        filtered = filtered[filtered.index <= date_to]
    return filtered


def write_rules_tmp(
    rules: list[dict[str, int | str]],
    ticker: str,
    sentiment_model: str,
    output_path: Path,
) -> None:
    """Перезаписывает временный YAML с правилами последнего WF-окна."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_rules_yaml(rules, ticker, sentiment_model),
        encoding="utf-8",
    )


def build_rules_for_window(train_window: pd.DataFrame, quantity: int) -> list[dict[str, int | str]]:
    """Строит правила follow/invert по train-окну без записи в основной rules.yaml."""
    trades = build_follow_trades(train_window, quantity)
    grouped = group_by_sentiment(trades)
    return build_rules_recommendation(grouped)


def _empty_trades_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source_date",
            "sentiment",
            "action",
            "direction",
            "next_body",
            "quantity",
            "pnl",
            "cum_pnl",
            "fold",
            "train_date_from",
            "train_date_to",
        ]
    )


def _empty_folds_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "fold",
            "test_date",
            "train_date_from",
            "train_date_to",
            "train_trades",
            "test_trades",
            "test_date_to",
            "pnl",
        ]
    )


def run_walk_forward(
    aggregated: pd.DataFrame,
    quantity: int,
    train_months: int,
    date_from: Optional[date],
    date_to: Optional[date],
    rules_tmp_path: Path,
    ticker: str = "",
    sentiment_model: str = "",
    test_days: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Строит out-of-sample сделки: N месяцев train -> следующие M дней test."""
    if train_months <= 0:
        raise typer.BadParameter("walk_forward_train_months должен быть больше 0")
    if test_days <= 0:
        raise typer.BadParameter("walk_forward_test_days должен быть больше 0")

    history = aggregated.sort_index()
    test_dates = list(_filter_by_date(history, date_from, date_to).index)
    if history.empty or len(test_dates) == 0:
        return _empty_trades_frame(), _empty_folds_frame()

    first_available_date = history.index.min()
    trade_frames: list[pd.DataFrame] = []
    fold_rows: list[dict] = []

    fold = 0
    index = 0
    while index < len(test_dates):
        test_date = test_dates[index]
        train_start = _minus_months(test_date, train_months)
        if first_available_date > train_start:
            index += 1
            continue

        train_window = history[(history.index >= train_start) & (history.index < test_date)]
        if train_window.empty:
            index += 1
            continue

        block_dates = test_dates[index:index + test_days]
        test_window = history.loc[block_dates]
        rules = build_rules_for_window(train_window, quantity)
        write_rules_tmp(rules, ticker, sentiment_model, rules_tmp_path)

        test_result = build_backtest(test_window, quantity, rules)
        fold += 1

        if test_result.empty:
            fold_pnl = 0.0
            test_trades = 0
        else:
            test_result = test_result.copy()
            test_result["fold"] = fold
            test_result["train_date_from"] = train_window.index.min()
            test_result["train_date_to"] = train_window.index.max()
            test_result = test_result.drop(columns=["cum_pnl"], errors="ignore")
            trade_frames.append(test_result)
            fold_pnl = float(test_result["pnl"].sum())
            test_trades = len(test_result)

        fold_rows.append(
            {
                "fold": fold,
                "test_date": test_date,
                "test_date_to": block_dates[-1],
                "train_date_from": train_window.index.min(),
                "train_date_to": train_window.index.max(),
                "train_trades": len(train_window),
                "test_trades": test_trades,
                "pnl": fold_pnl,
            }
        )
        index += test_days

    folds = pd.DataFrame(fold_rows) if fold_rows else _empty_folds_frame()
    if not trade_frames:
        return _empty_trades_frame(), folds

    result = pd.concat(trade_frames, ignore_index=True)
    result = result.sort_values("source_date").reset_index(drop=True)
    result["cum_pnl"] = result["pnl"].cumsum()
    return result, folds


def save_walk_forward_xlsx(result: pd.DataFrame, folds: pd.DataFrame, output_xlsx: Path) -> None:
    """Сохраняет сделки и fold-сводку без истории правил."""
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_xlsx) as writer:
        result.to_excel(writer, sheet_name="trades", index=False)
        folds.to_excel(writer, sheet_name="folds", index=False)


@app.command()
def main(
    quantity: Optional[int] = typer.Option(
        None,
        help="Количество контрактов на сделку. По умолчанию — quantity_test из settings.yaml.",
    ),
    train_months: Optional[int] = typer.Option(
        None,
        "--train-months",
        help="Размер обучающего окна в месяцах. По умолчанию — walk_forward_train_months из settings.yaml.",
    ),
    test_days: Optional[int] = typer.Option(
        None,
        "--test-days",
        help="Размер out-of-sample test-блока в днях. По умолчанию — walk_forward_test_days из settings.yaml.",
    ),
    date_from: Optional[str] = typer.Option(
        None,
        "--date-from",
        help="Нижняя граница периода WF (YYYY-MM-DD). По умолчанию — backtest_date_from.",
    ),
    date_to: Optional[str] = typer.Option(
        None,
        "--date-to",
        help="Верхняя граница периода WF (YYYY-MM-DD). По умолчанию — backtest_date_to.",
    ),
) -> None:
    """Запускает Walk-Forward: предыдущие N месяцев -> следующие M дней."""
    settings = load_settings_for(__file__, "model")

    ticker = str(settings.get("ticker", ""))
    model_name = str(settings.get("sentiment_model", ""))
    if quantity is None:
        quantity = int(settings.get("quantity_test", 1))
    if train_months is None:
        train_months = int(settings.get("walk_forward_train_months", 3))
    if test_days is None:
        test_days = int(settings.get("walk_forward_test_days", 1))

    d_from = _parse_date(date_from if date_from is not None else settings.get("backtest_date_from"))
    d_to = _parse_date(date_to if date_to is not None else settings.get("backtest_date_to"))

    sentiment_pkl = resolve_sentiment_pkl(settings)
    df = load_sentiment(sentiment_pkl)
    aggregated = index_by_date(df)

    walk_forward_dir = MODEL_DIR / "walk_forward"
    rules_tmp_path = walk_forward_dir / "rules_tmp.yaml"
    result, folds = run_walk_forward(
        aggregated=aggregated,
        quantity=quantity,
        train_months=train_months,
        date_from=d_from,
        date_to=d_to,
        rules_tmp_path=rules_tmp_path,
        ticker=ticker,
        sentiment_model=model_name,
        test_days=test_days,
    )

    if result.empty:
        typer.echo("Нет out-of-sample сделок для Walk-Forward. Проверьте период и train-окно.")
        raise typer.Exit(code=1)

    output_xlsx = walk_forward_dir / "sentiment_walk_forward_results.xlsx"
    save_walk_forward_xlsx(result, folds, output_xlsx)

    plots_dir = MODEL_DIR / "plots"
    output_html = plots_dir / "sentiment_walk_forward.html"
    wf_label = f"Walk-Forward {train_months}m->{test_days}d"
    build_report(result, ticker, f"{model_name} | {wf_label}", output_html, rules_tmp_path)

    output_qs_html = plots_dir / "sentiment_walk_forward_qs.html"
    notional_capital = float(settings.get("notional_capital", 1_000_000))
    build_qs_report(result, ticker, f"{model_name} | {wf_label}", output_qs_html, notional_capital)

    typer.echo(f"Готово: {output_xlsx} и {output_html}")
    typer.echo(f"Временные правила последнего окна: {rules_tmp_path}")
    typer.echo(f"Train window: {train_months} месяцев, OOS test block: {test_days} дн.")
    typer.echo(f"Fold-окон: {len(folds)}")
    typer.echo(f"Всего сделок: {len(result)}")
    typer.echo(f"Общий OOS PnL: {result['pnl'].sum():.2f}")
    typer.echo(f"Доля прибыльных сделок: {(result['pnl'] > 0).mean() * 100:.1f}%")
    typer.echo(f"Макс. просадка: {_max_drawdown(result):.2f}")


if __name__ == "__main__":
    app()
