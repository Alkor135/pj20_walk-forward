"""
Бэктест sentiment-стратегии на основе `sentiment_scores.pkl`.

Скрипт читает настройки из единого `rts/settings.yaml`, загружает
`sentiment` и `next_body` из PKL, применяет правила из `rules.yaml`
и строит сделки для follow / invert / skip логики.

P/L каждой сделки считается по `next_body`:
- `LONG`  -> `next_body * quantity`
- `SHORT` -> `-next_body * quantity`

Результаты сохраняются рядом со скриптом:
- HTML-отчёты в `plots/`
- XLSX с результатами в `backtest/`
"""

from __future__ import annotations

import pickle
import re
from datetime import date
from html import escape
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import quantstats_lumi as qs
import typer
import yaml

TICKER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TICKER_DIR.parent))
from config_loader import load_settings_for
from sentiment_forecast import build_next_month_forecast_html


def _parse_date(value) -> Optional[date]:
    """Преобразует входное значение в объект date или возвращает None."""
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    return pd.to_datetime(str(value)).date()


app = typer.Typer(help="Бэктест sentiment-стратегии по данным из PKL.")


VALID_ACTIONS = {"follow", "invert", "skip"}


def resolve_sentiment_pkl(settings: dict) -> Path:
    """Возвращает абсолютный путь к PKL-файлу с sentiment-оценками."""
    sentiment_path = Path(settings.get("sentiment_output_pkl", "sentiment_scores.pkl"))
    return sentiment_path if sentiment_path.is_absolute() else TICKER_DIR / sentiment_path


def load_sentiment(path: Path) -> pd.DataFrame:
    """Читает PKL, приводит типы и проверяет наличие колонок sentiment и next_body."""
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
    """Индексирует данные по source_date и проверяет уникальность дат."""
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


def load_rules(path: Path) -> list[dict]:
    """Загружает и валидирует список правил из YAML-файла."""
    if not path.exists():
        raise typer.BadParameter(f"Rules-yaml не найден: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    rules = data.get("rules") or []
    if not isinstance(rules, list) or not rules:
        raise typer.BadParameter(f"В {path} нет списка 'rules' или он пустой")
    for i, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise typer.BadParameter(f"Правило #{i} должно быть объектом: {rule}")
        for key in ("min", "max", "action"):
            if key not in rule:
                raise typer.BadParameter(f"Правило #{i} без поля '{key}': {rule}")
        if rule["action"] not in VALID_ACTIONS:
            raise typer.BadParameter(
                f"Правило #{i}: action должен быть одним из {sorted(VALID_ACTIONS)}, получено {rule['action']!r}"
            )
        if float(rule["min"]) > float(rule["max"]):
            raise typer.BadParameter(f"Правило #{i}: min > max ({rule})")
    return rules


def match_action(sentiment: float, rules: list[dict]) -> str:
    """Возвращает action из первого подходящего правила или skip по умолчанию."""
    for rule in rules:
        if float(rule["min"]) <= sentiment <= float(rule["max"]):
            return rule["action"]
    return "skip"


def direction_for_action(sentiment: float, action: str) -> str:
    """Возвращает направление: follow(0) = LONG, invert(0) = SHORT."""
    if action == "follow":
        return "LONG" if sentiment >= 0 else "SHORT"
    return "SHORT" if sentiment >= 0 else "LONG"


def build_backtest(
    aggregated: pd.DataFrame,
    quantity: int,
    rules: list[dict],
) -> pd.DataFrame:
    """Строит сделки бэктеста по sentiment, rules и целевому движению next_body."""
    rows = []
    for source_date, row in aggregated.iterrows():
        sentiment = float(row["sentiment"])
        next_body = float(row["next_body"])

        action = match_action(sentiment, rules)
        if action == "skip":
            continue
        direction = direction_for_action(sentiment, action)

        pnl = next_body * quantity if direction == "LONG" else -next_body * quantity

        rows.append(
            {
                "source_date": source_date,
                "sentiment": sentiment,
                "action": action,
                "direction": direction,
                "next_body": next_body,
                "quantity": quantity,
                "pnl": pnl,
            }
        )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("source_date").reset_index(drop=True)
    result["cum_pnl"] = result["pnl"].cumsum()
    return result


def _max_drawdown(result: pd.DataFrame) -> float:
    """Вычисляет максимальную просадку по колонке cum_pnl."""
    series = result["cum_pnl"]
    peak = series.cummax()
    drawdown = series - peak
    return float(drawdown.min())


def _max_consecutive(series: pd.Series, condition: int) -> int:
    """Возвращает максимальную длину серии значений, равных condition."""
    streaks = (series != condition).cumsum()
    filtered = series[series == condition]
    if filtered.empty:
        return 0
    return int(filtered.groupby(streaks[series == condition]).size().max())


def _drawdown_duration(drawdown: pd.Series) -> int:
    """Вычисляет максимальную длительность просадки в количестве сделок."""
    max_dd_duration = 0
    current_dd_start = None
    for i in range(len(drawdown)):
        if drawdown.iloc[i] < 0:
            if current_dd_start is None:
                current_dd_start = i
        else:
            if current_dd_start is not None:
                duration = i - current_dd_start
                if duration > max_dd_duration:
                    max_dd_duration = duration
                current_dd_start = None
    if current_dd_start is not None:
        duration = len(drawdown) - current_dd_start
        if duration > max_dd_duration:
            max_dd_duration = duration
    return max_dd_duration


def build_report(result: pd.DataFrame, ticker: str, model_name: str, output_html: Path, rules_path: Path) -> None:
    """Строит подробный Plotly HTML-отчёт по результатам backtest."""
    df = result.copy()
    df["source_date"] = pd.to_datetime(df["source_date"])
    df = df.sort_values("source_date").reset_index(drop=True)

    pl = df["pnl"].astype(float)
    cum = pl.cumsum()

    # ── Агрегации ─────────────────────────────────────────────────────────
    day_colors = ["#d32f2f" if v < 0 else "#2e7d32" for v in pl]

    df["Неделя"] = df["source_date"].dt.to_period("W")
    weekly = df.groupby("Неделя", as_index=False)["pnl"].sum()
    weekly["dt"] = weekly["Неделя"].apply(lambda p: p.start_time)
    week_colors = ["#d32f2f" if v < 0 else "#00838f" for v in weekly["pnl"]]

    df["Месяц"] = df["source_date"].dt.to_period("M")
    monthly = df.groupby("Месяц", as_index=False)["pnl"].sum()
    monthly["dt"] = monthly["Месяц"].dt.to_timestamp()
    month_colors = ["#d32f2f" if v < 0 else "#1565c0" for v in monthly["pnl"]]

    running_max = cum.cummax()
    drawdown = cum - running_max

    for w in (5, 10, 20):
        df[f"MA{w}"] = pl.rolling(w, min_periods=1).mean()

    sent_stats = (
        df.groupby("sentiment")
        .agg(trades=("pnl", "size"), pnl=("pnl", "sum"))
        .reset_index()
        .sort_values("sentiment")
    )
    action_stats = (
        df.groupby("action")
        .agg(
            trades=("pnl", "size"),
            pnl=("pnl", "sum"),
            winrate=("pnl", lambda s: (s > 0).mean() * 100),
        )
        .reset_index()
    )

    # ── Метрики ───────────────────────────────────────────────────────────
    total_profit = float(cum.iloc[-1])
    total_trades = len(df)
    win_trades = int((pl > 0).sum())
    loss_trades = int((pl < 0).sum())
    win_rate = win_trades / max(total_trades, 1) * 100
    max_dd = float(drawdown.min())
    best_trade = float(pl.max())
    worst_trade = float(pl.min())
    avg_trade = float(pl.mean())
    median_trade = float(pl.median())
    std_trade = float(pl.std()) if total_trades > 1 else 0.0

    gross_profit = float(pl[pl > 0].sum())
    gross_loss = float(abs(pl[pl < 0].sum()))
    avg_win = float(pl[pl > 0].mean()) if win_trades else 0.0
    avg_loss = float(abs(pl[pl < 0].mean())) if loss_trades else 0.0

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")
    recovery_factor = total_profit / abs(max_dd) if max_dd != 0 else float("inf")
    expectancy = (win_rate / 100) * avg_win - (1 - win_rate / 100) * avg_loss
    sharpe = (avg_trade / std_trade) * np.sqrt(252) if std_trade > 0 else 0.0

    downside = pl[pl < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = (avg_trade / downside_std) * np.sqrt(252) if downside_std > 0 else 0.0

    date_range_days = (df["source_date"].max() - df["source_date"].min()).days or 1
    annual_profit = total_profit * 365 / date_range_days
    calmar = annual_profit / abs(max_dd) if max_dd != 0 else float("inf")

    signs = pl.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    max_consec_wins = _max_consecutive(signs, 1)
    max_consec_losses = _max_consecutive(signs, -1)
    max_dd_duration = _drawdown_duration(drawdown)
    volatility = std_trade * np.sqrt(252)

    stats_text = (
        f"Итого: {total_profit:,.0f} | Сделок: {total_trades} | "
        f"Win: {win_trades} ({win_rate:.0f}%) | Loss: {loss_trades} | "
        f"PF: {profit_factor:.2f} | RF: {recovery_factor:.2f} | "
        f"Sharpe: {sharpe:.2f} | MaxDD: {max_dd:,.0f}"
    )
    test_period_text = (
        "Период тестирования: "
        f"{df['source_date'].min():%Y-%m-%d} - {df['source_date'].max():%Y-%m-%d}"
    )

    # ── Графики ───────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=(
            "P/L по сделкам",
            "Накопленная прибыль (equity)",
            "P/L по неделям",
            "P/L по месяцам",
            "Drawdown от максимума",
            "Распределение P/L сделок",
            "Скользящие средние P/L (5/10/20)",
            "P/L по action (follow/invert)",
            "P/L по значениям sentiment",
            "Кол-во сделок по sentiment",
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}],
        ],
        vertical_spacing=0.07,
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Bar(
            x=df["source_date"], y=pl, marker_color=day_colors,
            name="P/L сделки",
            hovertemplate="%{x|%Y-%m-%d}<br>P/L: %{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["source_date"], y=cum,
            mode="lines", fill="tozeroy",
            line=dict(color="#2e7d32", width=2),
            fillcolor="rgba(46,125,50,0.15)",
            name="Equity",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>",
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(
            x=weekly["dt"], y=weekly["pnl"], marker_color=week_colors,
            name="P/L неделя",
            hovertemplate="Нед. %{x|%Y-%m-%d}<br>P/L: %{y:,.0f}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=monthly["dt"], y=monthly["pnl"], marker_color=month_colors,
            name="P/L месяц",
            text=[f"{v:,.0f}" for v in monthly["pnl"]],
            textposition="outside",
            hovertemplate="%{x|%Y-%m}<br>P/L: %{y:,.0f}<extra></extra>",
        ),
        row=2, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df["source_date"], y=drawdown,
            mode="lines", fill="tozeroy",
            line=dict(color="#d32f2f", width=1.5),
            fillcolor="rgba(211,47,47,0.2)",
            name="Drawdown",
            hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:,.0f}<extra></extra>",
        ),
        row=3, col=1,
    )
    pl_pos = pl[pl > 0]
    pl_neg = pl[pl < 0]
    fig.add_trace(
        go.Histogram(x=pl_pos, marker_color="#2e7d32", opacity=0.7, name="Прибыль", nbinsx=20),
        row=3, col=2,
    )
    fig.add_trace(
        go.Histogram(x=pl_neg, marker_color="#d32f2f", opacity=0.7, name="Убыток", nbinsx=20),
        row=3, col=2,
    )
    for w, color in [(5, "#1565c0"), (10, "#ff6f00"), (20, "#7b1fa2")]:
        fig.add_trace(
            go.Scatter(
                x=df["source_date"], y=df[f"MA{w}"],
                mode="lines", line=dict(color=color, width=1.5),
                name=f"MA{w}",
                hovertemplate=f"MA{w}: " + "%{y:,.0f}<extra></extra>",
            ),
            row=4, col=1,
        )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)

    action_colors = ["#d32f2f" if v < 0 else "#2e7d32" for v in action_stats["pnl"]]
    fig.add_trace(
        go.Bar(
            x=action_stats["action"], y=action_stats["pnl"],
            marker_color=action_colors,
            text=[f"{v:,.0f}<br>{t} сд.<br>приб. {w:.0f}%"
                  for v, t, w in zip(action_stats["pnl"], action_stats["trades"], action_stats["winrate"])],
            textposition="outside",
            name="P/L по action",
            hovertemplate="%{x}<br>P/L: %{y:,.0f}<extra></extra>",
        ),
        row=4, col=2,
    )

    sent_colors = ["#d32f2f" if v < 0 else "#2e7d32" for v in sent_stats["pnl"]]
    fig.add_trace(
        go.Bar(
            x=sent_stats["sentiment"], y=sent_stats["pnl"],
            marker_color=sent_colors,
            text=[f"{v:,.0f}" for v in sent_stats["pnl"]],
            textposition="outside",
            name="P/L по sentiment",
            hovertemplate="sentiment: %{x}<br>P/L: %{y:,.0f}<extra></extra>",
        ),
        row=5, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=sent_stats["sentiment"], y=sent_stats["trades"],
            marker_color="#1565c0",
            name="Кол-во сделок",
            hovertemplate="sentiment: %{x}<br>сделок: %{y}<extra></extra>",
        ),
        row=5, col=2,
    )

    title = f"{ticker} | {model_name} | бэктест sentiment — правила: {rules_path.name}"
    fig.update_layout(
        height=2240,
        width=1500,
        title_text=f"{title}<br><sub>{test_period_text}</sub><br><sub>{stats_text}</sub>",
        title_x=0.5,
        margin=dict(t=140),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5),
        template="plotly_white",
        hovermode="x unified",
    )
    for r, c in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (4, 1), (4, 2), (5, 1), (5, 2)]:
        fig.update_yaxes(tickformat=",", row=r, col=c)

    # ── Таблица статистики ────────────────────────────────────────────────
    sec1 = [
        ["<b>ДОХОДНОСТЬ</b>", ""],
        ["Чистая прибыль", f"{total_profit:,.0f}"],
        ["Годовая прибыль (экстрапол.)", f"{annual_profit:,.0f}"],
        ["Средний P/L на сделку", f"{avg_trade:,.0f}"],
        ["Медианный P/L на сделку", f"{median_trade:,.0f}"],
        ["Лучшая сделка", f"{best_trade:,.0f}"],
        ["Худшая сделка", f"{worst_trade:,.0f}"],
    ]
    sec2 = [
        ["<b>РИСК</b>", ""],
        ["Max Drawdown", f"{max_dd:,.0f}"],
        ["Длит. макс. просадки", f"{max_dd_duration} сделок"],
        ["Волатильность (год.)", f"{volatility:,.0f}"],
        ["Std сделки", f"{std_trade:,.0f}"],
        ["VaR 95%", f"{np.percentile(pl, 5):,.0f}"],
        ["CVaR 95%", f"{pl[pl <= np.percentile(pl, 5)].mean():,.0f}"],
    ]
    sec3 = [
        ["<b>СТАТИСТИКА СДЕЛОК</b>", ""],
        ["Всего сделок", f"{total_trades}"],
        ["Win / Loss", f"{win_trades} / {loss_trades}"],
        ["Win rate", f"{win_rate:.1f}%"],
        ["Ср. выигрыш / проигрыш", f"{avg_win:,.0f} / {avg_loss:,.0f}"],
        ["Макс. серия побед", f"{max_consec_wins}"],
        ["Макс. серия убытков", f"{max_consec_losses}"],
    ]

    num_rows = max(len(sec1), len(sec2), len(sec3))
    for sec in (sec1, sec2, sec3):
        while len(sec) < num_rows:
            sec.append(["", ""])

    cols_values = [[], [], [], [], [], []]
    tbl_colors = [[], [], []]
    for i in range(num_rows):
        for j, sec in enumerate((sec1, sec2, sec3)):
            n, v = sec[i]
            is_hdr = v == "" and n.startswith("<b>")
            cols_values[j * 2].append(n)
            cols_values[j * 2 + 1].append(f"<b>{v}</b>" if v and not is_hdr else v)
            if is_hdr:
                tbl_colors[j].append("#e3f2fd")
            else:
                tbl_colors[j].append("#f5f5f5" if i % 2 == 0 else "white")

    fig_stats = go.Figure(
        go.Table(
            columnwidth=[200, 130, 180, 120, 220, 120],
            header=dict(
                values=["<b>Показатель</b>", "<b>Значение</b>"] * 3,
                fill_color="#1565c0",
                font=dict(color="white", size=14),
                align="left",
                height=32,
            ),
            cells=dict(
                values=cols_values,
                fill_color=[tbl_colors[0], tbl_colors[0], tbl_colors[1], tbl_colors[1],
                            tbl_colors[2], tbl_colors[2]],
                font=dict(size=13, color="#212121"),
                align=["left", "right", "left", "right", "left", "right"],
                height=26,
            ),
        )
    )
    fig_stats.update_layout(
        title_text=f"<b>{ticker} | {model_name} — Backtest: статистика стратегии</b>",
        title_x=0.5,
        title_font_size=18,
        height=32 + num_rows * 26 + 80,
        width=1500,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    # ── Таблица коэффициентов ─────────────────────────────────────────────
    coefficients = [
        {
            "name": "Recovery Factor", "formula": "Чистая прибыль / |Max Drawdown|",
            "value": f"{recovery_factor:.2f}",
            "description": "Коэффициент восстановления — во сколько раз прибыль превышает максимальную просадку. RF > 1 — стратегия заработала больше, чем потеряла в худший период.",
        },
        {
            "name": "Profit Factor", "formula": "Валовая прибыль / Валовый убыток",
            "value": f"{profit_factor:.2f}",
            "description": "Фактор прибыли. PF > 1 — прибыльность, 1.5–2.0 хорошо, > 2.0 отлично.",
        },
        {
            "name": "Payoff Ratio", "formula": "Средний выигрыш / Средний проигрыш",
            "value": f"{payoff_ratio:.2f}",
            "description": "При высоком payoff стратегия остаётся прибыльной даже при win rate < 50%.",
        },
        {
            "name": "Sharpe Ratio", "formula": "(Ср. P/L / Std) × √252",
            "value": f"{sharpe:.2f}",
            "description": "Отношение доходности к риску, приведённое к году. > 1 хорошо, > 2 отлично, > 3 исключительно.",
        },
        {
            "name": "Sortino Ratio", "formula": "(Ср. P/L / Downside Std) × √252",
            "value": f"{sortino:.2f}",
            "description": "Модификация Шарпа, учитывающая только нисходящую волатильность.",
        },
        {
            "name": "Calmar Ratio", "formula": "Годовая доходность / |Max Drawdown|",
            "value": f"{calmar:.2f}",
            "description": "Отношение годовой прибыли к макс. просадке. > 1 — прибыль превышает худшую просадку, > 3 отлично.",
        },
        {
            "name": "Expectancy", "formula": "Win% × Ср.выигрыш − Loss% × Ср.проигрыш",
            "value": f"{expectancy:,.0f}",
            "description": "Матожидание на одну сделку. Положительное — стратегия имеет преимущество (edge).",
        },
    ]

    fig_table = go.Figure(
        go.Table(
            columnwidth=[150, 250, 80, 450],
            header=dict(
                values=["<b>Коэффициент</b>", "<b>Формула</b>", "<b>Значение</b>", "<b>Расшифровка</b>"],
                fill_color="#1565c0",
                font=dict(color="white", size=14),
                align="left",
                height=36,
            ),
            cells=dict(
                values=[
                    [f"<b>{c['name']}</b>" for c in coefficients],
                    [c["formula"] for c in coefficients],
                    [f"<b>{c['value']}</b>" for c in coefficients],
                    [c["description"] for c in coefficients],
                ],
                fill_color=[["#f5f5f5" if i % 2 == 0 else "white" for i in range(len(coefficients))]] * 4,
                font=dict(size=13, color="#212121"),
                align=["left", "left", "center", "left"],
                height=60,
            ),
        )
    )
    fig_table.update_layout(
        title_text=f"<b>{ticker} | {model_name} — Backtest: ключевые коэффициенты</b>",
        title_x=0.5,
        title_font_size=18,
        height=560,
        width=1500,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    forecast_html = build_next_month_forecast_html(result)

    # ── Сохранение ────────────────────────────────────────────────────────
    output_html.parent.mkdir(parents=True, exist_ok=True)
    with output_html.open("w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html>\n<html><head><meta charset='utf-8'>\n")
        f.write(f"<title>{ticker} | {model_name} | sentiment backtest</title>\n</head><body>\n")
        f.write(fig.to_html(include_plotlyjs="cdn", full_html=False))
        f.write("\n<hr style='margin:30px 0; border:1px solid #ccc'>\n")
        f.write(fig_stats.to_html(include_plotlyjs=False, full_html=False))
        f.write("\n<hr style='margin:30px 0; border:1px solid #ccc'>\n")
        f.write(fig_table.to_html(include_plotlyjs=False, full_html=False))
        f.write("\n<hr style='margin:30px 0; border:1px solid #ccc'>\n")
        f.write(forecast_html)
        f.write("\n</body></html>")


def build_qs_report(result: pd.DataFrame, ticker: str, model_name: str, output_html: Path,
                    notional_capital: float = 1_000_000) -> None:
    """Генерирует QuantStats HTML tearsheet по серии доходностей backtest."""
    df = result.copy()
    df["source_date"] = pd.to_datetime(df["source_date"])
    returns = df.set_index("source_date")["pnl"] / notional_capital
    returns.index.name = None
    returns = returns.sort_index()
    report_title = f"{ticker} | {model_name} | бэктест sentiment (QuantStats)"
    qs.reports.html(returns, benchmark=None, output=str(output_html),
                    title=report_title, compounded=False)
    _replace_html_title(output_html, report_title)
    _insert_qs_notional_caption(output_html, notional_capital)


def _replace_html_title(output_html: Path, title: str) -> None:
    html = output_html.read_text(encoding="utf-8")
    title_tag = f"<title>{escape(title, quote=False)}</title>"
    html, count = re.subn(r"<title>.*?</title>", title_tag, html, count=1, flags=re.IGNORECASE | re.DOTALL)
    if count == 0:
        html = html.replace("</head>", f"{title_tag}\n</head>", 1)
    output_html.write_text(html, encoding="utf-8")

def _format_notional_capital(value: float) -> str:
    """Форматирует капитал для подписи QuantStats-отчета."""
    if float(value).is_integer():
        return f"{value:,.0f}".replace(",", " ")
    return f"{value:,.2f}".replace(",", " ")


def _insert_qs_notional_caption(output_html: Path, notional_capital: float) -> None:
    html = output_html.read_text(encoding="utf-8")
    caption = (
        '<p id="notional-capital-caption" '
        'style="font-size: 16px; margin: -8px 0 24px 0; color: #444;">'
        f"Бэктест 1 контрактом с начальным капиталом: "
        f"{escape(_format_notional_capital(notional_capital), quote=False)}"
        "</p>"
    )
    html = re.sub(
        r'\s*<p id="notional-capital-caption".*?</p>',
        "",
        html,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )
    html, count = re.subn(
        r"(<h1[^>]*>.*?</h1>)",
        rf"\1\n{caption}",
        html,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if count == 0:
        html = re.sub(
            r"(<body[^>]*>)",
            rf"\1\n{caption}",
            html,
            count=1,
            flags=re.IGNORECASE,
        )
    output_html.write_text(html, encoding="utf-8")

@app.command()
def main(
    quantity: Optional[int] = typer.Option(
        None,
        help="Количество контрактов на сделку. По умолчанию — quantity_test из settings.yaml.",
    ),
    rules_yaml: Path = typer.Option(
        None,
        "--rules-yaml",
        help="YAML-файл с правилами. По умолчанию rules.yaml рядом со скриптом.",
    ),
    date_from: Optional[str] = typer.Option(
        None,
        "--date-from",
        help="Нижняя граница окна (YYYY-MM-DD). Переопределяет settings.yaml:backtest_date_from.",
    ),
    date_to: Optional[str] = typer.Option(
        None,
        "--date-to",
        help="Верхняя граница окна (YYYY-MM-DD). Переопределяет settings.yaml:backtest_date_to.",
    ),
) -> None:
    """Запускает полный бэктест sentiment-стратегии и сохраняет отчёты."""
    # --- Загрузка настроек модели из единого {ticker}/settings.yaml ---
    settings = load_settings_for(__file__, "model")

    ticker = settings.get("ticker", "")
    model_name = str(settings.get("sentiment_model", "qwen3.6:35b"))

    sentiment_pkl = resolve_sentiment_pkl(settings)
    if quantity is None:
        quantity = int(settings.get("quantity_test", 1))

    if rules_yaml is None:
        rules_yaml = Path(__file__).resolve().parent / "rules.yaml"
    rules = load_rules(rules_yaml)

    d_from = _parse_date(date_from if date_from is not None else settings.get("backtest_date_from"))
    d_to = _parse_date(date_to if date_to is not None else settings.get("backtest_date_to"))

    df = load_sentiment(sentiment_pkl)
    aggregated = index_by_date(df)

    if d_from is not None:
        aggregated = aggregated[aggregated.index >= d_from]
    if d_to is not None:
        aggregated = aggregated[aggregated.index <= d_to]

    if aggregated.empty:
        typer.echo("После фильтра по дате не осталось записей. Проверьте окно.")
        raise typer.Exit(code=1)

    result = build_backtest(aggregated, quantity, rules)

    if result.empty:
        typer.echo("Нет доступных сделок для бэктеста. Проверьте pkl и правила.")
        raise typer.Exit(code=1)

    report_folder = Path(__file__).resolve().parent / "plots"
    output_html = report_folder / "sentiment_backtest.html"
    output_xlsx = Path(__file__).resolve().parent / "backtest" / "sentiment_backtest_results.xlsx"
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(output_xlsx, index=False)
    build_report(result, ticker, model_name, output_html, rules_yaml)

    output_qs_html = report_folder / "sentiment_backtest_qs.html"
    notional_capital = float(settings.get("notional_capital", 1_000_000))
    build_qs_report(result, ticker, model_name, output_qs_html, notional_capital)

    typer.echo(f"Готово: {output_xlsx} и {output_html}")
    typer.echo(f"Правила: {rules_yaml}")
    typer.echo(f"Всего сделок: {len(result)}")
    typer.echo(f"Общий PnL: {result['pnl'].sum():.2f}")
    typer.echo(f"Доля прибыльных сделок: {(result['pnl'] > 0).mean() * 100:.1f}%")
    typer.echo(f"Макс. просадка: {_max_drawdown(result):.2f}")


if __name__ == "__main__":
    app()
