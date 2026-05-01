"""
Бэктест комбинированной торговли по предсказаниям двух sentiment-моделей.

Скрипт читает настройки из единого `rts/settings.yaml` (`model_1`, `model_2`,
`notional_capital`), нормализует имена моделей в имена папок (замена `:` на `_`),
грузит для каждой модели XLSX из
`<ticker>/<model_dir>/backtest/sentiment_backtest_results.xlsx`.

Логика комбинирования: торгуем 1 контрактом только в дни, когда обе модели
выдали сигнал и направления совпадают (LONG+LONG или SHORT+SHORT). При
расхождении направлений или если хотя бы одна модель смолчала — сделки нет.
Размер P/L не удваивается — берётся P/L одной из моделей (model_1), так как
при совпадении направлений и одинаковой quantity P/L обеих моделей равен.

Сохраняет рядом со скриптом два отчёта:
- `plots/sentiment_combine.html`    — подробный Plotly HTML (по аналогии с
  sentiment_backtest.html, но без панелей по action/sentiment);
- `plots/sentiment_combine_qs.html` — QuantStats tearsheet.

Скрипт самодостаточен: тикер берётся из имени родительской папки
(`Path(__file__).resolve().parents[1]`), пути к моделям строятся
относительно этой же папки. Скопированный в другой тикер скрипт работает
без изменений — правится только секция в `<ticker>/settings.yaml`.
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import quantstats_lumi as qs
import typer
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
TICKER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(TICKER_DIR))
from config_loader import load_settings_for, load_model_settings as load_model_settings_from_config

app = typer.Typer(help="Бэктест комбинированной торговли по двум sentiment-моделям.")


def model_folder_name(model: str) -> str:
    """Имя папки модели: `gemma3:12b` -> `gemma3_12b`, `gemma3_12b` -> `gemma3_12b`."""
    return model.replace(":", "_")


def load_combine_settings() -> dict:
    """Загружает настройки combine из единого {ticker}/settings.yaml."""
    return load_settings_for(__file__, "combine")


def load_model_settings(model_dir: Path) -> dict:
    """Загружает настройки модели из единого {ticker}/settings.yaml."""
    return load_model_settings_from_config(TICKER_DIR, model_dir.name)


def load_strategy_xlsx(path: Path) -> pd.DataFrame:
    """Грузит XLSX бэктеста модели: source_date, direction, pnl."""
    if not path.exists():
        raise typer.BadParameter(f"XLSX с бэктестом не найден: {path}")
    df = pd.read_excel(path)
    required = {"source_date", "direction", "pnl"}
    missing = required - set(df.columns)
    if missing:
        raise typer.BadParameter(
            f"XLSX {path} не содержит обязательные колонки: {missing}"
        )
    out = pd.DataFrame(
        {
            "source_date": pd.to_datetime(df["source_date"], errors="coerce"),
            "direction": df["direction"].astype(str).str.upper(),
            "pnl": pd.to_numeric(df["pnl"], errors="coerce"),
        }
    )
    return (
        out.dropna(subset=["source_date", "pnl"])
        .sort_values("source_date")
        .reset_index(drop=True)
    )


def build_combined(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Inner-merge по дате; оставляет дни с совпадающими направлениями.

    Комбинированный P/L = P/L первой модели (== P/L второй при совпадении
    направления и одинаковой quantity).
    """
    merged = pd.merge(df1, df2, on="source_date", how="inner", suffixes=("_1", "_2"))
    agreed = merged[merged["direction_1"] == merged["direction_2"]].reset_index(drop=True)
    agreed["direction"] = agreed["direction_1"]
    agreed["pnl"] = agreed["pnl_1"]
    return agreed.sort_values("source_date").reset_index(drop=True)


def build_equity_axis(
    df1: pd.DataFrame, df2: pd.DataFrame, combined: pd.DataFrame
) -> pd.DataFrame:
    """Объединённая дата-ось со столбцами cum_<m1> / cum_<m2> / cum_combined.

    Используется только для графика сравнения equity (в бэктест-метрики не идёт).
    """
    union = (
        pd.merge(
            df1[["source_date"]],
            df2[["source_date"]],
            on="source_date",
            how="outer",
        )
        .sort_values("source_date")
        .reset_index(drop=True)
    )
    pl_1 = pd.merge(union, df1[["source_date", "pnl"]], on="source_date", how="left")["pnl"].fillna(0.0)
    pl_2 = pd.merge(union, df2[["source_date", "pnl"]], on="source_date", how="left")["pnl"].fillna(0.0)
    pl_c = pd.merge(union, combined[["source_date", "pnl"]], on="source_date", how="left")["pnl"].fillna(0.0)
    union["cum_1"] = pl_1.cumsum()
    union["cum_2"] = pl_2.cumsum()
    union["cum_combined"] = pl_c.cumsum()
    return union


def _max_consecutive(signs: pd.Series, target: int) -> int:
    """Максимальная длина подряд идущих значений, равных target."""
    best = current = 0
    for sign in signs:
        if sign == target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _drawdown_duration(drawdown: pd.Series) -> int:
    """Максимальная длительность просадки в количестве сделок."""
    max_dd_duration = 0
    current_dd_start = None
    for i, dd_value in enumerate(drawdown):
        if dd_value < 0:
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


def build_report(
    combined: pd.DataFrame,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    ticker: str,
    label_1: str,
    label_2: str,
    output_html: Path,
) -> None:
    """Строит подробный Plotly HTML-отчёт по комбинированной стратегии."""
    df = combined.copy()
    df["source_date"] = pd.to_datetime(df["source_date"])
    df = df.sort_values("source_date").reset_index(drop=True)

    pl = df["pnl"].astype(float)
    cum = pl.cumsum()
    equity = build_equity_axis(df1, df2, combined)

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

    # ── Метрики ───────────────────────────────────────────────────────────
    total_profit = float(cum.iloc[-1]) if not cum.empty else 0.0
    total_trades = int((pl != 0).sum())
    win_trades = int((pl > 0).sum())
    loss_trades = int((pl < 0).sum())
    win_rate = win_trades / max(total_trades, 1) * 100
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    best_trade = float(pl.max()) if not pl.empty else 0.0
    worst_trade = float(pl.min()) if not pl.empty else 0.0
    avg_trade = float(pl[pl != 0].mean()) if total_trades else 0.0
    median_trade = float(pl[pl != 0].median()) if total_trades else 0.0
    std_trade = float(pl[pl != 0].std()) if total_trades > 1 else 0.0

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
        rows=3, cols=2,
        subplot_titles=(
            "P/L по сделкам (комбинация)",
            "Накопленная прибыль (equity): комбинация vs модели",
            "P/L по неделям",
            "P/L по месяцам",
            "Drawdown от максимума",
            "Распределение P/L сделок",
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "histogram"}],
        ],
        vertical_spacing=0.09,
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Bar(
            x=df["source_date"], y=pl, marker_color=day_colors,
            name="P/L (комбинация)",
            hovertemplate="%{x|%Y-%m-%d}<br>P/L: %{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=equity["source_date"], y=equity["cum_1"],
            mode="lines",
            line=dict(color="#2e7d32", width=1.5),
            name=f"{label_1} (equity)",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>",
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=equity["source_date"], y=equity["cum_2"],
            mode="lines",
            line=dict(color="#1565c0", width=1.5),
            name=f"{label_2} (equity)",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>",
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=equity["source_date"], y=equity["cum_combined"],
            mode="lines", fill="tozeroy",
            line=dict(color="#6a1b9a", width=2.5),
            fillcolor="rgba(106,27,154,0.12)",
            name="Комбинация (equity)",
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
    fig.add_trace(
        go.Histogram(x=pl[pl > 0], marker_color="#2e7d32", opacity=0.7, name="Прибыль", nbinsx=20),
        row=3, col=2,
    )
    fig.add_trace(
        go.Histogram(x=pl[pl < 0], marker_color="#d32f2f", opacity=0.7, name="Убыток", nbinsx=20),
        row=3, col=2,
    )

    title = f"{ticker} | комбинация: {label_1} + {label_2} — sentiment backtest"
    fig.update_layout(
        height=1400,
        width=1500,
        title_text=f"{title}<br><sub>{test_period_text}</sub><br><sub>{stats_text}</sub>",
        title_x=0.5,
        margin=dict(t=140),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5),
        template="plotly_white",
        hovermode="x unified",
    )
    for r, c in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)]:
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
        title_text=f"<b>{ticker} | {label_1} + {label_2} — Combine Backtest: статистика</b>",
        title_x=0.5,
        title_font_size=18,
        height=32 + num_rows * 26 + 80,
        width=1500,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    # ── Сводная таблица: модель_1 (full) vs модель_2 (full) vs комбинация ─
    summary_rows = []
    for series, name in [
        (df1["pnl"].astype(float), label_1),
        (df2["pnl"].astype(float), label_2),
        (df["pnl"].astype(float), "Комбинация"),
    ]:
        s = series
        c = s.cumsum()
        rmax = c.cummax()
        dd = c - rmax
        gp = float(s[s > 0].sum())
        gl = float(abs(s[s < 0].sum()))
        n_trades = int((s != 0).sum())
        n_wins = int((s > 0).sum())
        traded = s[s != 0]
        avg_t = float(traded.mean()) if n_trades else 0.0
        std_t = float(traded.std()) if n_trades > 1 else 0.0
        sharpe_t = (avg_t / std_t) * np.sqrt(252) if std_t > 0 else 0.0
        downside_t = s[s < 0]
        downside_std_t = float(downside_t.std()) if len(downside_t) > 1 else 0.0
        sortino_t = (avg_t / downside_std_t) * np.sqrt(252) if downside_std_t > 0 else 0.0
        summary_rows.append({
            "Стратегия": name,
            "Сделок": n_trades,
            "Win%": f"{(n_wins / n_trades * 100) if n_trades else 0.0:.1f}",
            "Total P/L": f"{float(s.sum()):,.0f}",
            "Max DD": f"{float(dd.min()) if not dd.empty else 0.0:,.0f}",
            "PF": f"{(gp / gl) if gl > 0 else 0.0:.2f}",
            "Sharpe": f"{sharpe_t:.2f}",
            "Sortino": f"{sortino_t:.2f}",
        })
    summary_df = pd.DataFrame(summary_rows)
    fig_compare = go.Figure(
        go.Table(
            header=dict(
                values=[f"<b>{c}</b>" for c in summary_df.columns],
                fill_color="#37474f",
                font=dict(color="white", size=13),
                align="center",
                height=32,
            ),
            cells=dict(
                values=[summary_df[c].tolist() for c in summary_df.columns],
                fill_color=[["#f5f5f5" if i % 2 == 0 else "white" for i in range(len(summary_df))]] * len(summary_df.columns),
                font=dict(size=13, color="#212121"),
                align="center",
                height=28,
            ),
        )
    )
    fig_compare.update_layout(
        title_text=f"<b>{ticker} — сравнение стратегий</b>",
        title_x=0.5,
        title_font_size=16,
        height=32 + len(summary_df) * 28 + 80,
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
        title_text=f"<b>{ticker} | {label_1} + {label_2} — Combine Backtest: ключевые коэффициенты</b>",
        title_x=0.5,
        title_font_size=18,
        height=560,
        width=1500,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    # ── Сохранение ────────────────────────────────────────────────────────
    output_html.parent.mkdir(parents=True, exist_ok=True)
    with output_html.open("w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html>\n<html><head><meta charset='utf-8'>\n")
        f.write(f"<title>{ticker} | {label_1} + {label_2} | combine backtest</title>\n</head><body>\n")
        f.write(fig.to_html(include_plotlyjs="cdn", full_html=False))
        f.write("\n<hr style='margin:30px 0; border:1px solid #ccc'>\n")
        f.write(fig_compare.to_html(include_plotlyjs=False, full_html=False))
        f.write("\n<hr style='margin:30px 0; border:1px solid #ccc'>\n")
        f.write(fig_stats.to_html(include_plotlyjs=False, full_html=False))
        f.write("\n<hr style='margin:30px 0; border:1px solid #ccc'>\n")
        f.write(fig_table.to_html(include_plotlyjs=False, full_html=False))
        f.write("\n</body></html>")


def build_qs_report(
    merged: pd.DataFrame,
    ticker: str,
    label_1: str,
    label_2: str,
    output_html: Path,
    notional_capital: float,
) -> None:
    """Генерирует QuantStats HTML tearsheet по комбинированной серии доходностей."""
    df = merged.copy()
    df["source_date"] = pd.to_datetime(df["source_date"])
    returns = df.set_index("source_date")["pnl"] / notional_capital
    returns.index.name = None
    returns = returns.sort_index()
    report_title = f"{ticker} | {label_1} + {label_2} | combine backtest (QuantStats)"
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
    if float(value).is_integer():
        return f"{value:,.0f}".replace(",", " ")
    return f"{value:,.2f}".replace(",", " ")


def _insert_qs_notional_caption(output_html: Path, notional_capital: float) -> None:
    html = output_html.read_text(encoding="utf-8")
    caption = (
        '<p id="notional-capital-caption" '
        'style="font-size: 16px; margin: -8px 0 24px 0; color: #444;">'
        f"Бэктест комбинированной торговли с начальным капиталом: "
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
def main() -> None:
    """Запускает бэктест комбинированной торговли по двум моделям."""
    cfg = load_combine_settings()
    model_1_raw = cfg.get("model_1")
    model_2_raw = cfg.get("model_2")
    if not model_1_raw or not model_2_raw:
        raise typer.BadParameter("В settings.yaml должны быть указаны model_1 и model_2.")

    folder_1 = model_folder_name(str(model_1_raw))
    folder_2 = model_folder_name(str(model_2_raw))
    model_dir_1 = TICKER_DIR / folder_1
    model_dir_2 = TICKER_DIR / folder_2

    settings_1 = load_model_settings(model_dir_1)
    settings_2 = load_model_settings(model_dir_2)

    ticker = settings_1.get("ticker") or settings_2.get("ticker") or TICKER_DIR.name.upper()
    label_1 = str(settings_1.get("sentiment_model", folder_1))
    label_2 = str(settings_2.get("sentiment_model", folder_2))

    xlsx_1 = model_dir_1 / "backtest" / "sentiment_backtest_results.xlsx"
    xlsx_2 = model_dir_2 / "backtest" / "sentiment_backtest_results.xlsx"

    df1 = load_strategy_xlsx(xlsx_1)
    df2 = load_strategy_xlsx(xlsx_2)

    combined = build_combined(df1, df2)
    if combined.empty:
        typer.echo("Нет дат с совпадающими направлениями двух моделей. Проверьте XLSX.")
        raise typer.Exit(code=1)

    plots_dir = SCRIPT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "sentiment_combine.html"
    output_qs_html = plots_dir / "sentiment_combine_qs.html"

    build_report(combined, df1, df2, ticker, label_1, label_2, output_html)

    notional_capital = float(cfg.get("notional_capital", 1_000_000))
    build_qs_report(combined, ticker, label_1, label_2, output_qs_html, notional_capital)

    n_disagree = int(
        len(pd.merge(df1, df2, on="source_date", how="inner")) - len(combined)
    )
    typer.echo(f"Готово: {output_html}")
    typer.echo(f"Готово: {output_qs_html}")
    typer.echo(f"Сделок (согласие направлений): {len(combined)}")
    typer.echo(f"Пропущено из-за расхождения направлений: {n_disagree}")
    typer.echo(f"Total P/L (комбинация): {combined['pnl'].sum():,.0f}")
    typer.echo(f"Notional capital: {notional_capital:,.0f}")


if __name__ == "__main__":
    app()
