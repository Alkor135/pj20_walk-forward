from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd


def _fmt_num(value: float) -> str:
    """Форматирует число с пробелами между тысячами для HTML-отчета."""
    return f"{value:,.0f}".replace(",", " ")


def _forecast_interval_rows(mean_month: float, sigma_month: float) -> list[dict]:
    """Возвращает нормальные прогнозные интервалы для месячного PnL."""
    z_by_probability = [
        ("50%", 0.67448975),
        ("68%", 1.0),
        ("80%", 1.28155156),
        ("90%", 1.64485363),
        ("95%", 1.95996398),
        ("99%", 2.5758293),
    ]
    return [
        {
            "probability": probability,
            "low": mean_month - z * sigma_month,
            "high": mean_month + z * sigma_month,
        }
        for probability, z in z_by_probability
    ]


def _row_style(index: int) -> str:
    """Возвращает фон строки для легкого чередования в HTML-таблицах."""
    color = "#f7f7f7" if index % 2 == 0 else "#ffffff"
    return f"background:{color};"


def build_next_month_forecast_html(
    result: pd.DataFrame,
    forecast_days: int = 21,
    bootstrap_samples: int = 200_000,
) -> str:
    """Строит HTML-блок с прогнозным распределением PnL на следующий месяц."""
    pl = pd.to_numeric(result["pnl"], errors="coerce").dropna().astype(float)
    if len(pl) < 2:
        return ""

    avg_daily = float(pl.mean())
    std_daily = float(pl.std(ddof=1))
    mean_month = avg_daily * forecast_days
    sigma_month = std_daily * np.sqrt(forecast_days)
    normal_rows = _forecast_interval_rows(mean_month, sigma_month)

    rng = np.random.default_rng(42)
    bootstrap = rng.choice(pl.to_numpy(), size=(bootstrap_samples, forecast_days), replace=True).sum(axis=1)
    bootstrap_specs = [
        ("50%", 25, 75),
        ("68%", 16, 84),
        ("80%", 10, 90),
        ("90%", 5, 95),
        ("95%", 2.5, 97.5),
        ("99%", 0.5, 99.5),
    ]
    bootstrap_rows = [
        {
            "probability": probability,
            "low": float(np.percentile(bootstrap, low_pct)),
            "high": float(np.percentile(bootstrap, high_pct)),
        }
        for probability, low_pct, high_pct in bootstrap_specs
    ]

    threshold_rows = [
        ("P(PnL <= -10 000)", float((bootstrap <= -10_000).mean() * 100)),
        ("P(PnL <= -5 000)", float((bootstrap <= -5_000).mean() * 100)),
        ("P(PnL <= 0)", float((bootstrap <= 0).mean() * 100)),
        ("Вероятность прибыли, P(PnL > 0)", float((bootstrap > 0).mean() * 100)),
        ("P(PnL >= 5 000)", float((bootstrap >= 5_000).mean() * 100)),
        ("P(PnL >= 10 000)", float((bootstrap >= 10_000).mean() * 100)),
        ("P(PnL >= 20 000)", float((bootstrap >= 20_000).mean() * 100)),
    ]

    normal_rows_html = "\n".join(
        f"<tr style=\"{_row_style(i)}\">"
        f"<td>{row['probability']}</td>"
        f"<td>{_fmt_num(row['low'])} ... {_fmt_num(row['high'])}</td>"
        "</tr>"
        for i, row in enumerate(normal_rows)
    )
    bootstrap_rows_html = "\n".join(
        f"<tr style=\"{_row_style(i)}\">"
        f"<td>{row['probability']}</td>"
        f"<td>{_fmt_num(row['low'])} ... {_fmt_num(row['high'])}</td>"
        "</tr>"
        for i, row in enumerate(bootstrap_rows)
    )
    threshold_rows_html = "\n".join(
        f"<tr style=\"{_row_style(i)}\"><td>{escape(label, quote=False)}</td><td>{value:.1f}%</td></tr>"
        for i, (label, value) in enumerate(threshold_rows)
    )

    return f"""
<section id="next-month-forecast" style="width:1450px; margin:32px auto 44px auto; font-family:Arial, sans-serif; color:#212121;">
  <h2 style="text-align:center; margin:0 0 8px 0;">Прогноз на следующий месяц</h2>
  <p style="text-align:center; margin:0 0 22px 0; color:#555;">
    Оценка распределения PnL на {forecast_days} будущих сигналов/дней по историческому ряду сделок.
  </p>
  <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:18px; align-items:start;">
    <div style="border:1px solid #ddd; border-radius:6px; padding:16px;">
      <h3 style="margin:0 0 12px 0; font-size:18px;">Базовые параметры</h3>
      <table style="width:100%; border-collapse:collapse; font-size:14px;">
        <tr style="{_row_style(0)}"><td>Наблюдений</td><td style="text-align:right;"><b>{len(pl)}</b></td></tr>
        <tr style="{_row_style(1)}"><td>Средний дневной PnL</td><td style="text-align:right;"><b>{_fmt_num(avg_daily)}</b></td></tr>
        <tr style="{_row_style(2)}"><td>Дневная σ</td><td style="text-align:right;"><b>{_fmt_num(std_daily)}</b></td></tr>
        <tr style="{_row_style(3)}"><td>Ожидаемый PnL месяца</td><td style="text-align:right;"><b>{_fmt_num(mean_month)}</b></td></tr>
        <tr style="{_row_style(4)}"><td>Месячная σ</td><td style="text-align:right;"><b>{_fmt_num(sigma_month)}</b></td></tr>
      </table>
      <p style="font-size:13px; color:#555; line-height:1.4;">
        Нормальная модель: mean ± z × σ, где месячная σ = дневная σ × √N.
      </p>
    </div>
    <div style="border:1px solid #ddd; border-radius:6px; padding:16px;">
      <h3 style="margin:0 0 12px 0; font-size:18px;">Нормальные интервалы</h3>
      <table style="width:100%; border-collapse:collapse; font-size:14px;">
        <tr style="background:#1565c0; color:white;"><th style="text-align:left; padding:7px;">Вероятность</th><th style="text-align:right; padding:7px;">Диапазон PnL</th></tr>
        {normal_rows_html}
      </table>
    </div>
    <div style="border:1px solid #ddd; border-radius:6px; padding:16px;">
      <h3 style="margin:0 0 12px 0; font-size:18px;">Бутстрэп</h3>
      <table style="width:100%; border-collapse:collapse; font-size:14px;">
        <tr style="background:#1565c0; color:white;"><th style="text-align:left; padding:7px;">Вероятность</th><th style="text-align:right; padding:7px;">Диапазон PnL</th></tr>
        {bootstrap_rows_html}
      </table>
    </div>
  </div>
  <div style="border:1px solid #ddd; border-radius:6px; padding:16px; margin-top:18px;">
    <h3 style="margin:0 0 12px 0; font-size:18px;">Вероятности порогов по бутстрэпу</h3>
    <table style="width:100%; border-collapse:collapse; font-size:14px;">
      <tr style="background:#1565c0; color:white;"><th style="text-align:left; padding:7px;">Событие</th><th style="text-align:right; padding:7px;">Вероятность</th></tr>
      {threshold_rows_html}
    </table>
    <p style="font-size:13px; color:#555; line-height:1.4;">
      Это не прогноз рынка, а статистическая оценка следующего месяца при условии, что будущие сделки похожи на исторический бэктест.
    </p>
  </div>
</section>
"""
