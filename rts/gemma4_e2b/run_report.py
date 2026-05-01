"""
Отчётный оркестратор пайплайна для модели gemma4:e2b.

Последовательно запускает 4 скрипта в текущей папке:
1. sentiment_analysis.py     — расчёт sentiment-оценок и сохранение PKL.
2. sentiment_group_stats.py  — групповая статистика → group_stats/sentiment_group_stats.xlsx.
3. rules_recommendation.py   — генерация rules.yaml на основе XLSX.
4. sentiment_backtest.py     — бэктест по rules.yaml, отчёты в plots/ и backtest/.

Шаг sentiment_to_predict.py намеренно НЕ входит в этот оркестратор — генерация
прогнозов выполняется только торговым оркестратором run_trade.py на основной
машине, чтобы исключить расхождения предсказаний между разными компьютерами.

Каждый шаг выполняется тем же интерпретатором, в котором запущен оркестратор.
Аргументы CLI прозрачно прокидываются дальше — например, `--verbose` или
`--no-use-cache` уйдут только в шаги, которые такие опции принимают, поэтому
оркестратор знает только два режима: запустить весь pipeline или только
отдельные шаги через `--only`.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import typer

SCRIPT_DIR = Path(__file__).resolve().parent

PIPELINE: list[str] = [
    "sentiment_analysis.py",
    "sentiment_group_stats.py",
    "rules_recommendation.py",
    "sentiment_backtest.py",
]

app = typer.Typer(help="Отчётный пайплайн модели gemma4:e2b (без генерации прогноза).")


def run_step(script: str) -> None:
    """Запускает один скрипт пайплайна и пробрасывает его stdout/stderr в консоль."""
    script_path = SCRIPT_DIR / script
    if not script_path.exists():
        raise typer.BadParameter(f"Скрипт пайплайна не найден: {script_path}")

    typer.echo(f"\n=== {script} ===")
    started = time.monotonic()
    completed = subprocess.run([sys.executable, str(script_path)], cwd=str(SCRIPT_DIR))
    elapsed = time.monotonic() - started

    if completed.returncode != 0:
        typer.echo(f"[FAIL] {script} завершился с кодом {completed.returncode} ({elapsed:.1f} с)")
        raise typer.Exit(code=completed.returncode)

    typer.echo(f"[OK]   {script} ({elapsed:.1f} с)")


@app.command()
def main(
    only: Optional[str] = typer.Option(
        None,
        "--only",
        help="Запустить только указанные шаги через запятую (например: sentiment_analysis,sentiment_backtest).",
    ),
) -> None:
    """Прогоняет полный пайплайн или подмножество шагов."""
    if only:
        wanted = [s.strip() for s in only.split(",") if s.strip()]
        normalized = [s if s.endswith(".py") else f"{s}.py" for s in wanted]
        unknown = [s for s in normalized if s not in PIPELINE]
        if unknown:
            raise typer.BadParameter(
                f"Неизвестные шаги: {unknown}. Доступны: {PIPELINE}"
            )
        steps = [s for s in PIPELINE if s in normalized]
    else:
        steps = list(PIPELINE)

    typer.echo(f"Папка пайплайна: {SCRIPT_DIR}")
    typer.echo(f"Шаги к запуску: {steps}")

    total_started = time.monotonic()
    for script in steps:
        run_step(script)
    total_elapsed = time.monotonic() - total_started

    typer.echo(f"\nПайплайн завершён успешно за {total_elapsed:.1f} с.")


if __name__ == "__main__":
    app()
