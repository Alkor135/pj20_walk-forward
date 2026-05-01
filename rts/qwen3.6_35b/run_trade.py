"""
Торговый оркестратор пайплайна для модели qwen3.6:35b.

Последовательно запускает 5 скриптов в текущей папке:
1. sentiment_analysis.py     — расчёт sentiment-оценок и сохранение PKL.
2. sentiment_group_stats.py  — групповая статистика → group_stats/sentiment_group_stats.xlsx.
3. rules_recommendation.py   — генерация rules.yaml на основе XLSX.
4. sentiment_backtest.py     — бэктест по rules.yaml, отчёты в plots/ и backtest/.
5. sentiment_to_predict.py   — прогноз направления, файл YYYY-MM-DD.txt в predict_path.

Отличие от run_report.py — добавлен 5-й шаг (генерация прогноза). Этот
оркестратор должен запускаться только на основной торговой машине, чтобы
гарантировать единственный источник предсказаний (sentiment-результаты
локальных LLM могут несущественно отличаться между компьютерами).

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
    "sentiment_to_predict.py",
]

app = typer.Typer(help="Торговый пайплайн модели qwen3.6:35b (с генерацией прогноза).")


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
        help="Запустить только указанные шаги через запятую (например: sentiment_analysis,sentiment_to_predict).",
    ),
) -> None:
    """Прогоняет полный торговый пайплайн или подмножество шагов."""
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
