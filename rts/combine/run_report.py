"""
Отчётный оркестратор combine-пайплайна тикера.

Запускает 1 скрипт в текущей папке:
1. sentiment_combine.py      — бэктест согласия двух моделей, отчёты в plots/.

Шаг sentiment_to_predict.py намеренно НЕ входит в этот оркестратор — генерация
объединённого прогноза выполняется только торговым оркестратором run_trade.py
на основной машине.

Тикер выводится из имени родительской папки (Path(__file__).resolve().parents[1]),
скрипт самодостаточен и работает в любом тикере без изменений.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import typer

SCRIPT_DIR = Path(__file__).resolve().parent
TICKER = SCRIPT_DIR.parent.name

PIPELINE: list[str] = [
    "sentiment_combine.py",
]

app = typer.Typer(help=f"Отчётный combine-пайплайн тикера {TICKER} (без генерации прогноза).")


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
        help="Запустить только указанные шаги через запятую.",
    ),
) -> None:
    """Прогоняет combine-пайплайн в отчётном режиме."""
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

    typer.echo(f"\nCombine-пайплайн завершён успешно за {total_elapsed:.1f} с.")


if __name__ == "__main__":
    app()
