"""
Оркестратор всех модельных пайплайнов для тикера RTS.

Скрипт находит в подпапках `rts/<model>/` файлы `run_report.py`
(модельные оркестраторы) и запускает их по очереди.

Каждый модельный оркестратор сам прогоняет 5 шагов своего пайплайна
(sentiment_analysis → sentiment_group_stats → rules_recommendation →
sentiment_backtest → sentiment_to_predict). После всех моделей оркестратор
запускает combine-пайплайн в `rts/combine/` (если эта папка есть):
1. `combine/sentiment_combine.py`    — комбинированный бэктест по двум моделям;
2. `combine/sentiment_to_predict.py` — комбинированный прогноз на сегодня.
При запуске с `--only` combine выполняется только если явно указан в списке.

Запуск:
python rts/run_rts.py
python rts/run_rts.py --only gemma3_12b,gemma4_e2b,gemma4_e4b,qwen2.5_14b,qwen2.5_7b,qwen3_14b --keep-going
python rts/run_rts.py --keep-going  «продолжать дальше»
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import typer

TICKER_DIR = Path(__file__).resolve().parent
MODEL_RUNNER = "run_report.py"
COMBINE_NAME = "combine"

app = typer.Typer(help="Последовательный запуск пайплайнов всех моделей RTS.")


def discover_model_runners() -> list[Path]:
    """Возвращает отсортированный список оркестраторов вида rts/<model>/run_report.py."""
    runners: list[Path] = []
    for child in sorted(TICKER_DIR.iterdir()):
        if not child.is_dir() or child.name == COMBINE_NAME:
            continue
        candidate = child / MODEL_RUNNER
        if candidate.exists():
            runners.append(candidate)
    return runners


def build_run_plan(
    all_runners: list[Path],
    only: Optional[str],
    combine_dir: Path,
) -> tuple[list[Path], bool]:
    """Возвращает список модельных runner'ов и флаг запуска combine."""
    if not only:
        return all_runners, combine_dir.is_dir()

    wanted = {s.strip() for s in only.split(",") if s.strip()}
    model_names = {r.parent.name for r in all_runners}
    allowed = model_names | {COMBINE_NAME}
    unknown = wanted - allowed
    if unknown:
        available = sorted(allowed)
        raise typer.BadParameter(
            f"Неизвестные модели/шаги: {sorted(unknown)}. Доступны: {available}"
        )

    runners = [r for r in all_runners if r.parent.name in wanted]
    return runners, COMBINE_NAME in wanted and combine_dir.is_dir()


def run_model(
    runner: Path,
    stop_on_error: bool,
    label: str | None = None,
) -> tuple[bool, float]:
    """Запускает один скрипт (модельный оркестратор или combine) и возвращает (успех, время)."""
    name = label or runner.parent.name
    typer.echo(f"\n########## {name} ##########")
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, str(runner)],
        cwd=str(runner.parent),
    )
    elapsed = time.monotonic() - started

    if completed.returncode == 0:
        typer.echo(f"[OK]   {name} ({elapsed:.1f} с)")
        return True, elapsed

    typer.echo(f"[FAIL] {name} код={completed.returncode} ({elapsed:.1f} с)")
    if stop_on_error:
        raise typer.Exit(code=completed.returncode)
    return False, elapsed


@app.command()
def main(
    only: Optional[str] = typer.Option(
        None,
        "--only",
        help="Запустить только указанные модели/шаги через запятую; combine можно указать явно.",
    ),
    keep_going: bool = typer.Option(
        False,
        "--keep-going/--stop-on-error",
        help="Продолжать прогон при падении модели (по умолчанию — останавливаться).",
    ),
) -> None:
    """Прогоняет пайплайны всех моделей RTS или подмножества по --only."""
    all_runners = discover_model_runners()
    if not all_runners:
        typer.echo(f"Не найдено модельных оркестраторов в {TICKER_DIR}.")
        raise typer.Exit(code=1)

    combine_dir = TICKER_DIR / "combine"
    runners, run_combine = build_run_plan(
        all_runners=all_runners,
        only=only,
        combine_dir=combine_dir,
    )

    typer.echo(f"Корневая папка: {TICKER_DIR}")
    typer.echo(f"Моделей к запуску: {len(runners)}")
    for r in runners:
        typer.echo(f"  - {r.parent.name}")
    typer.echo(f"combine: {run_combine}")

    total_started = time.monotonic()
    summary: list[tuple[str, bool, float]] = []
    for runner in runners:
        ok, elapsed = run_model(runner, stop_on_error=not keep_going)
        summary.append((runner.parent.name, ok, elapsed))

    if run_combine:
        for script_name in ("sentiment_combine.py", "sentiment_to_predict.py"):
            script = combine_dir / script_name
            if not script.exists():
                continue
            label = f"combine/{script.stem}"
            ok, elapsed = run_model(
                script,
                stop_on_error=not keep_going,
                label=label,
            )
            summary.append((label, ok, elapsed))

    total_elapsed = time.monotonic() - total_started

    typer.echo("\n========== ИТОГ ==========")
    name_width = max(14, max((len(n) for n, _, _ in summary), default=14))
    for name, ok, elapsed in summary:
        marker = "OK  " if ok else "FAIL"
        typer.echo(f"  [{marker}] {name:{name_width}s} {elapsed:8.1f} с")
    typer.echo(f"Общее время: {total_elapsed:.1f} с")

    if any(not ok for _, ok, _ in summary):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
