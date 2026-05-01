"""
Торговый оркестратор всех модельных пайплайнов для тикера RTS.

Сначала скрипт запускает обязательные общие шаги подготовки данных:
1. `shared/download_minutes_to_db.py`
2. `shared/convert_minutes_to_days.py`
3. `shared/create_markdown_files.py`

После успешной подготовки скрипт находит в подпапках `rts/<model>/` файлы `run_trade.py`
(модельные торговые оркестраторы) и запускает их по очереди.

Каждый модельный run_trade.py прогоняет 5 шагов своего пайплайна
(sentiment_analysis → sentiment_group_stats → rules_recommendation →
sentiment_backtest → sentiment_to_predict). После всех моделей оркестратор
запускает combine-пайплайн в `rts/combine/run_trade.py` (если папка есть):
1. `combine/sentiment_combine.py`    — комбинированный бэктест по двум моделям;
2. `combine/sentiment_to_predict.py` — комбинированный прогноз на сегодня.

Этот оркестратор должен запускаться только на основной торговой машине, чтобы
гарантировать единственный источник предсказаний (sentiment-результаты
локальных LLM могут несущественно отличаться между компьютерами).
При запуске с `--only` combine выполняется только если явно указан в списке.

Запуск:
python rts/run_rts_trade.py
python rts/run_rts_trade.py --only gemma3_12b,gemma4_e2b --keep-going
python rts/run_rts_trade.py --keep-going
python rts/run_rts_trade.py --only gemma3_12b,gemma4_e2b,gemma4_e4b,qwen2.5_14b,qwen2.5_7b,qwen3_14b,combine
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import typer

TICKER_DIR = Path(__file__).resolve().parent
TICKER = TICKER_DIR.name

MODEL_RUNNER = "run_trade.py"
COMBINE_RUNNER = "run_trade.py"
COMBINE_NAME = "combine"
SHARED_STEPS = [
    Path("shared") / "download_minutes_to_db.py",
    Path("shared") / "convert_minutes_to_days.py",
    Path("shared") / "create_markdown_files.py",
]

app = typer.Typer(help=f"Торговый пайплайн всех моделей тикера {TICKER}.")


def discover_model_runners() -> list[Path]:
    """Возвращает отсортированный список модельных оркестраторов <model>/run_trade.py."""
    runners: list[Path] = []
    for child in sorted(TICKER_DIR.iterdir()):
        if not child.is_dir() or child.name == "combine":
            continue
        candidate = child / MODEL_RUNNER
        if candidate.exists():
            runners.append(candidate)
    return runners


def build_run_plan(
    all_runners: list[Path],
    only: Optional[str],
    combine_runner: Path,
) -> tuple[list[Path], bool]:
    """Возвращает список модельных runner'ов и флаг запуска combine."""
    if not only:
        return all_runners, combine_runner.exists()

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
    return runners, COMBINE_NAME in wanted and combine_runner.exists()


def run_script(
    runner: Path,
    stop_on_error: bool,
    label: str | None = None,
) -> tuple[bool, float]:
    """Запускает один скрипт (модельный или combine) и возвращает (успех, время)."""
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


def run_shared_steps() -> list[tuple[str, bool, float]]:
    """Запускает обязательную подготовку данных shared перед trade-пайплайном."""
    summary: list[tuple[str, bool, float]] = []
    typer.echo("\n========== SHARED PREP ==========")
    for rel_path in SHARED_STEPS:
        runner = TICKER_DIR / rel_path
        if not runner.exists():
            typer.echo(f"[FAIL] shared/{runner.name}: скрипт не найден: {runner}")
            raise typer.Exit(code=1)
        label = rel_path.as_posix()
        ok, elapsed = run_script(runner, stop_on_error=True, label=label)
        summary.append((label, ok, elapsed))
    return summary


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
    """Прогоняет торговые пайплайны всех моделей тикера или подмножества по --only."""
    all_runners = discover_model_runners()
    if not all_runners:
        typer.echo(f"Не найдено модельных оркестраторов в {TICKER_DIR}.")
        raise typer.Exit(code=1)

    combine_runner = TICKER_DIR / "combine" / COMBINE_RUNNER
    runners, run_combine = build_run_plan(
        all_runners=all_runners,
        only=only,
        combine_runner=combine_runner,
    )

    typer.echo(f"Корневая папка: {TICKER_DIR}")
    typer.echo(f"Режим: торговый (run_trade.py)")
    typer.echo(f"Моделей к запуску: {len(runners)}")
    for r in runners:
        typer.echo(f"  - {r.parent.name}")
    typer.echo(f"combine: {run_combine}")

    total_started = time.monotonic()
    summary: list[tuple[str, bool, float]] = run_shared_steps()

    for runner in runners:
        ok, elapsed = run_script(runner, stop_on_error=not keep_going)
        summary.append((runner.parent.name, ok, elapsed))

    if run_combine:
        label = f"combine/{combine_runner.stem}"
        ok, elapsed = run_script(
            combine_runner,
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
