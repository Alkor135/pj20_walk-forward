"""Открывает HTML-отчёты бэктеста всех моделей текущего тикера в Google Chrome.

Скрипт работает в любой папке-тикере (`rts/`, `mix/`, `ng/`, ...) — путь к
тикеру определяется как папка самого скрипта. Внутри ищутся файлы вида
`<model>/plots/*.html` для всех моделей. Найденные отчёты передаются в
один процесс Chrome `--new-window`, поэтому они открываются в одном окне
по разным вкладкам.

Порядок отчётов: сначала по имени папки модели (алфавитно), внутри модели —
по имени HTML-файла. Если ни одного отчёта не найдено — выводится подсказка
и скрипт завершается с кодом 0 (не считаем это ошибкой).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

TICKER_DIR = Path(__file__).resolve().parent
CHROME_PATH = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")


def collect_html_reports(ticker_dir: Path) -> list[Path]:
    """Возвращает все HTML-отчёты вида <model>/plots/*.html в папке тикера."""
    reports: list[Path] = []
    for model_dir in sorted(p for p in ticker_dir.iterdir() if p.is_dir()):
        plots_dir = model_dir / "plots"
        if not plots_dir.is_dir():
            continue
        reports.extend(sorted(plots_dir.glob("*.html")))
    return reports


def open_reports_in_chrome(chrome_path: Path, reports: list[Path]) -> None:
    """Открывает список HTML-отчётов в одном новом окне Google Chrome."""
    if not chrome_path.exists():
        print(f"Google Chrome не найден: {chrome_path}")
        raise SystemExit(1)

    subprocess.Popen([str(chrome_path), "--new-window", *[str(p) for p in reports]])


def main() -> None:
    """Находит HTML-отчёты бэктеста и открывает их в новом окне Chrome."""
    reports = collect_html_reports(TICKER_DIR)

    if not reports:
        print(f"HTML-отчёты не найдены в {TICKER_DIR}/<model>/plots/.")
        print("Сначала прогони пайплайн: python run_<ticker>.py")
        raise SystemExit(0)

    open_reports_in_chrome(CHROME_PATH, reports)

    print(f"Открываю {len(reports)} HTML-отчётов из {TICKER_DIR}:")
    for path in reports:
        print(f"  [ОТКРЫВАЮ] {path.relative_to(TICKER_DIR)}")


if __name__ == "__main__":
    main()
