"""
Строит рекомендованный rules.yaml из sentiment_group_stats.xlsx.

Скрипт читает настройки из единого `rts/settings.yaml`, загружает
`group_stats/sentiment_group_stats.xlsx` и для каждого значения sentiment
от -10 до +10 определяет рекомендованное действие:
- `follow`, если `total_pnl > 0`;
- `invert`, если `total_pnl < 0`;
- если `total_pnl == 0`, ищет ближайших соседей по sentiment с ненулевым
  `total_pnl`;
- если на одинаковом расстоянии есть оба соседа, выбирает того, у кого
  `abs(total_pnl)` больше;
- при полном равенстве переходит к следующим соседям.

Результат сохраняется в `rules.yaml` рядом со скриптом.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import typer
import yaml

TICKER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TICKER_DIR.parent))
from config_loader import load_settings_for
SENTIMENT_RANGE = range(-10, 11)

app = typer.Typer(help="Строит рекомендованный YAML с правилами по group_stats XLSX.")


def load_settings() -> dict:
    """Загружает настройки модели из единого {ticker}/settings.yaml."""
    return load_settings_for(__file__, "model")


def resolve_group_stats_input_xlsx(settings: dict, output_dir: Path) -> Path:
    """Возвращает путь к входному XLSX-файлу с групповой статистикой."""
    filename = str(settings.get("group_stats_output_xlsx", "sentiment_group_stats.xlsx"))
    return output_dir / filename


def resolve_rules_output_yaml(script_dir: Path) -> Path:
    """Возвращает путь к rules.yaml рядом со скриптом."""
    return script_dir / "rules.yaml"


def load_group_stats(path: Path) -> pd.DataFrame:
    """Загружает XLSX и проверяет обязательные колонки и диапазон sentiment."""
    if not path.exists():
        raise typer.BadParameter(f"Файл групповой статистики не найден: {path}")

    df = pd.read_excel(path)
    required = {"sentiment", "total_pnl"}
    missing = required - set(df.columns)
    if missing:
        raise typer.BadParameter(
            f"XLSX не содержит обязательные колонки: {sorted(missing)}"
        )

    df = df.copy()
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    df["total_pnl"] = pd.to_numeric(df["total_pnl"], errors="coerce")
    df = df.dropna(subset=["sentiment", "total_pnl"])
    df["sentiment"] = df["sentiment"].astype(int)

    duplicates = df["sentiment"].duplicated(keep=False)
    if duplicates.any():
        values = sorted(df.loc[duplicates, "sentiment"].unique().tolist())
        raise typer.BadParameter(f"В XLSX повторяются значения sentiment: {values}")

    by_sentiment = df.set_index("sentiment")["total_pnl"]
    missing_sentiments = [sentiment for sentiment in SENTIMENT_RANGE if sentiment not in by_sentiment.index]
    if missing_sentiments:
        raise typer.BadParameter(
            f"В XLSX отсутствуют значения sentiment: {missing_sentiments}"
        )

    return df.sort_values("sentiment").reset_index(drop=True)


def _action_from_total_pnl(total_pnl: float) -> str:
    """Преобразует знак total_pnl в follow/invert."""
    return "follow" if total_pnl > 0 else "invert"


def recommend_action(total_pnl_by_sentiment: pd.Series, sentiment: int) -> str:
    """Возвращает action для заданного sentiment по total_pnl и соседям."""
    total_pnl = float(total_pnl_by_sentiment.loc[sentiment])
    if total_pnl > 0:
        return "follow"
    if total_pnl < 0:
        return "invert"

    for distance in range(1, len(SENTIMENT_RANGE)):
        left_sentiment = sentiment - distance
        right_sentiment = sentiment + distance

        left_value = None
        right_value = None

        if left_sentiment in total_pnl_by_sentiment.index:
            candidate = float(total_pnl_by_sentiment.loc[left_sentiment])
            if candidate != 0:
                left_value = candidate

        if right_sentiment in total_pnl_by_sentiment.index:
            candidate = float(total_pnl_by_sentiment.loc[right_sentiment])
            if candidate != 0:
                right_value = candidate

        if left_value is None and right_value is None:
            continue
        if left_value is None:
            return _action_from_total_pnl(right_value)
        if right_value is None:
            return _action_from_total_pnl(left_value)
        if abs(left_value) > abs(right_value):
            return _action_from_total_pnl(left_value)
        if abs(right_value) > abs(left_value):
            return _action_from_total_pnl(right_value)
        continue

    raise typer.BadParameter(
        "Невозможно определить рекомендацию: все значения total_pnl равны 0."
    )


def build_rules_recommendation(grouped: pd.DataFrame) -> list[dict[str, int | str]]:
    """Строит список правил по одному правилу на каждый sentiment."""
    total_pnl_by_sentiment = grouped.set_index("sentiment")["total_pnl"]
    return [
        {
            "min": sentiment,
            "max": sentiment,
            "action": recommend_action(total_pnl_by_sentiment, sentiment),
        }
        for sentiment in SENTIMENT_RANGE
    ]


def render_rules_yaml(
    rules: list[dict[str, int | str]], ticker: str, sentiment_model: str
) -> str:
    """Рендерит рекомендации в компактный YAML-формат."""
    lines = [f"rules:  # {ticker} {sentiment_model}"]
    for rule in rules:
        lines.append(
            f"  - {{min: {rule['min']}, max: {rule['max']}, action: {rule['action']}}}"
        )
    return "\n".join(lines) + "\n"


@app.command()
def main() -> None:
    """Строит YAML-рекомендации на основе уже сохраненного group_stats XLSX."""
    settings = load_settings()
    script_dir = Path(__file__).resolve().parent
    group_stats_dir = script_dir / "group_stats"
    input_xlsx = resolve_group_stats_input_xlsx(settings, group_stats_dir)
    output_yaml = resolve_rules_output_yaml(script_dir)

    grouped = load_group_stats(input_xlsx)
    rules = build_rules_recommendation(grouped)

    ticker = str(settings.get("ticker", ""))
    sentiment_model = str(settings.get("sentiment_model", ""))
    output_yaml.write_text(
        render_rules_yaml(rules, ticker, sentiment_model), encoding="utf-8"
    )

    typer.echo(f"XLSX прочитан: {input_xlsx}")
    typer.echo(f"YAML сохранён: {output_yaml}")


if __name__ == "__main__":
    app()
