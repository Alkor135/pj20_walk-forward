from __future__ import annotations

import sys
import unittest
import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "rts" / "gemma3_12b"
sys.path.insert(0, str(MODEL_DIR))

import sentiment_backtest as bt


BACKTEST_FILES = [
    ROOT / "rts" / "gemma3_12b" / "sentiment_backtest.py",
    ROOT / "rts" / "gemma4_26b" / "sentiment_backtest.py",
    ROOT / "rts" / "gemma4_31b" / "sentiment_backtest.py",
    ROOT / "rts" / "gemma4_e2b" / "sentiment_backtest.py",
    ROOT / "rts" / "gemma4_e4b" / "sentiment_backtest.py",
    ROOT / "rts" / "qwen2.5_14b" / "sentiment_backtest.py",
    ROOT / "rts" / "qwen2.5_7b" / "sentiment_backtest.py",
    ROOT / "rts" / "qwen3.6_35b" / "sentiment_backtest.py",
    ROOT / "rts" / "qwen3_14b" / "sentiment_backtest.py",
]


def import_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BacktestForecastTests(unittest.TestCase):
    def test_builds_next_month_forecast_html(self) -> None:
        result = pd.DataFrame(
            {
                "source_date": pd.date_range("2026-01-01", periods=30, freq="D"),
                "pnl": [
                    1000, -500, 700, 300, -200, 900, 400, -800, 1200, 600,
                    -300, 500, 800, -100, 1100, 200, -600, 750, 350, -250,
                    950, 450, -700, 1300, 650, -350, 550, 850, -150, 1150,
                ],
            }
        )

        html = bt.build_next_month_forecast_html(result, forecast_days=21, bootstrap_samples=5000)

        self.assertIn("Прогноз на следующий месяц", html)
        self.assertIn("95%", html)
        self.assertIn("Вероятность прибыли", html)
        self.assertIn("mean ± z × σ", html)
        self.assertIn("Бутстрэп", html)
        self.assertIn("background:#f7f7f7", html)
        self.assertIn("background:#ffffff", html)

    def test_all_model_backtests_expose_next_month_forecast_html(self) -> None:
        for i, path in enumerate(BACKTEST_FILES):
            with self.subTest(path=path):
                module = import_from_path(path, f"sentiment_backtest_{i}")
                html = module.build_next_month_forecast_html(
                    pd.DataFrame({"pnl": [1000, -500, 700, 300, -200, 900]}),
                    forecast_days=5,
                    bootstrap_samples=1000,
                )
                self.assertIn("Прогноз на следующий месяц", html)
                self.assertIn("Вероятность прибыли", html)

    def test_combine_report_exposes_next_month_forecast_html(self) -> None:
        module = import_from_path(ROOT / "rts" / "combine" / "sentiment_combine.py", "sentiment_combine")
        html = module.build_next_month_forecast_html(
            pd.DataFrame({"pnl": [1000, -500, 700, 300, -200, 900]}),
            forecast_days=5,
            bootstrap_samples=1000,
        )

        self.assertIn("Прогноз на следующий месяц", html)
        self.assertIn("Вероятность прибыли", html)


if __name__ == "__main__":
    unittest.main()
