from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "rts" / "gemma3_12b"
sys.path.insert(0, str(MODEL_DIR))

import sentiment_walk_forward as wf


class WalkForwardTests(unittest.TestCase):
    def test_uses_previous_three_months_to_trade_next_day(self) -> None:
        aggregated = pd.DataFrame(
            [
                {"source_date": pd.to_datetime("2025-01-01").date(), "sentiment": 3, "next_body": 10},
                {"source_date": pd.to_datetime("2025-01-02").date(), "sentiment": 3, "next_body": 20},
                {"source_date": pd.to_datetime("2025-04-01").date(), "sentiment": 3, "next_body": 5},
            ]
        ).set_index("source_date")

        tmp_path = ROOT / "test_rules_tmp.yaml"
        try:
            result, folds = wf.run_walk_forward(
                aggregated=aggregated,
                quantity=1,
                train_months=3,
                date_from=pd.to_datetime("2025-01-01").date(),
                date_to=pd.to_datetime("2025-04-01").date(),
                rules_tmp_path=tmp_path,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        self.assertEqual(result["source_date"].tolist(), [pd.to_datetime("2025-04-01").date()])
        self.assertEqual(result["pnl"].tolist(), [5.0])
        self.assertEqual(result["action"].tolist(), ["follow"])
        self.assertEqual(folds["train_date_from"].tolist(), [pd.to_datetime("2025-01-01").date()])
        self.assertEqual(folds["train_date_to"].tolist(), [pd.to_datetime("2025-01-02").date()])
        self.assertEqual(folds["test_date"].tolist(), [pd.to_datetime("2025-04-01").date()])

    def test_date_from_limits_test_days_not_training_history(self) -> None:
        aggregated = pd.DataFrame(
            [
                {"source_date": pd.to_datetime("2025-01-01").date(), "sentiment": 3, "next_body": 10},
                {"source_date": pd.to_datetime("2025-01-02").date(), "sentiment": 3, "next_body": 20},
                {"source_date": pd.to_datetime("2025-04-01").date(), "sentiment": 3, "next_body": 5},
            ]
        ).set_index("source_date")

        tmp_path = ROOT / "test_rules_tmp.yaml"
        try:
            result, folds = wf.run_walk_forward(
                aggregated=aggregated,
                quantity=1,
                train_months=3,
                date_from=pd.to_datetime("2025-04-01").date(),
                date_to=pd.to_datetime("2025-04-01").date(),
                rules_tmp_path=tmp_path,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        self.assertEqual(result["source_date"].tolist(), [pd.to_datetime("2025-04-01").date()])
        self.assertEqual(folds["train_trades"].tolist(), [2])

    def test_test_days_groups_oos_days_without_overlap(self) -> None:
        aggregated = pd.DataFrame(
            [
                {"source_date": pd.to_datetime("2025-01-01").date(), "sentiment": 3, "next_body": 10},
                {"source_date": pd.to_datetime("2025-01-02").date(), "sentiment": 3, "next_body": 20},
                {"source_date": pd.to_datetime("2025-04-01").date(), "sentiment": 3, "next_body": 5},
                {"source_date": pd.to_datetime("2025-04-02").date(), "sentiment": 3, "next_body": 6},
                {"source_date": pd.to_datetime("2025-04-03").date(), "sentiment": 3, "next_body": 7},
            ]
        ).set_index("source_date")

        tmp_path = ROOT / "test_rules_tmp.yaml"
        try:
            result, folds = wf.run_walk_forward(
                aggregated=aggregated,
                quantity=1,
                train_months=3,
                test_days=2,
                date_from=pd.to_datetime("2025-01-01").date(),
                date_to=pd.to_datetime("2025-04-03").date(),
                rules_tmp_path=tmp_path,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        self.assertEqual(
            result["source_date"].tolist(),
            [
                pd.to_datetime("2025-04-01").date(),
                pd.to_datetime("2025-04-02").date(),
                pd.to_datetime("2025-04-03").date(),
            ],
        )
        self.assertEqual(folds["test_trades"].tolist(), [2, 1])
        self.assertEqual(folds["test_date"].tolist(), [pd.to_datetime("2025-04-01").date(), pd.to_datetime("2025-04-03").date()])
        self.assertEqual(folds["test_date_to"].tolist(), [pd.to_datetime("2025-04-02").date(), pd.to_datetime("2025-04-03").date()])

    def test_rewrites_only_tmp_rules_file(self) -> None:
        rules = [{"min": -10, "max": 10, "action": "follow"}]

        tmp_path = ROOT / "test_rules_tmp.yaml"
        main_rules_path = ROOT / "test_rules.yaml"
        try:
            main_rules_path.write_text("rules:\n  - {min: 0, max: 0, action: invert}\n", encoding="utf-8")

            wf.write_rules_tmp(
                rules=rules,
                ticker="RTS",
                sentiment_model="gemma3:12b",
                output_path=tmp_path,
            )

            saved_tmp = yaml.safe_load(tmp_path.read_text(encoding="utf-8"))
            saved_main = yaml.safe_load(main_rules_path.read_text(encoding="utf-8"))
        finally:
            tmp_path.unlink(missing_ok=True)
            main_rules_path.unlink(missing_ok=True)

        self.assertEqual(saved_tmp["rules"], rules)
        self.assertEqual(saved_main["rules"][0]["action"], "invert")


if __name__ == "__main__":
    unittest.main()
