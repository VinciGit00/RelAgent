"""Small regression tests; no API calls or benchmark datasets required."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from camel.toolkits import FunctionTool
from relagent.main import _infer_task_type
from relagent.scientist.agent import ScientistAgent
from relagent.scientist.eval_workspace import EvalWorkspace
from relagent.scientist.tools import make_validate_program_wrapped_tool
from relagent.scientist.usage import TokenBudgetExhausted, track_model_usage
from relagent.scientist.validation import (
    _determine_primary_metric_name,
    _instantiate_model,
    _primary_score,
)
from relbench.base import TaskType


class RuntimeContractTests(unittest.TestCase):
    def test_invalid_program_can_be_repaired_but_attempts_remain_bounded(self):
        tool = make_validate_program_wrapped_tool(
            conn=Mock(),
            train_df=None,
            val_df=None,
            evaluator=Mock(),
            task_type="entity_classification",
            entity_col="id",
            target_col="label",
            trial_logger=Mock(),
            max_trials=1,
        )
        bad = SimpleNamespace(error="invalid SQL", score=float("-inf"))
        good = SimpleNamespace(error=None, score=0.8)
        query = '[{"name":"x","sql":"SELECT id FROM eval_table"}]'
        with (
            patch(
                "relagent.scientist.tools.execute_and_validate",
                side_effect=[bad, bad, good],
            ) as evaluate,
            patch(
                "relagent.scientist.tools._format_validation_result",
                return_value="feedback",
            ),
        ):
            for _ in range(3):
                self.assertEqual(tool.func(query, "lightgbm"), "feedback")
            self.assertIn("limit reached", tool.func(query, "lightgbm"))
            self.assertEqual(evaluate.call_count, 3)

    def test_relbench3_nmae_is_a_finite_minimization_score(self):
        metrics = {"nmae": 0.42}
        self.assertEqual(_primary_score("entity_regression", metrics), -0.42)
        self.assertEqual(
            _determine_primary_metric_name("entity_regression", metrics), "nmae"
        )
        self.assertEqual(_primary_score("entity_regression", {"mae": 2}), -2)
        with self.assertRaises(ValueError):
            _primary_score("entity_regression", {"nmae": float("nan")})

    def test_relbench_task_types_do_not_depend_on_old_import_path(self):
        for task_type, expected in (
            (TaskType.REGRESSION, "entity_regression"),
            (TaskType.BINARY_CLASSIFICATION, "entity_classification"),
        ):
            self.assertEqual(
                _infer_task_type(
                    SimpleNamespace(task=SimpleNamespace(task_type=task_type))
                ),
                expected,
            )
        with self.assertRaises(ValueError):
            _infer_task_type(SimpleNamespace(task=SimpleNamespace(task_type="unknown")))

    def test_usage_is_live_and_budget_stops_next_request(self):
        with tempfile.TemporaryDirectory() as directory:
            request = Mock(
                return_value=SimpleNamespace(usage=SimpleNamespace(total_tokens=12))
            )
            model = SimpleNamespace(run=request)
            track_model_usage(model, Path(directory), token_budget=10)
            model.run([])
            with self.assertRaises(TokenBudgetExhausted):
                model.run([])
            self.assertEqual(request.call_count, 1)
            self.assertEqual(
                json.loads((Path(directory) / "usage.json").read_text())["tokens"], 12
            )

    def test_budget_fails_closed_if_provider_omits_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            model = SimpleNamespace(run=lambda *args: SimpleNamespace(usage=None))
            track_model_usage(model, Path(directory), token_budget=10)
            with self.assertRaisesRegex(RuntimeError, "omitted token usage"):
                model.run([])

    def test_workspace_rejects_mutations_and_multiple_statements(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = EvalWorkspace(Path(directory) / "eval.db")
            try:
                for sql in (
                    "DELETE FROM trials",
                    "SELECT 1; DROP TABLE trials",
                    "COPY trials TO '/tmp/forbidden.csv'",
                ):
                    with self.assertRaises(ValueError):
                        workspace.execute_query(sql)
                self.assertEqual(
                    len(workspace.execute_query("SELECT * FROM range(1000)")), 500
                )
                self.assertEqual(
                    len(workspace.execute_query("SELECT * FROM trials")), 0
                )
            finally:
                workspace.close()

    def test_sql_error_is_feedback_and_never_routes_by_literal(self):
        def execute_query(query: str) -> str:
            """Execute SQL.

            Args:
                query: SQL query.
            """
            if "bad_alias" in query:
                raise ValueError('Referenced table "e" not found')
            return query

        agent = ScientistAgent.__new__(ScientistAgent)
        agent.tool_output_limit = 5000
        agent.eval_workspace = Mock()
        tool = agent._wrap_sql_tool(FunctionTool(execute_query))
        self.assertIn("Repair this query", tool.func("SELECT bad_alias"))
        self.assertIn("eval_predictions", tool.func("SELECT 'eval_predictions'"))
        agent.eval_workspace.execute_query.assert_not_called()

    def test_classification_does_not_inherit_regression_objective(self):
        for choice in ("lightgbm", "gbdt"):
            model = _instantiate_model(
                choice, {"objective": "regression_l1"}, "entity_classification"
            )
            self.assertIsNone(model.get_params()["objective"])
            self.assertEqual(model.get_params()["n_jobs"], 1)
        model = _instantiate_model(
            "gbdt", {"objective": "regression_l1"}, "entity_regression"
        )
        self.assertEqual(model.get_params()["objective"], "regression_l1")

    def test_trial_limit_prevents_evaluation(self):
        connection = Mock()
        tool = make_validate_program_wrapped_tool(
            conn=connection,
            train_df=None,
            val_df=None,
            evaluator=Mock(),
            task_type="entity_classification",
            entity_col="id",
            target_col="label",
            trial_logger=Mock(),
            max_trials=0,
        )
        result = tool.func(
            '[{"name":"x","sql":"SELECT id FROM eval_table"}]', "lightgbm"
        )
        self.assertIn("limit reached", result)
        connection.execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
