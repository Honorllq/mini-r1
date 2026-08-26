import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
EVALUATE_SCRIPT = PROJECT_ROOT / "src" / "evaluate.py"
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "train.py"
DATA_PREP_SCRIPT = PROJECT_ROOT / "src" / "data_prep.py"


class TestEvaluationArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(EVALUATE_SCRIPT.read_text(encoding="utf-8"))
        result_appends = [
            node
            for node in ast.walk(cls.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "results"
            and node.func.attr == "append"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Dict)
        ]

        if len(result_appends) != 1:
            raise AssertionError("expected one results.append dictionary")

        record = result_appends[0].args[0]
        cls.fields = {
            ast.literal_eval(key): value
            for key, value in zip(record.keys, record.values)
        }

        summary_assignments = [
            node.value
            for node in ast.walk(cls.tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "summary"
            and isinstance(node.value, ast.Dict)
        ]
        if len(summary_assignments) != 1:
            raise AssertionError("expected one summary dictionary")

        summary = summary_assignments[0]
        cls.summary_fields = {
            ast.literal_eval(key): value
            for key, value in zip(summary.keys, summary.values)
        }

    def _assigned_value(self, name: str) -> ast.expr:
        assignments = [
            node.value
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ]
        self.assertEqual(len(assignments), 1)
        return assignments[0]

    def _single_string_call_argument(self, path: Path, name: str) -> str:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        arguments = [
            node.args[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ]
        self.assertEqual(len(arguments), 1)
        return ast.literal_eval(arguments[0])

    def test_evaluation_record_preserves_full_response(self):
        response = self.fields["response"]
        self.assertIsInstance(response, ast.Name)
        self.assertEqual(response.id, "response")

    def test_evaluation_record_stores_executed_code(self):
        code = self.fields["code"]
        self.assertIsInstance(code, ast.Name)
        self.assertEqual(code.id, "code")

    def test_saved_artifacts_follow_the_execution_path(self):
        response = self._assigned_value("response")
        self.assertIsInstance(response, ast.Call)
        self.assertIsInstance(response.func, ast.Attribute)
        self.assertEqual(response.func.attr, "decode")

        code = self._assigned_value("code")
        self.assertIsInstance(code, ast.Call)
        self.assertIsInstance(code.func, ast.Name)
        self.assertEqual(code.func.id, "extract_code")
        self.assertEqual(len(code.args), 1)
        self.assertIsInstance(code.args[0], ast.Name)
        self.assertEqual(code.args[0].id, "response")

        sandbox_calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_humaneval_test"
        ]
        self.assertEqual(len(sandbox_calls), 1)
        arguments = {
            keyword.arg: keyword.value
            for keyword in sandbox_calls[0].keywords
        }
        self.assertIsInstance(arguments["code"], ast.Name)
        self.assertEqual(arguments["code"].id, "code")

    def test_summary_labels_reused_training_tasks(self):
        expected_metadata = {
            "dataset": "openai/openai_humaneval",
            "split": "test",
            "evaluation_scope": "in_sample_same_tasks",
        }
        for field, expected in expected_metadata.items():
            self.assertIn(field, self.summary_fields)
            self.assertEqual(ast.literal_eval(self.summary_fields[field]), expected)

    def test_scope_metadata_matches_dataset_loaders(self):
        dataset = ast.literal_eval(self.summary_fields["dataset"])
        split = ast.literal_eval(self.summary_fields["split"])
        self.assertEqual(
            self._single_string_call_argument(DATA_PREP_SCRIPT, "load_dataset"),
            dataset,
        )
        self.assertEqual(
            self._single_string_call_argument(TRAIN_SCRIPT, "load_humaneval"),
            split,
        )
        self.assertEqual(
            self._single_string_call_argument(EVALUATE_SCRIPT, "load_humaneval"),
            split,
        )


if __name__ == "__main__":
    unittest.main()
