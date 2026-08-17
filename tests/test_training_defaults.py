import ast
import unittest
from pathlib import Path


TRAIN_SCRIPT = Path(__file__).parents[1] / "src" / "train.py"


class TestTrainingDefaults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(TRAIN_SCRIPT.read_text(encoding="utf-8"))

    def test_cli_defaults_to_best_v3_epoch_count(self):
        epoch_arguments = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--num_train_epochs"
        ]

        self.assertEqual(len(epoch_arguments), 1)
        defaults = {
            keyword.arg: keyword.value
            for keyword in epoch_arguments[0].keywords
        }
        self.assertIn("default", defaults)
        self.assertEqual(ast.literal_eval(defaults["default"]), 2)

    def test_grpo_config_uses_cli_epoch_count(self):
        configs = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "GRPOConfig"
        ]

        self.assertEqual(len(configs), 1)
        values = {keyword.arg: keyword.value for keyword in configs[0].keywords}
        epoch_value = values["num_train_epochs"]
        self.assertIsInstance(epoch_value, ast.Attribute)
        self.assertIsInstance(epoch_value.value, ast.Name)
        self.assertEqual(epoch_value.value.id, "args")
        self.assertEqual(epoch_value.attr, "num_train_epochs")


if __name__ == "__main__":
    unittest.main()
