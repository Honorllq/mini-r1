import ast
import io
import runpy
import sys
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


TRAIN_SCRIPT = Path(__file__).parents[1] / "src" / "train.py"


def _stub_module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


class _FakeModel:
    def parameters(self):
        return ()


class _FakeTokenizer:
    pad_token = "pad"
    eos_token = "eos"


class _FakeDataset:
    def __init__(self, size: int = 164):
        self.size = size

    def select(self, indices):
        return _FakeDataset(len(list(indices)))

    def __len__(self):
        return self.size


class _FakeTrainer:
    def train(self):
        return None

    def save_model(self, output_dir):
        return None


def _run_train_cli(arguments: list[str]):
    model_loader = mock.Mock(return_value=_FakeModel())
    config_factory = mock.Mock(
        side_effect=lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    stubs = {
        "torch": _stub_module("torch", bfloat16=object()),
        "transformers": _stub_module(
            "transformers",
            AutoModelForCausalLM=types.SimpleNamespace(
                from_pretrained=model_loader
            ),
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=mock.Mock(return_value=_FakeTokenizer())
            ),
        ),
        "peft": _stub_module(
            "peft",
            LoraConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
        ),
        "trl": _stub_module(
            "trl",
            GRPOConfig=config_factory,
            GRPOTrainer=lambda **kwargs: _FakeTrainer(),
        ),
        "data_prep": _stub_module(
            "data_prep", load_humaneval=lambda split: _FakeDataset()
        ),
        "reward_funcs": _stub_module(
            "reward_funcs",
            code_reward_humaneval_partial=mock.Mock(),
            format_reward=mock.Mock(),
        ),
    }

    exit_code = None
    with mock.patch.object(sys, "argv", ["train.py", *arguments]):
        with mock.patch.object(sys, "path", sys.path.copy()):
            with mock.patch.dict(sys.modules, stubs):
                with redirect_stdout(io.StringIO()), redirect_stderr(
                    io.StringIO()
                ):
                    try:
                        runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")
                    except SystemExit as exc:
                        exit_code = exc.code
    return model_loader, config_factory, exit_code


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

    def test_grpo_config_explicitly_disables_kl_penalty(self):
        configs = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "GRPOConfig"
        ]

        self.assertEqual(len(configs), 1)
        values = {keyword.arg: keyword.value for keyword in configs[0].keywords}
        self.assertIn("beta", values)
        beta = ast.literal_eval(values["beta"])
        self.assertIs(type(beta), float)
        self.assertEqual(beta, 0.0)

    def test_cli_rejects_invalid_max_steps_before_model_load(self):
        for raw_value in ("0", "-2", "1.5", "not-an-int"):
            with self.subTest(raw_value=raw_value):
                model_loader, config_factory, exit_code = _run_train_cli(
                    ["--max_steps", raw_value]
                )
                self.assertEqual(exit_code, 2)
                model_loader.assert_not_called()
                config_factory.assert_not_called()

    def test_cli_preserves_debug_max_steps_behavior(self):
        cases = (
            ([], -1),
            (["--debug"], 2),
            (["--debug", "--max_steps", "-1"], 2),
            (["--debug", "--max_steps", "5"], 5),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                _, config_factory, exit_code = _run_train_cli(arguments)
                self.assertIsNone(exit_code)
                self.assertEqual(
                    config_factory.call_args.kwargs["max_steps"], expected
                )


if __name__ == "__main__":
    unittest.main()
