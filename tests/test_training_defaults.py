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


def _run_train_cli(
    arguments: list[str],
    *,
    tokenizer: _FakeTokenizer | None = None,
    trainer: mock.Mock | None = None,
):
    model_loader = mock.Mock(return_value=_FakeModel())
    config_factory = mock.Mock(
        side_effect=lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    dataset = _FakeDataset()
    dataset.select = mock.Mock(wraps=dataset.select)
    runtime = {
        "tokenizer_loader": mock.Mock(
            return_value=tokenizer if tokenizer is not None else _FakeTokenizer()
        ),
        "dataset_loader": mock.Mock(return_value=dataset),
        "lora_factory": mock.Mock(
            side_effect=lambda **kwargs: types.SimpleNamespace(**kwargs)
        ),
        "trainer_factory": mock.Mock(
            return_value=trainer if trainer is not None else mock.Mock(spec=_FakeTrainer)
        ),
        "partial_reward": mock.Mock(),
        "format_reward": mock.Mock(),
    }
    stubs = {
        "torch": _stub_module("torch", bfloat16=mock.sentinel.bfloat16),
        "transformers": _stub_module(
            "transformers",
            AutoModelForCausalLM=types.SimpleNamespace(
                from_pretrained=model_loader
            ),
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=runtime["tokenizer_loader"]
            ),
        ),
        "peft": _stub_module(
            "peft",
            LoraConfig=runtime["lora_factory"],
        ),
        "trl": _stub_module(
            "trl",
            GRPOConfig=config_factory,
            GRPOTrainer=runtime["trainer_factory"],
        ),
        "data_prep": _stub_module(
            "data_prep", load_humaneval=runtime["dataset_loader"]
        ),
        "reward_funcs": _stub_module(
            "reward_funcs",
            code_reward_humaneval_partial=runtime["partial_reward"],
            format_reward=runtime["format_reward"],
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
    return model_loader, config_factory, exit_code, runtime


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
                model_loader, config_factory, exit_code, _ = _run_train_cli(
                    ["--max_steps", raw_value]
                )
                self.assertEqual(exit_code, 2)
                model_loader.assert_not_called()
                config_factory.assert_not_called()

    def test_cli_rejects_invalid_epoch_counts_before_model_load(self) -> None:
        for raw_value in ("0", "-1", "-2", "1.5", "not-an-int"):
            with self.subTest(raw_value=raw_value):
                model_loader, config_factory, exit_code, _ = _run_train_cli(
                    ["--num_train_epochs", raw_value]
                )
                self.assertEqual(exit_code, 2)
                model_loader.assert_not_called()
                config_factory.assert_not_called()

    def test_cli_preserves_valid_epoch_counts_and_step_limits(self) -> None:
        cases = (
            ([], 2, -1),
            (["--num_train_epochs", "1"], 1, -1),
            (["--num_train_epochs", "3"], 3, -1),
            (["--debug", "--num_train_epochs", "3"], 3, 2),
            (["--num_train_epochs", "3", "--max_steps", "5"], 3, 5),
        )
        for arguments, expected_epochs, expected_steps in cases:
            with self.subTest(arguments=arguments):
                model_loader, config_factory, exit_code, _ = _run_train_cli(arguments)
                self.assertIsNone(exit_code)
                model_loader.assert_called_once()
                self.assertEqual(
                    config_factory.call_args.kwargs["num_train_epochs"], expected_epochs
                )
                self.assertEqual(
                    config_factory.call_args.kwargs["max_steps"], expected_steps
                )

    def test_cli_preserves_debug_max_steps_behavior(self):
        cases = (
            ([], -1),
            (["--debug"], 2),
            (["--debug", "--max_steps", "-1"], 2),
            (["--debug", "--max_steps", "5"], 5),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                _, config_factory, exit_code, _ = _run_train_cli(arguments)
                self.assertIsNone(exit_code)
                self.assertEqual(
                    config_factory.call_args.kwargs["max_steps"], expected
                )


class TestTrainingRuntime(unittest.TestCase):
    def test_full_training_wires_resources_and_saves_after_training(self) -> None:
        model_name = "example/custom-model"
        output_dir = "outputs/custom-adapter"
        model_loader, config_factory, exit_code, runtime = _run_train_cli(
            ["--model_name", model_name, "--output_dir", output_dir]
        )
        self.assertIsNone(exit_code)
        model_loader.assert_called_once_with(
            model_name, torch_dtype=mock.sentinel.bfloat16, device_map="auto"
        )
        runtime["tokenizer_loader"].assert_called_once_with(model_name)
        runtime["dataset_loader"].assert_called_once_with("test")
        dataset = runtime["dataset_loader"].return_value
        dataset.select.assert_not_called()
        runtime["lora_factory"].assert_called_once_with(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )
        runtime["trainer_factory"].assert_called_once()
        trainer_args = runtime["trainer_factory"].call_args.kwargs
        self.assertIs(trainer_args["model"], model_loader.return_value)
        self.assertIs(
            trainer_args["processing_class"], runtime["tokenizer_loader"].return_value
        )
        self.assertIs(trainer_args["train_dataset"], dataset)
        self.assertEqual(
            trainer_args["reward_funcs"],
            [runtime["partial_reward"], runtime["format_reward"]],
        )
        self.assertEqual(
            vars(trainer_args["args"]), config_factory.call_args.kwargs
        )
        self.assertEqual(trainer_args["args"].output_dir, output_dir)
        self.assertEqual(
            vars(trainer_args["peft_config"]), runtime["lora_factory"].call_args.kwargs
        )
        trainer = runtime["trainer_factory"].return_value
        self.assertEqual(
            trainer.mock_calls, [mock.call.train(), mock.call.save_model(output_dir)]
        )

    def test_debug_passes_the_first_four_tasks_to_trainer(self) -> None:
        _, _, exit_code, runtime = _run_train_cli(["--debug"])
        self.assertIsNone(exit_code)
        dataset = runtime["dataset_loader"].return_value
        dataset.select.assert_called_once()
        self.assertEqual(list(dataset.select.call_args.args[0]), [0, 1, 2, 3])
        trainer_args = runtime["trainer_factory"].call_args.kwargs
        self.assertIsNot(trainer_args["train_dataset"], dataset)
        self.assertEqual(len(trainer_args["train_dataset"]), 4)
        self.assertEqual(trainer_args["args"].max_steps, 2)

    def test_tokenizer_padding_fallback_is_passed_to_trainer(self) -> None:
        for pad_token in (None, "custom-pad"):
            with self.subTest(pad_token=pad_token):
                tokenizer = _FakeTokenizer()
                tokenizer.pad_token = pad_token
                _, _, exit_code, runtime = _run_train_cli([], tokenizer=tokenizer)
                self.assertIsNone(exit_code)
                trainer_tokenizer = runtime["trainer_factory"].call_args.kwargs[
                    "processing_class"
                ]
                self.assertIs(trainer_tokenizer, tokenizer)
                self.assertEqual(
                    trainer_tokenizer.pad_token,
                    tokenizer.eos_token if pad_token is None else pad_token,
                )

    def test_training_failure_propagates_without_saving(self) -> None:
        trainer = mock.Mock(spec=_FakeTrainer)
        trainer.train.side_effect = RuntimeError("training failed")
        with self.assertRaisesRegex(RuntimeError, "training failed"):
            _run_train_cli([], trainer=trainer)
        trainer.train.assert_called_once_with()
        trainer.save_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
