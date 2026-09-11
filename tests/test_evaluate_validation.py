import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from unittest import mock


EVALUATE_SCRIPT = Path(__file__).parents[1] / "src" / "evaluate.py"


def _stub_module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def _load_evaluate_module() -> tuple[types.ModuleType, mock.Mock]:
    model_loader = mock.Mock(
        side_effect=AssertionError("model must not load for invalid input")
    )
    stubs = {
        "torch": _stub_module("torch", bfloat16=object()),
        "tqdm": _stub_module("tqdm", tqdm=lambda values: values),
        "transformers": _stub_module(
            "transformers",
            AutoModelForCausalLM=types.SimpleNamespace(
                from_pretrained=model_loader
            ),
            AutoTokenizer=types.SimpleNamespace(from_pretrained=mock.Mock()),
        ),
        "data_prep": _stub_module("data_prep", load_humaneval=mock.Mock()),
        "reward_funcs": _stub_module("reward_funcs", extract_code=mock.Mock()),
        "local_sandbox": _stub_module(
            "local_sandbox", run_humaneval_test=mock.Mock()
        ),
    }

    spec = importlib.util.spec_from_file_location(
        "mini_r1_evaluate_under_test", EVALUATE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load evaluate.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.object(sys, "path", sys.path.copy()):
        with mock.patch.dict(sys.modules, stubs):
            spec.loader.exec_module(module)
    return module, model_loader


class TestEvaluateInputValidation(unittest.TestCase):
    def test_evaluate_rejects_non_positive_samples_before_model_load(self):
        module, model_loader = _load_evaluate_module()

        for num_samples in (0, -1):
            with self.subTest(num_samples=num_samples):
                with self.assertRaisesRegex(
                    ValueError, "num_samples must be greater than 0"
                ):
                    module.evaluate("unused-model", num_samples=num_samples)

        model_loader.assert_not_called()

    def test_evaluate_rejects_non_integer_samples_before_model_load(self):
        module, model_loader = _load_evaluate_module()

        for num_samples in (True, 1.5, None, "1"):
            with self.subTest(num_samples=num_samples):
                with self.assertRaisesRegex(
                    TypeError, "num_samples must be an integer"
                ):
                    module.evaluate("unused-model", num_samples=num_samples)

        model_loader.assert_not_called()

    def test_evaluate_rejects_non_positive_generation_lengths_before_model_load(self):
        module, model_loader = _load_evaluate_module()

        for max_new_tokens in (0, -1):
            with self.subTest(max_new_tokens=max_new_tokens):
                with self.assertRaisesRegex(
                    ValueError, "max_new_tokens must be greater than 0"
                ):
                    module.evaluate(
                        "unused-model", max_new_tokens=max_new_tokens
                    )

        model_loader.assert_not_called()

    def test_evaluate_rejects_non_integer_generation_lengths_before_model_load(self):
        module, model_loader = _load_evaluate_module()

        for max_new_tokens in (True, 1.5, None, "1"):
            with self.subTest(max_new_tokens=max_new_tokens):
                with self.assertRaisesRegex(
                    TypeError, "max_new_tokens must be an integer"
                ):
                    module.evaluate(
                        "unused-model", max_new_tokens=max_new_tokens
                    )

        model_loader.assert_not_called()

    def test_cli_rejects_invalid_sample_counts(self):
        module, _ = _load_evaluate_module()

        for num_samples in ("0", "-1", "1.5", "abc", ""):
            with self.subTest(num_samples=num_samples):
                with mock.patch.object(
                    sys, "argv", ["evaluate.py", "--num_samples", num_samples]
                ):
                    with mock.patch.object(module, "evaluate") as evaluate:
                        with mock.patch("sys.stderr", new=io.StringIO()):
                            with self.assertRaises(SystemExit) as raised:
                                module.main()
                self.assertEqual(raised.exception.code, 2)
                evaluate.assert_not_called()

    def test_cli_accepts_positive_samples(self):
        module, _ = _load_evaluate_module()

        with mock.patch.object(
            sys, "argv", ["evaluate.py", "--num_samples", "1"]
        ):
            with mock.patch.object(module, "evaluate") as evaluate:
                module.main()

        self.assertEqual(evaluate.call_args.kwargs["num_samples"], 1)

    def test_cli_rejects_invalid_generation_lengths(self):
        module, _ = _load_evaluate_module()

        for max_new_tokens in ("0", "-1", "1.5", "abc", ""):
            with self.subTest(max_new_tokens=max_new_tokens):
                with mock.patch.object(
                    sys,
                    "argv",
                    ["evaluate.py", "--max_new_tokens", max_new_tokens],
                ):
                    with mock.patch.object(module, "evaluate") as evaluate:
                        with mock.patch("sys.stderr", new=io.StringIO()):
                            with self.assertRaises(SystemExit) as raised:
                                module.main()
                self.assertEqual(raised.exception.code, 2)
                evaluate.assert_not_called()

    def test_cli_accepts_positive_generation_length(self):
        module, _ = _load_evaluate_module()

        with mock.patch.object(
            sys, "argv", ["evaluate.py", "--max_new_tokens", "1"]
        ):
            with mock.patch.object(module, "evaluate") as evaluate:
                module.main()

        self.assertEqual(evaluate.call_args.kwargs["max_new_tokens"], 1)


class _EvaluationInputs(dict[str, object]):
    def to(self, device: str) -> "_EvaluationInputs":
        self.device = device
        return self


class TestEvaluateRuntime(unittest.TestCase):
    """Exercise evaluation and real JSON I/O without model downloads or a GPU."""

    def setUp(self) -> None:
        self.module, self.model_loader = _load_evaluate_module()
        self.model = mock.Mock(device="base-device")
        self.model_loader.side_effect = None
        self.model_loader.return_value = self.model
        self.model.generate.side_effect = (
            [[101, 102, 201]], [[101, 102, 202]], [[101, 102, 203]]
        )
        self.module.torch.no_grad = mock.Mock(side_effect=nullcontext)

        self.inputs = _EvaluationInputs(
            input_ids=types.SimpleNamespace(shape=(1, 2))
        )
        self.tokenizer = mock.Mock(
            pad_token=None, eos_token="<eos>", pad_token_id=7,
            return_value=self.inputs,
        )
        self.tokenizer.apply_chat_template.return_value = "formatted prompt"
        self.responses = (
            "```python\nreturn '正确'\n```", "No code here", "```python\nreturn 0\n```"
        )
        self.codes = ("return '正确'\n", "", "return 0\n")
        self.tokenizer.decode.side_effect = self.responses
        self.module.AutoTokenizer.from_pretrained.return_value = self.tokenizer
        self.module.extract_code.side_effect = self.codes
        self.module.run_humaneval_test.side_effect = (True, False)

        self.rows = [
            {
                "task_id": f"HumanEval/{index}",
                "prompt": [{"role": "user", "content": f"task {index}"}],
                "verification_info": {
                    "test_code": f"test {index}", "entry_point": f"solve_{index}"
                },
            }
            for index in range(3)
        ]
        self.dataset = self.module.load_humaneval.return_value
        self.dataset.select.side_effect = lambda indices: [self.rows[i] for i in indices]
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output_dir = Path(temporary.name) / "nested" / "eval"

    def _evaluate(self, num_samples: int = 3, lora_path: str | None = None) -> dict:
        with redirect_stdout(io.StringIO()):
            summary = self.module.evaluate(
                "base-model", num_samples=num_samples, max_new_tokens=7,
                lora_path=lora_path, label="unit_test", output_dir=str(self.output_dir),
            )
        artifact = self.output_dir / "eval_unit_test.json"
        self.assertEqual(json.loads(artifact.read_text(encoding="utf-8")), summary)
        return summary

    def test_mixed_results_are_aggregated_and_saved_without_losing_samples(self) -> None:
        summary = self._evaluate()

        self.assertEqual(summary["num_samples"], 3)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["no_code"], 1)
        self.assertAlmostEqual(summary["pass_at_1"], 1 / 3)
        self.assertEqual(summary["model"], "base-model")
        self.assertIsNone(summary["lora_path"])
        self.assertEqual(summary["evaluation_scope"], "in_sample_same_tasks")
        self.assertEqual(
            summary["results"],
            [
                {
                    "task_id": f"HumanEval/{i}", "passed": i == 0,
                    "has_code": i != 1, "response": self.responses[i], "code": self.codes[i],
                }
                for i in range(3)
            ],
        )
        self.module.load_humaneval.assert_called_once_with("test")
        self.dataset.select.assert_called_once_with(range(3))
        self.model.eval.assert_called_once_with()
        self.assertEqual(self.inputs.device, "base-device")
        self.assertEqual(self.tokenizer.pad_token, "<eos>")
        self.assertEqual(
            self.model.generate.call_args_list,
            [mock.call(**self.inputs, max_new_tokens=7, do_sample=False, pad_token_id=7)] * 3,
        )
        self.assertEqual(
            self.tokenizer.apply_chat_template.call_args_list,
            [
                mock.call(row["prompt"], tokenize=False, add_generation_prompt=True)
                for row in self.rows
            ],
        )
        self.assertEqual(
            self.tokenizer.decode.call_args_list,
            [mock.call([token], skip_special_tokens=True) for token in (201, 202, 203)],
        )
        self.assertEqual(
            self.module.extract_code.call_args_list,
            [mock.call(response) for response in self.responses],
        )
        self.assertEqual(
            self.module.run_humaneval_test.call_args_list,
            [
                mock.call(code=self.codes[i], test_code=f"test {i}", entry_point=f"solve_{i}")
                for i in (0, 2)
            ],
        )

    def test_lora_model_is_used_and_existing_padding_is_preserved(self) -> None:
        adapter = mock.Mock(device="adapter-device")
        adapter.generate.return_value = [[101, 102, 201]]
        adapter_loader = mock.Mock(return_value=adapter)
        peft = _stub_module(
            "peft", PeftModel=types.SimpleNamespace(from_pretrained=adapter_loader)
        )
        self.tokenizer.pad_token = "<existing-pad>"

        with mock.patch.dict(sys.modules, {"peft": peft}):
            summary = self._evaluate(num_samples=1, lora_path="adapter-path")

        adapter_loader.assert_called_once_with(self.model, "adapter-path")
        adapter.eval.assert_called_once_with()
        adapter.generate.assert_called_once_with(
            **self.inputs, max_new_tokens=7, do_sample=False, pad_token_id=7
        )
        self.model.generate.assert_not_called()
        self.model.eval.assert_not_called()
        self.assertEqual(self.inputs.device, "adapter-device")
        self.assertEqual(self.tokenizer.pad_token, "<existing-pad>")
        self.assertEqual(summary["lora_path"], "adapter-path")
        self.assertEqual(summary["num_samples"], 1)
        self.assertEqual(summary["pass_at_1"], 1.0)
        self.assertEqual(len(summary["results"]), 1)


if __name__ == "__main__":
    unittest.main()
