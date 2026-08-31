import importlib.util
import io
import sys
import types
import unittest
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


if __name__ == "__main__":
    unittest.main()
