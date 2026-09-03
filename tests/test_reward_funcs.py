import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


REWARD_FUNCS_SCRIPT = Path(__file__).parents[1] / "src" / "reward_funcs.py"
METADATA_REWARD_FUNCTIONS = (
    "code_reward",
    "code_reward_humaneval",
    "code_reward_humaneval_partial",
)


def _load_reward_funcs_module() -> tuple[types.ModuleType, dict[str, mock.Mock]]:
    sandbox_functions = {
        "compute_pass_rate": mock.Mock(),
        "run_humaneval_test": mock.Mock(),
        "compute_humaneval_pass_rate": mock.Mock(),
    }
    local_sandbox = types.ModuleType("local_sandbox")
    for name, function in sandbox_functions.items():
        setattr(local_sandbox, name, function)

    spec = importlib.util.spec_from_file_location(
        "mini_r1_reward_funcs_under_test", REWARD_FUNCS_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load reward_funcs.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.object(sys, "path", sys.path.copy()):
        with mock.patch.dict(sys.modules, {"local_sandbox": local_sandbox}):
            spec.loader.exec_module(module)
    return module, sandbox_functions


def _completion(code: str) -> list[dict[str, str]]:
    return [{"role": "assistant", "content": f"```python\n{code}```"}]


class TestRewardBatchAlignment(unittest.TestCase):
    def test_metadata_rewards_reject_misaligned_batches_before_sandbox(self):
        batch_sizes = ((2, 1), (1, 2))

        for function_name in METADATA_REWARD_FUNCTIONS:
            for completion_count, metadata_count in batch_sizes:
                with self.subTest(
                    function=function_name,
                    completions=completion_count,
                    metadata=metadata_count,
                ):
                    module, sandbox_functions = _load_reward_funcs_module()
                    reward_function = getattr(module, function_name)
                    completions = [_completion("pass") for _ in range(completion_count)]
                    verification_info = [
                        {
                            "test_cases": [],
                            "test_code": "def check(candidate): pass",
                            "entry_point": "candidate",
                        }
                        for _ in range(metadata_count)
                    ]

                    with self.assertRaisesRegex(
                        ValueError,
                        "completions and verification_info must have the same length",
                    ):
                        reward_function(
                            completions,
                            verification_info=verification_info,
                        )

                    for sandbox_function in sandbox_functions.values():
                        sandbox_function.assert_not_called()

    def test_code_reward_preserves_batch_order(self):
        module, sandbox_functions = _load_reward_funcs_module()
        compute_pass_rate = sandbox_functions["compute_pass_rate"]
        compute_pass_rate.side_effect = (0.25, 0.75)
        test_cases = ([{"input": "1", "output": "2"}], [{"input": "2", "output": "3"}])

        rewards = module.code_reward(
            [_completion("first()"), _completion("second()")],
            verification_info=[
                {"test_cases": test_cases[0]},
                {"test_cases": test_cases[1]},
            ],
        )

        self.assertEqual(rewards, [0.25, 0.75])
        self.assertEqual(
            compute_pass_rate.call_args_list,
            [mock.call("first()", test_cases[0]), mock.call("second()", test_cases[1])],
        )

    def test_binary_humaneval_reward_preserves_batch_order(self):
        module, sandbox_functions = _load_reward_funcs_module()
        run_humaneval_test = sandbox_functions["run_humaneval_test"]
        run_humaneval_test.side_effect = (True, False)
        verification_info = [
            {"test_code": "first test", "entry_point": "first"},
            {"test_code": "second test", "entry_point": "second"},
        ]

        rewards = module.code_reward_humaneval(
            [_completion("first()"), _completion("second()")],
            verification_info=verification_info,
        )

        self.assertEqual(rewards, [1.0, 0.0])
        self.assertEqual(
            run_humaneval_test.call_args_list,
            [
                mock.call(
                    code="first()", test_code="first test", entry_point="first"
                ),
                mock.call(
                    code="second()", test_code="second test", entry_point="second"
                ),
            ],
        )

    def test_partial_humaneval_reward_preserves_batch_order(self):
        module, sandbox_functions = _load_reward_funcs_module()
        compute_pass_rate = sandbox_functions["compute_humaneval_pass_rate"]
        compute_pass_rate.side_effect = (0.25, 0.75)
        verification_info = [
            {"test_code": "first test", "entry_point": "first"},
            {"test_code": "second test", "entry_point": "second"},
        ]

        rewards = module.code_reward_humaneval_partial(
            [_completion("first()"), _completion("second()")],
            verification_info=verification_info,
        )

        self.assertEqual(rewards, [0.25, 0.75])
        self.assertEqual(
            compute_pass_rate.call_args_list,
            [
                mock.call(
                    code="first()", test_code="first test", entry_point="first"
                ),
                mock.call(
                    code="second()", test_code="second test", entry_point="second"
                ),
            ],
        )

    def test_metadata_rewards_accept_empty_batches(self):
        module, sandbox_functions = _load_reward_funcs_module()

        for function_name in METADATA_REWARD_FUNCTIONS:
            with self.subTest(function=function_name):
                reward_function = getattr(module, function_name)
                self.assertEqual(
                    reward_function([], verification_info=[]),
                    [],
                )

        for sandbox_function in sandbox_functions.values():
            sandbox_function.assert_not_called()


class TestFormatReward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, _ = _load_reward_funcs_module()

    @staticmethod
    def _completion_with_text(text: str) -> list[dict[str, str]]:
        return [{"role": "assistant", "content": text}]

    def assert_format_score(self, text: str, expected: float):
        rewards = self.module.format_reward([self._completion_with_text(text)])
        self.assertEqual(rewards, [expected])

    def test_accepts_one_canonical_block_with_optional_outer_whitespace(self):
        valid_outputs = (
            "<reasoning>work</reasoning><answer>result</answer>",
            " \n\t<reasoning>work</reasoning><answer>result</answer>\r\n ",
        )

        for output in valid_outputs:
            with self.subTest(output=output):
                self.assert_format_score(output, 0.5)

    def test_preserves_reward_for_empty_sections(self):
        self.assert_format_score(
            "<reasoning></reasoning><answer></answer>",
            0.5,
        )

    def test_rejects_non_whitespace_outside_or_multiple_complete_blocks(self):
        invalid_outputs = (
            "prefix<reasoning>work</reasoning><answer>result</answer>",
            "<reasoning>work</reasoning><answer>result</answer>suffix",
            (
                "<reasoning>first</reasoning><answer>one</answer>"
                "<reasoning>second</reasoning><answer>two</answer>"
            ),
        )

        for output in invalid_outputs:
            with self.subTest(output=output):
                self.assert_format_score(output, 0.0)

    def test_rejects_nested_or_crossed_structural_tags(self):
        invalid_outputs = (
            (
                "<reasoning>outer <reasoning>inner</reasoning></reasoning>"
                "<answer>result</answer>"
            ),
            (
                "<reasoning>work</reasoning>"
                "<answer>outer <answer>inner</answer></answer>"
            ),
            (
                "<reasoning>work <answer>crossed</reasoning>"
                "<answer>result</answer>"
            ),
            (
                "<reasoning>work</reasoning>"
                "<answer>result </reasoning> crossed</answer>"
            ),
            (
                "<reasoning>work <answer >nested</answer > continued</reasoning>"
                "<answer>result</answer>"
            ),
            (
                "<reasoning>work</reasoning>"
                "<answer>result <answer\n>nested</answer\n></answer>"
            ),
            (
                "<reasoning>work</reasoning>"
                "<answer><answer data-x='1'>nested</answer>"
            ),
        )

        for output in invalid_outputs:
            with self.subTest(output=output):
                self.assert_format_score(output, 0.0)


if __name__ == "__main__":
    unittest.main()
